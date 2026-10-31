"""
БЛОК 3 — ФИЗИКА МОНТЕ-КАРЛО (с массивами и Numba).

Ядро симуляции нейтронного транспорта:
  - выборка длин пробега          (sample_distances_batch)
  - выборка типов реакций         (sample_reactions_batch)
  - генерация направлений рассеяния (scatter_directions)
  - транспорт партии нейтронов   (transport_batch)
  - оркестрация поколения        (run_generation_batch)
  - отношение воспроизводства источника по поколениям (estimate_k_full)

Соответствует БЛОКУ 3 исходного монолита NSPM MCNP 3000.py.
Модель транспорта сохранена; статистика учитывает всё поколение отдельно от выборки путей.
"""

import numpy as np
import sys
from dataclasses import asdict
from enum import IntEnum
from threading import Lock
from numbers import Real
from time import perf_counter

class HistoryStatus(IntEnum):
    MAX_STEPS_REACHED = -1
    FISSION = 0
    ABSORPTION = 1
    LEAK = 2

class SimulationCancelled(Exception):
    """Cooperative cancellation; no completed result is returned."""

_rng_lock = Lock()
DEFAULT_MAX_MEMORY_BYTES = 512 * 2**20


def _check_memory(n, samples, steps, budget, secondary=0, retained=0):
    """Conservative array/storage planning guard, not a process-RSS limit."""
    if budget is None:
        return
    _integer("max_memory_bytes", budget)
    # Includes input/output ownership, boxed events, work arrays and both
    # scratch/trimmed paths. Python, JIT and allocator overhead are not bounded.
    required = int(n) * 512 + min(int(n), int(samples)) * (int(steps) + 1) * 48
    required += int(secondary) * 128 + int(retained)
    if required > budget:
        raise MemoryError(f"Simulation storage estimate {required} bytes exceeds "
                          f"max_memory_bytes={budget}; reduce populations/samples/steps "
                          "or explicitly increase the budget")

def _integer(name, value, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")

def _positive(name, value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")

def _arrays(positions, directions):
    positions = np.asarray(positions, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or directions.shape != positions.shape:
        raise ValueError("positions and directions must have matching shape (N, 3)")
    if not np.isfinite(positions).all() or not np.isfinite(directions).all():
        raise ValueError("particle arrays must contain finite values")
    if len(directions) and not np.allclose(np.linalg.norm(directions, axis=1), 1.0):
        raise ValueError("directions must be unit vectors")
    return positions, directions


from numba_compat import njit, HAS_NUMBA
from material import MATERIALS, resolve_transport_data, resolve_medium_rates


@njit
def _sample_distances_batch(mfp, size):
    """Генерирует расстояния для партии нейтронов (Numba-ускорено)"""
    if not np.isfinite(mfp) or mfp <= 0 or size < 0:
        raise ValueError("mfp must be finite and positive; size must be nonnegative")
    return -mfp * np.log(np.random.random(size) + 1e-15)

@njit
def _sample_reactions_batch(sigma_f, sigma_a, sigma_s, size):
    """Генерирует типы реакций для партии: 0=fission, 1=absorption, 2=scatter"""
    sigma_t = sigma_f + sigma_a + sigma_s
    if (not np.isfinite(sigma_t) or sigma_f < 0 or sigma_a < 0 or sigma_s < 0
            or sigma_t <= 0 or size < 0):
        raise ValueError("cross sections must be finite and nonnegative with positive total")
    p_f = sigma_f / sigma_t
    p_a = sigma_a / sigma_t

    xi = np.random.random(size)
    reactions = np.zeros(size, dtype=np.int32)

    for i in range(size):
        if xi[i] < p_f:
            reactions[i] = 0
        elif xi[i] < p_f + p_a:
            reactions[i] = 1
        else:
            reactions[i] = 2
    return reactions

@njit
def _scatter_directions(directions, size):
    """Генерирует новые направления после рассеяния (Numba-ускорено)"""
    if directions.ndim != 2 or directions.shape[1] != 3 or size < 0 or size > len(directions):
        raise ValueError("directions must have shape (N, 3), with 0 <= size <= N")
    if not np.isfinite(directions).all():
        raise ValueError("directions must be finite")
    for j in range(size):
        if abs(np.sum(directions[j] ** 2) - 1.0) > 1e-5:
            raise ValueError("directions must be unit vectors")
    new_dirs = directions.copy()

    for i in range(size):
        mu = 1.0 - 2.0 * np.random.random()
        phi = 2.0 * np.pi * np.random.random()

        sin_theta = np.sqrt(max(0.0, 1.0 - mu**2))
        u, v, w = directions[i]

        if abs(w) < 0.9999:
            denom = np.sqrt(1.0 - w**2)
            new_dir = np.array([
                u*mu + sin_theta*(u*w*np.cos(phi) - v*np.sin(phi)) / denom,
                v*mu + sin_theta*(v*w*np.cos(phi) + u*np.sin(phi)) / denom,
                w*mu - sin_theta*denom*np.cos(phi)
            ])
        else:
            new_dir = np.array([
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                mu * np.sign(w)
            ])

        norm = np.sqrt(new_dir[0]**2 + new_dir[1]**2 + new_dir[2]**2)
        if norm > 0:
            new_dir = new_dir / norm
        new_dirs[i] = new_dir

    return new_dirs

def sample_distances_batch(mfp, size):
    _positive("mfp", mfp)
    _integer("size", size)
    return _sample_distances_batch(mfp, size)


def sample_reactions_batch(sigma_f, sigma_a, sigma_s, size):
    _integer("size", size)
    xs = np.asarray([sigma_f, sigma_a, sigma_s], dtype=float)
    if not np.isfinite(xs).all() or (xs < 0).any() or not np.isfinite(xs.sum()) or xs.sum() <= 0:
        raise ValueError("cross sections must be finite and nonnegative with positive total")
    return _sample_reactions_batch(sigma_f, sigma_a, sigma_s, size)


def scatter_directions(directions, size):
    _integer("size", size)
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1:] != (3,) or size > len(directions):
        raise ValueError("directions must have shape (N, 3), with size <= N")
    return _scatter_directions(directions, size)


@njit(nogil=True)
def _transport_batch(positions, directions, mfp, sigma_f, sigma_a, sigma_s,
                     nu_f, reactor_radius, max_steps=500, sample_count=40, seed=-1):
    """
    Транспортирует партию нейтронов одновременно.
    Возвращает: final_positions, final_directions, fates, fission_indices, paths, lengths
    fates: 0=fission, 1=absorption, 2=leak, -1=unfinished (step limit)
    paths: список истории позиций для каждого нейтрона
    """
    if seed >= 0:
        np.random.seed(seed)
    n_neutrons = positions.shape[0]
    alive = np.ones(n_neutrons, dtype=np.bool_)
    fates = np.full(n_neutrons, -1, dtype=np.int32)
    fission_indices = []

    # Evenly spaced source indices: deterministic and independent of transport RNG.
    sample_count = min(sample_count, n_neutrons)
    paths = np.empty((sample_count, max_steps + 1, 3), dtype=np.float64)
    lengths = np.ones(sample_count, dtype=np.int64)
    slots = np.full(n_neutrons, -1, dtype=np.int64)
    for j in range(sample_count):
        i = j * n_neutrons // sample_count
        slots[i] = j
        paths[j, 0] = positions[i]

    for step in range(max_steps):
        # Кто ещё живой?
        alive_count = np.sum(alive)
        if alive_count == 0:
            break

        # Расстояния для живых нейтронов
        dists = _sample_distances_batch(mfp, n_neutrons)

        # Сдвигаем позиции
        for i in range(n_neutrons):
            if alive[i]:
                for axis in range(3):
                    positions[i, axis] += directions[i, axis] * dists[i]
                j = slots[i]
                if j >= 0:
                    paths[j, lengths[j]] = positions[i]
                    lengths[j] += 1

        # Проверяем утечку
        for i in range(n_neutrons):
            if alive[i]:
                r = np.sqrt(positions[i, 0]**2 + positions[i, 1]**2 + positions[i, 2]**2)
                if r > reactor_radius:
                    alive[i] = False
                    fates[i] = 2  # leak

        # Генерируем реакции для живых
        reactions = _sample_reactions_batch(sigma_f, sigma_a, sigma_s, n_neutrons)

        for i in range(n_neutrons):
            if not alive[i]:
                continue

            if reactions[i] == 0:  # fission
                alive[i] = False
                fates[i] = 0
                # A dead particle never moves again. Preserve chronological
                # event order without allocating a small coordinate array here.
                fission_indices.append(i)
            elif reactions[i] == 1:  # absorption
                alive[i] = False
                fates[i] = 1
            else:  # scatter
                mu = 1.0 - 2.0 * np.random.random()
                phi = 2.0 * np.pi * np.random.random()
                sin_theta = np.sqrt(max(0.0, 1.0 - mu**2))

                u, v, w = directions[i]
                if abs(w) < 0.9999:
                    denom = np.sqrt(1.0 - w**2)
                    x = u*mu + sin_theta*(u*w*np.cos(phi) - v*np.sin(phi)) / denom
                    y = v*mu + sin_theta*(v*w*np.cos(phi) + u*np.sin(phi)) / denom
                    z = w*mu - sin_theta*denom*np.cos(phi)
                else:
                    x = sin_theta * np.cos(phi)
                    y = sin_theta * np.sin(phi)
                    z = mu * np.sign(w)
                norm = np.sqrt(x*x + y*y + z*z)
                if norm > 0:
                    directions[i, 0] = x / norm
                    directions[i, 1] = y / norm
                    directions[i, 2] = z / norm

    # A step-limited history is terminal for this run, not a completed interaction.
    for i in range(n_neutrons):
        if alive[i]:
            fates[i] = -1  # HistoryStatus.MAX_STEPS_REACHED
            alive[i] = False

    return positions, directions, fates, fission_indices, paths, lengths

def transport_batch(positions, directions, mfp, sigma_f, sigma_a, sigma_s,
                    nu_f, reactor_radius, max_steps=500, max_samples=40, seed=None,
                    max_memory_bytes=DEFAULT_MAX_MEMORY_BYTES):
    """Transport copies of input arrays; paths contain only sampled histories.

    Fate -1 explicitly denotes MAX_STEPS_REACHED, never an active history.
    Fission points remain a chronological list of owning coordinate arrays.
    """
    positions, directions = _arrays(positions, directions)
    _positive("mfp", mfp)
    _positive("reactor_radius", reactor_radius)
    _integer("max_steps", max_steps)
    _integer("max_samples", max_samples)
    _check_memory(len(positions), max_samples, max_steps, max_memory_bytes)
    xs = np.asarray([sigma_f, sigma_a, sigma_s, nu_f], dtype=float)
    if not np.isfinite(xs).all() or (xs < 0).any() or not np.isfinite(xs[:3].sum()) or xs[:3].sum() <= 0:
        raise ValueError("cross sections and nu must be finite and nonnegative; total must be positive")
    if seed is not None:
        _integer("seed", seed)
        if seed > 2**32 - 1:
            raise ValueError("seed must be <= 2**32 - 1")
    # Serializes fallback global RNG access. Numba seeds its own thread-local RNG.
    with _rng_lock:
        saved_state = np.random.get_state() if not HAS_NUMBA and seed is not None else None
        try:
            out = _transport_batch(positions.copy(), directions.copy(), mfp,
                                   sigma_f, sigma_a, sigma_s, nu_f, reactor_radius,
                                   max_steps, max_samples, -1 if seed is None else seed)
        finally:
            if saved_state is not None:
                np.random.set_state(saved_state)
    pos, dirs, fates, fiss, paths, lengths = out
    # Preserve the public list of owning coordinate arrays, in event order.
    fiss = [pos[i].copy() for i in fiss]
    # Copies prevent short paths retaining the full scratch allocation.
    return pos, dirs, fates, fiss, [paths[j, :lengths[j]].copy() for j in range(len(lengths))]


def run_generation_batch(positions, directions, material, radius, max_samples_per_gen=40,
                         max_steps=500, rng=None, seed=None,
                         max_memory_bytes=DEFAULT_MAX_MEMORY_BYTES):
    """Return source arrays, three terminal counts, sampled paths, fission points,
    and complete outcome counts (including unfinished particles).
    """
    _integer("max_samples_per_gen", max_samples_per_gen)
    data = resolve_transport_data(material)
    rng = np.random.default_rng(seed) if rng is None else rng

    # Транспортируем нейтроны
    final_pos, final_dirs, fates, fiss_pts, all_paths = transport_batch(
        positions, directions,
        data.mean_free_path,
        data.sigma_f, data.sigma_a, data.sigma_s,
        data.nu,
        radius,
        max_steps=max_steps, max_samples=max_samples_per_gen, seed=seed,
        max_memory_bytes=max_memory_bytes
    )
    # The public transport boundary validates and copies the inputs once.
    n_neutrons = len(final_pos)

    # Подсчитываем результаты
    fissions = np.sum(fates == 0)
    absorptions = np.sum(fates == 1)
    leakages = np.sum(fates == 2)
    outcome_counts = {
        "fission": int(fissions),
        "absorption": int(absorptions),
        "leak": int(leakages),
        "unfinished": int(np.sum(fates == -1)),
    }
    if sum(outcome_counts.values()) != n_neutrons:
        raise RuntimeError("Generation outcome counts do not match processed particles")

    # Формируем новый источник (вторичные нейтроны)
    secondary_expectation = float(fissions) * data.nu
    if not np.isfinite(secondary_expectation):
        raise MemoryError("Secondary population exceeds finite allocation capacity")
    n_secondary = int(secondary_expectation)
    _check_memory(n_neutrons, max_samples_per_gen, max_steps, max_memory_bytes,
                  secondary=n_secondary)

    if n_secondary > 0:
        # Позиции вторичных — там же, где произошли деления
        new_directions = rng.standard_normal((n_secondary, 3))

        # Нормализуем направления
        norms = np.linalg.norm(new_directions, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        new_directions = new_directions / norms

        # Распределяем вторичные нейтроны
        fiss_idx = np.where(fates == 0)[0]
        if len(fiss_idx) > 0:
            # Случайно выбираем позиции делений для вторичных
            # Standard NumPy Generators preserve scalar int64 draws and their
            # continuation state for this bounded-integer batch on supported
            # versions (covered by compatibility tests). Keep custom RNGs on
            # the scalar protocol accepted by this public function.
            if type(rng) is np.random.Generator:
                indices = rng.integers(0, len(fiss_idx), size=n_secondary)
                new_positions = final_pos[fiss_idx[indices]]
            else:
                new_positions = np.empty((n_secondary, 3), dtype=float)
                for i in range(n_secondary):
                    fiss_pos_idx = rng.integers(0, len(fiss_idx))
                    new_positions[i] = final_pos[fiss_idx[fiss_pos_idx]]
    else:
        new_positions = np.zeros((0, 3), dtype=float)
        new_directions = np.zeros((0, 3), dtype=float)

    # Формируем пути для визуализации (семпл до max_samples_per_gen)
    sample_count = min(n_neutrons, max_samples_per_gen)
    sample_idx = np.arange(sample_count) * n_neutrons // max(1, sample_count)
    path_frames = []
    fate_map = {0: "fission", 1: "absorption", 2: "leak", -1: "unfinished"}

    for slot, idx in enumerate(sample_idx):
        # Используем полную историю пути из transport_batch
        path = all_paths[slot]
        fate = fate_map.get(fates[idx], "unknown")
        path_frames.append((path, fate))

    fiss_pts_array = np.array(fiss_pts) if fiss_pts else np.zeros((0, 3))

    return (new_positions, new_directions,
            fissions, absorptions, leakages,
            path_frames, fiss_pts_array, outcome_counts)

def estimate_k_full(n_gen=30, n_source=150, radius=25.0,
                    material=None, n_burnin=8, callback=None, seed=None,
                    max_samples_per_gen=40, max_steps=500, cancel_event=None,
                    max_memory_bytes=DEFAULT_MAX_MEMORY_BYTES):
    """Run generations with the compatible source-production ratio.

    k_g = int(F_g * nu) / N_g before population control. N_g includes
    step-limited histories; dropping them would change the estimator. Unresolved
    histories can still interact, so this ratio is incomplete when any remain.
    k_std is a population spread across selected generations, not a standard
    error or a validated physical uncertainty. Cancellation raises, as do failures.

    Material data are resolved once and frozen for the run. Existing result
    fields are preserved; run_info, estimator and warning codes describe the
    calculation independently of any GUI. Warning codes are documented in the
    architecture audit. This is not a validated k_eff estimator.
    """
    for name, value in (("n_gen", n_gen), ("n_source", n_source),
                        ("n_burnin", n_burnin), ("max_steps", max_steps),
                        ("max_samples_per_gen", max_samples_per_gen)):
        _integer(name, value)
    _positive("radius", radius)
    _check_memory(n_source, max_samples_per_gen, max_steps, max_memory_bytes)
    if seed is not None:
        _integer("seed", seed)
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable or None")
    rng = np.random.default_rng(seed)
    if material is None:
        material = MATERIALS["U-235 (чистый)"]

    data = resolve_transport_data(material)

    # Инициализация исходного поколения как массивы
    positions = np.zeros((n_source, 3), dtype=float)
    directions = rng.standard_normal((n_source, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    directions = directions / norms

    k_history = []
    path_frames = []
    fiss_frames = []
    generation_stats = []
    retained_bytes = 0

    for gen in range(n_gen):
        if cancel_event is not None and cancel_event.is_set():
            raise SimulationCancelled("Simulation cancelled")
        if positions.shape[0] == 0:
            break

        # Транспортируем поколение
        (next_pos, next_dirs,
         fiss, abso, leak,
         paths, fpts, counts) = run_generation_batch(
             positions, directions, data, radius, max_samples_per_gen, max_steps,
             rng=rng, seed=int(rng.integers(0, 2**32)),
             max_memory_bytes=(None if max_memory_bytes is None else
                               max_memory_bytes - retained_bytes))

        # Source-production ratio before capping; not a validated k_eff.
        k = len(next_pos) / max(1, positions.shape[0]) if positions.shape[0] > 0 else 0.0
        k_history.append(k)
        path_frames.append(paths)
        fiss_frames.append(fpts)
        retained_bytes += fpts.nbytes + sum(path.nbytes for path, _ in paths)
        generation_stats.append({"processed": int(positions.shape[0]),
                                 "completed": int(positions.shape[0]) - counts["unfinished"],
                                 "step_limited": counts["unfinished"],
                                 "histories_complete": counts["unfinished"] == 0,
                                 "outcome_counts": counts})

        if callback:
            callback(gen + 1, k, fiss, abso, leak)

        # Обновляем источник
        if len(next_pos) == 0:
            break
        elif len(next_pos) > n_source:
            # Выборка для контроля взрыва
            idx = rng.choice(len(next_pos), n_source, replace=False)
            positions = next_pos[idx]
            directions = next_dirs[idx]
        else:
            positions = next_pos
            directions = next_dirs

    if cancel_event is not None and cancel_event.is_set():
        raise SimulationCancelled("Simulation cancelled")

    k_active = k_history[n_burnin:] if len(k_history) > n_burnin else k_history
    k_mean = float(np.mean(k_active)) if k_active else 0.0
    k_std = float(np.std(k_active)) if k_active else 0.0
    step_limited = sum(s["step_limited"] for s in generation_stats)
    warnings = []
    if step_limited:
        warnings.append("step_limit_reached")
    if not k_history:
        warnings.append("no_estimate")
    elif n_burnin and len(k_history) <= n_burnin:
        warnings.append("burnin_not_applied")

    return {
        "k_history":   k_history,
        "k_mean":      k_mean,
        "k_std":       k_std,
        "path_frames": path_frames,
        "fiss_frames": fiss_frames,
        "generation_stats": generation_stats,
        # Execution success and history completeness are separate concepts.
        "execution_status": "completed",
        "histories_complete": step_limited == 0,
        "completed_histories": sum(s["completed"] for s in generation_stats),
        "step_limited_histories": step_limited,
        "processed_histories": sum(s["processed"] for s in generation_stats),
        "warnings": warnings,
        "estimator": {
            "name": "source_production_ratio",
            "definition": "int(fissions * nu) / processed, before population control",
            "validated_keff": False,
            "active_generation_start": int(n_burnin) if len(k_history) > n_burnin else 0,
            "active_generation_count": len(k_active),
            "std_kind": "population_standard_deviation",
        },
        "run_info": {
            "seed": None if seed is None else int(seed),
            "backend": "numba" if HAS_NUMBA else "python",
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "numba_version": getattr(sys.modules.get("numba"), "__version__", None) if HAS_NUMBA else None,
            "n_gen": int(n_gen), "n_source": int(n_source),
            "n_burnin": int(n_burnin), "radius_cm": float(radius),
            "max_steps": int(max_steps),
            "max_samples_per_gen": int(max_samples_per_gen),
            "max_memory_bytes": None if max_memory_bytes is None else int(max_memory_bytes),
            "material": asdict(data),
        },
        "termination_reason": ("empty_source" if n_source == 0 else
                               "no_generations" if n_gen == 0 else
                               "no_secondary_source" if len(next_pos) == 0 else
                               "generation_limit"),
    }


# Independent educational models below do not use fission or generation ratios.
def _real_scalar(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def _confidence(value):
    value = _real_scalar("confidence", value)
    if not 0 < value < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    return value


def summarize_binomial(successes, trials, confidence=.95, unfinished=0):
    """Marginal Clopper-Pearson interval for independent Bernoulli histories.

    Censored histories are not discarded or treated as failures: full-outcome
    estimates/intervals are unavailable. Completion bounds describe the possible
    empirical fraction, NOT confidence bounds on a population probability.
    Multiple outcome intervals are marginal, not simultaneous multinomial CIs.
    """
    from scipy.special import betaincinv, betainccinv

    for name, value in (("successes", successes), ("trials", trials), ("unfinished", unfinished)):
        _integer(name, value)
    successes, trials, unfinished = int(successes), int(trials), int(unfinished)
    if trials > np.iinfo(np.int64).max:
        raise ValueError("trials exceeds int64 counting capacity")
    if successes + unfinished > trials:
        raise ValueError("successes + unfinished must not exceed trials")
    confidence = _confidence(confidence)
    result = dict(trials=int(trials), successes=int(successes), unfinished=int(unfinished),
                  estimate=None, standard_error=None, confidence_interval=None,
                  confidence=confidence, method="clopper_pearson_marginal",
                  standard_error_kind="bernoulli_plugin",
                  observed_fraction=successes / trials if trials else None,
                  completion_bounds=None, unavailable_reason=None)
    if not trials:
        result["unavailable_reason"] = "empty_sample"
    elif unfinished:
        result["unavailable_reason"] = "censored_histories"
        result["completion_bounds"] = (successes / trials, (successes + unfinished) / trials)
    else:
        p = successes / trials
        tail = (1 - confidence) / 2
        # Binomial-tail inversion via the beta-binomial identity. Complemented
        # inversion avoids forming 1-tail near one; endpoints need no zero shapes.
        lo = 0. if successes == 0 else float(betaincinv(successes, trials-successes+1, tail))
        hi = 1. if successes == trials else float(betainccinv(successes+1, trials-successes, tail))
        result["estimate"] = p
        if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi <= 1:
            result["unavailable_reason"] = "interval_resolution_limit"
        else:
            result.update(standard_error=float(np.sqrt(p * (1 - p) / trials)),
                          confidence_interval=(lo, hi))
    return result


def summarize_samples(samples, confidence=.95, assumption="descriptive"):
    """Mean/sample variance; Student-t CI ONLY under explicit IID normal inputs.

    Descriptive mode makes no independence claim and provides no SE/CI. This
    function must not be applied to correlated legacy generation ratios as IID.
    Constant samples cannot justify an unknown-variance t interval.
    """
    from scipy.stats import t

    confidence = _confidence(confidence)
    if assumption not in ("descriptive", "iid_normal"):
        raise ValueError("assumption must be descriptive or iid_normal")
    raw = np.asanyarray(samples)
    if np.ma.isMaskedArray(raw) or raw.dtype.kind not in "iuf":
        raise ValueError("samples must be real numeric data without masks")
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("samples must be a finite one-dimensional array")
    n = len(x)
    result = dict(n=n, mean=None, sample_variance=None, standard_error=None,
                  confidence_interval=None, confidence=confidence, assumption=assumption,
                  unavailable_reason="empty_sample")
    if not n:
        return result
    # Center first for close, large-offset observations; scale only where needed.
    with np.errstate(over="ignore", invalid="ignore"):
        delta = x - x[0]
    if np.isfinite(delta).all():
        scale = float(np.max(np.abs(delta))) or 1.
        mean = float(x[0] + np.mean(delta / scale) * scale)
    else:
        scale = float(np.max(np.abs(x))) or 1.
        mean = float(np.mean(x / scale) * scale)
    variance = None
    if n > 1:
        with np.errstate(over="ignore", invalid="ignore"):
            delta = x - mean
            if np.isfinite(delta).all():
                scale = float(np.max(np.abs(delta))) or 1.
                normalized = delta / scale
            else:
                scale = float(np.max(np.abs(x))) or 1.
                normalized = x / scale - mean / scale
            std = float(np.sqrt(np.sum(normalized**2) / (n - 1)) * scale)
            variance = float(np.float64(std) * std)
        if not np.isfinite(variance):
            raise ValueError("sample variance exceeds float64 representability")
        if std > 0 and variance == 0:
            raise ValueError("sample variance underflows float64 representability")
    result.update(mean=mean, sample_variance=variance,
                  unavailable_reason="independence_and_distribution_not_asserted")
    if assumption == "iid_normal":
        if n < 2:
            result["unavailable_reason"] = "insufficient_samples"
        elif variance == 0:
            result["unavailable_reason"] = "zero_sample_variance"
        else:
            se = std / np.sqrt(n)
            # isf avoids rounding (1+confidence)/2 to 1 near the endpoint.
            margin = float(t.isf((1 - confidence) / 2, n - 1) * se)
            interval = (mean - margin, mean + margin)
            if not np.isfinite(interval).all() or interval[0] >= interval[1]:
                raise ValueError("confidence interval exceeds float64 representability")
            result.update(standard_error=float(se), confidence_interval=interval, unavailable_reason=None)
    return result


@njit(nogil=True)
def _slab_batch(edges, absorption, scattering, n, max_collisions, rng,
                sample_count, source_x, source_mu):
    """Sequential independent random flights; only sampled x vertices retained.

    Counts: absorption, left exit, right exit, collision limit, numeric limit.
    Optical depth Exp(1) is carried across interfaces until the next collision.
    Scattering chooses an isotropic 3-D direction through its uniform x cosine.
    """
    counts = np.zeros(5, dtype=np.int64)
    paths = np.empty((sample_count, max_collisions + 2 if sample_count else 1), dtype=np.float64)
    lengths = np.ones(sample_count, dtype=np.int64)
    fates = np.full(sample_count, -1, dtype=np.int64)
    for i in range(n):
        x, mu = source_x, source_mu
        layer = np.searchsorted(edges, x, side="right") - 1
        if mu < 0 and x == edges[layer]:
            layer -= 1
        collisions = 0
        if i < sample_count:
            paths[i, 0] = x
        fate = -1
        while fate < 0:
            if layer < 0:
                fate = 1
                break
            if layer >= len(absorption):
                fate = 2
                break
            tau = -np.log1p(-rng.random())
            while True:
                rate = absorption[layer] + scattering[layer]
                if mu == 0. and rate == 0.:
                    fate = 4  # A grazing vacuum trajectory cannot be resolved.
                    break
                boundary = edges[layer + 1] if mu > 0 else edges[layer]
                distance = abs((boundary - x) / mu) if mu != 0 else np.inf
                optical_width = rate * distance if rate > 0 else 0.
                if rate > 0 and tau < optical_width:
                    displacement = mu * (tau / rate)
                    next_x = x + displacement
                    if not np.isfinite(next_x) or (tau > 0. and mu != 0. and next_x == x):
                        fate = 4
                        break
                    x = next_x
                    if i < sample_count:
                        paths[i, lengths[i]] = x
                        lengths[i] += 1
                    if collisions == max_collisions:
                        fate = 3
                        break
                    collisions += 1
                    if rng.random() < absorption[layer] / rate:
                        fate = 0
                    else:
                        mu = 2. * rng.random() - 1.
                    break
                tau = max(0., tau - optical_width)
                x = boundary
                layer += 1 if mu > 0 else -1
                if layer < 0 or layer == len(absorption):
                    fate = 1 if layer < 0 else 2
                    if i < sample_count:
                        paths[i, lengths[i]] = x
                        lengths[i] += 1
                    break
        counts[fate] += 1
        if i < sample_count:
            fates[i] = fate
    return counts, paths, lengths, fates


def simulate_slab(layers, edges_cm, n_histories=10000, *, seed=None,
                  max_collisions=1000, max_samples=0, batch_size=4096,
                  source_x=0., source_mu=1., confidence=.95, backend="auto",
                  cancel_event=None, max_memory_bytes=DEFAULT_MAX_MEMORY_BYTES):
    """Nonbranching random flights in an abstract piecewise-constant slab.

    Layers use explicit absorption_rate/scattering_rate in cm^-1. No nuclear
    channels, energy transport or multiplication. Interior interfaces preserve
    remaining optical depth; outer boundaries absorb by escape. The collision
    budget censors unresolved histories; it does not prevent ballistic escape.
    Memory is O(layers + samples*(max_collisions+2)), independent of population.
    Cancellation is cooperative between batches and raises without partial output.
    """
    import scipy

    started = perf_counter()
    for name, value in (("n_histories", n_histories), ("max_collisions", max_collisions),
                        ("max_samples", max_samples), ("batch_size", batch_size)):
        _integer(name, value, 1 if name == "batch_size" else 0)
    if n_histories > np.iinfo(np.int64).max:
        raise ValueError("n_histories exceeds int64 counting capacity")
    if max_collisions > np.iinfo(np.int64).max - 2:
        raise ValueError("max_collisions exceeds int64 indexing capacity")
    if seed is not None:
        _integer("seed", seed)
    confidence = _confidence(confidence)
    if backend not in ("auto", "numba", "python"):
        raise ValueError("backend must be auto, numba or python")
    if backend == "numba" and not HAS_NUMBA:
        raise RuntimeError("Numba backend is unavailable")
    use_numba = HAS_NUMBA and backend != "python"
    kernel = _slab_batch if use_numba else getattr(_slab_batch, "py_func", _slab_batch)
    if not isinstance(layers, (tuple, list)) or not layers:
        raise ValueError("layers must be a nonempty sequence of MaterialDataset")
    layers = tuple(layers)
    samples = min(max_samples, n_histories)
    # Check before layer/source/scratch allocations, including owning path copies.
    required = 4096 + len(layers) * 4096 + samples * ((max_collisions + 2) * 16 + 256)
    if max_memory_bytes is not None:
        _integer("max_memory_bytes", max_memory_bytes)
        if required > max_memory_bytes:
            raise MemoryError(f"Slab storage estimate {required} exceeds max_memory_bytes={max_memory_bytes}")
    rates = np.array([resolve_medium_rates(layer) for layer in layers], dtype=np.float64)
    raw_edges = np.asanyarray(edges_cm)
    if np.ma.isMaskedArray(raw_edges) or raw_edges.dtype.kind not in "iuf":
        raise ValueError("edges_cm must contain real numeric values without masks")
    edges = np.array(raw_edges, dtype=np.float64, copy=True)
    if edges.shape != (len(layers) + 1,) or not np.isfinite(edges).all() or edges[0] != 0 or not np.all(np.diff(edges) > 0):
        raise ValueError("edges_cm must start at 0, be finite, strictly increasing, and bound every layer")
    source_x, source_mu = _real_scalar("source_x", source_x), _real_scalar("source_mu", source_mu)
    if not 0 <= source_x < edges[-1] or not -1 <= source_mu <= 1 or source_mu == 0:
        raise ValueError("source_x must be inside [0, width); source_mu must be nonzero in [-1, 1]")
    effective_seed = int(np.random.SeedSequence(seed).entropy)
    rng = np.random.Generator(np.random.PCG64(effective_seed))
    counts = np.zeros(5, dtype=np.int64)
    paths, path_fates = [], []
    transport_started = perf_counter()
    for offset in range(0, n_histories, batch_size):
        if cancel_event is not None and cancel_event.is_set():
            raise SimulationCancelled("Slab simulation cancelled")
        n = min(batch_size, n_histories - offset)
        sampled = min(n, max(0, samples - offset))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            outcomes, scratch, lengths, fates = kernel(edges, rates[:, 0], rates[:, 1], n,
                max_collisions, rng, sampled, source_x, source_mu)
        counts += outcomes
        paths.extend(scratch[j, :lengths[j]].copy() for j in range(sampled))
        path_fates.extend(int(fates[j]) for j in range(sampled))
    if cancel_event is not None and cancel_event.is_set():
        raise SimulationCancelled("Slab simulation cancelled")
    transport_seconds = perf_counter() - transport_started
    if int(counts.sum()) != n_histories:
        raise RuntimeError("Slab outcome counts do not reconcile")
    names = ("absorbed", "escaped_left", "escaped_right", "collision_limit", "numeric_limit")
    outcomes = {name: int(value) for name, value in zip(names, counts)}
    unfinished = int(counts[3] + counts[4])
    estimates = {name: summarize_binomial(int(counts[i]), n_histories, confidence, unfinished)
                 for i, name in enumerate(names[:3])}
    warnings = []
    if not n_histories:
        warnings.append("no_histories")
    if counts[3]:
        warnings.append("collision_limit_reached")
    if counts[4]:
        warnings.append("numerical_resolution_limit")
    result = dict(algorithm_version="abstract_slab_v1", execution_status="completed",
        outcome_counts=outcomes, processed_histories=int(n_histories),
        completed_histories=int(n_histories) - unfinished, unfinished_histories=unfinished,
        histories_complete=unfinished == 0, estimates=estimates,
        sampled_paths_cm=paths, sampled_path_fates=path_fates, warnings=warnings,
        limitations=["abstract_nonreactive_model", "piecewise_constant_rates",
                     "isotropic_scattering", "marginal_not_simultaneous_intervals"],
        timing=dict(transport_seconds=transport_seconds),
        run_info=dict(seed=None if seed is None else int(seed), effective_seed=effective_seed,
            rng="PCG64", n_histories=int(n_histories), max_collisions=int(max_collisions),
            max_samples=int(max_samples), batch_size=int(batch_size), source_x_cm=source_x,
            source_mu=source_mu, confidence=confidence,
            max_memory_bytes=None if max_memory_bytes is None else int(max_memory_bytes),
            edges_cm=edges.tolist(), layers=[asdict(layer) for layer in layers],
            backend="numba" if use_numba else "python_numpy", python_version=sys.version.split()[0],
            numpy_version=np.__version__, scipy_version=scipy.__version__,
            numba_version=getattr(sys.modules.get("numba"), "__version__", None) if use_numba else None))
    result["timing"]["total_seconds"] = perf_counter() - started
    return result
