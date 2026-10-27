"""
БЛОК 3 — ФИЗИКА МОНТЕ-КАРЛО (с массивами и Numba).

Ядро симуляции нейтронного транспорта:
  - выборка длин пробега          (sample_distances_batch)
  - выборка типов реакций         (sample_reactions_batch)
  - генерация направлений рассеяния (scatter_directions)
  - транспорт партии нейтронов   (transport_batch)
  - оркестрация поколения        (run_generation_batch)
  - оценка k_eff по поколениям   (estimate_k_full)

Соответствует БЛОКУ 3 исходного монолита NSPM MCNP 3000.py.
Модель транспорта сохранена; статистика учитывает всё поколение отдельно от выборки путей.
"""

import numpy as np

from numba_compat import njit, HAS_NUMBA
from material import MATERIALS


@njit
def sample_distances_batch(mfp, size):
    """Генерирует расстояния для партии нейтронов (Numba-ускорено)"""
    return -mfp * np.log(np.random.random(size) + 1e-15)

@njit
def sample_reactions_batch(sigma_f, sigma_a, sigma_s, size):
    """Генерирует типы реакций для партии: 0=fission, 1=absorption, 2=scatter"""
    sigma_t = sigma_f + sigma_a + sigma_s
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
def scatter_directions(directions, size):
    """Генерирует новые направления после рассеяния (Numba-ускорено)"""
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

@njit
def transport_batch(positions, directions, mfp, sigma_f, sigma_a, sigma_s,
                     nu_f, reactor_radius, max_steps=500):
    """
    Транспортирует партию нейтронов одновременно.
    Возвращает: final_positions, final_directions, fates, fission_positions, paths
    fates: 0=fission, 1=absorption, 2=leak, -1=unfinished (step limit)
    paths: список истории позиций для каждого нейтрона
    """
    n_neutrons = positions.shape[0]
    alive = np.ones(n_neutrons, dtype=np.bool_)
    fates = np.full(n_neutrons, -1, dtype=np.int32)
    fission_positions = []

    # Инициализация историй путей
    paths = []
    for i in range(n_neutrons):
        paths.append([positions[i].copy()])

    for step in range(max_steps):
        # Кто ещё живой?
        alive_count = np.sum(alive)
        if alive_count == 0:
            break

        # Расстояния для живых нейтронов
        dists = sample_distances_batch(mfp, n_neutrons)

        # Сдвигаем позиции
        for i in range(n_neutrons):
            if alive[i]:
                positions[i] += directions[i] * dists[i]
                paths[i].append(positions[i].copy())

        # Проверяем утечку
        for i in range(n_neutrons):
            if alive[i]:
                r = np.sqrt(positions[i, 0]**2 + positions[i, 1]**2 + positions[i, 2]**2)
                if r > reactor_radius:
                    alive[i] = False
                    fates[i] = 2  # leak

        # Генерируем реакции для живых
        reactions = sample_reactions_batch(sigma_f, sigma_a, sigma_s, n_neutrons)

        for i in range(n_neutrons):
            if not alive[i]:
                continue

            if reactions[i] == 0:  # fission
                alive[i] = False
                fates[i] = 0
                fission_positions.append(positions[i].copy())
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
                    directions[i] = new_dir / norm

    return positions, directions, fates, fission_positions, paths

def run_generation_batch(positions, directions, material, radius, max_samples_per_gen=40,
                         max_steps=500):
    """Return source arrays, three terminal counts, sampled paths, fission points,
    and complete outcome counts (including unfinished particles).
    """
    n_neutrons = positions.shape[0]

    # Транспортируем нейтроны
    final_pos, final_dirs, fates, fiss_pts, all_paths = transport_batch(
        positions.copy(), directions.copy(),
        material.mean_free_path(),
        material.sigma_f, material.sigma_a, material.sigma_s,
        material.nu,
        radius,
        max_steps=max_steps
    )

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
    n_secondary = int(fissions * material.nu)

    if n_secondary > 0:
        # Позиции вторичных — там же, где произошли деления
        new_positions = np.zeros((n_secondary, 3), dtype=float)
        new_directions = np.random.randn(n_secondary, 3)

        # Нормализуем направления
        norms = np.linalg.norm(new_directions, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        new_directions = new_directions / norms

        # Распределяем вторичные нейтроны
        fiss_idx = np.where(fates == 0)[0]
        if len(fiss_idx) > 0:
            # Случайно выбираем позиции делений для вторичных
            for i in range(n_secondary):
                fiss_pos_idx = np.random.randint(0, len(fiss_idx))
                new_positions[i] = final_pos[fiss_idx[fiss_pos_idx]].copy()
    else:
        new_positions = np.zeros((0, 3), dtype=float)
        new_directions = np.zeros((0, 3), dtype=float)

    # Формируем пути для визуализации (семпл до max_samples_per_gen)
    sample_count = min(n_neutrons, max_samples_per_gen)
    sample_idx = np.random.choice(n_neutrons, size=sample_count, replace=False)
    path_frames = []
    fate_map = {0: "fission", 1: "absorption", 2: "leak", -1: "unfinished"}

    for idx in sample_idx:
        # Используем полную историю пути из transport_batch
        path = all_paths[idx]
        fate = fate_map.get(fates[idx], "unknown")
        path_frames.append((path, fate))

    fiss_pts_array = np.array(fiss_pts) if fiss_pts else np.zeros((0, 3))

    return (new_positions, new_directions,
            fissions, absorptions, leakages,
            path_frames, fiss_pts_array, outcome_counts)

def estimate_k_full(n_gen=30, n_source=150, radius=25.0,
                    material=None, n_burnin=8, callback=None):
    """Запускает симуляцию Монте-Карло с использованием массивов"""
    if material is None:
        material = MATERIALS["U-235 (чистый)"]

    # Инициализация исходного поколения как массивы
    positions = np.zeros((n_source, 3), dtype=float)
    directions = np.random.randn(n_source, 3)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    directions = directions / norms

    k_history = []
    path_frames = []
    fiss_frames = []
    generation_stats = []

    for gen in range(n_gen):
        if positions.shape[0] == 0:
            break

        # Транспортируем поколение
        (next_pos, next_dirs,
         fiss, abso, leak,
         paths, fpts, counts) = run_generation_batch(positions, directions, material, radius)

        # k-эффективное = (нейтроны в следующем поколении) / (нейтроны в текущем)
        k = len(next_pos) / max(1, positions.shape[0]) if positions.shape[0] > 0 else 0.0
        k_history.append(k)
        path_frames.append(paths)
        fiss_frames.append(fpts)
        generation_stats.append({"processed": int(positions.shape[0]),
                                 "outcome_counts": counts})

        if callback:
            callback(gen + 1, k, fiss, abso, leak)

        # Обновляем источник
        if len(next_pos) == 0:
            break
        elif len(next_pos) > n_source:
            # Выборка для контроля взрыва
            idx = np.random.choice(len(next_pos), n_source, replace=False)
            positions = next_pos[idx].copy()
            directions = next_dirs[idx].copy()
        else:
            positions = next_pos.copy()
            directions = next_dirs.copy()

    k_active = k_history[n_burnin:] if len(k_history) > n_burnin else k_history
    k_mean = float(np.mean(k_active)) if k_active else 0.0
    k_std = float(np.std(k_active)) if k_active else 0.0

    return {
        "k_history":   k_history,
        "k_mean":      k_mean,
        "k_std":       k_std,
        "path_frames": path_frames,
        "fiss_frames": fiss_frames,
        "generation_stats": generation_stats,
    }


