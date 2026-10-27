# HeliosMC

Launch the modular Matplotlib application from this directory:

```sh
python main.py
```

Requires NumPy and Matplotlib. Numba is optional; `numba_compat.py` keeps the
Python fallback available. Select the intended Python interpreter in VS Code,
then use the **NeutronMC** launch configuration.

## Module relationships

- `main.py`: application startup; initializes `gui.NeutronSimGUI`.
- `gui.py`: Matplotlib controls, trajectories, heatmap, k history and balance chart;
  imports `physics.estimate_k_full` and `material.MATERIALS`.
- `physics.py`: distance/reaction sampling, scattering, batch transport,
  generation orchestration and k estimation; imports materials and Numba compatibility.
- `material.py`: `Material`, its cross-section/mean-free-path methods and the
  existing `MATERIALS` library. Constants are unchanged.
- `neutron.py`: standalone `Neutron` model with movement/scattering methods;
  retained for single-particle use. The batch engine operates on arrays.
- `numba_compat.py`: optional JIT decorator and availability flag.
- `NSPM MCNP 3000.py`: unchanged legacy backup containing duplicate definitions.
  The modular application does not import or execute it.

The extracted modules already contained the model and GUI. The missing entry point
and discarded aggregate results were repaired without replacing the transport model.
Dependencies run from startup to GUI to physics/materials; no module imports startup.

## Generation results

`run_generation_batch` returns the existing seven values followed by an eighth
`outcome_counts` dictionary: `fission`, `absorption`, `leak`, `unfinished`.
Its optional `max_steps` defaults to the original 500. A fate of -1 means the
particle reached the step limit without a terminal outcome.

`estimate_k_full` retains its previous result keys and callback signature, and adds
`generation_stats`, aligned with `k_history`. Each entry contains `processed` and
`outcome_counts`. The four counts must sum to `processed`; an inconsistency raises
an error. This counts the actual population transported in each generation before
secondary-source population control.

`path_frames` remains a visualization sample of at most 40 trajectories per
generation. The histogram reads only `generation_stats`, including unfinished
particles labelled **не завершены**. Trajectory sampling does not determine counts.

## Verification

```sh
python -m unittest -v test_generation_stats
python -m compileall -q .
```

Tests cover mixed outcomes above the sample limit, zero/one transport steps,
unfinished labels, empty batches, multi-generation accounting and Matplotlib
histogram rendering independent of sampled paths. The suite uses the Agg backend.
`verification_gui.png` is a rendered simulation preview.

These checks verify software behavior; they do not establish physical validation.
