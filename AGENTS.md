# HeliosMC — Development Rules

## Project Overview

HeliosMC is an educational Monte Carlo neutron
transport simulator written in Python.

The project is developed for a scientific research
competition.

Maintain a clean, understandable, modular architecture.

## 1. General Principles

- Prefer simplicity over unnecessary abstraction.
- Preserve existing functionality.
- Reuse existing modules whenever possible.
- Avoid unnecessary refactoring.
- Keep the code understandable to its student developer.
- Explain important architectural decisions.
- Do not introduce speculative features.

## 2. File Creation Policy

Do not create new files unless necessary.

Before creating a file, determine whether an
existing file can serve the same purpose.

Allowed directories:

- Root: existing application modules and configuration.
- tests/: automated tests.
- benchmarks/: performance benchmarks.
- benchmarks/results/: benchmark outputs.
- docs/: project documentation.
- data/: approved datasets.

Do not create temporary files in the project root.

Do not create duplicate benchmark scripts.

Do not generate a new Markdown report for every task.

Prefer updating existing documentation.

Do not rename, move or delete existing files without
checking imports, references and tests.

## 3. Dependency Policy

Currently approved third-party dependencies:

- numpy
- numba
- matplotlib
- scipy
- pytest

Python standard-library modules are allowed.

Do not install or introduce new third-party
dependencies without explicit user approval.

Potential future dependencies requiring approval:

- PyQt6
- pandas
- h5py
- torch
- openmc

Prefer existing dependencies and standard-library
solutions when technically reasonable.

Do not modify requirements.txt without explaining why.

Do not run pip install automatically without approval.

## 4. Architecture Rules

Keep the existing modular structure.

Do not introduce additional abstraction layers
without demonstrated necessity.

Avoid unnecessary classes, wrappers and managers.

Keep the physics engine independent from GUI code.

Keep benchmarks and tests separate from application code.

Preserve existing public APIs unless a change
is explicitly approved.

Do not modify physical assumptions or formulas
during performance-only tasks.

## 5. Temporary Files

Use temporary directories outside the repository
when possible.

Do not leave temporary test directories after
successful execution.

Do not delete user-created files.

Never commit:

- __pycache__/
- .pytest_cache/
- .venv/
- .test-tmp-*/
- *.pyc
- API keys or secrets

## 6. Development Workflow

Before editing:

1. Inspect relevant existing files.
2. Check Git status.
3. Identify the smallest set of files requiring changes.
4. Explain the intended modifications.
5. Identify any proposed new files or dependencies.

During implementation:

1. Make focused changes.
2. Avoid unrelated modifications.
3. Reuse existing functionality.
4. Run relevant tests.
5. Avoid generating unnecessary artifacts.

After implementation:

Provide a concise summary containing:

- Files modified.
- Files created.
- Files deleted.
- Dependencies added.
- Tests executed.
- Benchmark results, if relevant.
- Remaining problems.

## 7. Approval Requirements

Explicit approval is required before:

- Installing new third-party libraries.
- Creating new production modules.
- Restructuring directories.
- Deleting existing files.
- Changing public APIs.
- Replacing major algorithms.
- Modifying physical models.
- Running destructive Git commands.
- Pushing commits to GitHub.

Small test fixtures and temporary benchmark outputs
may be created inside their designated directories
without separate approval.

## 8. Code Quality

- Use descriptive names.
- Add type hints where useful.
- Avoid unnecessary complexity.
- Avoid premature optimization.
- Document non-obvious numerical assumptions.
- Maintain backward compatibility where practical.
- Add regression tests for bug fixes.

## 9. Scientific Integrity

Do not claim scientific accuracy without validation.

Distinguish software correctness from physical validity.

Document assumptions, units and limitations.

Do not fabricate benchmark results.

## 10. Task Completion

A task is complete only when:

- The requested behavior is implemented.
- Relevant tests pass.
- No unnecessary files remain.
- No unapproved dependencies were introduced.
- Changes are explained clearly.

Do not automatically commit or push changes.