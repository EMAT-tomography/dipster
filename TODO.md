# TODO — dipster_bare Improvements

Prioritized list of changes identified during code review (Sep 2026).

---

## P0 — Blocking (breaks API or gives wrong results)

### 1. `tomo.bp` backward is not a proper adjoint
- **Where:** `tomo.py:55-62`
- **Issue:** The SIRT branch (one iteration: `rec += C * A.T(R * (sino - A(rec)))`) is used as the autograd backward, but backward should be the pure adjoint `A.T(sino)` (the `iters=0` branch). This makes the gradient of `fp` with respect to the volume mathematically incorrect.
- **Todo:**
  - [x] Make `bp`'s backward path always use `rec_temp = A.T(sino_temp)` (force `iters=0` semantics).
  - [ ] Add a numerical gradient-checker test (`torch.autograd.gradcheck` or finite-difference) to validate `fp`/`bp` adjoint.

### 2. `custom_grad_func.backward` returns 7 gradients for 2 inputs
- **Where:** `grad.py:49` — `return grad_output, None, None, None, None, None, None`
- **Issue:** `forward(ctx, input_r, angle)` has 2 user inputs; PyTorch expects exactly 2 gradient outputs. Returning 7 is a latent `RuntimeError` or silent mis-assignment.
- **Todo:**
  - [x] Change to `return grad_output, None`.

### 3. `Manifold` loses `manifold_size` on save/load
- **Where:** `nets.py:47-57`
- **Issue:** `to_dict()` and `load_state_dict()` do not persist `manifold_size`. After `Solver.from_state_dict`, calling `get_value` raises `AttributeError: 'Manifold' object has no attribute 'manifold_size'`.
- **Todo:**
  - [ ] Add `manifold_size` to `to_dict()` and `load_state_dict()`.
  - [ ] Add a `from_state_dict` → `get_value` round-trip test.

### 4. `Sinogram.from_file` calls nonexistent `cls.from_mat`
- **Where:** `data.py:36`
- **Issue:** Any `.mat` file load triggers `AttributeError: 'Sinogram' object has no attribute 'from_mat'`.
- **Todo:**
  - [ ] Implement `from_mat` (using `scipy.io.loadmat`) or remove the `.mat` branch and update the error message.

### 5. Hard-coded device assumption (GPU 0 / CUDA)
- **Where:** `grad.py:86` (`.cuda()`), `tomo.py:25` (`.to(0)`), `tomo.py:56-57` (`device=0`)
- **Issue:** CPU-only setups and GPU indices ≠ 0 are broken. Contradicts `params.dev` which allows selection.
- **Todo:**
  - [ ] Thread `params.dev` (or equivalent device object) through `CustomGradient`, `tomo.fp`, `tomo.bp`.
  - [ ] Accept a `device` parameter from the caller rather than hard-coding.

### 6. `_validation` warps with wrong affine index
- **Where:** `solver.py:380` — `affine.warp(rec[i], i)`
- **Issue:** `i` is the enumerate ordinal, not the frame index. Should use `val_i` (=`val_indices[i]`) to fetch the correct per-frame correction.
- **Todo:**
  - [ ] Change to `self.affine.warp(rec[i], val_i).squeeze()`.

---

## P1 — Dead or broken features (import-time or runtime failures)

### 7. `tune.py` Optuna path is non-functional
- **Where:** `tune.py:106,118`
- **Issue:** Passes `train_frames` and `eval_callback` to `Solver.train`, but `Solver.train` (`solver.py:98`) **ignores both parameters** — the signature accepts them but never uses them. Also `solver.params.max_steps` is never read anywhere, so the `n_steps` budget has no effect.
- **Todo:**
  - [ ] Honor `train_frames` in `Solver.train` (override `self.train_indices` when non-None).
  - [ ] Wire `eval_callback` into the loss-check loop (in addition to / in place of the built-in intval).
  - [ ] Expose a `max_steps` param that caps total training steps.
  - [ ] Verify `tune.py` end-to-end with a small synthetic tilt series.

### 8. `hyper.py` Ray path is fully broken
- **Where:** `hyper.py:27-28, 65-67`
- **Issue:**
  - `solver.results._tables` doesn't exist (should be `.tables`).
  - `tune.report(loss=...['losses_losses'], ...)` uses wrong keys.
  - `best_trained_model` (line 67) is an undefined name → `NameError`.
- **Todo:**
  - [ ] Rewrite `_hypertrain` against the real `Report` API (`.tables`, correct metric keys).
  - [ ] Remove the dangling `best_trained_model` block or reconstruct it from state.
  - [ ] Decide: keep Ray path or delete in favour of `tune.py` (Optuna).

### 9. Top-level `import mrcfile` in `util.py` breaks imports
- **Where:** `util.py:12`
- **Issue:** `mrcfile` is not in `install_requires`. `import dipster_bare` fails on any machine without mrcfile.
- **Todo:**
  - [ ] Move `import mrcfile` inside the `read_mrcfile` function (lazy import).
  - [ ] Add `mrcfile` to `install_requires` in `setup.cfg` (or document as optional).

### 10. Unused `import astra` in `tomo.py`
- **Where:** `tomo.py:4`
- **Issue:** `astra` is imported at module top but never used in this file.
- **Todo:**
  - [ ] Remove `import astra` from `tomo.py` (if it's not needed elsewhere for tomosipo internals).

### 11. `tune.py` `style_sizes` parameter is ignored
- **Where:** `tune.py:84`
- **Issue:** `trial.suggest_categorical('style_size', [8, 16])` hard-codes the list, ignoring the `style_sizes` parameter passed into `make_objective` and `tune`.
- **Todo:**
  - [ ] Use the `style_sizes` list from the parameter instead of `[8, 16]`.

---

## P2 — Correctness & robustness

### 12. `save_model` rarely saves the true best checkpoint
- **Where:** `solver.py:489, 494` — `if values_arr[-1, 1] == best_score[1]:`
- **Issue:** Float equality gate: the best checkpoint is only saved if the latest logged value happens to match all-time best. Otherwise the save silently no-ops and the best is lost.
- **Todo:**
  - [ ] Always save the `state_dict` when the latest value equals the best (within tolerance or exact for metrics).
  - [ ] Alternatively: track best explicitly and save unconditionally when improved, logging step at which best was achieved.

### 13. `Params()` design: most attributes undefined when `ts=None`
- **Where:** `params.py:7-63`
- **Issue:** `Params.__init__` only defines attributes inside `if ts is not None:`. `Params.from_dict` calls `cls()` (no `ts`) then manually assigns from state — any field missing from a legacy state dict is silently absent, leading to `AttributeError` later.
- **Todo:**
  - [ ] Define all attributes unconditionally in `Params.__init__` with defaults.
  - [ ] Only override data-dependent ones (`proj_size`, `frames`) when `ts` is provided.

### 14. `wandb_local_dir` default is a Windows path
- **Where:** `params.py:57` — `r"..\..\wandb"`
- **Issue:** Windows-style relative path; wrong/dangerous on Linux.
- **Todo:**
  - [ ] Default to a platform-neutral path (e.g. `"./wandb"` or `os.path.expanduser("wandb")`).
  - [ ] Optionally warn on startup if the path does not exist and cannot be created.

### 15. Division by zero in `normalize` / `reporting.quantify`
- **Where:** `data.py:74`, `reporting.py:31`
- **Issue:** Image with `max == min` (flat) → ZeroDivisionError.
- **Todo:**
  - [ ] Guard: `rng = max(d.max() - d.min(), 1e-8)`.

### 16. No input validation in `Sinogram.__init__`
- **Where:** `data.py:20-28`
- **Issue:** No checks that `data.ndim == 3`, `len(angles) == data.shape[0]`, `len(times) == data.shape[2]`.
- **Todo:**
  - [ ] Add assertions / `raise ValueError` for shape and length consistency.

### 17. Shape convention is inconsistent between docstring and actual usage
- **Where:** `data.py` docstring says `(n_projections, x, y)`; `params.py:10` reads `shape[0]` as `proj_size`; `solver` indexes `ts.data[:, depth, frame, :]`.
- **Issue:** The canonical shape is `(x, depth, frames[, ch])` but it's never stated clearly.
- **Todo:**
  - [ ] Write a clear, unambiguous shape docstring at the top of `Sinogram`.
  - [ ] Add a `# NOTE: data shape is (x, y, frames)` comment in `solver.py` where indexing happens.

---

## P3 — Performance

### 18. `Report._update_values` rebuilds wandb plot each call
- **Where:** `reporting.py:61`
- **Issue:** Reconstructs full `wandb.Table` + `wandb.plot.line` from **entire history** on every scalar logged → O(n²) I/O/memory.
- **Todo:**
  - [ ] Use plain `wandb.log({key: value})` per step.
  - [ ] Build plots/tables only at `publish()` time or once per epoch.

### 19. Python-level double loops in `fp` / `bp` / `custom_grad_func`
- **Where:** `grad.py:28-31`, `tomo.py:26-29`, `tomo.py:45-65`
- **Issue:** Per-depth Python loops calling tomosipo operator repeatedly.
- **Todo:**
  - [ ] Vectorize the diagonal extraction in `grad.py` using numpy/torch advanced indexing (`np.diagonal` or `torch.diagonal`).
  - [ ] Batch per-depth tomosipo calls where possible (check tomosipo API for vectorized angles).

---

## P4 — Maintainability & hygiene

### 20. Unused imports
- **Where:**
  - `solver.py`: `sys`, `nn`, `F`, `trange`
  - `util.py`: duplicate `import numpy as np` (line 6 & 9), `pi` from math
- **Todo:**
  - [ ] Remove all unused imports.

### 21. No `__all__`, no `__version__` in package
- **Where:** `dipster_bare/__init__.py`
- **Todo:**
  - [ ] Add `__version__ = "0.1.0"` and `__all__ = ["Solver", "hypertrain", "Params", "Sinogram", "normalize"]`.

### 22. No test suite
- **Where:** (entire project)
- **Issue:** Only `notebooks/Train.ipynb` exercises the happy path. No automated tests for edge cases, adjoint correctness, save/load round-trip, `tune.py`, or `hyper.py`.
- **Todo:**
  - [ ] Add `tests/` directory.
  - [ ] Minimum test cases:
    - `fp`/`bp` adjoint (numerical gradient check)
    - `Solver.from_state_dict` → `state_dict` round-trip
    - `Sinogram` shape validation
    - `Params.to_dict` / `from_dict` round-trip
    - `tune.py` objective on a 32×32 synthetic tilt series (smoke)
  - [ ] Wire into `pytest`; add to CI when available.

---

## Suggested execution order

| Batch | Items | Rationale |
|-------|-------|-----------|
| 1 | P0-1, P0-2, P0-3, P0-6, P0-4, P2-12 | Correctness of the reconstruction and checkpoint saving |
| 2 | P0-5, P1-9, P1-10 | Remove import failures so `import dipster_bare` works out of the box |
| 3 | P1-7, P1-11, P1-8 | Make `tune.py` functional and decide fate of `hyper.py` |
| 4 | P2-13, P2-14, P2-15, P2-16, P2-17 | Robustness hardening |
| 5 | P3-18, P3-19 | Performance if runs are long |
| 6 | P4-20, P4-21, P4-22 | Hygiene + test suite |
