"""Optuna-based hyperparameter tuning for dipster_bare.

Tunes four hyperparameters of the reconstruction network:

    style_size         -- latent grid size (up_factor is DERIVED as
                          proj_size // style_size so the CNN output stays
                          proj_size x proj_size, e.g. 128x128)
    depth              -- number of MappingNet fully-connected layers
    hidden_dim         -- MappingNet FC width
    noise_regularizer  -- total-variation weight

Objective: mean projection-space PSNR on a held-out set of tilt frames
(maximised). A few frames are held out of training; each candidate is scored by
reconstructing those frames, re-projecting the full volume, and comparing to the
measured projections -- so noise_regularizer is rewarded for generalising to
unseen angles.

This module imports optuna; install it (``pip install optuna plotly``) before
importing. It is intentionally NOT imported from ``dipster_bare.__init__`` so the
base package keeps importing without optuna present.
"""

import numpy as np
import torch
import optuna
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from . import tomo, util
from .solver import Solver


def split_frames(n_frames, n_val=3, seed=0):
    """Split frame indices into (train_idx, val_idx) numpy int arrays.

    Validation frames are spread evenly across the range so they cover the
    angle/time span; every other frame is used for training.
    """
    n_frames = int(n_frames)
    n_val = max(1, min(int(n_val), n_frames - 1))
    val_idx = np.unique(np.linspace(0, n_frames - 1, n_val, dtype=int))
    val_set = set(val_idx.tolist())
    train_idx = np.array([i for i in range(n_frames) if i not in val_set], dtype=int)
    return train_idx, val_idx


@torch.no_grad()
def projection_ssim(solver, ts, frames):
    """Mean projection-space SSIM over ``.

    For each frame this assembles the FULL re-projected tilt image, looping over
    every depth slice and compares it to the
    measured projection ts.data[:, :, f, 0].
    """
    p = solver.params
    dev = p.dev
    scores = []
    for f in np.atleast_1d(frames):
        idx = torch.tensor([int(f)], device=dev)
        ang = ts.angles[idx]
        tim = ts.times[idx]
        rec = np.zeros((p.proj_size, p.proj_size))
        for i in range(p.proj_size):
            val = tomo.fp(solver.reconstruct_slices(ang, [i], tim), ang).squeeze()
            rec[:, i] = util.torch_to_np(val)
        ref = util.torch_to_np(ts.data[:, :, idx, 0].squeeze())
        scores.append(ssim(rec, ref, data_range=ref.max() - ref.min()))
    return float(np.mean(scores))


def _apply_base_params(params, base_params):
    """Copy fixed (non-tuned) settings onto a fresh Params object."""
    if base_params:
        for k, v in base_params.items():
            setattr(params, k, v)


def make_objective(ts, base_params, train_frames, val_frames, n_steps, save_period, style_sizes = (8,16)):
    """Build an Optuna objective: train one candidate, return held-out PSNR."""
    style_sizes = list(style_sizes)

    def objective(trial):
        lr = trial.suggest_float('lr', 5e-6, 5e-4, log=True)
        style_size = trial.suggest_categorical('style_size', [8,16])
        depth = trial.suggest_int('depth', 3, 6)
        hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512])
        noise_regularizer = trial.suggest_float('noise_regularizer', 5e-6, 5e-4, log=True)

        solver = Solver(ts)
        _apply_base_params(solver.params, base_params)

        # Lock CNN output to proj_size x proj_size: up_factor is derived, not tuned.
        proj_size = solver.params.proj_size
        up_factor = proj_size // style_size
        if (proj_size % style_size != 0) or ((up_factor & (up_factor - 1)) != 0) or up_factor < 1:
            raise optuna.TrialPruned(
                f"style_size={style_size} incompatible with proj_size={proj_size} "
                f"(proj_size // style_size must be a power of two)")

        solver.params.style_size = style_size
        solver.params.lr = lr
        solver.params.up_factor = up_factor
        solver.params.depth = depth
        solver.params.hidden_dim = hidden_dim
        solver.params.noise_regularizer = noise_regularizer
        solver.params.max_steps = n_steps
        solver.params.save_period = save_period
        solver.params.evaluate = False     # no wandb run, no per-step .pkl saving
        solver.params.hypertrain = True
        trial.set_user_attr('up_factor', up_factor)

        def cb(step):
            v = projection_ssim(solver, ts, val_frames)
            trial.report(v, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        solver.train(ts, train_frames=train_frames, eval_callback=cb)
        return projection_ssim(solver, ts, val_frames)

    return objective


def tune(ts, trials=30, n_steps=3000, save_period=500, n_val=3, base_params=None,
         style_sizes=(4, 8, 16, 32), storage=None, study_name='dipster_bare', seed=0):
    """Run an Optuna study over (style_size, depth, hidden_dim, noise_regularizer).

    Objective = mean held-out projection PSNR (maximised). Returns the study.

    Parameters
    ----------
    ts          : Sinogram with .data (proj, depth, frames[, 1]), .angles, .times
                  already on the training device.
    trials      : number of Optuna trials (sequential, one GPU).
    n_steps     : training steps per trial (tuning budget; keep small).
    save_period : eval/prune cadence in steps.
    n_val       : number of held-out validation frames.
    base_params : dict of fixed Params attributes (e.g. lr, gamma, batch_size,
                  step_size) applied to every trial.
    style_sizes : candidate latent grid sizes; up_factor = proj_size // style_size.
    storage     : optional Optuna storage URL (e.g. 'sqlite:///dipster_tune.db')
                  to persist / resume the study.
    """
    n_frames = int(ts.data.shape[2]) if ts.data.ndim >= 3 else int(ts.data.shape[0])
    train_frames, val_frames = split_frames(n_frames, n_val=n_val, seed=seed)
    print(f"Frames: {n_frames} total -> {len(train_frames)} train, "
          f"{len(val_frames)} val {val_frames.tolist()}")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage),
    )
    objective = make_objective(ts, base_params, train_frames, val_frames,
                               n_steps=n_steps, save_period=save_period,
                               style_sizes=style_sizes)
    study.optimize(objective, n_trials=trials)

    print(f"\nBest value (held-out projection PSNR): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"  derived up_factor: {study.best_trial.user_attrs.get('up_factor')}")
    return study
