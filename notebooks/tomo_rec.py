import torch
import numpy as np

from tqdm.notebook import tqdm, trange
from kornia import losses as L
from dipster_bare import grad, util
import tomosipo as tp
import math

def make_projector(P, angles):
    """Differentiable multi-angle parallel-beam projector matching tomo.fp's geometry,
    built for a fixed set of angles.

    project(x): x [P, P, P, 1] ([H, depth, W, 1]) -> sino [det, depth, N].
    Backward is the true adjoint A.T, i.e. the exact least-squares gradient
    (unlike grad.single_angle_fp, whose backward is a SIRT-preconditioned backprojection).
    """
    angs = (np.asarray(util.torch_to_np(angles), dtype=float) + 90) * np.pi / 180
    pg = tp.parallel(angles=angs, shape=(P, P), size=(1, 1))
    vg = tp.volume(shape=(P, P, P), size=(1, 1, 1))
    A = tp.operator(vg, pg)

    class _FP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, vol):
            v = vol.permute(1, 0, 2, 3)[..., 0].contiguous()   # [depth, H, W]
            return A(v).permute(2, 0, 1)                        # [det, depth, N]

        @staticmethod
        def backward(ctx, g):
            v = A.T(g.permute(1, 2, 0).contiguous())           # [depth, H, W]
            return v.permute(1, 0, 2).unsqueeze(-1)            # [H, depth, W, 1]

    return _FP.apply


def tv3d(x):
    """Anisotropic (L1) total variation over all three spatial axes.

    x: [H, depth, W, 1]; sums |first differences| along H, depth, and W
    """
    v = x[..., 0]                                  # [H, depth, W]
    return (v[1:, :, :] - v[:-1, :, :]).abs().sum() \
         + (v[:, 1:, :] - v[:, :-1, :]).abs().sum() \
         + (v[:, :, 1:] - v[:, :, :-1]).abs().sum()


def tv_reconstruct(ts, frames, iters=500, lr=1e-2, tv_weight=1e-7,
                   nonneg=True, init=None, verbose=True):
    """TV-regularized tomographic reconstruction of one volume from selected frames.

    Minimizes  sum_f ||A(x)_f - ts.data[:, :, f]||^2 + tv_weight * TV(x) over a volume x.
    The forward projection is batched: a multi-angle tomosipo operator 
    is built once and projects all chosen frames in one call, with its
    true adjoint as the gradient. Regularizer is 3-D total variation (H, depth, W), so it
    also smooths across slices.

    Args:
        ts:        GPU torch Sinogram (ts.data [P, P, frames], ts.angles, ts.times).
        frames:    iterable of frame indices to fit (the chosen projections).
        iters:     optimizer iterations.
        lr:        learning rate.
        tv_weight: regularization weight.
        nonneg:    clamp the volume to physical non-negative densities each step.
        init:      optional start volume [P, P, P, 1].
        verbose:   print the data / TV terms periodically.

    Returns:
        Detached volume tensor of shape [P, P, P, 1] = [H, depth, W, 1].
    """
    P = ts.data.shape[0]
    dev = ts.angles.device
    frames = list(frames)
    data = ts.data[..., 0] if ts.data.ndim == 4 else ts.data   # [P, P, frames]
    data_stack = data[:, :, frames].contiguous()               # [det, depth, N]
    project = make_projector(P, ts.angles[frames])             

    x = (torch.zeros(P, P, P, 1, device=dev) if init is None
         else init.detach().clone().to(dev)).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)

    for it in range(iters):
        opt.zero_grad()
        sino = project(x)                                      # all chosen angles in one call
        data_loss = ((sino - data_stack) ** 2).mean(dim=(0, 1)).sum()   # mean per frame, summed
        tv = tv3d(x)
        tv_loss = tv_weight * tv                                   # 3-D TV (H, depth, W)
        loss = data_loss + tv_loss
        loss.backward()
        opt.step()
        if nonneg:
            with torch.no_grad():
                x.clamp_(min=0)
        if verbose and (it % max(1, iters // 10) == 0 or it == iters - 1):
            print(f"it {it:4d}  data {float(data_loss.detach()):.3e}  tv {float(tv.detach()):.3e}", end = "\r")

    return x.detach(), data_loss.detach(), tv.detach()

def em(A, y, num_iterations, x_init=None, volume_mask=None,
    projection_mask=None, progress_bar=False, callbacks=()):
    """Execute the (Maximum Likelihood) Expectation-Maximization algorithm from https://github.com/ahendriksen/ts_algorithms/blob/master/ts_algorithms/em.py

    If `y` is located on GPU, the entire algorithm is executed on a single GPU.

    IF `y` is located in RAM (CPU in PyTorch parlance), then only the
    forward and backprojection are executed on GPU.

    :param A: `tomosipo.Operator`
        Projection operator
    :param y: `torch.Tensor`
        Projection data
    :param num_iterations: `int`
        Number of iterations
    :param x_init: `torch.Tensor`
        Initial value for the solution. Setting to None will start with ones.
        Setting x_init to a previously found solution can be useful to
        continue with more iterations of EM.
    :param volume_mask: `torch.Tensor`
        Mask for the reconstruction volume. All voxels outside of the mask will
        be assumed to not contribute to the projection data.
        Setting to None will result in using the whole volume.
    :param projection_mask: `torch.Tensor`
        Mask for the projection data. All pixels outside of the mask will
        be assumed to not contribute to the reconstruction.
        Setting to None will result in using the whole projection data.
    :param progress_bar: `bool`
        Whether to show a progress bar on the command line interface.
        Default: False
    :param callbacks:
        Iterable containing functions or callable objects. Each callback will
        be called every iteration with the current estimate and iteration
        number as arguments. If any callback returns True, the algorithm stops
        after this iteration. This can be used for logging, tracking or
        alternative stopping conditions.
    :returns: `torch.Tensor`
        A reconstruction of the volume using num_iterations iterations of EM
    :rtype:

    """
    dev = y.device

    # Compute C
    y_tmp = torch.ones(A.range_shape, device=dev)
    C = A.T(y_tmp)
    C[C < tp.epsilon] = math.inf
    C.reciprocal_()

    if x_init is None:
        x_cur = torch.ones(A.domain_shape, device=dev)
    else:
        with torch.cuda.device_of(y):
            x_cur = x_init.clone()

    if volume_mask is not None:
        x_cur *= volume_mask

    x_tmp = torch.empty(A.domain_shape, device=dev)
    for iteration in trange(num_iterations, disable=not progress_bar):
        A(x_cur, out=y_tmp)
        if projection_mask is not None:
            y_tmp *= projection_mask
        y_tmp[y_tmp < tp.epsilon] = math.inf
        torch.div(y, y_tmp, out=y_tmp)
        A.T(y_tmp, out=x_tmp)
        x_cur *= x_tmp
        x_cur *= C

    return x_cur

def sirt(A, y, num_iterations, min_constraint=None, max_constraint=None, x_init=None, volume_mask=None,
    projection_mask=None, progress_bar=False, callbacks=()):
    """Execute the SIRT algorithm from https://github.com/ahendriksen/ts_algorithms/blob/master/ts_algorithms/sirt.py

    If `y` is located on GPU, the entire algorithm is executed on a single GPU.

    IF `y` is located in RAM (CPU in PyTorch parlance), then only the
    forward and backprojection are executed on GPU.

    :param A: `tomosipo.Operator`
        Projection operator
    :param y: `torch.Tensor`
        Projection data
    :param num_iterations: `int`
        Number of iterations
    :param min_constraint: `float`
        Minimum value enforced at each iteration. Setting to None skips this step.
    :param max_constraint: `float`
        Maximum value enforced at each iteration. Setting to None skips this step.
    :param x_init: `torch.Tensor`
        Initial value for the solution. Setting to None will start with zeros.
        Setting x_init to a previously found solution can be useful to
        continue with more iterations of SIRT.
    :param volume_mask: `torch.Tensor`
        Mask for the reconstruction volume. All voxels outside of the mask will
        be assumed to not contribute to the projection data.
        Setting to None will result in using the whole volume.
    :param projection_mask: `torch.Tensor`
        Mask for the projection data. All pixels outside of the mask will
        be assumed to not contribute to the reconstruction.
        Setting to None will result in using the whole projection data.
    :param progress_bar: `bool`
        Whether to show a progress bar on the command line interface.
        Default: False
    :param callbacks:
        Iterable containing functions or callable objects. Each callback will
        be called every iteration with the current estimate and iteration
        number as arguments. If any callback returns True, the algorithm stops
        after this iteration. This can be used for logging, tracking or
        alternative stopping conditions.
    :returns: `torch.Tensor`
        A reconstruction of the volume using num_iterations iterations of SIRT
    :rtype:

    """
    dev = y.device

    # Compute C
    y_tmp = torch.ones(A.range_shape, device=dev)
    C = A.T(y_tmp)
    C[C < tp.epsilon] = math.inf
    C.reciprocal_()
    # Compute R
    x_tmp = torch.ones(A.domain_shape, device=dev)
    R = A(x_tmp)
    R[R < tp.epsilon] = math.inf
    R.reciprocal_()

    if x_init is None:
        x_cur = torch.zeros(A.domain_shape, device=dev)
    else:
        with torch.cuda.device_of(y):
            x_cur = x_init.clone()

    if volume_mask is not None:
        x_cur *= volume_mask
        C *= volume_mask

    if projection_mask is not None:
        R *= projection_mask

    for iteration in trange(num_iterations, disable=not progress_bar):
        A(x_cur, out=y_tmp)
        y_tmp -= y
        y_tmp *= R
        A.T(y_tmp, out=x_tmp)
        x_tmp *= C
        x_cur -= x_tmp
        if (min_constraint is not None) or (max_constraint is not None):
            x_cur.clamp_(min_constraint, max_constraint)

    return x_cur