'''
DIP-STER optimisation gradients.

Provides the differentiable diagonal projection.
'''
import torch
from torch import nn
from . import tomo

class ProjectionGradFunc(torch.autograd.Function):
    '''
    Differentiable projection of a reconstruction volume.
    To get a single call of the forward and backward projection function,
    they are calculated at every angles at once.
    Only the necessary ones are kept with the diagonal cut.

    forward:
        vol  [x, y, z, c] -> diagonal sinogram [x, y, 1, c]
        tomo.fp produces the full multi-angle sinogram [x, y, nb_angles, c];
        the diagonal over (y, nb_angles) keeps slice j projected at its own
        assigned angle.

    backward:
        sino [x, y, 1, c] -> vol gradient [x, y, z, c]
        Exact adjoint of forward: the diagonal is re-expanded into the full
        [x, y, y, c] sinogram, then tomo.bp is applied.
    '''

    @staticmethod
    def forward(ctx, vol, angle):
        '''
        Forward: project a volume at per-slice angles and extract the diagonal.

        Parameters
        ----------
        vol : torch.Tensor
            Volume, shape [x, y, z, c]
            x and z direction of the volume must have the same size.

        angle : torch.Tensor
            Tilt angle for each of the 'y' slices, in DEGREES (microscope
            convention, -90 to 90 deg).
            Length must equal to y.

        Returns
        -------
        torch.Tensor
            Diagonal sinogram, shape [x, y, 1, c]
        '''
        ctx.angle = angle

        # Compress all the batch processing into a single fp call
        y = tomo.fp(vol, angle)               # [x, y, nb_angles, c]
        y_cut = y[:, torch.arange(y.shape[1]),
                  torch.arange(y.shape[1]), :]  # [x, y, 1, c]
        return y_cut

    @staticmethod
    def backward(ctx, sino):
        '''
        Backward: exact adjoint of the diagonal projection.

        Re-expands the sinogram gradient onto the diagonal of the full
        [x, y, y, c] sinogram, then applies tomo.bp, so the result is the exact
        gradient of forward with respect to the volume.

        Parameters
        ----------
        sino : torch.Tensor
            Sinogram gradient, shape [x, y, 1, c]

        Returns
        -------
        tuple
            Volume gradient, shape [x, y, z, c]
            then None (angle is not differentiable).
        '''
        angle = ctx.angle

        # Refill sinogram to do a single bp call
        x, y, _, c = sino.shape  # [x, y, 1, c]
        sino_full = torch.zeros(x, y, y, c,
                                device=sino.device,
                                dtype=sino.dtype)  # [x, y, nb_angles, c]
        sino_full[:, torch.arange(y), torch.arange(y), :] = sino

        return tomo.bp(sino_full, angle), None


class ProjectionGradient(nn.Module):
    '''
    Trainable reconstruction volume with differentiable projection.

    Holds the shared volume ``self.vol_rec`` (the optimised parameter).
    ``forward`` projects it (via ProjectionGradFunc) at the given angles and
    returns the diagonal sinogram used in the data loss.
    '''
    def __init__(self, image_size, batch, channels, device):
        super().__init__()
        self.rec_vol = torch.zeros((image_size, batch, image_size, channels),
                                   device=device)

    def forward(self, angles):
        '''
        Project the stored volume at the given angles.

        Parameters
        ----------
        angles : torch.Tensor
            Tilt angle for each of the ``batch`` slices, in DEGREES
            (microscope convention, -90 to 90 deg). Length must equal ``batch``

        Returns
        -------
        torch.Tensor
            Diagonal sinogram, shape [image_size, batch, 1, channels].
        '''
        return ProjectionGradFunc.apply(self.rec_vol, angles)
