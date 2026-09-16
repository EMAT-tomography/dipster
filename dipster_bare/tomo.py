'''
Forward and backward projection methods of the volumes.
'''
import tomosipo as tp
import torch

from . import util


def fp(volume, angles):
    '''
    Forward projection of a volume into a sinogram at known angles in torch.

    Maps the differentiable reconstruction into the projection space.

    Parameters
    ----------
    volume : torch.Tensor
        Volume, shape [inplane_size, outplane_size, inplane_size, channels],
        equivalent to [     x      ,       y      ,      z      ,    c    ]
        x and z direction of the volume must have the same size.

    angles : torch.Tensor or float
        Tilt angle(s), in DEGREES. A single scalar (one angle for the whole
        volume) or a 1-D tensor of length nb_angles.
        The angles should be given in the microscope convention (-90 to 90 deg)

    Returns
    -------
    torch.Tensor
        Sinogram, shape [inplane_size, outplane_size, nb_angles, channels],
        equivalent to   [     x      ,       y      , nb_angles,     c   ].
    '''
    # Extract the relevant sizes from the volume
    inplane_size, outplane_size, channels = (volume.shape[0],
                                             volume.shape[1],
                                             volume.shape[3])

    # Preprocessing and setup of the variables
    device = volume.device
    volume = torch.permute(volume, (1, 0, 2, 3))  # volume into [y, x, z, c]
    angles = (angles + 90) * torch.tensor(torch.pi / 180)
    angles = util.torch_to_np(angles)

    proj_geometry = tp.parallel(angles=angles,
                                shape=(outplane_size, inplane_size),
                                size=(1, 1))
    vol_geometry = tp.volume(shape=(outplane_size, inplane_size, inplane_size),
                             size=(1, 1, 1))

    A = tp.operator(vol_geometry, proj_geometry)

    # Run forward projection
    sino_temp = torch.zeros(A.range_shape).to(device)  # [y, nb_angles, x]
    sino = torch.zeros(A.range_shape[0],
                       A.range_shape[1],
                       A.range_shape[2],
                       channels).to(device)  # [y, nb_angles, x, c]

    for i in range(channels):
        sino_temp = A(volume[..., i])

        sino[..., i] = sino_temp  # [y, nb_angles, x]

    sino = torch.permute(sino, (2, 0, 1, 3))  # reorder to [x, y, nb_angles, c]
    return sino


def bp(sino, angles, iters=0):
    '''
    Back projection of a sinogram into a volume at known angle in torch.

    Maps a projection space gradient back to volume space.

    Can be used in two ways:
    - Differentiation, the backward gradient of fp (default, iters=0)
    - Reconstruction, basic SIRT reconstruction (iters>0)

    Parameters
    ----------
    sino : torch.Tensor
        Sinogram, shape  [inplane_size, outplane_size, nb_angles, channels],
         equivalent to   [     x      ,       y      , nb_angles,     c   ].
        x and z direction of the volume must have the same size.

    angles : torch.Tensor or float
        Tilt angle(s), in DEGREES. A single scalar (one angle for the whole
        volume) or a 1-D tensor of length nb_angles.
        The angles should be given in the microscope convention (-90 to 90 deg)

    Returns
    -------
    rec_vol : torch.Tensor
        Volume, shape [inplane_size, outplane_size, inplane_size, channels],
        equivalent to [     x      ,       y      ,      z      ,    c    ]
    '''
    # Extract the relevant sizes from the volume
    inplane_size, outplane_size, channels = (sino.shape[0],
                                             sino.shape[1],
                                             sino.shape[3])

    device = sino.device
    sino = torch.permute(sino, (1, 2, 0, 3))  # sino into [y, nb_angles, x, c]
    angles = (angles + 90) * torch.tensor(torch.pi / 180)
    angles = util.torch_to_np(angles)

    vol_geometry = tp.volume(shape=(outplane_size, inplane_size, inplane_size),
                             size=(1, 1, 1))
    proj_geometry = tp.parallel(angles=angles,
                                shape=(outplane_size, inplane_size),
                                size=(1, 1))

    A = tp.operator(vol_geometry, proj_geometry)

    # Run backward projection
    rec_temp = torch.zeros(A.domain_shape).to(device)
    rec_vol = torch.zeros(outplane_size,
                          inplane_size,
                          inplane_size,
                          channels).to(device)  # [y, x, z, c]

    for i in range(channels):
        sino_temp = sino[..., i]  # [y, nb_angles, x]

        if iters == 0:
            rec_temp = A.T(sino_temp)
        else:
            R = 1 / A(torch.ones(A.domain_shape, device=device))
            C = 1 / A.T(torch.ones(A.range_shape, device=device))
            torch.clamp(R, max=1 / tp.epsilon, out=R)
            torch.clamp(C, max=1 / tp.epsilon, out=C)

            for _ in range(iters):
                rec_temp += C * A.T(R * (sino_temp - A(rec_temp)))

        rec_vol[..., i] = rec_temp  # [y, x, z]

    rec_vol = torch.permute(rec_vol, (1, 0, 2, 3))  # [x, y, z, c]
    return rec_vol
