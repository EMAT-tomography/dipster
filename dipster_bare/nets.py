import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from . import util

def get_params(net):
    params = []
    params += [x for x in net.parameters() ]
    return params


class Manifold():
    def __init__(self, times, depth, manifold_size, device):
        #cast times and depth to float
        self.times = float(times)
        self.depth = float(depth)-1
        self.device = device

        self.manifold_size = manifold_size

    def get_value(self, ang, z, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t).to(self.device)
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z).to(self.device).to(torch.float32)
        t_norm = t / self.times
        z_norm = z / self.depth

        if self.manifold_size == 2:
            manifold = torch.stack([z_norm, t_norm])

        elif self.manifold_size == 3:
            ang_norm = (ang + 180) / 180
            # print(z.shape, t.shape, ang_norm.shape, z_norm.shape, t_norm.shape)
            manifold = torch.stack([ang_norm, z_norm, t_norm])
        else:
            x = torch.cos((ang + 90) * torch.pi / 180).to(self.device)
            y = torch.sin((ang + 90) * torch.pi / 180).to(self.device)
            manifold = torch.stack([x, y, z_norm, t_norm])

        if len(manifold.shape) < 2:
            manifold = manifold[:, None]
        manifold = manifold.permute(1, 0).float().to(self.device)
        return manifold

    def to_dict(self):
        return {
            'times': self.times,
            'depth': self.depth,
            'device': self.device
        }

    def load_state_dict(self, d):
        self.times = d['times']
        self.depth = d['depth']
        self.device = d['device']


def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MappingNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # latent_dim is the manifold
        latent_dim = opt.latent_dim

        # style dim is the latent vector between mapnet and cnn
        # set as style_size^2 so it can be resized for the cnn
        style_dim = opt.style_size**2 * opt.input_nch  # Added input_nch

        # size of the FC layers
        hidden_dim = opt.hidden_dim

        # number of FC layers
        depth = opt.depth

        # build the net
        layers = []
        if depth == -1:
            layers += [nn.Linear(latent_dim, style_dim)]
        else:
            layers += [nn.Linear(latent_dim, hidden_dim)]
            layers += [nn.ReLU()]
            for _ in range(depth):
                layers += [nn.Linear(hidden_dim, hidden_dim)]
                layers += [nn.ReLU()]
            layers += [nn.Linear(hidden_dim, style_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        out = self.net(z)
        return out


class Net(nn.Module):
    def __init__(self, opt):
        super().__init__()
        inp_ch=opt.input_nch
        ndf=opt.proj_size
        out_ch=opt.output_nch
        Nr=opt.Nr
        num_ups=int(np.log2(opt.up_factor))
        need_bias=opt.need_bias
        upsample_mode=opt.upsample_mode

        # Build conv layers
        layers = [conv(inp_ch, ndf, 3, bias=need_bias),
                  nn.BatchNorm2d(ndf),
                  nn.ReLU(True)]

        for _ in range(Nr):
            layers += [conv(ndf, ndf, 3, bias=need_bias),
                       nn.BatchNorm2d(ndf),
                       nn.ReLU(True)]

        for _ in range(num_ups):
            layers += [nn.Upsample(scale_factor=2, mode=upsample_mode),
                       conv(ndf, ndf, 3, bias=need_bias),
                       nn.BatchNorm2d(ndf),
                       nn.ReLU(True)]
            for _ in range(Nr):
                layers += [conv(ndf, ndf, 3, bias=need_bias),
                           nn.BatchNorm2d(ndf),
                           nn.ReLU(True)]

        # Output
        layers += [conv(ndf, out_ch, 3, bias=need_bias)]

        # Final optional activation to physically bound the values
        act = getattr(opt, 'output_activation', 'none')
        if act == 'sigmoid':
            layers += [nn.Sigmoid()]            # smooth bound to (0, 1)
        elif act in ('hardtanh', 'clamp'):
            layers += [nn.Hardtanh(0.0, 1.0)]   # hard clip to [0, 1] (dead grad outside)
        elif act == 'relu':
            layers += [nn.ReLU(True)]
        elif act not in ('none', None):
            raise ValueError(f"Unknown output_activation '{act}' "
                             f"(expected 'none', 'relu', 'sigmoid', or 'hardtanh').")

        self.net = nn.Sequential(*layers)

    def forward(self, z, s=None):
        out = self.net(z)
        return out


class AffineCorrection(nn.Module):
    """One learnable 2x3 affine per frame, initialised to the identity.

    Warps a clean model projection towards the measured
    projection, so the reconstruction network can output particles that better 
    adhere to the spatiotemporal priors of the manifold while the affine absorbs
    residual acquisition artefacts (per-frame misalignment / shift / scale / skew). 
    One transform per frame because each frame is a unique (angle, time) pair.
    """

    def __init__(self, n_frames):
        super().__init__()
        ident = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        self.theta = nn.Parameter(ident.repeat(n_frames, 1, 1).clone())  # [F, 2, 3]

    def warp(self, img, frame_idx):
        """Warp ``img`` ([N, C, H, W]) by the affines of ``frame_idx`` ([N])."""
        theta = self.theta[frame_idx]                                    # [N, 2, 3]
        grid = F.affine_grid(theta, img.shape, align_corners=False)
        return F.grid_sample(img, grid, align_corners=False, padding_mode='zeros')

    def identity_penalty(self):
        """Mean squared deviation from the identity transform (keeps corrections minimal)."""
        ident = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=self.theta.device)
        return ((self.theta - ident) ** 2).mean()

    @torch.no_grad()
    def decompose(self, proj_size=None):
        """Decompose each frame's 2x3 affine into interpretable components"""
        th = self.theta.detach()
        a, b, tx = th[:, 0, 0], th[:, 0, 1], th[:, 0, 2]
        c, d, ty = th[:, 1, 0], th[:, 1, 1], th[:, 1, 2]
        sx = torch.sqrt(a * a + c * c)
        det = a * d - b * c
        sy = det / sx
        rotation = torch.rad2deg(torch.atan2(c, a))
        skew = torch.rad2deg(torch.atan2(a * b + c * d, det))
        s = (proj_size / 2.0) if proj_size else 1.0
        return {'translation_x': (tx * s).cpu(), 'translation_y': (ty * s).cpu(),
                'rotation_deg': rotation.cpu(), 'scale_x': sx.cpu(),
                'scale_y': sy.cpu(), 'skew_deg': skew.cpu()}
