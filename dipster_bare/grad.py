import torch
from torch import nn, optim
from torch import Tensor
import numpy as np
from . import tomo, util

class custom_grad_func(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    NuFFT from https://github.com/jyhmiinlin/pynufft
    """

    @staticmethod
    def forward(ctx,input_r, angle):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.dev = input_r.get_device()
        ctx.angle=angle
        x_val = input_r
        y = tomo.fp(x_val, angle)
        y_cut = torch.zeros(y.shape[0],y.shape[1],1,y.shape[3]).to(ctx.dev)
        for i in range(y.shape[3]):
            for j in range(y.shape[1]):
                y_cut[:,j,0,i] = y[:,j,j,i]
        return y_cut


    @staticmethod
    def backward(ctx,grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        angle = ctx.angle
        if torch.numel(angle) == 1:
            angle = torch.tensor([angle.item()])

        yc = grad_output
        grad_output = tomo.bp(yc, angle, 1)

        #return grad_output
        return grad_output, None, None, None, None, None, None

class single_angle_fp_func(torch.autograd.Function):
    """Differentiable projection of a whole volume at one angle.

    forward:  vol [s, d, s, c]  -> proj [det=s, depth=d, 1, c]   (tomo.fp, 1 angle)
    backward: proj-grad         -> vol-grad                       (tomo.bp, per-depth)

    Unlike custom_grad_func this never builds the [s, d, d] (det x slice x angle)
    diagonal sinogram: every depth row of the volume is projected once at the frame's
    single angle, which is exactly the per-slice projection (tomosipo parallel beam
    projects each depth row independently). Used by the affine phase, where all of a
    frame's depth-slices share one angle, to avoid the ~proj_size x memory increase.
    """

    @staticmethod
    def forward(ctx, vol, angle):
        ctx.dev = vol.get_device()
        ctx.angle = angle
        ctx.depth = vol.shape[1]
        return tomo.fp(vol, angle.reshape(1))            # single-angle 2-D projection

    @staticmethod
    def backward(ctx, grad_output):
        ang = ctx.angle.reshape(1).expand(ctx.depth)     # one (equal) angle per depth row
        grad_vol = tomo.bp(grad_output.contiguous(), ang, 1)
        return grad_vol, None


def single_angle_fp(vol, angle):
    """Project ``vol`` ([s, d, s, c]) at a single ``angle`` -> ``[det, depth, 1, c]``."""
    return single_angle_fp_func.apply(vol, angle)


class CustomGradient(nn.Module):
    def __init__(self,ImageSize,batch,channels):
        super(CustomGradient,self).__init__()
        self.X=Tensor(ImageSize,batch, ImageSize,channels).fill_(0).cuda()
        #self.X=Tensor(ImageSize,ImageSize).fill_(0)

    def forward(self,angles):
        return  custom_grad_func.apply(self.X, angles)
