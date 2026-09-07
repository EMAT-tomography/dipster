import tomosipo as tp
import os
import torch
import astra
import sys
from collections.abc import Iterable
from . import util

def fp(rec, ang):
    s, d, c = rec.shape[0], rec.shape[1], rec.shape[3]

    # get device of reconstruction
    dev = rec.device
    rec = torch.permute(rec, (1,0,2,3))
    angs = (ang+90) * torch.tensor(torch.pi / 180)

    angs = util.torch_to_np(angs)
    pg = tp.parallel(angles = angs, shape = (d,s), size = (1,1))
    vg = tp.volume(shape = (d,s,s), size = (1,1,1))
    A = tp.operator(vg,pg)


    # Run forward projection
    sino_temp = torch.zeros(A.range_shape).to(dev)
    sino = torch.zeros(A.range_shape[0], A.range_shape[1], A.range_shape[2], c).to(0)
    for i in range(c):
        sino_temp = A(rec[:,:,:,i])
        sino[:,:,:,i] = sino_temp
    sino = torch.permute(sino, (2,0,1,3))

    return sino


def bp(sino, ang, iters = 100):
    s, d, c = sino.shape[0], sino.shape[1], sino.shape[3]
    dev = sino.device
    sino = torch.permute(sino, (3,2,0,1))
    angs = (ang+90)* torch.tensor(torch.pi / 180)

    rec = torch.zeros(c,s,s,d).to(dev)
    if not isinstance(angs, Iterable):
        angs = torch.tensor([angs])

    angs = util.torch_to_np(angs)
    for i in range(d):
        if isinstance(angs[i], Iterable):
            sino_temp = sino[:,:,:,i]
        else:
            sino_temp = sino[:,:,:,i]
        vg = tp.volume(shape = (c,s,s), size = (1,1,1))
        pg = tp.parallel(angles = angs[i], shape = (c,s), size = (1,1))
        A = tp.operator(vg,pg)
        rec_temp = torch.zeros(A.domain_shape).to(dev)

        if iters > 0:
            R = 1 / A(torch.ones(A.domain_shape, device=0))
            C = 1 / A.T(torch.ones(A.range_shape, device=0))
            torch.clamp(R, max=1 / tp.epsilon, out=R)
            torch.clamp(C, max=1 / tp.epsilon, out=C)

            for k in range(iters):
                rec_temp += C * A.T(R * (sino_temp - A(rec_temp)))
        else:
            rec_temp = A.T(sino_temp)
        rec[:,:,:,i] = rec_temp
    rec = torch.permute(rec, (1,3,2,0))
    return rec
