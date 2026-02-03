# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import sys
import torch
from tqdm import tqdm
import numpy as np

import easier as esr

swe_dir = os.path.dirname(__file__)
if swe_dir not in sys.path:
    sys.path.append(swe_dir)
from swe_main import ShallowWaterEquation  # type: ignore

class Swap(esr.Module):
    def __init__(self, eqn: ShallowWaterEquation):
        super().__init__()
        self.eqn = eqn
    
    def forward(self):
        self.eqn.h[:] = self.eqn.h_new
        self.eqn.uh[:] = self.eqn.uh_new
        self.eqn.vh[:] = self.eqn.vh_new

class Obj(esr.Module):
    def __init__(self, eqn: ShallowWaterEquation):
        super().__init__()
        self.eqn = eqn
        self.loss = esr.Tensor(torch.tensor([0.0], dtype=torch.float64), mode='replicate')
    
    def forward(self):
        loss = esr.sum(self.eqn.h)
        self.loss[:] = loss

class Optimizer(esr.Module):
    def __init__(self, h: esr.Tensor, grad_h: esr.Tensor):
        super().__init__()
        self.h = h
        self.grah_h = grad_h
    
    def forward(self):
        self.h.sub_(self.grah_h * 100)


if __name__ == "__main__":
    """
    Usage:

    mkdir res
    torchrun --nnodes=1 --nproc_per_node=4 \
        tutorial/shallow_water_equation/swe_main.py --backend=cpu res/ \
        ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default="cpu"
    )
    parser.add_argument(
        "--backend", type=str, choices=["none", "torch", "cpu", "cuda"],
        default='torch'
    )
    parser.add_argument(
        "--comm_backend", type=str, choices=["gloo", "nccl"],
        default='gloo'
    )
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--sim_step", type=int, default=1000)
    parser.add_argument("--train_step", type=int, default=1000)
    parser.add_argument("--output", type=str)
    parser.add_argument("mesh", type=str)
    parser.add_argument("shallow_water", type=str)
    args = parser.parse_args()

    esr.init(args.comm_backend)

    eqn = ShallowWaterEquation(
        args.mesh, args.shallow_water, args.dt, args.device, for_backprop=True
    )

    swap = Swap(eqn)

    obj = Obj(eqn)
    dloss = esr.Tensor(torch.tensor([1.0], dtype=torch.float64), mode='replicate')
    obj_vjp = esr.vjp(obj, [obj.eqn.h], [obj.loss], vectors=[dloss])

    eqn_vjp = esr.vjp(eqn, [eqn.h], [eqn.h_new], vectors=obj_vjp.products)

    # h and grad_h never meet, so elempart may be not the same.
    opt = Optimizer(eqn.h, eqn_vjp.products[0])

    [eqn, eqn_vjp, swap, obj_vjp, opt] = esr.compile([eqn, eqn_vjp, swap, obj_vjp, opt], args.backend)

    for ti in range(args.train_step):

        # esr.compile together so that partitioned h uh vh share the same ElemParts
        hs = []
        uhs = []
        vhs = []

        for i in tqdm(range(args.sim_step)):
            # TODO esr.Tensor do not support .to, we can only do it on the
            # underlying .data whose element order is decided by ElemPart
            hs.append(eqn.h.data.to('cpu', copy=True))
            uhs.append(eqn.uh.data.to('cpu', copy=True))
            vhs.append(eqn.vh.data.to('cpu', copy=True))

            eqn()
            swap()

        obj_vjp()
        loss = obj_vjp.loss.collect()
        
        for i in tqdm(reversed(range(args.sim_step))):
            eqn_vjp.h.copy_(hs[i])
            eqn_vjp.uh.copy_(uhs[i])
            eqn_vjp.vh.copy_(vhs[i])

            eqn_vjp()

            eqn_vjp.vectors[0].copy_(eqn_vjp.products[0])
        
        eqn.h.copy_(hs[0])
        eqn.uh.zero_()
        eqn.vh.zero_()
        opt()

        print(ti, loss)