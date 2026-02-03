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


class InitH(esr.Module):
    def __init__(self, h: esr.Tensor, x: esr.Tensor, y: esr.Tensor):
        super().__init__()
        self.h = h
        self.x = x
        self.y = y
        self.p0 = esr.Tensor(torch.tensor([0, 0], dtype=h.dtype), mode='replicate')
    
    def forward(self):
        w = 2
        self.h[:] = 0.05 / (1 + torch.exp(
            10 * torch.sin(3.14* w *self.x) * torch.sin(3.14* w *self.y)
        ))

        self.h[:] += 1 + 0.1 * torch.exp(
            -100 * ((self.x - self.p0[0])**2 + (self.y - self.p0[1])**2)
        )


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
    parser.add_argument("--output", type=str)
    parser.add_argument("mesh", type=str)
    parser.add_argument("shallow_water", type=str)
    args = parser.parse_args()

    esr.init(args.comm_backend)

    eqn = ShallowWaterEquation(
        args.mesh, args.shallow_water, args.dt, args.device
    )

    init_h = InitH(eqn.h, eqn.x, eqn.y)

    p0_v = torch.zeros_like(init_h.p0)
    p0_v[0] = 1.0  # component x of gradient

    h_t = esr.Tensor(esr.zeros_like(eqn.h), mode='partition')
    p0_t = esr.Tensor(p0_v, mode='replicate')

    init_h_jvp = esr.jvp(init_h, [init_h.p0, eqn.h], [], vectors=[p0_t, h_t])
    eqn_jvp = esr.jvp(eqn, [init_h.p0, eqn.h], [], vectors=[p0_t, h_t])

    [eqn, init_h] = esr.compile([eqn_jvp, init_h_jvp], args.backend)

    init_h()

    for i in tqdm(range(1000)):
        if i % 10 == 0:
            x = eqn.x.collect().cpu().numpy(),
            y = eqn.y.collect().cpu().numpy(),
            z = eqn.h.collect().cpu().numpy(),
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                np.savez(f'{args.output}/data{i//10:03d}.npz', x=x, y=y, z=z)

        eqn()
