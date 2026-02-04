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
        self.target_h = esr.Tensor(esr.hdf5(TARGET_H_HDF5, 'h'), mode='partition')
    
    def forward(self):
        loss = esr.norm(self.eqn.h - self.target_h)
        self.loss[:] = loss

class Optimizer(esr.Module):
    def __init__(self, h: esr.Tensor, grad_h: esr.Tensor):
        super().__init__()
        self.h = h
        self.grah_h = grad_h
    
    def forward(self):
        self.h.sub_(self.grah_h * 10.0**55)


class InitTarget(esr.Module):
    def __init__(self, eqn: ShallowWaterEquation, img_arr: torch.Tensor):
        super().__init__()
        self.eqn = eqn
        self.img_arr = esr.Tensor(img_arr, mode='replicate')
        self.target_h = esr.Tensor(esr.zeros_like(eqn.h), mode='partition')
    
    def forward(self):
        x = self.eqn.x
        y = self.eqn.y
        img_x = (x * self.img_arr.shape[0]).long()
        img_y = (y * self.img_arr.shape[1]).long()
        gray = self.img_arr[img_x, img_y]

        target_h = (gray / 255.0 - 0.5) * 0.1 + 1.0
        self.target_h[:] = target_h


TARGET_H_HDF5 = os.path.join(swe_dir, 'target_h.hdf5')

def init_target(eqn: ShallowWaterEquation) -> None:
    from PIL import Image
    logo = Image.open(os.path.join(swe_dir, '../logo.png')).convert('L')

    logo = logo.crop((0, 245, 1300, 1560))
    logo = logo.resize((1000, 1000))
    logo.save(os.path.join(swe_dir, 'logo_gray.png'))

    img_arr = torch.from_numpy(np.array(logo)).double()

    init_target = InitTarget(eqn, img_arr)
    [init_target] = esr.compile([init_target], backend='none') # type: ignore
    init_target: InitTarget

    init_target()
    
    init_target.target_h.save(TARGET_H_HDF5, 'h')

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

    if not os.path.exists(TARGET_H_HDF5):
        init_target(eqn)
        print("Init target H. Rerun this torchrun command")
        exit(0)

    swap = Swap(eqn)

    obj = Obj(eqn)
    dloss = esr.Tensor(torch.tensor([1.0], dtype=torch.float64), mode='replicate')
    obj_vjp = esr.vjp(obj, [obj.eqn.h], [obj.loss], vectors=[dloss])

    eqn_vjp = esr.vjp(eqn, [eqn.h], [eqn.h_new], vectors=obj_vjp.products)

    # h and grad_h never meet, so elempart may be not the same.
    opt = Optimizer(eqn.h, eqn_vjp.products[0])

    [eqn, eqn_vjp, swap, obj_vjp, opt] = esr.compile([eqn, eqn_vjp, swap, obj_vjp, opt], args.backend)

    # remove assembled initial height
    eqn.h.data[:] = 1.0

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

        print(ti, loss.item(), eqn_vjp.products[0].aminmax())
    

    print("Final simulation")
    for i in tqdm(range(args.sim_step)):
        if i % 10 == 0:
            x = eqn.x.collect().cpu().numpy(),
            y = eqn.y.collect().cpu().numpy(),
            z = eqn.h.collect().cpu().numpy(),
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                np.savez(f'{args.output}/data{i//10:03d}.npz', x=x, y=y, z=z)

        eqn()
