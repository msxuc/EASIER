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

class SwapCot(esr.Module):
    def __init__(self, prev_product: esr.Tensor, next_vector: esr.Tensor):
        super().__init__()
        self.prev_product = prev_product
        self.next_vector = next_vector
    
    def forward(self):
        self.next_vector[:] = self.prev_product

class Obj(esr.Module):
    def __init__(self, eqn: ShallowWaterEquation):
        super().__init__()
        self.eqn = eqn
        self.loss = esr.Tensor(torch.tensor([0.0], dtype=torch.float64), mode='replicate')
        self.target_h = esr.Tensor(esr.hdf5(TARGET_H_HDF5, 'h'), mode='partition')

        self.nv = self.target_h.shape[0]

    
    def forward(self):
        loss = esr.norm(self.eqn.h - self.target_h)
        self.loss[:] = loss

class Optimizer(esr.Module):
    def __init__(self, h0: esr.Tensor, grad_h0: esr.Tensor, learning_rate: float):
        super().__init__()
        self.h0 = h0
        self.grad_h0 = grad_h0
        self.learning_rate = learning_rate
    
    def forward(self):
        self.h0.sub_(self.grad_h0 * self.learning_rate)


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

        target_h = gray / 255.0 * 0.1 - 0.05 + 1.0
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
    parser.add_argument("--learning_rate", type=str, default='1e3')
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

    # unless we are storing/restoring the same esr.Tensor, for exchanging data
    # between two esr.Tensors like d_h_next and d_h_prev, we need a dedicated
    # esr.Module like Swap or SwapCot to ensure the element identities,
    # i.e. ElemParts, between these two esr.Tensors match.
    swap = Swap(eqn)

    obj = Obj(eqn)
    dloss = esr.Tensor(torch.tensor([1.0], dtype=torch.float64), mode='replicate')
    obj_vjp = esr.vjp(obj, [obj.eqn.h], [obj.loss], vectors=[dloss])

    d_h_next_vector = obj_vjp.products[0]

    eqn_vjp = esr.vjp(eqn, [eqn.h], [eqn.h_new], vectors=[d_h_next_vector])
    d_h_prev_product = eqn_vjp.products[0]
    swap_cot = SwapCot(d_h_prev_product, d_h_next_vector)

    # h and grad_h never meet, so elempart may be not the same.
    opt = Optimizer(eqn.h, d_h_prev_product, float(args.learning_rate))

    [eqn, eqn_vjp, swap, swap_cot, obj_vjp, opt] = esr.compile(
        [eqn, eqn_vjp, swap, swap_cot, obj_vjp, opt],
        args.backend
    )

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
        
        print(f"BP for train step {ti}")
        for i in tqdm(range(args.sim_step)):
            sim_i = args.sim_step - i - 1
            eqn_vjp.h.data.copy_(hs[sim_i])
            eqn_vjp.uh.data.copy_(uhs[sim_i])
            eqn_vjp.vh.data.copy_(vhs[sim_i])

            eqn_vjp()
            swap_cot()
        
        opt()

        dhmin, dhmax = d_h_prev_product.data.aminmax()
        hmin, hmax = eqn.h.data.aminmax()
        print(ti, loss.item(), hmin.item(), hmax.item(), dhmin.item(), dhmax.item())
    

    print("Final simulation")
    # for i in tqdm(range(args.sim_step)):
    for i in tqdm(range(100)):
        # if i % 10 == 0:
        if i % 1 == 0:
            x = eqn.x.collect().cpu().numpy(),
            y = eqn.y.collect().cpu().numpy(),
            z = eqn.h.collect().cpu().numpy(),
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                # np.savez(f'{args.output}/data{i//10:03d}.npz', x=x, y=y, z=z)
                np.savez(f'{args.output}/data{i:03d}.npz', x=x, y=y, z=z)

        eqn()
        swap()
