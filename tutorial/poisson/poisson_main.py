# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import torch

import matplotlib.pyplot as plt
from matplotlib import tri

import easier as esr
from easier.numeric import Linsys
from easier.numeric.solver import CG, GMRES


class Poisson(esr.Module):
    def __init__(self, mesh: str, poisson: str, device='cpu', x=None) -> None:
        super().__init__()

        # src (torch.LongTensor): src cell indices, with shape `(ne,)`
        self.src = esr.hdf5(mesh, 'src', dtype=torch.long)
        # dst (torch.LongTensor): dst cell indices, with shape `(ne,)`
        self.dst = esr.hdf5(mesh, 'dst', dtype=torch.long)

        cells = esr.hdf5(mesh, 'cells', dtype=torch.long)
        self.nc = cells.shape[0]

        self.reducer = esr.Reducer(self.src, self.nc)
        self.selector = esr.Selector(self.dst)

        self.x = esr.Tensor(
            esr.zeros((self.nc,), dtype=torch.double), mode='partition'
        ) if x is None else x
        # b: (nc,)
        self.b = esr.Tensor(
            esr.hdf5(poisson, 'b', dtype=torch.double), mode='partition')
        # Ac: (nc,)
        self.Ac = esr.Tensor(
            esr.hdf5(poisson, 'Ac', dtype=torch.double), mode='partition')
        # Af: (src.shape[0],)
        self.Af = esr.Tensor(
            esr.hdf5(poisson, 'Af', dtype=torch.double), mode='partition'
        )
        self.A = Linsys(self.Ac, self.Af, self.selector, self.reducer)

        self.rho = esr.Tensor(
            esr.hdf5(poisson, 'rho', dtype=torch.double), mode='partition')
        # centroid: (nc, 2)
        self.centroid = esr.Tensor(
            esr.hdf5(poisson, 'centroid', dtype=torch.double), mode='partition'
        )

        self.to(device)


if __name__ == '__main__':
    """
    Usage:

    torchrun --nproc_per_node=4 tutorial/poisson/poisson_main.py \
        --solver=cg --backend=cpu --comm_backend=gloo \
        ~/.easier/triangular_100.hdf5 ~/.easier/Poisson_100.hdf5
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", type=str, choices=["cg", "gmres"], default="cg"
    )
    parser.add_argument("--plot", type=bool, default=False)
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
    parser.add_argument("mesh", type=str)
    parser.add_argument("poisson", type=str)
    args = parser.parse_args()

    print("Compile Poisson:")
    print("mesh HDF5 file:   ", args.mesh)
    print("poisson HDF5 file:", args.poisson)

    esr.init(args.comm_backend)

    eqn = Poisson(args.mesh, args.poisson, args.device)

    if args.solver == 'cg':
        sol = CG(eqn.A, eqn.b, eqn.x)
    else:
        sol = GMRES(eqn.A, eqn.b, eqn.x)

    # eqn, sol = esr.compile([eqn, sol], args.backend, load_dir=f'~/.easier/dump/poisson_gmres_100_w4')
    eqn, sol = esr.compile([eqn, sol], args.backend)

    tol = 1e-5
    info = sol.solve(atol=tol, maxiter=20, debug_iter=10)
    # assert info["residual"] < tol

    if args.plot:
        rho = eqn.rho.collect().cpu().numpy()
        x_synced = eqn.x.collect().cpu().numpy()
        centroid = eqn.centroid.collect().cpu().numpy()

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            cells = tri.Triangulation(centroid[::1, 0], centroid[::1, 1])

            plt.figure(figsize=(6, 5))
            im = plt.tricontourf(cells, rho[::1], levels=50, cmap='jet')
            # plt.triplot(points, linewidth=0.1)
            plt.colorbar(im)
            plt.tight_layout()
            plt.savefig('rho.jpeg', dpi=300)
            plt.cla()

            plt.figure(figsize=(6, 5))
            im = plt.tricontourf(cells, x_synced[::1], levels=50, cmap='jet')
            # plt.triplot(points, linewidth=0.1)
            plt.colorbar(im)
            plt.tight_layout()
            plt.savefig('phi.jpeg', dpi=300)
            plt.cla()
