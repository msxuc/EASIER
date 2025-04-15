# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import h5py
import torch

import easier as esr
import easier.cpp_extension as ext


def get_triangular_mesh(mesh_size=100) -> str:
    data_dir = os.path.expanduser('~/.easier')
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f'triangular_{mesh_size}.hdf5')

    if os.path.exists(path):
        return path

    src, dst, cells, bcells, bpoints, points = \
        ext.generate_triangular_mesh(torch.tensor(mesh_size))
    src = src.numpy()
    dst = dst.numpy()
    cells = cells.reshape(-1, 3).numpy()
    bcells = bcells.numpy()
    points = points.reshape(-1, 2).numpy()
    bpoints = bpoints.reshape(-1, 2).numpy()

    with h5py.File(path, 'w') as h5f:
        h5f.create_dataset('src', data=src)
        h5f.create_dataset('dst', data=dst)
        h5f.create_dataset('cells', data=cells)
        h5f.create_dataset('bcells', data=bcells)
        h5f.create_dataset('points', data=points)
        h5f.create_dataset('bpoints', data=bpoints)

    return path


class TriangularMeshGenerator(esr.Module):
    def __init__(self, mesh_size: int, device='cpu'):
        super().__init__()
        n = mesh_size
        self.mesh_size = n
        self.delta = 1. / mesh_size
        self.nv = n * n * 4
        self.ne = n * n * 12 - n * 4;
        self.np = (n + 1) * (n + 1) + n * n

        self.src = esr.Tensor(
            esr.zeros([self.ne], dtype=torch.int64, device=device),
            mode='partition'
        )
        self.dst = esr.Tensor(
            esr.zeros([self.ne], dtype=torch.int64, device=device),
            mode='partition'
        )
        self.cells = esr.Tensor(
            esr.zeros([self.nv, 3], dtype=torch.int64, device=device),
            mode='partition'
        )
        self.points = esr.Tensor(
            esr.zeros([self.np, 2], dtype=torch.float64, device=device),
            mode='partition'
        )
        self.bcells = esr.Tensor(
            esr.zeros([n * 4], dtype=torch.int64, device=device),
            mode='partition'
        )
        self.bpoints = esr.Tensor(
            esr.zeros([n * 4 * 2], dtype=torch.int64, device=device),
            mode='partition'
        )

        # Auxiliary data to explicitly provide the IDs of cells/edges/points.
        # We can use such IDs as "thread IDs".
        self.cell_ids = esr.Tensor(
            esr.arange(self.ne, dtype=torch.int64, device=device),
            mode='partition'
        )
        self.edge_ids = esr.Tensor(
            esr.arange(self.nv, dtype=torch.int64, device=device),
            mode='partition'
        )
        self.point_ids = esr.Tensor(
            esr.arange(self.np, dtype=torch.int64, device=device),
            mode='partition'
        )
        self.bcell_ids = esr.Tensor(
            esr.arange(n * 4, dtype=torch.int64, device=device),
            mode='partition'
        )
    
    def forward(self):
        1