# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import h5py
import torch

import easier.cpp_extension as ext

def get_triangular_mesh(nx, ny=None, data_dir='~/.easier') -> str:
    # type: (int, int|None, str) -> str
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f'triangular_{nx}_{ny}.hdf5')

    if os.path.exists(path):
        return path
    
    src, dst, cells, bcells, bpoints, points = \
        ext.generate_triangular_mesh(nx, ny)
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

if __name__ == '__main__':
    """
    Usage:

    python tutorial/create_triangular_mesh.py 100 [100] [~/.easier]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("nx", type=int)
    parser.add_argument("ny", nargs="?", default=None, type=int)
    parser.add_argument("data_dir", type=str, nargs="?", default='~/.easier')
    args = parser.parse_args()

    data_dir: str = os.path.expanduser(args.data_dir)
    nx: int = args.nx
    if args.ny is None:
        ny: int = nx
    else:
        ny: int = args.ny

    mesh = get_triangular_mesh(nx, ny, data_dir)

    print("Create triangular mesh:")
    print("nx:", args.nx)
    print("ny:", args.ny)
    print("output HDF5 file:", mesh)