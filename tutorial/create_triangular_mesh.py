# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import h5py

import easier.cpp_extension as ext


def get_triangular_mesh(n, data_dir='~/.easier') -> str:
    # type: (int, str) -> str
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f'triangular_{n}.hdf5')

    if os.path.exists(path):
        return path

    src, dst, cells, bcells, bpoints, points = \
        ext.generate_triangular_mesh(n)
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

    python tutorial/create_triangular_mesh.py 100 ~/.easier
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_size", type=int)
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()

    data_dir: str = os.path.expanduser(args.data_dir)
    mesh_size: int = args.mesh_size

    mesh = get_triangular_mesh(mesh_size, data_dir)

    print("Create triangular mesh:")
    print("mesh size:", mesh_size)
    print("output HDF5 file:", mesh)
