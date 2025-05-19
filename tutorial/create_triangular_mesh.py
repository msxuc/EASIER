# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os

def get_triangular_mesh(nx, ny=None, data_dir='~/.easier') -> str:
    # type: (int, int|None, str) -> str
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f'triangular_{nx}_{ny}.hdf5')

    if os.path.exists(path):
        return path

    return ""

if __name__ == '__main__':
    """

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("nx", type=int)
    parser.add_argument("ny", nargs="?", default=None, type=int)
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()

    data_dir: str = os.path.expanduser(args.data_dir)
    nx: int = args.nx
    if args.ny is None:
        ny: int = nx
    else:
        ny: int = args.ny

    print()

    get_triangular_mesh(nx, ny, data_dir)