# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import h5py

import numpy as np
import torch
import torch.distributed
from tqdm import tqdm

import easier as esr
from easier.examples.mesh import get_triangular_mesh


class ShallowWaterMeshComponentsCollector(esr.Module):
    def __init__(self, mesh: str):
        super().__init__()

        # (nc, 3)
        self.cells = esr.Tensor(
            esr.hdf5(mesh, 'cells', dtype=torch.long), mode='partition'
        )

        # (nbc, 2)
        self.bpoints = esr.Tensor(
            esr.hdf5(mesh, 'bpoints', dtype=torch.long), mode='partition'
        )

        self.selector_src = esr.Selector(
            esr.hdf5(mesh, 'src', dtype=torch.long)
        )
        self.selector_dst = esr.Selector(
            esr.hdf5(mesh, 'dst', dtype=torch.long)
        )

        nc = self.cells.shape[0]
        ne = self.selector_src.idx.shape[0]
        nbc = self.bpoints.shape[0]

        #
        # Output
        #
        self.src_p = torch.nn.ParameterList([
            esr.Tensor(
                esr.zeros([ne], dtype=torch.long), mode='partition'
            ) for i in range(3)
        ])
        self.dst_p = torch.nn.ParameterList([
            esr.Tensor(
                esr.zeros([ne], dtype=torch.long), mode='partition'
            ) for i in range(3)
        ])
        self.cells_p = torch.nn.ParameterList([
            esr.Tensor(
                esr.zeros([nc], dtype=torch.long), mode='partition'
            ) for i in range(3)
        ])

        # bp{i}: boundary points indices in each boundary cell,
        #   with shape `(nbc,)`, `nbc` means number of boundary cell
        self.bp = torch.nn.ParameterList([
            esr.Tensor(
                esr.zeros([nbc], dtype=torch.long), mode='partition'
            ) for i in range(2)
        ])

    def forward(self):
        # (ne, 3)
        src_p = self.selector_src(self.cells)
        dst_p = self.selector_dst(self.cells)

        for i in range(3):
            # (ne,)
            self.src_p[i].set_(src_p[:, i])
            self.dst_p[i].set_(dst_p[:, i])

            # (nc,)
            self.cells_p[i].set_(self.cells[:, i])

        for i in range(2):
            # (nbc,)
            self.bp[i].set_(self.bpoints[:, i])


class ShallowWaterInitializer(esr.Module):
    def __init__(self, shallow_water: str, mesh: str):
        super().__init__()

        self.points = esr.Tensor(
            esr.hdf5(mesh, 'points', dtype=torch.double), mode='partition'
        )

        cells = esr.hdf5(mesh, 'cells', dtype=torch.long)
        nc = cells.shape[0]

        self.selector_src_p = torch.nn.ModuleList([
            esr.Selector(
                esr.hdf5(shallow_water, f'src_p{i}', dtype=torch.long),
            ) for i in range(3)
        ])
        self.selector_dst_p = torch.nn.ModuleList([
            esr.Selector(
                esr.hdf5(shallow_water, f'dst_p{i}', dtype=torch.long),
            ) for i in range(3)
        ])
        self.selector_cells_p = torch.nn.ModuleList([
            esr.Selector(
                esr.hdf5(shallow_water, f'cells_p{i}', dtype=torch.long),
            ) for i in range(3)
        ])

        ne: int = self.selector_src_p[0].idx.shape[0]  # type: ignore

        # bcells: boundary cell indices, with shape `(nbc,)`,
        #   `nbc` means number of boundary cell
        bcells = esr.hdf5(mesh, 'bcells', dtype=torch.long)
        nbc = bcells.shape[0]

        self.bselector = esr.Selector(bcells)
        self.selector_bp = torch.nn.ModuleList([
            esr.Selector(
                # bp{i}: boundary points indices in each boundary cell,
                #   with shape `(nbc,)`, `nbc` means number of boundary cell
                esr.hdf5(shallow_water, f'bp{i}', dtype=torch.long)
            ) for i in range(2)
        ])

        #
        # Output
        #
        self.x = esr.Tensor(
            esr.zeros([nc], dtype=torch.double), mode='partition'
        )
        self.y = esr.Tensor(
            esr.zeros([nc], dtype=torch.double), mode='partition'
        )
        self.area = esr.Tensor(
            esr.zeros([nc], dtype=torch.double), mode='partition'
        )
        self.sx = esr.Tensor(
            esr.zeros([ne], dtype=torch.double), mode='partition'
        )
        self.sy = esr.Tensor(
            esr.zeros([ne], dtype=torch.double), mode='partition'
        )
        self.bsx = esr.Tensor(
            esr.zeros([nbc], dtype=torch.double), mode='partition'
        )
        self.bsy = esr.Tensor(
            esr.zeros([nbc], dtype=torch.double), mode='partition'
        )
        self.h = esr.Tensor(
            esr.zeros([nc], dtype=torch.double), mode='partition'
        )
        self.alpha = esr.Tensor(
            esr.zeros([ne], dtype=torch.double), mode='partition'
        )

    def get_alpha(self, sc, dc, p0, p1):
        x1 = sc[:, 0]
        y1 = sc[:, 1]
        x2 = dc[:, 0]
        y2 = dc[:, 1]
        x3 = p0[:, 0]
        y3 = p0[:, 1]
        x4 = p1[:, 0]
        y4 = p1[:, 1]

        y21 = y2 - y1
        y43 = y4 - y3
        y31 = y3 - y1
        x31 = x3 - x1
        x21 = x2 - x1
        x43 = x4 - x3

        return (x31 * y43 - y31 * x43) / (x21 * y43 - y21 * x43)

    def get_face_norm(self, p0, p1, p2):
        a1 = p0[:, 0]
        a2 = p0[:, 1]
        b1 = p1[:, 0]
        b2 = p1[:, 1]
        c1 = p2[:, 0]
        c2 = p2[:, 1]

        s = torch.sign((b1 - c1) * (a2 - c2) - (b2 - c2) * (a1 - c1))

        return s * (b2 - c2), -s * (b1 - c1)

    def forward(self):
        src_p0 = self.selector_src_p[0](self.points)
        src_p1 = self.selector_src_p[1](self.points)
        src_p2 = self.selector_src_p[2](self.points)

        dst_p0 = self.selector_dst_p[0](self.points)
        dst_p1 = self.selector_dst_p[1](self.points)
        dst_p2 = self.selector_dst_p[2](self.points)

        src_cent = (src_p0 + src_p1 + src_p2) / 3.
        dst_cent = (dst_p0 + dst_p1 + dst_p2) / 3.

        dist = dst_cent - src_cent

        norm01_x, norm01_y = self.get_face_norm(src_p2, src_p0, src_p1)
        norm12_x, norm12_y = self.get_face_norm(src_p0, src_p1, src_p2)
        norm20_x, norm20_y = self.get_face_norm(src_p1, src_p2, src_p0)

        norm01_x_, norm01_y_ = self.get_face_norm(dst_cent, src_p0, src_p1)
        norm12_x_, norm12_y_ = self.get_face_norm(dst_cent, src_p1, src_p2)
        norm20_x_, norm20_y_ = self.get_face_norm(dst_cent, src_p2, src_p0)

        condition = (norm01_x * norm01_x_ + norm01_y * norm01_y_) < 0
        self.sx[:] = torch.where(condition, norm01_x, 0.)
        self.sy[:] = torch.where(condition, norm01_y, 0.)
        alpha = self.get_alpha(src_cent, dst_cent, src_p0, src_p1)
        self.alpha[:] = torch.where(condition, alpha, 0.)

        condition = (norm12_x * norm12_x_ + norm12_y * norm12_y_) < 0
        self.sx[:] = torch.where(condition, norm12_x, self.sx)
        self.sy[:] = torch.where(condition, norm12_y, self.sy)
        alpha = self.get_alpha(src_cent, dst_cent, src_p1, src_p2)
        self.alpha[:] = torch.where(condition, alpha, self.alpha)

        condition = (norm20_x * norm20_x_ + norm20_y * norm20_y_) < 0
        self.sx[:] = torch.where(condition, norm20_x, self.sx)
        self.sy[:] = torch.where(condition, norm20_y, self.sy)
        alpha = self.get_alpha(src_cent, dst_cent, src_p2, src_p0)
        self.alpha[:] = torch.where(condition, alpha, self.alpha)

        p0 = self.selector_cells_p[0](self.points)
        x0 = p0[:, 0]
        y0 = p0[:, 1]
        p1 = self.selector_cells_p[1](self.points)
        x1 = p1[:, 0]
        y1 = p1[:, 1]
        p2 = self.selector_cells_p[2](self.points)
        x2 = p2[:, 0]
        y2 = p2[:, 1]

        self.area[:] = 0.5 * torch.abs(
            x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
        centroid = (p0 + p1 + p2) / 3.

        self.x[:] = centroid[:, 0]
        self.y[:] = centroid[:, 1]

        self.h[:] = 1 + 0.1 * torch.exp(
            -100 * ((self.x - 0.0)**2 + (self.y - 0.0)**2)
        )

        # boundary condition
        b_p0 = self.selector_bp[0](self.points)
        b_p1 = self.selector_bp[1](self.points)
        b_cell_cent = self.bselector(centroid)

        bnorm_x, bnorm_y = self.get_face_norm(b_cell_cent, b_p0, b_p1)
        self.bsx[:] = -bnorm_x
        self.bsy[:] = -bnorm_y


def _assemble_shallow_water(mesh: str, shallow_water: str, device='cpu'):
    components = ShallowWaterMeshComponentsCollector(mesh)
    components.to(device)

    [components] = esr.compile(
        [components], 'torch', partition_mode='evenly'
    )  # type: ignore
    components: ShallowWaterMeshComponentsCollector
    components()

    for i in range(3):
        components.src_p[i].save(shallow_water, f'src_p{i}')
        components.dst_p[i].save(shallow_water, f'dst_p{i}')
        components.cells_p[i].save(shallow_water, f'cells_p{i}')
    for i in range(2):
        components.bp[i].save(shallow_water, f'bp{i}')

    initializer = ShallowWaterInitializer(shallow_water, mesh)
    initializer.to(device)

    [initializer] = esr.compile(
        [initializer], 'torch', partition_mode='evenly'
    )  # type: ignore
    initializer: ShallowWaterInitializer
    initializer()

    initializer.x.save(shallow_water, 'x')
    initializer.y.save(shallow_water, 'y')
    initializer.area.save(shallow_water, 'area')
    initializer.sx.save(shallow_water, 'sx')
    initializer.sy.save(shallow_water, 'sy')
    initializer.bsx.save(shallow_water, 'bsx')
    initializer.bsy.save(shallow_water, 'bsy')
    initializer.h.save(shallow_water, 'h')
    initializer.alpha.save(shallow_water, 'alpha')

    return shallow_water


class ShallowWaterEquation(esr.Module):
    def __init__(self, mesh_size: int = 100, dt=0.005, device='cpu') -> None:
        super().__init__()

        mesh_path: str
        sw_path: str
        sw_exists: bool
        if torch.distributed.get_rank() == 0:
            mesh_path = get_triangular_mesh(mesh_size)

            data_dir = os.path.expanduser('~/.easier')
            os.makedirs(data_dir, exist_ok=True)
            sw_path = os.path.join(data_dir, f'SW_{mesh_size}.hdf5')
            sw_exists = os.path.exists(sw_path)

            torch.distributed.broadcast_object_list(
                [mesh_path, sw_path, sw_exists], 0
            )
        else:
            recv_objs = [None, None, None]
            torch.distributed.broadcast_object_list(recv_objs, 0)
            [mesh_path, sw_path, sw_exists] = recv_objs  # type: ignore

        if not sw_exists:
            _assemble_shallow_water(mesh_path, sw_path, device)

        self.dt = dt
        # src (torch.LongTensor): src cell indices, with shape `(ne,)`
        self.src = esr.hdf5(mesh_path, 'src', dtype=torch.long)
        # dst (torch.LongTensor): dst cell indices, with shape `(ne,)`
        self.dst = esr.hdf5(mesh_path, 'dst', dtype=torch.long)
        self.ne = self.src.shape[0]

        # cells (torch.LongTensor): three point indices for each triangle
        #   cells, with shape `(nc, 3)`, `nc` means number of cells
        self.cells = esr.hdf5(mesh_path, 'cells', dtype=torch.long)
        self.nc = self.cells.shape[0]

        # points (torch.DoubleTensor): point coordinates on a plane,
        #   with shape `(np, 2)`, `np` means number of points
        self.points = esr.hdf5(mesh_path, 'points', dtype=torch.long)
        self.np = self.points.shape[0]

        # bcells (torch.LongTensor): boundary cell indices, with shape `(nbc,)`,
        #   `nbc` means number of boundary cell
        self.bcells = esr.hdf5(mesh_path, 'bcells', dtype=torch.long)
        self.nbc = self.bcells.shape[0]

        # bpoints (torch.LongTensor): boundary points indices in each boundary
        #   cell, with shape `(nbc, 2)`, `nbc` means number of boundary cell
        self.bpoints = esr.hdf5(mesh_path, 'bpoints', dtype=torch.long)

        self.scatter = esr.Reducer(self.dst, self.nc)
        self.gather_src = esr.Selector(self.src)
        self.gather_dst = esr.Selector(self.dst)
        self.scatter_b = esr.Reducer(self.bcells, self.nc)
        self.gather_b = esr.Selector(self.bcells)

        self.x = esr.Tensor(
            esr.hdf5(sw_path, 'x', dtype=torch.double), mode='partition'
        )
        self.y = esr.Tensor(
            esr.hdf5(sw_path, 'y', dtype=torch.double), mode='partition'
        )
        self.area = esr.Tensor(
            esr.hdf5(sw_path, 'area', dtype=torch.double), mode='partition'
        )
        self.sx = esr.Tensor(
            esr.hdf5(sw_path, 'sx', dtype=torch.double), mode='partition'
        )
        self.sy = esr.Tensor(
            esr.hdf5(sw_path, 'sy', dtype=torch.double), mode='partition'
        )
        self.bsx = esr.Tensor(
            esr.hdf5(sw_path, 'bsx', dtype=torch.double), mode='partition'
        )
        self.bsy = esr.Tensor(
            esr.hdf5(sw_path, 'bsy', dtype=torch.double), mode='partition'
        )
        self.h = esr.Tensor(
            esr.hdf5(sw_path, 'h', dtype=torch.double), mode='partition'
        )
        self.alpha = esr.Tensor(
            esr.hdf5(sw_path, 'alpha', dtype=torch.double), mode='partition'
        )

        self.uh = esr.Tensor(
            esr.zeros((self.nc,), dtype=torch.double), mode='partition'
        )
        self.vh = esr.Tensor(
            esr.zeros((self.nc,), dtype=torch.double), mode='partition'
        )

        self.to(device)

    def face_reconstruct(self, phi):
        return (1 - self.alpha) * self.gather_src(phi) + \
            self.alpha * self.gather_dst(phi)

    def delta(self, h, uh, vh):
        h_f = self.face_reconstruct(h)
        uh_f = self.face_reconstruct(uh)
        vh_f = self.face_reconstruct(vh)

        u_f = uh_f / h_f
        v_f = vh_f / h_f

        h_f_square = 0.5 * h_f * h_f
        uh_f_times_sx = uh_f * self.sx
        vh_f_times_sy = vh_f * self.sy
        gather_b_h_square = 0.5 * self.gather_b(h)**2

        delta_h = - (
            self.scatter(uh_f_times_sx + vh_f_times_sy)
        ) / self.area

        delta_uh = - (
            self.scatter((u_f * uh_f + h_f_square) * self.sx +
                         u_f * vh_f_times_sy) +
            self.scatter_b(gather_b_h_square * self.bsx)
        ) / self.area

        delta_vh = - (
            self.scatter(v_f * uh_f_times_sx +
                         (v_f * vh_f + h_f_square) * self.sy) +
            self.scatter_b(gather_b_h_square * self.bsy)
        ) / self.area

        return delta_h, delta_uh, delta_vh

    def forward(self):
        delta_h1, delta_uh1, delta_vh1 = self.delta(self.h, self.uh, self.vh)
        delta_h2, delta_uh2, delta_vh2 = self.delta(
            self.h + 0.5 * self.dt * delta_h1,
            self.uh + 0.5 * self.dt * delta_uh1,
            self.vh + 0.5 * self.dt * delta_vh1,)
        delta_h3, delta_uh3, delta_vh3 = self.delta(
            self.h + 0.5 * self.dt * delta_h2,
            self.uh + 0.5 * self.dt * delta_uh2,
            self.vh + 0.5 * self.dt * delta_vh2,)
        delta_h4, delta_uh4, delta_vh4 = self.delta(
            self.h + self.dt * delta_h3,
            self.uh + self.dt * delta_uh3,
            self.vh + self.dt * delta_vh3,)

        self.h[:] += self.dt / 6 * (
            delta_h1 + delta_h2 + delta_h3 + delta_h4)
        self.uh[:] += self.dt / 6 * (
            delta_uh1 + delta_uh2 + delta_uh3 + delta_uh4)
        self.vh[:] += self.dt / 6 * (
            delta_vh1 + delta_vh2 + delta_vh3 + delta_vh4)


if __name__ == "__main__":
    """
    mkdir res
    torchrun --nnodes=1 --nproc_per_node=4 \
        shallow_water_equation.py --backend=cpu --output=res
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
    parser.add_argument("--scale", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    esr.init(args.comm_backend)

    eqn = ShallowWaterEquation(args.scale, args.dt, args.device)
    [eqn] = esr.compile([eqn], args.backend)

    for i in tqdm(range(1000)):
        if i % 10 == 0:
            x = eqn.x.collect().cpu().numpy(),
            y = eqn.y.collect().cpu().numpy(),
            z = eqn.h.collect().cpu().numpy(),
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                np.savez(f'{args.output}/data{i//10:03d}.npz', x=x, y=y, z=z)

        eqn()
