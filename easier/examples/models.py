# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import torch
import torch.distributed
import h5py

import easier as esr
from easier.numeric import Linsys

from .mesh import get_triangular_mesh


class PoissonMeshComponentsCollector(esr.Module):
    def __init__(self, mesh: str, device='cpu'):
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


class PoissonInitializer(esr.Module):
    def __init__(self, poisson: str, mesh: str, device='cpu'):
        super().__init__()

        cells = esr.hdf5(mesh, 'cells', dtype=torch.long)
        nc = cells.shape[0]

        self.points = esr.Tensor(
            esr.hdf5(mesh, 'points', dtype=torch.double), mode='partition'
        )

        self.reducer = esr.Reducer(esr.hdf5(mesh, 'src', dtype=torch.long), nc)

        ne = self.reducer.idx.shape[0]

        self.selector_src_p = torch.nn.ModuleList([
            esr.Selector(
                esr.hdf5(poisson, f'src_p{i}', dtype=torch.long),
            ) for i in range(3)
        ])
        self.selector_dst_p = torch.nn.ModuleList([
            esr.Selector(
                esr.hdf5(poisson, f'dst_p{i}', dtype=torch.long),
            ) for i in range(3)
        ])
        self.selector_cells_p = torch.nn.ModuleList([
            esr.Selector(
                esr.hdf5(poisson, f'cells_p{i}', dtype=torch.long),
            ) for i in range(3)
        ])

        # bcells: boundary cell indices, with shape `(nbc,)`,
        #   `nbc` means number of boundary cell
        bcells = esr.hdf5(mesh, 'bcells', dtype=torch.long)
        self.bselector = esr.Selector(bcells)
        self.breducer = esr.Reducer(bcells, nc)

        self.selector_bp = torch.nn.ModuleList([
            esr.Selector(
                # bp{i}: boundary points indices in each boundary cell,
                #   with shape `(nbc,)`, `nbc` means number of boundary cell
                esr.hdf5(poisson, f'bp{i}', dtype=torch.long)
            ) for i in range(2)
        ])

        # TODO make this a replica esr.Tensor so that we can use `Mod.to()`
        # to move it and change dtype. Only torch parameters and buffers are
        # affected.
        # OTAH, `self.center = torch.tensor()` also works, but users must
        # change it dtype specifically and manually. (EASIER will move it).
        # Inline `centroid - torch.tensor()` works too. But no way to change
        # dtype. During FX tracing we can only get symbolic proxy for `.dtype`,
        # but the tensor() call requires a real dtype object.
        self.center = esr.Tensor(
            torch.tensor([[0.5, 0.5]], dtype=torch.double), mode='replicate'
        )

        #
        # Output
        #
        self.b = esr.Tensor(
            esr.zeros([nc], dtype=torch.double), mode='partition'
        )
        self.Ac = esr.Tensor(
            esr.zeros([nc], dtype=torch.double), mode='partition'
        )
        self.Af = esr.Tensor(
            esr.zeros([ne], dtype=torch.double), mode='partition'
        )
        self.rho = esr.Tensor(
            esr.zeros([nc], dtype=torch.double), mode='partition'
        )
        self.centroid = esr.Tensor(
            esr.zeros([nc, 2], dtype=torch.double), mode='partition'
        )

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
        norm_x = torch.where(condition, norm01_x, 0.)
        norm_y = torch.where(condition, norm01_y, 0.)

        condition = (norm12_x * norm12_x_ + norm12_y * norm12_y_) < 0
        norm_x = torch.where(condition, norm12_x, norm_x)
        norm_y = torch.where(condition, norm12_y, norm_y)

        condition = (norm20_x * norm20_x_ + norm20_y * norm20_y_) < 0
        norm_x = torch.where(condition, norm20_x, norm_x)
        norm_y = torch.where(condition, norm20_y, norm_y)

        dist = dist / (dist**2).sum(dim=1, keepdim=True)
        self.Af[:] = dist[:, 0] * norm_x + dist[:, 1] * norm_y
        self.Ac[:] = - self.reducer(self.Af)

        p0 = self.selector_cells_p[0](self.points)
        x0 = p0[:, 0]
        y0 = p0[:, 1]
        p1 = self.selector_cells_p[1](self.points)
        x1 = p1[:, 0]
        y1 = p1[:, 1]
        p2 = self.selector_cells_p[2](self.points)
        x2 = p2[:, 0]
        y2 = p2[:, 1]

        area = 0.5 * torch.abs(
            x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)
        )

        self.centroid[:] = (p0 + p1 + p2) / 3.
        self.rho[:] = torch.exp(
            -0.5 * 400 * ((self.centroid - self.center)**2).sum(1)
        )
        self.b[:] = self.rho * area

        # boundary condition
        b_p0 = self.selector_bp[0](self.points)
        b_p1 = self.selector_bp[1](self.points)
        b_cell_cent = self.bselector(self.centroid)
        b_face_cent = (b_p0 + b_p1) / 2.

        bnorm_x, bnorm_y = self.get_face_norm(b_cell_cent, b_p0, b_p1)
        bdist = b_face_cent - b_cell_cent
        bdist = bdist / (bdist**2).sum(dim=-1, keepdim=True)

        self.Ac.sub_(self.breducer(
            bdist[:, 0] * bnorm_x + bdist[:, 1] * bnorm_y
        ))


def _assemble_poisson(mesh: str, poisson: str, device='cpu'):
    components = PoissonMeshComponentsCollector(mesh)
    components.to(device)

    [components] = esr.compile(
        [components], 'none'
    )  # type: ignore
    components: PoissonMeshComponentsCollector
    components()

    for i in range(3):
        components.src_p[i].save(poisson, f'src_p{i}')
        components.dst_p[i].save(poisson, f'dst_p{i}')
        components.cells_p[i].save(poisson, f'cells_p{i}')
    for i in range(2):
        components.bp[i].save(poisson, f'bp{i}')

    initializer = PoissonInitializer(poisson, mesh)
    initializer.to(device)

    [initializer] = esr.compile(
        [initializer], 'none'
    )  # type: ignore
    initializer: PoissonInitializer
    initializer()

    initializer.b.save(poisson, 'b')
    initializer.Ac.save(poisson, 'Ac')
    initializer.Af.save(poisson, 'Af')
    initializer.rho.save(poisson, 'rho')
    initializer.centroid.save(poisson, 'centroid')

    return poisson


class Poisson(esr.Module):
    def __init__(self, mesh_size=100, device='cpu', x=None) -> None:
        super().__init__()

        mesh: str
        poisson: str
        poisson_exists: bool
        if torch.distributed.get_rank() == 0:
            mesh = get_triangular_mesh(mesh_size)

            data_dir = os.path.expanduser('~/.easier')
            os.makedirs(data_dir, exist_ok=True)
            poisson = os.path.join(data_dir, f'Poisson_{mesh_size}.hdf5')
            poisson_exists = os.path.exists(poisson)

            torch.distributed.broadcast_object_list(
                [mesh, poisson, poisson_exists], 0
            )
        else:
            recv_objs = [None, None, None]
            torch.distributed.broadcast_object_list(recv_objs, 0)
            [mesh, poisson, poisson_exists] = recv_objs  # type: ignore

        if not poisson_exists:
            _assemble_poisson(mesh, poisson, device)

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


class Poisson1D(esr.Module):

    def __init__(self, n):
        super().__init__()
        Ac = torch.ones((n, 2)) * -2
        self.Ac = esr.Tensor(Ac.reshape(-1, 2).double(), mode='partition')

        Af = torch.concat(
            (torch.ones((n - 1)),
             torch.ones((n - 1))))
        self.Af = esr.Tensor(Af.reshape(-1, 1).double(), mode='partition')

        src = torch.concat(
            (torch.linspace(1, n - 1, n - 1, dtype=torch.long),
             torch.linspace(0, n - 2, n - 1, dtype=torch.long)))
        self.selector = esr.Selector(src)

        dst = torch.concat(
            (torch.linspace(0, n - 2, n - 1, dtype=torch.long),
             torch.linspace(1, n - 1, n - 1, dtype=torch.long)))
        self.reducer = esr.Reducer(dst, n)

        self.x = esr.Tensor(torch.ones((n, 2)).double(), mode='partition')
        b = torch.exp(
            -0.5 * 100 * (torch.linspace(0, 1, n) - 0.5)**2) / (n - 1)**2
        self.b = esr.Tensor(b.reshape(-1, 1).double(), mode='partition')
        self.A = Linsys(self.Ac, self.Af, self.selector, self.reducer)


class Circuit(esr.Module):

    def __init__(self):
        super().__init__()
        b = torch.tensor(
            [-0.001, 0.001, 0, 0, 0.001, 0, 0, 0, 0, 0, 2, 0.2, 2])
        Ac = torch.tensor(
            [10, 0.001, 10, 5 / 3, 32 / 3, 32 / 3, 5 / 3, 0, -10, -50,
             0, 0, 0])
        Af = torch.tensor(
            [-10, 1, -10, 1, -2 / 3, -1, -1, -1, -2 / 3, -10, -1, -10, -2 / 3,
             1, 1, -1, -2 / 3, 1, 1, -1, 1, -1, 1, 1, -1, 1])
        src = torch.tensor(
            [2, 8, 0, 11, 4, 6, 9, 11, 3, 5, 10, 4, 6, 9, 12, 3, 5, 10, 0, 3,
             5, 4, 7, 2, 3, 5])
        dst = torch.tensor(
            [0, 0, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 9, 9,
             10, 10, 11, 11, 12])

        n = 13
        self.reducer = esr.Reducer(dst, n)
        self.selector = esr.Selector(src)
        self.Af = esr.Tensor(Af.reshape(-1, 1).double(), mode='partition')
        self.Ac = esr.Tensor(Ac.reshape(-1, 1).double(), mode='partition')
        self.x = esr.Tensor(torch.zeros((n, 2)).double(), mode='partition')
        self.b = esr.Tensor(b.reshape(-1, 1).double(), mode='partition')
        self.A = Linsys(self.Ac, self.Af, self.selector, self.reducer)
