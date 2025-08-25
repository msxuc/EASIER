# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import torch

import easier as esr
from easier.core.runtime.data_loader.ops import ConcatDataLoader


class PoissonMeshComponentsCollector(esr.Module):
    def __init__(self, scale: int, device='cpu'):
        super().__init__()

        mesh = esr.Mesh(
            esr.linspace(0, 1, scale),
            esr.linspace(0, 1, scale)
        )


        self.selector_src = esr.Selector(
            mesh.src
        )
        self.selector_dst = esr.Selector(
            mesh.dst
        )

        ne = mesh.ne
        # nbc = self.bpoints.shape[0]

        self.cells_p = torch.nn.ParameterList([
            esr.Tensor(
                [
                    mesh.indices[:-1, :-1],
                    mesh.indices[1:, :-1],
                    mesh.indices[1:, 1:],
                    mesh.indices[:-1, 1:],
                ][i],
                mode='partition'
            ) for i in range(4)
        ])

        #
        # Output
        #
        self.src_p = torch.nn.ParameterList([
            esr.Tensor(
                esr.zeros([ne], dtype=torch.long), mode='partition'
            ) for i in range(4)
        ])
        self.dst_p = torch.nn.ParameterList([
            esr.Tensor(
                esr.zeros([ne], dtype=torch.long), mode='partition'
            ) for i in range(4)
        ])

    def forward(self):
        for i in range(4):
            # (ne,)
            self.src_p[i].copy_(self.selector_src(self.cells_p[i]))
            self.dst_p[i].copy_(self.selector_dst(self.cells_p[i]))


class PoissonInitializer(esr.Module):
    def __init__(self, poisson: str, scale: int, device='cpu'):
        super().__init__()

        mesh = esr.Mesh(
            esr.linspace(0, 1, scale),
            esr.linspace(0, 1, scale)
        )

        self.points = esr.Tensor(
            mesh.vertices,
            mode='partition'
        )

        self.reducer = esr.Reducer(
            mesh.src,
            mesh.nc
        )

        nc = mesh.nc
        ne = mesh.ne

        self.selector_src_p = torch.nn.ModuleList([
            esr.Selector(
                esr.hdf5(poisson, f'src_p{i}', dtype=torch.long),
            ) for i in range(4)
        ])
        self.selector_dst_p = torch.nn.ModuleList([
            esr.Selector(
                esr.hdf5(poisson, f'dst_p{i}', dtype=torch.long),
            ) for i in range(4)
        ])

        
        self.selector_cells_p = torch.nn.ModuleList([
            esr.Selector(
                [
                    mesh.indices[:-1, :-1],
                    mesh.indices[1:, :-1],
                    mesh.indices[1:, 1:],
                    mesh.indices[:-1, 1:],
                ][i],
            ) for i in range(4)
        ])

        bcells = ConcatDataLoader([
            mesh.cell_indices[0, :],
            mesh.cell_indices[:, -1],
            mesh.cell_indices[-1, :],
            mesh.cell_indices[:, 0],
        ])
        self.bselector = esr.Selector(bcells)
        self.breducer = esr.Reducer(bcells, nc)

        self.selector_bp = torch.nn.ModuleList([
            esr.Selector(
                [
                    ConcatDataLoader([
                        mesh.indices[0, :-1],
                        mesh.indices[:-1, -1],
                        mesh.indices[-1, :-1],
                        mesh.indices[:-1, 0]
                    ]),
                    ConcatDataLoader([
                        mesh.indices[0, 1:],
                        mesh.indices[1:, -1],
                        mesh.indices[-1, 1:],
                        mesh.indices[1:, 0]
                    ]),
                ][i]
            ) for i in range(2)
        ])

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
        src_p3 = self.selector_src_p[3](self.points)

        dst_p0 = self.selector_dst_p[0](self.points)
        dst_p1 = self.selector_dst_p[1](self.points)
        dst_p2 = self.selector_dst_p[2](self.points)
        dst_p3 = self.selector_dst_p[3](self.points)

        src_cent = (src_p0 + src_p1 + src_p2 + src_p3) / 4.
        dst_cent = (dst_p0 + dst_p1 + dst_p2 + dst_p3) / 4.

        dist = dst_cent - src_cent

        norm01_x, norm01_y = self.get_face_norm(src_p2, src_p0, src_p1)
        norm12_x, norm12_y = self.get_face_norm(src_p0, src_p1, src_p2)
        norm23_x, norm23_y = self.get_face_norm(src_p1, src_p2, src_p3)
        norm30_x, norm30_y = self.get_face_norm(src_p2, src_p3, src_p0)

        norm01_x_, norm01_y_ = self.get_face_norm(dst_cent, src_p0, src_p1)
        norm12_x_, norm12_y_ = self.get_face_norm(dst_cent, src_p1, src_p2)
        norm23_x_, norm23_y_ = self.get_face_norm(dst_cent, src_p2, src_p3)
        norm30_x_, norm30_y_ = self.get_face_norm(dst_cent, src_p3, src_p0)

        condition = (norm01_x * norm01_x_ + norm01_y * norm01_y_) < 0
        norm_x = torch.where(condition, norm01_x, 0.)
        norm_y = torch.where(condition, norm01_y, 0.)

        condition = (norm12_x * norm12_x_ + norm12_y * norm12_y_) < 0
        norm_x = torch.where(condition, norm12_x, norm_x)
        norm_y = torch.where(condition, norm12_y, norm_y)

        condition = (norm23_x * norm23_x_ + norm23_y * norm23_y_) < 0
        norm_x = torch.where(condition, norm23_x, norm_x)
        norm_y = torch.where(condition, norm23_y, norm_y)

        condition = (norm30_x * norm30_x_ + norm30_y * norm30_y_) < 0
        norm_x = torch.where(condition, norm30_x, norm_x)
        norm_y = torch.where(condition, norm30_y, norm_y)

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
        p3 = self.selector_cells_p[3](self.points)
        x3 = p3[:, 0]
        y3 = p3[:, 1]

        area = torch.abs(
            (x0 - x1) * (y0 - y3)
        )

        self.centroid[:] = (p0 + p1 + p2 + p3) / 4.
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


def assemble_poisson(scale: int, poisson: str, device='cpu'):
    components = PoissonMeshComponentsCollector(scale)
    components.to(device)

    [components] = esr.compile(
        [components], 'none', partition_mode='evenly'
    )  # type: ignore
    components: PoissonMeshComponentsCollector
    components()

    for i in range(4):
        components.src_p[i].save(poisson, f'src_p{i}')
        components.dst_p[i].save(poisson, f'dst_p{i}')

    initializer = PoissonInitializer(poisson, scale)
    initializer.to(device)

    [initializer] = esr.compile(
        [initializer], 'none', partition_mode='evenly'
    )  # type: ignore
    initializer: PoissonInitializer
    initializer()

    initializer.b.save(poisson, 'b')
    initializer.Ac.save(poisson, 'Ac')
    initializer.Af.save(poisson, 'Af')
    initializer.rho.save(poisson, 'rho')
    initializer.centroid.save(poisson, 'centroid')

    return poisson


if __name__ == '__main__':
    """
    Usage:

    torchrun --nproc_per_node=4 tutorial/poisson/assemble_poisson.py \
        100 ~/.easier/Poisson_100.hdf5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default="cpu"
    )
    parser.add_argument(
        "--comm_backend", type=str, choices=["gloo", "nccl"],
        default='gloo'
    )
    parser.add_argument("scale", type=int)
    parser.add_argument("poisson", type=str)
    args = parser.parse_args()

    print("Assemble Poisson:")
    print("output HDF5 file:", args.poisson)

    esr.init(args.comm_backend)

    assemble_poisson(args.scale, args.poisson, args.device)
