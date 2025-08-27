# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import torch

import easier as esr
from easier.core.runtime.data_loader.mesh import Mesh


class ShallowWaterMeshComponentsCollector(esr.Module):
    def __init__(self, scale: int):
        super().__init__()

        mesh = esr.Mesh(
            esr.linspace(0, 1, scale + 1),
            esr.linspace(0, 1, scale + 1)
        )


        self.selector_src = esr.Selector(
            mesh.src
        )
        self.selector_dst = esr.Selector(
            mesh.dst
        )

        ne = mesh.ne

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


class ShallowWaterInitializer(esr.Module):
    def __init__(self, shallow_water: str, scale: int):
        super().__init__()

        vmesh = esr.Mesh(
            esr.linspace(0, 1, scale + 1),
            esr.linspace(0, 1, scale + 1)
        )

        cmesh = esr.Mesh(
            esr.arange(scale, dtype=torch.int64),
            esr.arange(scale, dtype=torch.int64),
        )

        nc = cmesh.nv
        ne = vmesh.ne

        self.points = esr.Tensor(
            vmesh.vertices,
            mode='partition'
        )

        self.reducer = esr.Reducer(
            vmesh.src,
            nc
        )

        self.selector_src_p = torch.nn.ModuleList([
            esr.Selector(
                esr.hdf5(shallow_water, f'src_p{i}', dtype=torch.long),
            ) for i in range(4)
        ])
        self.selector_dst_p = torch.nn.ModuleList([
            esr.Selector(
                esr.hdf5(shallow_water, f'dst_p{i}', dtype=torch.long),
            ) for i in range(4)
        ])

        
        self.selector_cells_p = torch.nn.ModuleList([
            esr.Selector(
                [
                    vmesh.indices[:-1, :-1],
                    vmesh.indices[1:, :-1],
                    vmesh.indices[1:, 1:],
                    vmesh.indices[:-1, 1:],
                ][i],
            ) for i in range(4)
        ])

        # Corner cells are counted twice
        bcells = esr.concat([
            cmesh.indices[0, :],
            cmesh.indices[:, -1],
            cmesh.indices[-1, ::-1],
            cmesh.indices[::-1, 0],
        ])
        nbc = bcells.shape[0]

        self.bselector = esr.Selector(bcells)
        self.breducer = esr.Reducer(bcells, nc)

        self.selector_bp = torch.nn.ModuleList([
            esr.Selector(
                [
                    esr.concat([
                        vmesh.indices[0, :-1],
                        vmesh.indices[:-1, -1],
                        vmesh.indices[-1, -1:0:-1],
                        vmesh.indices[-1:0:-1, 0]
                    ]),
                    esr.concat([
                        vmesh.indices[0, 1:],
                        vmesh.indices[1:, -1],
                        vmesh.indices[-1, -2::-1],
                        vmesh.indices[-2::-1, 0]
                    ]),
                ][i]
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
        src_p3 = self.selector_src_p[3](self.points)

        dst_p0 = self.selector_dst_p[0](self.points)
        dst_p1 = self.selector_dst_p[1](self.points)
        dst_p2 = self.selector_dst_p[2](self.points)
        dst_p3 = self.selector_dst_p[3](self.points)

        src_cent = (src_p0 + src_p1 + src_p2 + src_p3) / 4.
        dst_cent = (dst_p0 + dst_p1 + dst_p2 + dst_p3) / 4.

        norm01_x, norm01_y = self.get_face_norm(src_p3, src_p0, src_p1)
        norm12_x, norm12_y = self.get_face_norm(src_p0, src_p1, src_p2)
        norm23_x, norm23_y = self.get_face_norm(src_p1, src_p2, src_p3)
        norm30_x, norm30_y = self.get_face_norm(src_p2, src_p3, src_p0)

        norm01_x_, norm01_y_ = self.get_face_norm(dst_cent, src_p0, src_p1)
        norm12_x_, norm12_y_ = self.get_face_norm(dst_cent, src_p1, src_p2)
        norm23_x_, norm23_y_ = self.get_face_norm(dst_cent, src_p2, src_p3)
        norm30_x_, norm30_y_ = self.get_face_norm(dst_cent, src_p3, src_p0)

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

        condition = (norm23_x * norm23_x_ + norm23_y * norm23_y_) < 0
        self.sx[:] = torch.where(condition, norm23_x, self.sx)
        self.sy[:] = torch.where(condition, norm23_y, self.sy)
        alpha = self.get_alpha(src_cent, dst_cent, src_p2, src_p3)
        self.alpha[:] = torch.where(condition, alpha, self.alpha)

        condition = (norm30_x * norm30_x_ + norm30_y * norm30_y_) < 0
        self.sx[:] = torch.where(condition, norm30_x, self.sx)
        self.sy[:] = torch.where(condition, norm30_y, self.sy)
        alpha = self.get_alpha(src_cent, dst_cent, src_p3, src_p0)
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
        p3 = self.selector_cells_p[3](self.points)

        self.area[:] = torch.abs((x0 - x1) * (y1 - y2))
        centroid = (p0 + p1 + p2 + p3) / 4.

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


def assemble_shallow_water(scale: int, shallow_water: str, device='cpu'):
    components = ShallowWaterMeshComponentsCollector(scale)
    components.to(device)

    [components] = esr.compile(
        [components], 'torch', partition_mode='evenly'
    )  # type: ignore
    components: ShallowWaterMeshComponentsCollector
    components()

    for i in range(4):
        components.src_p[i].save(shallow_water, f'src_p{i}')
        components.dst_p[i].save(shallow_water, f'dst_p{i}')

    initializer = ShallowWaterInitializer(shallow_water, scale)
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


if __name__ == '__main__':
    """
    Usage:

    torchrun --nproc_per_node=4 \
        tutorial/shallow_water_equation/assemble_shallow_water.py \
        ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default="cpu"
    )
    parser.add_argument(
        "--comm_backend", type=str, choices=["gloo", "nccl"],
        default='gloo'
    )
    # parser.add_argument("mesh", type=str)
    parser.add_argument("scale", type=int)
    parser.add_argument("shallow_water", type=str)
    args = parser.parse_args()

    print("Assemble Poisson:")
    # print("mesh HDF5 file:  ", args.mesh)
    print("output HDF5 file:", args.shallow_water)

    esr.init(args.comm_backend)

    assemble_shallow_water(args.scale, args.shallow_water, args.device)
