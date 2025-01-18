import os
import argparse
from typing_extensions import Literal

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import easier as esr
from easier.core.module import VertexSet
from easier.examples import Poisson
from easier.examples.mesh import get_triagular_mesh


class ShallowWaterEquation(esr.Module):
    def __init__(self, mesh_size=100, dt=0.005) -> None:
        super().__init__()
        mesh_path = mesh = get_triagular_mesh(mesh_size)
        path, name = os.path.split(mesh_path)
        path = os.path.join(path, 'Poisson_' + name)
        if os.path.exists(path):
            return path

        with h5py.File(mesh_path, 'r') as mesh:
            src = torch.from_numpy(mesh['src'][...]).long()
            dst = torch.from_numpy(mesh['dst'][...]).long()
            cells = torch.from_numpy(mesh['cells'][...]).long()
            points = torch.from_numpy(mesh['points'][...]).double()
            bcells = torch.from_numpy(mesh['bcells'][...]).long()
            bpoints = torch.from_numpy(mesh['bpoints'][...]).long()


        self.dt = 0.005

        mesh = Poisson(type).mesh

        # cells (torch.LongTensor): three point indices for each triangle
        #   cells, with shape `(nc, 3)`, `nc` means number of cells
        cells = torch.from_numpy(mesh['cells']).long()
        # points (torch.DoubleTensor): point coordinates on a plane,
        #   with shape `(np, 2)`, `np` means number of points
        points = torch.from_numpy(mesh['points']).double()

        pset = VertexSet(points.shape[0])
        cset = esr.VertexSet(cells.shape[0])

        # src (torch.LongTensor): src cell indices, with shape `(ne,)`
        src = torch.from_numpy(mesh['src']).long()
        # dst (torch.LongTensor): dst cell indices, with shape `(ne,)`
        dst = torch.from_numpy(mesh['dst']).long()
        # bcells (torch.LongTensor): boundary cell indices, with shape `(nbc,)`,
        #   `nbc` means number of boundary cell
        bcells = torch.from_numpy(mesh['bcells']).long()
        # bpoints (torch.LongTensor): boundary points indices in each boundary
        #   cell, with shape `(nbc, 2)`, `nbc` means number of boundary cell
        bpoints = torch.from_numpy(mesh['bpoints']).long()

        self.scatter = esr.Scatter(dst, cset)
        self.gather_src = esr.Gather(src, cset)
        self.gather_dst = esr.Gather(dst, cset)
        self.scatter_b = esr.Scatter(bcells, cset)
        self.gather_b = esr.Gather(bcells, cset)

        self.x = esr.VertexTensor(torch.zeros(cset.nv).double(), cset)
        self.y = esr.VertexTensor(torch.zeros(cset.nv).double(), cset)
        self.area = esr.VertexTensor(torch.zeros(cset.nv).double(), cset)
        self.sx = esr.EdgeTensor(torch.zeros(src.shape[0]).double())
        self.sy = esr.EdgeTensor(torch.zeros(src.shape[0]).double())
        self.bsx = esr.EdgeTensor(torch.zeros(bcells.shape[0]).double())
        self.bsy = esr.EdgeTensor(torch.zeros(bcells.shape[0]).double())
        self.h = esr.VertexTensor(torch.zeros(cset.nv).double(), cset)
        self.alpha = esr.EdgeTensor(torch.zeros(src.shape[0]).double())

        self.cells = cells
        self.cset = cset
        self.points = points
        self.pset = pset
        self.src = src
        self.dst = dst
        self.bcells = bcells
        self.bpoints = bpoints

        self.uh = esr.VertexTensor(torch.zeros(cset.nv).double(), cset)
        self.vh = esr.VertexTensor(torch.zeros(cset.nv).double(), cset)

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
            self.scatter(uh_f_times_sx + vh_f_times_sy) +
            self.scatter_b(self.gather_b(h * 0))
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


class Initializer(torch.nn.Module):
    def __init__(self, cells, cset, points, pset, src, dst, bcells, bpoints,
                 x, y, area, sx, sy, bsx, bsy, h, alpha):
        super().__init__()
        self.x = x
        self.y = y
        self.area = area
        self.sx = sx
        self.sy = sy
        self.bsx = bsx
        self.bsy = bsy
        self.h = h
        self.alpha = alpha

        src_p = cells[src]
        dst_p = cells[dst]

        self.points = esr.VertexTensor(points[:, :2], pset)
        self.cell_points = esr.VertexTensor(points[cells, :2], cset)

        self.gather_src_p0 = esr.Gather(src_p[:, 0], pset)
        self.gather_src_p1 = esr.Gather(src_p[:, 1], pset)
        self.gather_src_p2 = esr.Gather(src_p[:, 2], pset)

        self.gather_dst_p0 = esr.Gather(dst_p[:, 0], pset)
        self.gather_dst_p1 = esr.Gather(dst_p[:, 1], pset)
        self.gather_dst_p2 = esr.Gather(dst_p[:, 2], pset)

        self.bscatter = esr.Scatter(bcells, cset)
        self.bgather = esr.Gather(bcells, cset)
        self.bgather_p0 = esr.Gather(bpoints[:, 0], pset)
        self.bgather_p1 = esr.Gather(bpoints[:, 1], pset)

    def forward(self):
        src_p0 = self.gather_src_p0(self.points)
        src_p1 = self.gather_src_p1(self.points)
        src_p2 = self.gather_src_p2(self.points)

        dst_p0 = self.gather_dst_p0(self.points)
        dst_p1 = self.gather_dst_p1(self.points)
        dst_p2 = self.gather_dst_p2(self.points)

        dst_cent = (dst_p0 + dst_p1 + dst_p2) / 3.
        src_cent = (src_p0 + src_p1 + src_p2) / 3.

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

        x0 = self.cell_points[:, 0, 0]
        y0 = self.cell_points[:, 0, 1]
        x1 = self.cell_points[:, 1, 0]
        y1 = self.cell_points[:, 1, 1]
        x2 = self.cell_points[:, 2, 0]
        y2 = self.cell_points[:, 2, 1]

        self.area[:] = 0.5 * torch.abs(
            x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
        centroid = self.cell_points.sum(dim=1) / 3.
        self.x[:] = centroid[:, 0]
        self.y[:] = centroid[:, 1]

        self.h[:] = 1 + 0.1 * torch.exp(
            -100 * ((self.x - 0.0)**2 + (self.y - 0.0)**2))

        # boundary condition
        b_p0 = self.bgather_p0(self.points)
        b_p1 = self.bgather_p1(self.points)
        b_cell_cent = self.bgather(centroid)

        bnorm_x, bnorm_y = self.get_face_norm(b_cell_cent, b_p0, b_p1)
        self.bsx[:] = -bnorm_x
        self.bsy[:] = -bnorm_y

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


if __name__ == "__main__":
    # torchrun --nnodes=1 --nproc-per-node=4 \
    #   shallow_water_equation.py --backend=gpu --plot=true
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--backend", type=str, choices=["none", "torch", "cpu", "gpu"],
        default="none")
    parser.add_argument(
        "--scale", type=str, choices=["small", "medium", "large"],
        default="small")
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()

    eqn = ShallowWaterEquation(args.scale).to(args.device)
    init = Initializer(
            eqn.cells, eqn.cset, eqn.points, eqn.pset, eqn.src, eqn.dst,
            eqn.bcells, eqn.bpoints, eqn.x, eqn.y, eqn.area, eqn.sx, eqn.sy,
            eqn.bsx, eqn.bsy, eqn.h, eqn.alpha)
    init()
    esr.jit([eqn], args.backend)

    images = []
    for i in tqdm(range(1000)):
        if args.plot:
            x = eqn.x.sync().cpu().numpy(),
            y = eqn.y.sync().cpu().numpy(),
            c = eqn.h.sync().cpu().numpy(),
            if int(os.environ.get("LOCAL_RANK", 0)) == 0 and i % 10 == 0:
                plt.cla()
                fig = plt.figure(figsize=(6, 5))
                im = plt.scatter(x, y, c=c, s=3.5, cmap='Blues', vmax=1.02,
                                 vmin=0.98, marker='s')
                plt.colorbar(im)
                plt.xlabel(f't={i * eqn.dt:.2f}')
                filename = '/tmp/tmp.png'
                plt.savefig(filename, dpi=300)
                images.append(imageio.imread(filename))
                plt.close()
        eqn()

    if images:
        imageio.mimsave('movie.gif', images, 'GIF', duration=0.2)
