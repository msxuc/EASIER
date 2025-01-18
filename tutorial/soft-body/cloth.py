import torch
import numpy as np
import matplotlib.pyplot as plt

import easier as esr

import meshio
from tqdm import trange


def cross_product(a, b):
    r0 = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    r1 = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    r2 = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    return torch.stack((r0, r1, r2), dim=1)


def prosses_mesh(path: str):
    mesh = meshio.read(path)

    for cell in mesh.cells:
        if cell.type == 'triangle':
            faces = cell.data

    edges = np.vstack((
        faces[:, (0, 1)],
        faces[:, (0, 2)],
        faces[:, (1, 0)],
        faces[:, (1, 2)],
        faces[:, (2, 0)],
        faces[:, (2, 1)],
    ))

    return (
        torch.from_numpy(mesh.points).double(),
        torch.from_numpy(edges).to(dtype=torch.int64),
        torch.from_numpy(faces).to(dtype=torch.int64)
    )


class Cloth(esr.Module):
    def __init__(self, points, edges, faces):
        super().__init__()

        self.dt = 1e-6
        self.Ks = 1e1
        self.M = 1e-6
        self.Kd = 1e1 * (self.M * self.Ks) ** 0.5

        # x_0
        x1 = points
        self.x1 = esr.Tensor(x1.cuda(), mode='partition')

        # v_-1/2
        v = torch.zeros(points.shape).double()  # * \
        #     torch.tensor([[0, 0, 4.9 * self.dt]]).double()
        self.v = esr.Tensor(v.cuda(), mode='partition')

        x0 = x1 - v * self.dt
        self.x0 = esr.Tensor(x0.cuda(), mode='partition')

        L = torch.norm(
            points[edges[:, 0]] - points[edges[:, 1]], dim=1).reshape(-1, 1)
        self.L = esr.Tensor(L.cuda(), mode='partition')

        # self.select_1 = esr.Selector(faces[:, 0])
        # self.select_2 = esr.Selector(faces[:, 1])
        # self.select_3 = esr.Selector(faces[:, 2])

        self.select_i = esr.Selector(edges[:, 0].cuda())
        self.select_j = esr.Selector(edges[:, 1].cuda())
        self.reduce_i = esr.Reducer(edges[:, 0].cuda(), n=points.shape[0])

    def f(self, x, v):
        x_i = self.select_i(x)
        x_j = self.select_j(x)

        v_i = self.select_i(v)
        v_j = self.select_j(v)

        dx = x_i - x_j
        L = torch.norm(dx, dim=1).reshape(-1, 1)
        ex = dx / L

        dl = L - self.L
        dv = torch.einsum('ij,ij->i', v_i - v_j, ex).reshape(-1, 1)

        f_ij = - self.Ks * dl * ex \
            - self.Kd * dv * ex \
            + torch.tensor([[0., 0., -self.M * 9.8]]).cuda().double()
        # f_ij = - self.Ks * dl * ex + \
        #     torch.tensor([[0., 0., -self.M * 9.8]]).cuda().double()

        f_i = self.reduce_i(f_ij)

        f_i[1, 0] = 0.
        f_i[1, 1] = 0.
        f_i[1, 2] = 0.

        f_i[2, 0] = 0.
        f_i[2, 1] = 0.
        f_i[2, 2] = 0.

        return f_i

    # def _forward(self):
    #     v1 = self.v + self.dt * self.f(self.x1, self.v) / self.M
    #     x2 = self.x1 + self.dt * v1
    #     v0 = (x2 - self.x0) / 2 / self.dt
    #     self.v[:] = self.v + self.dt * self.f(self.x1, v0) / self.M
    #     self.x0[:] = self.x1[:]
    #     self.x1[:] = self.x1 + self.dt * self.v

    def forward(self):
        v = self.v + 0.5 * self.dt * self.f(self.x0, self.v) / self.M
        x = self.x0 + 0.5 * self.dt * v
        self.v[:] += self.dt * self.f(x, v) / self.M
        self.x0[:] += self.dt * self.v


if __name__ == "__main__":
    points, edges, faces = prosses_mesh("mesh.vtk")

    cloth = Cloth(points, edges, faces)
    cloth, = esr.compile([cloth,], backend='none')

    for t in trange(1000000):

        cloth()

        if t % 10000 != 0:
            continue

        points = cloth.x0.collect().cpu().numpy()

        plt.cla()
        plt.figure(figsize=(3, 3))
        plt.scatter(points[:, 0], points[:, 2], linewidths=0.1, s=2.)
        plt.title(f't={cloth.dt * t:.04f}')
        plt.savefig(f'res/fig{t//10000:05d}.jpg', dpi=300)
        plt.close()
