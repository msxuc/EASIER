import torch
import numpy as np
import easier as esr

import gmsh
import meshio
import pyvista as pv
from tqdm import trange


def prosses_mesh(path: str):
    mesh = meshio.read(path)

    tetra = None
    faces = None
    for cell in mesh.cells:
        if cell.type == 'tetra':
            tetra = cell.data
        if cell.type == 'triangle':
            faces = cell.data

    edges = np.vstack((
        tetra[:, (0, 1)],
        tetra[:, (0, 2)],
        tetra[:, (0, 3)],
        tetra[:, (1, 0)],
        tetra[:, (1, 2)],
        tetra[:, (1, 3)],
        tetra[:, (2, 0)],
        tetra[:, (2, 1)],
        tetra[:, (2, 3)],
        tetra[:, (3, 0)],
        tetra[:, (3, 1)],
        tetra[:, (3, 2)],
    ))

    faces = np.hstack(([[3] for _ in range(len(faces))], faces))

    return mesh.points, edges, faces

    # faces = torch.tensor(mesh.faces.reshape(-1, 4))[:, 1:]

    # return points, edges, faces


class Ball(esr.Module):
    def __init__(self, points, edges):
        super().__init__()

        L = torch.norm(
            points[edges[:, 0]] - points[edges[:, 1]], dim=1).reshape(-1, 1)

        # points[0, 2] += 0.05

        self.x = esr.Tensor(points, mode='partition')
        # self.v = esr.zeros(points.shape)
        v = torch.tensor([[0., 0., -0.01] for i in range(points.shape[0])]).double()
        self.v = esr.Tensor(v, mode='partition')
        self.L = esr.Tensor(L, mode='partition')

        self.K = 1.
        self.M = 1.
        self.dt = 0.01

        # self.select_1 = esr.Selector(faces[:, 0])
        # self.select_2 = esr.Selector(faces[:, 1])
        # self.select_3 = esr.Selector(faces[:, 2])

        self.select_i = esr.Selector(edges[:, 0])
        self.select_j = esr.Selector(edges[:, 1])
        self.reduce_i = esr.Reducer(edges[:, 0], n=points.shape[0])

    # def volume(self):
    def delta(self, x, v):
        x_i = self.select_i(x)
        x_j = self.select_j(x)

        v_i = self.select_i(v)
        v_j = self.select_j(v)

        dx = x_i - x_j
        norm_dx = torch.norm(dx, dim=1).reshape(-1, 1)
        ex = dx / norm_dx

        force_ij = -self.K * (norm_dx - self.L) * ex
        # force_ij = torch.einsum('ij,ij->i', v_i - v_j, ex).reshape(-1, 1) * 0.01
        force_ij -= torch.einsum('ij,ij->i', v_i - v_j, ex).reshape(-1, 1) * ex * 0.1

        spring_force = self.reduce_i(force_ij)
        # gravity = -self.M * 0.001

        force = spring_force
        # force[:, 2] += gravity

        v[:, 2] = torch.where(x[:, 2] <= 0, -v[:, 2], v[:, 2])
        delta_x = v
        delta_v = force / self.M

        return delta_x, delta_v

    def forward(self):
        delta_x1, delta_v1 = self.delta(self.x, self.v)
        delta_x2, delta_v2 = self.delta(
            self.x + 0.5 * self.dt * delta_x1,
            self.v + 0.5 * self.dt * delta_v1)
        delta_x3, delta_v3 = self.delta(
            self.x + 0.5 * self.dt * delta_x2,
            self.v + 0.5 * self.dt * delta_v2)
        delta_x4, delta_v4 = self.delta(
            self.x + self.dt * delta_x3,
            self.v + self.dt * delta_v3)

        self.x[:] += self.dt / 6 * (
            delta_x1 + 2 * delta_x2 + 2 * delta_x3 + delta_x4)
        self.v[:] += self.dt / 6 * (
            delta_v1 + 2 * delta_v2 + 2 * delta_v3 + delta_v4)


if __name__ == "__main__":
    # mesh generateion
    # gmsh.initialize()
    # gmsh.model.add("sphere_model")
    # gmsh.model.occ.addSphere(0, 0, 0.5, 0.5)
    # gmsh.model.occ.synchronize()
    # gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
    # gmsh.model.mesh.generate(3)
    # gmsh.write("sphere.msh")
    # gmsh.finalize()

    points, edges, faces = prosses_mesh("sphere.msh")

    ball = Ball(torch.from_numpy(points), torch.from_numpy(edges))
    ball, = esr.compile([ball], backend='none')

    mesh = pv.PolyData(points, faces)
    plane = pv.Plane()

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(plane, color='lightgray', opacity=0.5)
    plotter.add_mesh(
        mesh, show_edges=False,
        color='skyblue', opacity=1.0, specular=0.8, specular_power=30
    )
    plotter.camera.zoom(0.5)
    plotter.camera_position = 'xz'

    # Open a GIF file for saving frames
    plotter.open_gif("wave_animation.gif")

    for i in trange(10000):

        ball()

        if i % 20 != 0:
            continue

        points = ball.x.collect()
        mesh.points[:] = points[:]
        plotter.update_coordinates(plane.points, plane)
        plotter.update_coordinates(mesh.points, mesh)
        plotter.add_mesh(plane, color='lightgray', opacity=0.5)
        plotter.write_frame()

    plotter.close()
