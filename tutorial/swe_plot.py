# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
# import pyvista as pv
# import imageio.v2 as imageio
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--filename", type=str)
args = parser.parse_args()


import matplotlib.animation as anime
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import matplotlib.colors as colors


fig = plot.figure()
ax = fig.add_subplot(projection='3d')
writer = anime.PillowWriter(fps=15)

# xs = np.linspace(0, 1, 1000)
# ys = np.linspace(0, 1, 1000)
# xs, ys = np.meshgrid(xs, ys)

# def func(x,y,r,t):
#     return np.cos(r/2+t)*np.exp(-np.square(r)/50)
# rlist = np.sqrt( np.square(xs) + np.square(ys) )

with writer.saving(fig, args.filename, dpi=72):
    for i in tqdm(range(5)):
        data = np.load(f"{args.data_dir}/data{i:03d}.npz")
        x, y, z = (data['x'], data['y'], data['z']*2)

        # print(
        #     np.min(x), np.max(x),
        #     np.min(y), np.max(y),
        #     np.min(z), np.max(z),
        # )

        # x = x.reshape(200, 200)
        # y = y.reshape(200, 200)
        # z = z.reshape(200, 200)

        ax.azim = 45
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(1.9, 2.3)

        # ax.plot_surface(x, y, z, rstride=2, cstride=2)
        ax.scatter(x, y, z, c=cm.Blues((z - 1.9) / 0.4 / 2 + 0.5).reshape(-1, 4))

        writer.grab_frame()
        plot.cla()

# frames = []
# for i in range(100):
#     data = np.load(f"{args.data_dir}/data{i:03d}.npz")
#     points = np.concatenate(
#         (data['x'], data['y'], data['z']*2), axis=0).transpose()
#     point_cloud = pv.PolyData(points)
#     surface = point_cloud.delaunay_2d().extract_geometry()

#     plotter = pv.Plotter(off_screen=True)
#     plotter.add_mesh(surface, color='skyblue', smooth_shading=True)
#     plotter.render()
#     frame = plotter.screenshot()
#     frames.append(frame)
#     plotter.clear()

# gif_filename = args.gif_filename
# imageio.mimwrite(gif_filename, frames, loop=0, duration=0.2)
