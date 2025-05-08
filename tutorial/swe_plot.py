# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import argparse
from tqdm import tqdm

import matplotlib.animation as anime
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import matplotlib.colors as colors

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--filename", type=str)
args = parser.parse_args()


fig = plot.figure()
ax = fig.add_subplot(projection='3d')
writer = anime.PillowWriter(fps=5)

zmin, zmax = 0.97, 1.03


with writer.saving(fig, args.filename, dpi=72):
    # for i in tqdm(range(100)):
    for i in tqdm(range(100)):
        data = np.load(f"{args.data_dir}/data{i:03d}.npz")
        x, y, z = (data['x'], data['y'], data['z'])

        ax.azim = 45
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(zmin, zmax + 0.1)

        norm = colors.TwoSlopeNorm(vcenter=zmin, vmin=0, vmax=zmax)

        ax.scatter(x, y, z, c=z, cmap=cm.Blues, norm=norm)

        writer.grab_frame()
        plot.cla()