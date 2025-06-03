# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import argparse
from tqdm import tqdm

import matplotlib.animation as anime
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import matplotlib.colors as colors

if __name__ == '__main__':
    """
    Usage:

    python tutorial/shallow_water_equation/swe_plot.py \
        --data_dir res --filename swe.gif
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()

    fig = plot.figure()
    ax = fig.add_subplot(projection='3d')
    writer = anime.PillowWriter(fps=5)

    z_range = 0.1

    with writer.saving(fig, args.filename, dpi=72):
        for i in tqdm(range(100)):
            data = np.load(f"{args.data_dir}/data{i:03d}.npz")
            x, y, z = (data['x'], data['y'], data['z'])

            ax.azim = 45
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(1 - z_range, 1 + z_range)

            norm = colors.Normalize(vmin=1 - z_range, vmax=1 + z_range)
            ax.scatter(x, y, z, c=z, cmap=cm.Blues, norm=norm)

            writer.grab_frame()
            plot.cla()
