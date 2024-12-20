"""Manually build a mesh from points and faces"""
# from vedo import Mesh, Plotter, Sphere
import pyvista as pv

# Define the vertices and faces that make up the mesh
verts = [(1.,1.,0.), (1.,-1.,0.), (-1.,1.,0.), (-1.,-1.,0.), (0.,0.,2.)]
cells = [
    3, 0, 1, 4,
    3, 0, 2, 4,
    3, 2, 3, 4,
    3, 1, 3, 4,
    4, 0, 1, 3, 2
]  # (2,3,7,6), (4,5,6,7)] # cells same as faces

wall = pv.PolyData(verts, cells)
ball = pv.Icosphere(radius=1.0, center=(3, 3, 0), nsub=3)
edges = ball.extract_all_edges()

import numpy as np
print(np.array(edges.lines.reshape(-1, 3)[:, 1:]))


# Create a plotter object
plotter = pv.Plotter(off_screen=False)
# cubemap = pv.examples.download_sky_box_cube_map()
# plotter.set_environment_texture(cubemap)

# Add the mesh to the plotter
plotter.add_mesh(ball, color='pink', opacity=1.0, specular=0.8, specular_power=30)
plotter.add_mesh(wall, color='tan')
plotter.add_mesh(edges, color='black')

# Create a light source
# light = pv.Light(position=(50, 50, 50), focal_point=(0, 0, 0), color='white', intensity=0.2)

# Add the light to the plotter
# plotter.add_light(light)

plotter.show_axes()

# Render the scene and save it as a PNG
# plotter.show(screenshot="mesh.png")
plotter.show(screenshot="mesh.png")


# # Build the polygonal Mesh object from the vertices and faces
# msh = Mesh([verts, cells]).c("lightblue").lighting(roughness=0.05)
# # msh = Sphere().c("lightblue").alpha(0.9).phong().lighting("glossy")
# plt = Plotter(offscreen=True, screensize=(1920, 1080), bg="linen")
#
# num_frames = 10
# angle_per_frame = 360 / num_frames
#
# for i in range(num_frames):
#     plt.show(msh, viewup='z', axes=1, interactive=False)
#     plt.camera.Azimuth(angle_per_frame * i)
#     plt.screenshot(f"frame_{i:03d}.png", scale=5.0)


# styles = ['default', 'metallic', 'plastic', 'shiny', 'glossy', 'ambient', 'off']
#
# for i,s in enumerate(styles):
#     msh_copy = msh.clone(deep=False).lighting(s)
#     plt.at(i).show(msh_copy, axes=1, viewup="z")

# msh = Mesh(dataurl+"beethoven.ply").c('gold').subdivide()
#
# plt = Plotter(N=len(styles), bg='bb')
#
# for i,s in enumerate(styles):
#     msh_copy = msh.clone(deep=False).lighting(s)
#     plt.at(i).show(msh_copy, s)
#
# plt.interactive().close()