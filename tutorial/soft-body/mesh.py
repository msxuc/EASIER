import gmsh

# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("Square")

# Define the corner points of the square
lc = 1e-2  # Characteristic length for meshing
gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(0, 0, 1, lc, 2)
gmsh.model.geo.addPoint(1, 0, 1, lc, 3)
gmsh.model.geo.addPoint(1, 0, 0, lc, 4)

# Define the lines forming the square's edges
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

# Create a loop and a surface
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

# Synchronize the model and generate the mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

# Optionally, save the mesh to a file
gmsh.write("mesh.vtk")

# Finalize Gmsh
gmsh.finalize()
