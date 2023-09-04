import open3d as o3d
import numpy as np
import time
# Read the point cloud from a PLY file

mesh_file = "your_mesh_with_colors.obj" 
 
# Load the mesh from the file
mesh = o3d.io.read_triangle_mesh(mesh_file)

# Create a visualization window
o3d.visualization.draw_geometries([mesh])




