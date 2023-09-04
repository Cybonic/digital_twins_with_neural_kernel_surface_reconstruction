

import nksr
import torch
import open3d as o3d
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Calculate the angle between a vertex and the camera viewpoint
def calculate_vertex_angle(vertices, camera_position):
    vertex_positions = np.column_stack((vertices[:,0], vertices[:,1], vertices[:,2]))
    vectors_to_camera = vertex_positions - camera_position
    norms = np.linalg.norm(vectors_to_camera, axis=1)
    normalized_vectors = vectors_to_camera / norms[:, np.newaxis]
    
    # Assume camera is looking along the positive z-axis
    dot_products = normalized_vectors[:, 2]
    angles = np.arccos(dot_products)
    
    return angles

def generate_colors(vertices, camera_position):
    vertex_angles = calculate_vertex_angle(vertices, camera_position)
    
    # Normalize angles to [0, 1] for color mapping
    normalized_angles = (vertex_angles - vertex_angles.min()) / (vertex_angles.max() - vertex_angles.min())
    
    # Use colormap to convert normalized angles to colors
    cmap = plt.get_cmap('viridis')  # You can choose a different colormap
    colors = cmap(normalized_angles)
    
    return colors


device = torch.device("cuda:0")
reconstructor = nksr.Reconstructor(device)

if __name__=='__main__':

    file = "data/bunny/data/bun000.ply"
    pcd = o3d.io.read_point_cloud(file)  # Replace with the actual path to the .ply file if it's not in the same directory

    # Save the point cloud as a PCD file
    output_file = "pcd_raw.pcd"
    o3d.io.write_point_cloud(output_file, pcd)
    
    #o3d.visualization.draw_geometries([pcd])
    
    # Estimate the normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=500), fast_normal_computation=True)

    # Normalize the normals for visualization
    normals = np.asarray(pcd.normals)
    normals = torch.tensor(normals / np.linalg.norm(normals, axis=1, keepdims=True),dtype=torch.float32).to((device))

    xyz = torch.tensor(pcd.points,dtype=torch.float32).to((device))
    
    camera_position = np.array([0, 0, 10])  # Set the camera position as needed
    vertex_colors = torch.tensor(generate_colors(xyz.cpu().numpy(), camera_position),dtype=torch.float32).to(device)
    
    #vertex_colors -> RGBA (Red, Green, Blue, Alpha) 
    # Note that input_xyz and input_normal are torch tensors of shape [N, 3] and [N, 3] respectively.
    field = reconstructor.reconstruct(xyz, normals)
    #field = reconstructor.reconstruct()
    # input_color is also a tensor of shape [N, 3]
    field.set_texture_field(nksr.fields.PCNNField(xyz, vertex_colors[:,:3]))
    # Increase the dual mesh's resolution.
    mesh = field.extract_dual_mesh(mise_iter=2)

    output_file = "point_cloud.pcd"

    vertices = np.array(mesh.v.cpu().numpy())

    # Create an Open3D point cloud and assign the vertices
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)

    # Optionally, assign colors if available in the mesh data
    if hasattr(mesh, 'c'):
        colors = np.array(mesh.c.cpu().numpy())
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Save the point cloud as a PCD file
    output_file = "mesh_as_point_cloud.pcd"
    o3d.io.write_point_cloud(output_file, point_cloud)

    # Visualizing
    
    #from pycg import vis
    # Visualize the mesh using pycg
    #vis.show_3d([vis.mesh(mesh.v, mesh.f, color=mesh.c)])

        