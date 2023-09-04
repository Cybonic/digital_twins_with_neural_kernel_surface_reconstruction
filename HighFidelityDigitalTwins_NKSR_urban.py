
import nksr
import torch
import open3d as o3d
import numpy as np

import matplotlib.pyplot as plt
import utils.velo_utils as velo_utils

class load_from_files():
    def __init__(self,root):
        self.root = root
  
        dt =None
        if 'kitti' in root:
            dt = velo_utils.VELODYNE_64_STRUCTURE
        elif 'on-foot' in root:
            dt = velo_utils.OUSTER_16_STRUCTURE
        else:
            assert dt != None
        self.lidar_parser = velo_utils.parser(dt)
    
    def load_pcds(self,file):
        return self.lidar_parser.velo_read(file)[:,:3]
        

# https://developer.nvidia.com/blog/recreate-high-fidelity-digital-twins-with-neural-kernel-surface-reconstruction/?ncid=so-link-508891-vt37&=&linkId=100000214326019#cid=_so-link_en-us

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

    #from syntetic_data_2D_pcd import generate_shape
    # pcd_file = 'data/kitti/000000.bin'
    pcd_file = 'data/on-foot/0000030.bin'
    velo = load_from_files('on-foot')
    pcd_raw = velo.load_pcds(pcd_file)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_raw)
    # Save the point cloud as a PCD file
    output_file = "pcd_raw.pcd"
    o3d.io.write_point_cloud(output_file, pcd)
    
    #o3d.visualization.draw_geometries([pcd])
    down_pc = pcd.voxel_down_sample(voxel_size=0.01)
    
    # Estimate the normals
    down_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50), fast_normal_computation=True)

    # Normalize the normals for visualization
    normals = np.asarray(down_pc.normals)
    normals = torch.tensor(normals / np.linalg.norm(normals, axis=1, keepdims=True),dtype=torch.float32).to((device))


    xyz = torch.tensor(down_pc.points,dtype=torch.float32).to((device))
    
    camera_position = np.array([0, 0, 10])  # Set the camera position as needed
    vertex_colors = torch.tensor(generate_colors(xyz.cpu().numpy(), camera_position),dtype=torch.float32).to(device)
    
    #vertex_colors -> RGBA (Red, Green, Blue, Alpha) 
    # Note that input_xyz and input_normal are torch tensors of shape [N, 3] and [N, 3] respectively.
    field = reconstructor.reconstruct(xyz, normals)
    #field = reconstructor.reconstruct()
    # input_color is also a tensor of shape [N, 3]
    field.set_texture_field(nksr.fields.PCNNField(xyz, vertex_colors[:,:3]))
    # Increase the dual mesh's resolution.
    mesh = field.extract_dual_mesh(mise_iter=0)

    output_file = "point_cloud.pcd"

    vertices = np.array(mesh.v.cpu().numpy())
    faces = np.array(mesh.f.cpu().numpy())
    colors = np.array(mesh.c.cpu().numpy())

    # Create an Open3D point cloud and assign the vertices
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)



    # Save the point cloud as a PCD file
    output_file = "mesh_as_point_cloud.pcd"
    #o3d.io.write_point_cloud(output_file, point_cloud)
  
    # Visualizing
    
    from pycg import vis
    import pymeshlab as ml
  
    import open3d as o3d
    from meshpy import triangle

  
    import numpy as np
    output_file = "your_mesh.obj"
    
    
    
    
        # Create an Open3D TriangleMesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Create an Open3D VertexColor object
    vertex_colors_o3d = o3d.utility.Vector3dVector(colors)

    # Set vertex colors for the mesh
    mesh.vertex_colors = vertex_colors_o3d

    # Specify the output file path
    output_file = "your_mesh_with_colors.obj"  # You can change the format (e.g., .ply, .stl) as needed

    # Save the mesh with colors to the specified file
    o3d.io.write_triangle_mesh(output_file, mesh)

  





        