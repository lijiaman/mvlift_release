import os
import rerun as rr
import numpy as np

# from matplotlib import colormaps

from rerun.components import Material

# _index_to_color = lambda x, cmap="tab10": colormaps[cmap](x % colormaps[cmap].N)

def compute_vertex_normals(vertices, faces):
    # Initialize normals array with zeros
    normals = np.zeros(vertices.shape, dtype=np.float32)

    # Iterate through each face and calculate the normal
    for face in faces:
        vert_indices = face[:3]
        v1, v2, v3 = vertices[vert_indices, :]

        # Compute the vectors from v1 to v2 and v1 to v3
        edge1 = v2 - v1
        edge2 = v3 - v1

        # Use the cross product to get the face normal
        face_normal = np.cross(edge1, edge2)

        # Normalize the face normal
        face_normal /= np.linalg.norm(face_normal)

        # Add the face normal to each vertex that makes up the face
        normals[vert_indices] += face_normal

    # Normalize all vertex normals
    for i in range(len(normals)):
        normals[i] /= np.linalg.norm(normals[i])

    return normals

def log_meshes_to_rerun(mesh_dir, mesh_dir_w_vposer, overlap_meshes=False):
    obj_files = sorted([f for f in os.listdir(mesh_dir) if f.endswith('.obj')])

    obj_files_w_vposer = sorted([f for f in os.listdir(mesh_dir_w_vposer) if f.endswith('.obj')])
    
    # rr.init("human_mesh_visualization", spawn=True)

    rr.init("human_mesh_visualization")

    # rr.connect() 

    min_z = None 
    for frame_id, obj_file in enumerate(obj_files):
        mesh_path = os.path.join(mesh_dir, obj_file)
        vertices, faces = load_mesh_from_obj(mesh_path)

        if min_z is None:
            min_z = vertices[:, 2].min()

        vertices[:, 2] -= min_z 

        vertex_normals = compute_vertex_normals(vertices, faces) 

        rr.set_time_sequence("frame_id", frame_id)
        
        # Log the mesh data
        rr.log(
            f"world/human_mesh",
            rr.Mesh3D(
                vertex_positions=vertices,
                vertex_normals=vertex_normals, 
                triangle_indices=faces,
                # mesh_material=Material(albedo_factor=_index_to_color(0))  # White color
                mesh_material=Material(albedo_factor=[3, 244, 252])  # White color
            )
        )

    min_z = None 
    for frame_id, obj_file in enumerate(obj_files_w_vposer):
        mesh_path = os.path.join(mesh_dir_w_vposer, obj_file)
        vertices, faces = load_mesh_from_obj(mesh_path)

        if not overlap_meshes:
            # Move vertices in x direction to visualize simultaneously 
            vertices[:, 1] += 1  

        if min_z is None:
            min_z = vertices[:, 2].min()

        vertices[:, 2] -= min_z 

        vertex_normals = compute_vertex_normals(vertices, faces) 

        rr.set_time_sequence("frame_id", frame_id)
        
        # Log the mesh data
        rr.log(
            f"world/human_mesh_w_vposer",
            rr.Mesh3D(
                vertex_positions=vertices,
                vertex_normals=vertex_normals, 
                triangle_indices=faces,
                # mesh_material=Material(albedo_factor=_index_to_color(0))  # White color
                mesh_material=Material(albedo_factor=[152, 248, 137])  # White color
            )
        )

    rr.save("./test_vposer_cmp.rrd")

def load_mesh_from_obj(file_path: str):
    """ A simple parser to extract vertices and faces from an OBJ file. """
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:4])))
            elif line.startswith('f '):
                face = [int(idx.split('/')[0]) - 1 for idx in line.strip().split()[1:4]]
                faces.append(face)
    return np.array(vertices), np.array(faces)

if __name__ == "__main__":
    # mesh_directory = "/viscam/projects/vimotion/opt3d_w_multiview_gen2d_vposer_cmp/AIST/0_sample_0_objs"
    # mesh_directory_w_vposer = "/viscam/projects/vimotion/opt3d_w_multiview_gen2d_vposer_cmp/AIST/0_sample_0_objs_vposer"
    
    mesh_directory = "/viscam/projects/vimotion/work_opt3d_w_multiview_gen2d/AIST_all_view/8_sample_0_objs"
    mesh_directory_w_vposer = "/viscam/projects/vimotion/work_opt3d_w_multiview_gen2d/AIST_all_view_w_vposer/8_sample_0_objs"
    log_meshes_to_rerun(mesh_directory, mesh_directory_w_vposer)

    # mesh_directory = "/viscam/projects/vimotion/work_opt3d_w_multiview_gen2d/AIST_all_view_w_vposer/8_sample_0_objs"
    # mesh_directory_w_vposer = "/viscam/projects/vimotion/work_opt3d_w_multiview_gen2d/AIST_all_view_w_multi_step_recon_w_vposer/8_sample_0_objs"
    # log_meshes_to_rerun(mesh_directory, mesh_directory_w_vposer)

    # mesh_directory = "/viscam/projects/vimotion/work_opt3d_w_multiview_gen2d/AIST_all_view_w_multi_step_recon_w_vposer/8_sample_0_objs"
    # mesh_directory_w_vposer = "/viscam/projects/vimotion/work_opt3d_w_multiview_gen2d/AIST_all_view_w_multi_step_recon_w_vposer/8_sample_0_objs_gt"
    # log_meshes_to_rerun(mesh_directory, mesh_directory_w_vposer, overlap_meshes=True)

    # mesh_directory = "/viscam/projects/vimotion/opt3d_w_sds_work/AIST_all_view_w_sds_w_vposer/8_sample_0_objs"
    # mesh_directory_w_vposer = "/viscam/projects/vimotion/opt3d_w_sds_work/AIST_all_view_w_sds_w_vposer/8_sample_0_objs_gt"
    # log_meshes_to_rerun(mesh_directory, mesh_directory_w_vposer)

'''
save to .rrd file. 
rerun --web-viewer test.rrd 
replace localhost with the machine name (move3.stanford.edu) (need to connect to VPN) 
'''