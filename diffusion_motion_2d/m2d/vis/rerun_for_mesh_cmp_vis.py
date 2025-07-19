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

def log_meshes_to_rerun(mesh_dir_list, mesh_dir_name_list, dest_rrd_path):
    rr.init("human_mesh_visualization")

    color_list = [[189, 232, 202], [215, 195, 241], [255, 173, 96], [30, 144, 255]]
    # green, purple, orange  
    for idx in range(len(mesh_dir_list)):
        mesh_dir = mesh_dir_list[idx]

        obj_files = sorted([f for f in os.listdir(mesh_dir) if f.endswith('.obj')])

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
            tag_name = mesh_dir_name_list[idx]
           
            rr.log(
                tag_name,
                rr.Mesh3D(
                    vertex_positions=vertices,
                    vertex_normals=vertex_normals, 
                    triangle_indices=faces,
                    # mesh_material=Material(albedo_factor=_index_to_color(0))  # White color
                    mesh_material=Material(albedo_factor=color_list[idx])  # White color
                )
            )
    

    rr.save(dest_rrd_path)

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
    # res_root_folder = "/viscam/projects/vimotion/hope_final_work_opt3d_res_seq50/AIST_debug_sample36_scale_opt_w_vposer"
    # res_root_folder = "/viscam/projects/vimotion/hope_final_work_opt3d_res_seq50/AIST_debug_sample36_scale_opt_sds_refine_w_vposer"
    # res_root_folder = "/viscam/projects/vimotion/hope_final_work_opt3d_res_seq50/AIST_set1_scale_opt_sds_refine_w_vposer"
    # res_root_folder = "/viscam/projects/vimotion/hope_final_work_opt3d_res_seq50/AIST_set1_hard_seq_w_vposer"
    # res_root_folder = "/viscam/projects/vimotion/hope_final_work_opt3d_res_seq50/AIST_set1_debug_vposer_smooth_w_vposer"
    # res_root_folder = "/viscam/projects/vimotion/hope_final_work_opt3d_res_seq50/AIST_set1_seq360_all_w_vposer"
    # res_root_folder = "/viscam/projects/vimotion/hope_final_work_opt3d_res_seq300/AIST_set11_seq300_almost_there_opt2d_first"
    # res_root_folder = "/viscam/projects/vimotion/hope_final_work_opt3d_res_seq300/AIST_set11_seq300_almost_there_hybrid"
    # res_root_folder = "/viscam/projects/vimotion/cvpr25_opt_3d_w_multiview_diffusion_aist/AIST"
    # res_root_folder = "/viscam/projects/vimotion/cvpr25_opt_3d_w_multiview_diffusion_aist/FineGym"
    # res_root_folder = "/viscam/projects/vimotion/datasets/FineGym/synthetic_gen_3d_data_new_vis_check"
    # res_root_folder = "/viscam/projects/vimotion/cvpr25_opt_3d_w_multiview_diffusion_aist/FineGym_set3_syn3d_model9"
    # res_root_folder = "/viscam/projects/vimotion/cvpr25_opt_3d_w_multiview_diffusion_aist/steezy_set2_model9"

    # res_root_folder = "/viscam/projects/vimotion/cvpr25_opt_3d_w_multiview_diffusion_aist/nicole"

    # res_root_folder = "/viscam/projects/vimotion/hope_final_work_opt3d_res_seq300/cat_opt2d_first"
    # res_root_folder = "/viscam/projects/mofitness/datasets/bb_data/synthetic_gen_3d_data_vis_check_debug"
    # res_root_folder = "/viscam/projects/mofitness/datasets/bb_data/synthetic_gen_3d_data_vis_check_w_vposer_val"
    # res_root_folder = "/viscam/projects/vimotion/final_cvpr25_opt_3d_w_multiview_diffusion/bb_data"
    # res_root_folder = "/viscam/projects/mofitness/final_cvpr25_opt_3d_w_multiview_diffusion/bb_data"

    # res_root_folder = "/viscam/projects/mofitness/final_cvpr25_opt_3d_w_multiview_diffusion/cat_data"

    # res_root_folder = "/viscam/projects/mofitness/final_cvpr25_opt_3d_w_multiview_diffusion/FineGym"
    res_root_folder = "/viscam/projects/mofitness/datasets/cat_data/synthetic_gen_3d_data_vis_check_val"

    dest_rrd_folder = res_root_folder + "_rrd" 
    if not os.path.exists(dest_rrd_folder):
        os.makedirs(dest_rrd_folder)  

    file_names = os.listdir(res_root_folder) 

    file_names.sort() 

    for f_name in file_names:
        # if "0_sample_0" not in f_name:
        #     continue 

        # if "sample_0_objs_wham" not in f_name:
        #     continue 

        if "_objs" not in f_name:
            continue 

        # if int(f_name.split("_")[0]) not in [10, 13, 41, 42, 43, 45]:
        #     continue 

        # For human data 
        # gt_mesh_folder = os.path.join(res_root_folder, f_name)
        # ours_mesh_folder = os.path.join(res_root_folder, f_name)
        # mbert_mesh_folder = os.path.join(res_root_folder, f_name.replace("_objs_wham", "_objs_mbert"))
        # smplify_mesh_folder = os.path.join(res_root_folder, f_name.replace("_objs_wham", "_objs_smplify"))
        # wham_mesh_folder = os.path.join(res_root_folder, f_name.replace("_objs_wham", "_objs_wham"))

        # mesh_dir_list = [ours_mesh_folder, mbert_mesh_folder, smplify_mesh_folder, wham_mesh_folder] 
        # mesh_dir_name_list = ["ours", "mbert", "smplify", "wham"] 

        # # dest_rrd_path = os.path.join(dest_rrd_folder, f_name.replace("_objs", ".rrd")) 
        # dest_rrd_path = os.path.join(dest_rrd_folder, f_name.replace("_objs_wham", ".rrd")) 

        # mesh_dir_list = [ours_mesh_folder, wham_mesh_folder] 
        # mesh_dir_name_list = ["ours", "wham"] 

        # For animal data 
        ours_mesh_folder = os.path.join(res_root_folder, f_name)
        mesh_dir_list = [ours_mesh_folder] 
        mesh_dir_name_list = ["ours"] 

        dest_rrd_path = os.path.join(dest_rrd_folder, f_name.replace("_objs", ".rrd")) 

        if not os.path.exists(dest_rrd_path):
            log_meshes_to_rerun(mesh_dir_list, mesh_dir_name_list, dest_rrd_path) 
    

'''
save to .rrd file. 
rerun --web-viewer test.rrd 
replace localhost with the machine name (move3.stanford.edu) (need to connect to VPN) 
'''