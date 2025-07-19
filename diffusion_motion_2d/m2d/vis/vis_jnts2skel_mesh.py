import os 
import numpy as np 
import trimesh 

from blender_vis_mesh_motion import run_blender_rendering_and_save2video 

if __name__ == "__main__":
    # root_folder = "/viscam/projects/vimotion/opt_res3d"
    
    # root_data_folder = os.path.join(root_folder, "amass_objs")

    # dest_image_folder = os.path.join(root_folder, "amass_res_imgs")
    # dest_vid_folder = os.path.join(root_folder, "amass_res_videos")
    
    # root_data_folder = os.path.join(root_folder, "AIST_objs")

    # dest_image_folder = os.path.join(root_folder, "AIST_res_imgs")
    # dest_vid_folder = os.path.join(root_folder, "AIST_res_videos")

    # root_data_folder = "/viscam/projects/vimotion/opt_skel_recon_loss/AIST_new_wo_noise_tmin200_random"
    # ori_root_data_folder = "/viscam/projects/vimotion/opt3d_w_sds_epi_line_conditioned_motion2d_res_w_smpl"

    # ori_root_data_folder = "/viscam/projects/vimotion/opt3d_w_sds_epipoles_epi_line_conditioned_motion2d_res_w_smpl"
    
    # ori_root_data_folder = "/viscam/projects/vimotion/opt3d_w_sds_work"
    # ori_root_data_folder = "/viscam/projects/vimotion/work_opt3d_w_multiview_gen2d"
    # ori_root_data_folder = "/viscam/projects/vimotion/work_opt3d_w_sds"

    # ori_root_data_folder = "/viscam/projects/vimotion/hope_work_opt3d_res"

    # root_data_folder = "/viscam/projects/mofitness/datasets/cat_data/synthetic_gen_3d_data_vis_check"

   

    # tmp_folder_names = os.listdir(ori_root_data_folder)
    # for root_data_folder_name in tmp_folder_names:
    #     if "blender" in root_data_folder_name:
    #         continue 

    #     # if "AIST_latest_wo_init3d_for_eval_complete" not in root_data_folder_name:
    #     #     continue 

    #     # if "_complete" in root_data_folder_name:
    #     #     continue 

    #     # if "amass" not in root_data_folder_name:
    #     #     continue 

    #     # if "FineGym" not in root_data_folder_name:
    #     #     continue 

    #     # if "AIST_all_view_w_vposer" not in root_data_folder_name:
    #     #     continue 

    #     # if "FineGym_w_vposer" not in root_data_folder_name:
    #     #     continue 
            
    #     root_data_folder = os.path.join(ori_root_data_folder, root_data_folder_name)

    # root_data_folder = "/viscam/projects/mofitness/final_cvpr25_opt_3d_w_multiview_diffusion/AIST_for_vis"
    # root_data_folder = "/viscam/projects/mofitness/final_cvpr25_opt_3d_w_multiview_diffusion/nicole_for_vis"
    root_data_folder = "/viscam/projects/mofitness/final_cvpr25_opt_3d_w_multiview_diffusion/steezy_for_vis"

    # root_data_folder = "/viscam/projects/mofitness/final_cvpr25_opt_3d_w_multiview_diffusion/cat_data_for_vis"

    dest_data_folder = root_data_folder 
    dest_image_folder = os.path.join(dest_data_folder, "for_eval_blender_imgs")
    dest_vid_folder = os.path.join(dest_data_folder, "for_eval_blender_videos")

    if not os.path.exists(dest_vid_folder):
        os.makedirs(dest_vid_folder)

    seq_folders = os.listdir(root_data_folder)
    seq_folders.sort() 
    for seq_name in seq_folders:
        if "_objs" not in seq_name:
            continue 

        if "elepose" not in seq_name:
            continue 

        if int(seq_name.split("_")[0]) not in [13, 14, 19, 33, 34, 44, 45, 46, 47, 49, 50, 54]:
            continue
            
        seq_folder_path = os.path.join(root_data_folder, seq_name)

        out_folder_path = os.path.join(dest_image_folder, seq_name)
        if not os.path.exists(out_folder_path):
            os.makedirs(out_folder_path)
        out_vid_path = os.path.join(dest_vid_folder, \
            seq_name+".mp4")
        if not os.path.exists(out_vid_path):
            run_blender_rendering_and_save2video(seq_folder_path, out_folder_path, out_vid_path) 
