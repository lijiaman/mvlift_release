import os 
import numpy as np 

import trimesh 

from blender_vis_mesh_motion import run_blender_rendering_and_save2video 

import shutil 

if __name__ == "__main__":

    # For final CVPR demo 
    res_root_folder = "/viscam/projects/mofitness/final_cvpr25_opt_3d_w_multiview_diffusion/AIST_for_vis"
    dest_res_folder = "/move/u/jiamanli/final_cvpr25_for_demo/AIST_for_vis_fast_more_res"


    if "omomo" in res_root_folder:
        vis_object = True 
    else:
        vis_object = False 

    dest_blender_res_folder = dest_res_folder + "_blender_img_res" 
    if not os.path.exists(dest_blender_res_folder):
        os.makedirs(dest_blender_res_folder)  

    dest_blender_res_video_folder = dest_res_folder + "_blender_video_res" 
    if not os.path.exists(dest_blender_res_video_folder):
        os.makedirs(dest_blender_res_video_folder)  

    file_names = os.listdir(res_root_folder) 
    file_names.sort() 

    for f_name in file_names:
        seq_folder_path = os.path.join(res_root_folder, f_name)
        
        num_cam = 1
        for cam_idx in range(num_cam):
            out_folder_path = os.path.join(dest_blender_res_folder, f_name+"_cam"+str(cam_idx))
            out_vid_path = os.path.join(dest_blender_res_video_folder, f_name+"_cam"+str(cam_idx)+".mp4")  

            if not os.path.exists(out_vid_path):
                run_blender_rendering_and_save2video(seq_folder_path, out_folder_path, out_vid_path, \
                    scene_blend_path="./for_res_cmp_cam"+str(cam_idx)+"_fast.blend", \
                    vis_object=vis_object) 

                shutil.rmtree(out_folder_path) 

