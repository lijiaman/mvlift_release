import sys 
sys.path.append("../../")

import os
import random
import numpy as np
import pickle as pkl 

import json 
import joblib 

import torch
from torch.utils.data.dataset import Dataset

import pytorch3d.transforms as transforms 

from human_body_prior.body_model.body_model import BodyModel

from m2d.vis.vis_jnts import plot_3d_motion, reverse_perspective_projection, gen_2d_motion_vis 
from m2d.lafan1.utils import rotate_at_frame, normalize, quat_normalize, quat_between, quat_mul, quat_inv, quat_mul_vec 
from m2d.data.utils_multi_view_2d_motion import extract_smplx2openpose_jnts19, single_seq_get_multi_view_2d_motion_from_3d

from m2d.vis.vis_jnts import plot_3d_motion, gen_2d_motion_vis

from m2d.data.utils_2d_pose_normalize import move_init_pose2d_to_center, normalize_pose2d, de_normalize_pose2d 
from m2d.data.utils_multi_view_2d_motion import cal_min_seq_len_from_mask_torch 

from m2d.data.utils_multi_view_2d_motion import cal_min_seq_len_from_mask_torch, get_multi_view_cam_extrinsic  

from m2d.lafan1.utils import quat_mul, quat_mul_vec, quat_inv, quat_normalize 

from m2d.data.youtube_motion2d_dataset import compute_epipoles, simulate_epipolar_lines 

from smplx import SMPL 

import trimesh 

def load_smpl_models_dict():
    # Prepare SMPLX model 
    data_root_folder = "/move/u/jiamanli/datasets/semantic_manip/processed_data"
    soma_work_base_dir = os.path.join(data_root_folder, 'smpl_all_models')
    support_base_dir = soma_work_base_dir 
    surface_model_type = "smpl"
    # surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, "male", 'model.npz')
    # surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "female", 'model.npz')
    surface_model_female_fname = os.path.join(support_base_dir, "basicModel_f_lbs_10_207_0_v1.0.0.pkl")
    surface_model_male_fname = os.path.join(support_base_dir, "basicModel_m_lbs_10_207_0_v1.0.0.pkl")
    dmpl_fname = None
    num_dmpls = None 
    num_expressions = None
    num_betas = 10

    male_bm = BodyModel(bm_fname=surface_model_male_fname,
                    num_betas=num_betas,
                    num_expressions=num_expressions,
                    num_dmpls=num_dmpls,
                    dmpl_fname=dmpl_fname)
    female_bm = BodyModel(bm_fname=surface_model_female_fname,
                    num_betas=num_betas,
                    num_expressions=num_expressions,
                    num_dmpls=num_dmpls,
                    dmpl_fname=dmpl_fname)

    for p in male_bm.parameters():
        p.requires_grad = False
    for p in female_bm.parameters():
        p.requires_grad = False 

    male_bm = male_bm.cuda()
    female_bm = female_bm.cuda()
    
    bm_dict = {'male' : male_bm, 'female' : female_bm}

    return bm_dict 

def convert_coco17_to_openpose18(keypoints_info):
    # keypoints_info = np.concatenate(
    #         (keypoints, scores[..., None]), axis=-1)

    # compute neck joint
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    new_keypoints_info = np.insert(
        keypoints_info, 17, neck, axis=1)
    mmpose_idx = [
        17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
    ]
    openpose_idx = [
        1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
    ]
    new_keypoints_info[:, openpose_idx] = \
        new_keypoints_info[:, mmpose_idx]
   
    return new_keypoints_info

def convert_h36m17_to_openpose18(keypoints):
    # keypoints: T X 17 X 3 

    # coco17: 
# {0,  "Nose"},
#     {1,  "LEye"},
#     {2,  "REye"},
#     {3,  "LEar"},
#     {4,  "REar"},
#     {5,  "LShoulder"},
#     {6,  "RShoulder"},
#     {7,  "LElbow"},
#     {8,  "RElbow"},
#     {9,  "LWrist"},
#     {10, "RWrist"},
#     {11, "LHip"},
#     {12, "RHip"},
#     {13, "LKnee"},
#     {14, "Rknee"},
#     {15, "LAnkle"},
#     {16, "RAnkle"},
   
   
# hm3617:                     0: 'root',
#                             1: 'rhip',
#                             2: 'rkne',
#                             3: 'rank',
#                             4: 'lhip',
#                             5: 'lkne',
#                             6: 'lank',
#                             7: 'belly',
#                             8: 'neck',
#                             9: 'nose',
#                             10: 'head',
#                             11: 'lsho',
#                             12: 'lelb',
#                             13: 'lwri',
#                             14: 'rsho',
#                             15: 'relb',
#                             16: 'rwri'
    hm2coco_idx = [9, -1, -1, -1, -1, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3] 

    new_keypoints = keypoints[:, hm2coco_idx, :] # T X 17 X 3 
    
    jnts18 = convert_coco17_to_openpose18(new_keypoints) # T X 18 X 3 

    return jnts18  

def run_smpl_forward(smpl_poses, smpl_trans, smpl_scaling, smpl_model, betas=None):
   
    smpl_output = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
        scaling=torch.from_numpy(smpl_scaling.reshape(1, 1)).float(),
        betas=betas, 
        )

    keypoints3d = smpl_output.joints # T X 45 X 3 
    verts = smpl_output.vertices 

    return keypoints3d, verts

def convert_smpl_jnts_to_openpose18(smpl_jnts):
    # smpl_jnts: T X 45 X 3 
    # smpl_verts: T X 6890 X 3 

    # Extract selected verex index 
    # 'nose':		    332,
    # 'reye':		    6260,
    # 'leye':		    2800,
    # 'rear':		    4071,
    # 'lear':		    583,

    # selected_vertex_idx = [332, 6260, 2800, 4071, 583] 
    # selected_verts = smpl_verts[:, selected_vertex_idx, :] # T X 5 X 3 

    # smpl2openpose_idx = [22, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, \
    #             23, 24, 25, 26] 

    # smplx_jnts27 = np.concatenate((smpl_jnts[:, :22, :], selected_verts), \
    #                     axis=1) # T X (22+5) X 3 
    # smplx_for_openpose_jnts19 = smplx_jnts27[:, smpl2openpose_idx, :] # T X 19 X 3  

    # # Remove pelvis joint to be aligned with DWPose results T X 18 X 3 
    # smplx_jnts18 = np.concatenate((smplx_for_openpose_jnts19[:, :8], \
    #     smplx_for_openpose_jnts19[:, 9:]), axis=1) # T X 18 X 3 

    # This is the correct version that align with the 2D pose model. 
    smpl2jnts18_idx = [24, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28]

    # new version, but using this version leads to worse results, why? 
    # smpl2jnts18_idx = [24, 12, 16, 18, 20, 17, 19, 21, 1, 4, 7, 2, 5, 8, 26, 25, 28, 27]
    smpl_jnts18 = smpl_jnts[:, smpl2jnts18_idx, :] # T X 18 X 3 

    return smpl_jnts18 

    # smpl_jnts18 = torch.from_numpy(smpl_jnts18).float() 

    # # Insert neck joint index 
    # lshoulder_idx = 4
    # rshoulder_idx = 1
    # neck_jnt = (smpl_jnts17[:, lshoulder_idx:lshoulder_idx+1, :]+\
    #         smpl_jnts17[:, rshoulder_idx:rshoulder_idx+1, :])/2. # T X 1 X 3 

    # smpl_jnts18 = torch.cat((smpl_jnts17[:, 0:1, :], neck_jnt, smpl_jnts17[:, 1:, :]), dim=1)

    # return smpl_jnts18.detach().cpu().numpy()  

def get_SMPL_rest_pose_jonts_offsets(smpl_poses, smpl_trans, smpl_scaling, smpl_model): 
    smpl_poses = np.zeros_like(smpl_poses)
    smpl_trans = np.zeros_like(smpl_trans)

    smpl_jnts, smpl_verts = run_smpl_forward(smpl_poses[0:1], smpl_trans[0:1], \
        smpl_scaling[0:1], smpl_model)

    parents = [0,          0,          0,          0,          1,
                2,          3,          4,          5,          6,
                7,          8,          9,          9,          9,
               12,         13,         14,         16,         17,
               18,         19, 
               15,         22,         22,          23,        24]
    
    # Add additional 5 joints nose, right eye, left eye,  right ear, left ear, 

    parents_for_fk = [-1,          0,          0,          0,          1,
                2,          3,          4,          5,          6,
                7,          8,          9,          9,          9,
               12,         13,         14,         16,         17,
               18,         19, 
               15,         22,         22,          23,        24]

    
    selected_vertex_list = [332, 6260, 2800, 4071, 583] 

    jnts27 = torch.cat((smpl_jnts[:, :22, :], smpl_verts[:, selected_vertex_list, :]), dim=1) # 1 X 27 X 3 

    # Compute rest joint offsets 
    parent_jnts27 = jnts27[:, parents, :].clone() # 1 X 27 X 3 
    rest_smpl_offsets = jnts27 - parent_jnts27 # 1 X 27 X 3 

    return rest_smpl_offsets

class AISTPose3D(Dataset):
    def __init__(self, sample_n_frames=120, center_all_jnts2d=False, use_decomposed_traj_rep=False,):

        self.center_all_jnts2d = center_all_jnts2d 

        self.use_decomposed_traj_rep = use_decomposed_traj_rep 

        self.pkl_folder = "/viscam/projects/vimotion/datasets/AIST++/aist_plusplus_final/motions"
        processed_data_folder = "/viscam/projects/vimotion/datasets/AIST/processed_data"
        train_val_json_path = os.path.join(processed_data_folder, "train_val_split.json")

        self.wham_res_folder = "/viscam/projects/vimotion/datasets/AIST/clips_wham_res_complete"

        self.vit_pose2d_folder = "/viscam/projects/vimotion/datasets/AIST/clip_vit_pose2d_res"

        self.seq_names = json.load(open(train_val_json_path, 'r'))['test']

        smpl_dir = "/move/u/jiamanli/datasets/semantic_manip/processed_data/smpl_all_models/smpl"
        self.smpl = SMPL(model_path=smpl_dir, gender='MALE', batch_size=1)

        self.processed_data_path = os.path.join(processed_data_folder, "test_aist_3d_motion_for_init.p")
        if os.path.exists(self.processed_data_path):
            self.data_dict = joblib.load(self.processed_data_path) 
        else:
            self.data_dict = self.load_pkl_files()  
            joblib.dump(self.data_dict, self.processed_data_path)

        self.sample_n_frames = sample_n_frames 
        self.processed_window_data_path = os.path.join(processed_data_folder, \
            "test_aist_3d_motion_for_init_window_"+str(self.sample_n_frames)+".p")
        if os.path.exists(self.processed_window_data_path):
            self.new_window_dict = joblib.load(self.processed_window_data_path)
        else:
            self.window_dict = self.compute_window_data_for_eval() 
            self.new_window_dict = self.proccess_window_data_to_align() 
            joblib.dump(self.new_window_dict, self.processed_window_data_path)

        self.clean_window_dict = self.filter_bad_gt_seq() 

        print("The number of testing sequences for init 3D pose: {0}".format(len(self.data_dict)))
        print("The number of testing windows for AIST evaluation: {0}".format(len(self.new_window_dict)))
        print("The number of clean testing windows for AIST evaluation: {0}".format(len(self.clean_window_dict))) # 308 

        cam_extrinsic_list = get_multi_view_cam_extrinsic(num_views=30, farest=True)  # K X 4 X 4  
        ref_cam_extrinsic = cam_extrinsic_list[0:1] # 1 X 4 X 4 
        target_cam_extrinsic = cam_extrinsic_list[1:] # (K-1) X 4 X 4 
        ref_cam_extrinsic = ref_cam_extrinsic.repeat(target_cam_extrinsic.shape[0], 1, 1) # (K-1) X 4 X 4 
        self.epipoles = compute_epipoles(ref_cam_extrinsic, target_cam_extrinsic).cpu() # (K-1) X 2  
    
    def filter_bad_gt_seq(self):
        clean_cnt = 0
        bad_seq_list = [34, 38, 45, 62, 64, 66, 73, 78, 79, 83, 85, 112, \
                    113, 121, 125, 126, 129, 133, 148, 154, 163, 173, \
                    174, 176, 182, 183, 187,  192, 193, 208, 217, 231, \
                    242, 246, 256, 257, 259, 268, 269, 280, 283, 315, \
                    322, 327, 331, 334, 336, 343]
        
        clean_window_dict = {} 
        for k in self.new_window_dict:
            if k in bad_seq_list:
                continue 

            if self.new_window_dict[k]['smpl_jnts18'].shape[0] < self.sample_n_frames:
                continue

            motion_data = torch.from_numpy(self.new_window_dict[k]['pose2d'].copy()).float() 

            # Centralize 2d pose sequence based on the firs 2D pose 
            motion_data, vis_2d_mask, scale_factors_for_input2d = move_init_pose2d_to_center(motion_data[None], \
                                for_eval=True, \
                                center_all_jnts2d=self.center_all_jnts2d) # 1 X T X 18 X 2  
            motion_data = motion_data.squeeze(0)

            actual_seq_len = cal_min_seq_len_from_mask_torch(vis_2d_mask) 

            if actual_seq_len < self.sample_n_frames:
                continue

            clean_window_dict[clean_cnt] = self.new_window_dict[k] 
            clean_cnt += 1 
        
        return clean_window_dict

    def load_pkl_files(self):
        data_dict = {}

        cnt = 0
        debug_cnt = 0 
        for s_name in self.seq_names:
            camera_tag = s_name.split("_")[2]

            # if "gMH_sBM_c08_d24_mMH3_ch04" not in s_name:
            #     continue 

            # Load WHAM results 
            wham_seq_folder = os.path.join(self.wham_res_folder, s_name.replace(".npz", ""))
            curr_wham_seq_path = os.path.join(wham_seq_folder, "wham_output.pkl")

            # Load corrsponding ViTpose2d result 
            curr_vit_seq_path = os.path.join(self.vit_pose2d_folder, s_name)

            pkl_s_name = s_name.replace(camera_tag, "cAll").replace(".npz", ".pkl")
            data_pkl_path = os.path.join(self.pkl_folder, pkl_s_name)

            if os.path.exists(data_pkl_path):
                pkl_data = pkl.load(open(data_pkl_path, 'rb'))

                data_dict[cnt] = {}
                data_dict[cnt]['seq_name'] = s_name 

                # data_dict[cnt]['smpl_jnts17'] = pkl_data['keypoints3d_optim'] # T X 17 X 3 
                smpl_trans = pkl_data['smpl_trans'][::2]
                smpl_poses = pkl_data['smpl_poses'][::2].reshape(-1, 24, 3)
                smpl_scaling = pkl_data['smpl_scaling']

                smpl_trans = smpl_trans / smpl_scaling 
                smpl_scaling[:] = 1. 

                jnts3d_smpl, verts_smpl = run_smpl_forward(smpl_poses, smpl_trans, smpl_scaling, self.smpl) 
                # T X 45 X 3 

                rest_offsets = \
                    get_SMPL_rest_pose_jonts_offsets(smpl_poses, smpl_trans, smpl_scaling, self.smpl)
                
                # Convert joints to 18 joints 
                jnts18 = convert_smpl_jnts_to_openpose18(jnts3d_smpl.detach().cpu().numpy())

                vit_pose2d_data = np.load(curr_vit_seq_path)
               
                if os.path.exists(curr_wham_seq_path):
                    wham_data = joblib.load(curr_wham_seq_path)
                    wham_poses_world = wham_data[0]['pose_world'][::2] # T X 72 
                    wham_trans_world = wham_data[0]['trans_world'][::2] # T X 3 
                    wham_betas = wham_data[0]['betas'][::2] # T X 10 
                    wham_betas = torch.from_numpy(wham_betas).float() 

                    # Load wham local pose resuslts to compare 2D reprojection error with our model 
                    local_wham_poses = wham_data[0]['pose'][::2] # T X 72 
                    local_wham_trans = wham_data[0]['trans'][::2] # T X 3 

                    wham_jnts3d_smpl, wham_verts_smpl = run_smpl_forward(wham_poses_world, \
                        wham_trans_world, smpl_scaling, self.smpl, betas=wham_betas) 
                    # T X 45 X 3 

                    local_wham_jnts3d_smpl, _ = run_smpl_forward(local_wham_poses, \
                        local_wham_trans, smpl_scaling, self.smpl, betas=wham_betas) 
                    # T X 45 X 3 

                    # Convert joints to 18 joints 
                    wham_jnts18 = convert_smpl_jnts_to_openpose18(wham_jnts3d_smpl.detach().cpu().numpy())

                    local_wham_jnts18 = convert_smpl_jnts_to_openpose18(local_wham_jnts3d_smpl.detach().cpu().numpy())

                    if wham_jnts18.shape[0] != jnts18.shape[0]:
                        debug_cnt += 1 
                        print("WHAM seq len:{0}".format(wham_jnts18.shape[0]))
                        print("SMPL GT seq len:{0}".format(jnts18.shape[0]))
                        print("Seq 2D len:{0}".format(vit_pose2d_data['body_pose'][::2].shape[0]))
                        print("seq name:{0}".format(s_name))

                        # Usually because the human move out of the view of the camera 
                        wham_seq_len = wham_jnts18.shape[0] 
                        jnts18 = jnts18[:wham_seq_len]
                        smpl_poses = smpl_poses[:wham_seq_len]
                        smpl_trans = smpl_trans[:wham_seq_len]
                        jnts3d_smpl = jnts3d_smpl[:wham_seq_len]

                    # Save GT 
                    data_dict[cnt]['smpl_jnts18'] = jnts18 
                    data_dict[cnt]['rest_offsets'] = rest_offsets # 1 X 27 X 3 

                    data_dict[cnt]['smpl_poses'] = smpl_poses 
                    data_dict[cnt]['smpl_trans'] = smpl_trans 
                    data_dict[cnt]['smpl_jnts45'] = jnts3d_smpl.detach().cpu().numpy()

                    # Save pose2d results 
                    data_dict[cnt]['pose2d'] = vit_pose2d_data['body_pose'][::2] # T X 18 X 2 

                    # Save WHAM results 
                    data_dict[cnt]['wham_poses']= wham_poses_world 
                    data_dict[cnt]['wham_trans'] = wham_trans_world 
                    data_dict[cnt]['wham_jnts45'] = wham_jnts3d_smpl.detach().cpu().numpy()
                    data_dict[cnt]['wham_jnts18'] = wham_jnts18 

                    data_dict[cnt]['local_wham_jnts18'] = local_wham_jnts18

                    data_dict[cnt]['wham_betas'] = wham_betas.detach().cpu().numpy()  

                    assert wham_jnts18.shape[0] == jnts18.shape[0]
                    assert wham_jnts18.shape[0] == vit_pose2d_data['body_pose'][::2].shape[0]

                    cnt += 1 

        print("Unequal seq len cnt:{0}".format(debug_cnt))
        print("Total seq cnt:{0}".format(len(data_dict)))
     
        return data_dict 

    def compute_window_data_for_eval(self):
        window_dict = {}
        window_len = self.sample_n_frames 

        window_cnt = 0
        for seq_cnt in self.data_dict:
            curr_seq_data = self.data_dict[seq_cnt]

            curr_seq_len = curr_seq_data['smpl_jnts18'].shape[0]

            for t_idx in range(0, curr_seq_len, window_len):
                window_dict[window_cnt] = {}

                for k in curr_seq_data:
                    if "rest_offsets" in k or "seq_name" in k:
                        window_dict[window_cnt][k] = curr_seq_data[k]
                    else:
                        window_dict[window_cnt][k] = curr_seq_data[k][t_idx:t_idx+window_len]
                
                window_dict[window_cnt]['s_idx'] = t_idx 
                window_dict[window_cnt]['e_idx'] = t_idx + window_len 

                window_cnt += 1 

                break # For evaluation, we only need the first window. 

                # if window_len >= 120 or t_idx > 120:
                #     break 

        return window_dict 

    def proccess_window_data_to_align(self):
        new_window_dict = {}
        cnt = 0
        for k in self.window_dict:
            curr_window_data = self.window_dict[k]

            # Apply rotation back to put human on floor z = 0 
            smpl_poses = curr_window_data['smpl_poses']
            smpl_trans = curr_window_data['smpl_trans'] 

            if smpl_poses.shape[0] < self.sample_n_frames:
                continue 

            wham_poses = curr_window_data['wham_poses']
            wham_trans = curr_window_data['wham_trans'] 

            wham_betas = curr_window_data['wham_betas'] 

            smpl_poses, smpl_trans, smpl_jnts18, wham_poses, wham_trans, wham_jnts18 = \
                self.align_wham_seq_to_gt_init(smpl_poses, smpl_trans, \
                wham_poses, wham_trans, wham_betas)

            new_window_dict[cnt] = {}
            new_window_dict[cnt]['smpl_poses'] = smpl_poses
            new_window_dict[cnt]['smpl_trans'] = smpl_trans
            new_window_dict[cnt]['smpl_jnts18'] = smpl_jnts18

            new_window_dict[cnt]['wham_poses'] = wham_poses 
            new_window_dict[cnt]['wham_trans'] = wham_trans 
            new_window_dict[cnt]['wham_betas'] = wham_betas 
            new_window_dict[cnt]['wham_jnts18'] =  wham_jnts18 

            new_window_dict[cnt]['local_wham_jnts18'] = curr_window_data['local_wham_jnts18'] 

            new_window_dict[cnt]['pose2d'] = curr_window_data['pose2d'] 

            new_window_dict[cnt]['seq_name'] = curr_window_data['seq_name']
            new_window_dict[cnt]['s_idx'] = curr_window_data['s_idx']
            new_window_dict[cnt]['e_idx'] = curr_window_data['e_idx'] 

            cnt += 1  

        return new_window_dict 

    def __len__(self):
        return len(self.clean_window_dict)

    def align_wham_seq_to_gt_init(self, smpl_poses, smpl_trans, \
        wham_poses, wham_trans, wham_betas):

        smpl_poses = torch.from_numpy(smpl_poses).float() # T X 24 X 3 
        smpl_trans = torch.from_numpy(smpl_trans).float() # T X 3 
       
        wham_poses = torch.from_numpy(wham_poses).float().reshape(-1, 24, 3) # T X 24 X 3 
        wham_trans = torch.from_numpy(wham_trans).float() # T X 3 
      
        # First, apply rotation back to put human on floor z = 0 
        align_rot_mat = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        seq_len, _, _ = smpl_poses.shape 
        align_rot_mat = torch.from_numpy(align_rot_mat).float()[None].repeat(seq_len,\
            1, 1) # T X 3 X 3 

        smpl_trans = torch.matmul(align_rot_mat, \
            smpl_trans[:, :, None])
        smpl_trans = smpl_trans.squeeze(-1) # T X 3 

        wham_trans = torch.matmul(align_rot_mat, \
            wham_trans[:, :, None])
        wham_trans = wham_trans.squeeze(-1) # T X 3 

        wham_global_rot_aa_rep = wham_poses[:, 0, :] # T X 3 
        wham_global_rot_mat = transforms.axis_angle_to_matrix(wham_global_rot_aa_rep) # T X 3 X 3 
        wham_global_rot_mat = torch.matmul(align_rot_mat, wham_global_rot_mat) # T X 3 X 3 

        # Then, apply rotation to make the facing diretion of the first frame consistent 
        wham_global_rot_quat = transforms.matrix_to_quaternion(wham_global_rot_mat) # T X 4 
        key_glob_Q = wham_global_rot_quat[0:1][:, None, None, :].detach().cpu().numpy() # 1 X 1 X 1 X 4 

        wham_forward = np.array([1, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :] * quat_mul_vec(
            key_glob_Q, np.array([1, 0, 0])[np.newaxis, np.newaxis, np.newaxis, :]
        ) # In rest pose, x direction is the body left direction, root joint point to left hip joint.  

        wham_forward = normalize(wham_forward)

        # Get GT facing direction 
        gt_smpl_global_rot_aa_rep = smpl_poses[:, 0, :] 
        gt_smpl_global_rot_mat = transforms.axis_angle_to_matrix(gt_smpl_global_rot_aa_rep) # T X 3 X 3 
        gt_smpl_global_rot_mat = torch.matmul(align_rot_mat, gt_smpl_global_rot_mat) # T X 3 X 3 

        gt_smpl_global_rot_quat = transforms.matrix_to_quaternion(gt_smpl_global_rot_mat) # T X 4 
        gt_key_glob_Q = gt_smpl_global_rot_quat[0:1][:, None, None, :].detach().cpu().numpy() # 1 X 1 X 1 X 4 

        # Replace the global rotation with new global orientation in WHAM 
        new_gt_rot_mat = transforms.quaternion_to_matrix(gt_smpl_global_rot_quat) 
        new_gt_rot_aa_rep = transforms.matrix_to_axis_angle(new_gt_rot_mat)

        smpl_poses[:, 0, :] = new_gt_rot_aa_rep 

        gt_forward = np.array([1, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :] * quat_mul_vec(
            gt_key_glob_Q, np.array([1, 0, 0])[np.newaxis, np.newaxis, np.newaxis, :]
        ) # In rest pose, x direction is the body left direction, root joint point to left hip joint.  

        gt_forward = normalize(gt_forward)

        yrot = quat_normalize(quat_between(gt_forward, wham_forward))[0, 0] # 1 X 4 

        wham_global_rot_quat = quat_mul(quat_inv(yrot), wham_global_rot_quat) # T X 4 
        wham_trans = quat_mul_vec(quat_inv(yrot), wham_trans) # T X 3
        # wham_jnts18 = quat_mul_vec(quat_inv(yrot), wham_jnts18) # T X J X 3 

        # Replace the global rotation with new global orientation in WHAM 
        wham_global_rot_quat = torch.from_numpy(wham_global_rot_quat).float() 
        new_wham_rot_mat = transforms.quaternion_to_matrix(wham_global_rot_quat) 
        new_wham_rot_aa_rep = transforms.matrix_to_axis_angle(new_wham_rot_mat)

        # Update wham result 
        wham_poses[:, 0, :] = new_wham_rot_aa_rep 

        # Run forward to get SMPL joints again. 
        smpl_scaling = np.asarray([1.])
        wham_betas = torch.from_numpy(wham_betas).float()
        wham_jnts3d_smpl, wham_verts_smpl = run_smpl_forward(wham_poses.detach().cpu().numpy(), \
                        wham_trans.detach().cpu().numpy(), smpl_scaling, self.smpl, betas=wham_betas) # T X 45 X 3 
        
        # Run forward to get SMPL joints again. 
        gt_jnts3d_smpl, gt_verts_smpl = run_smpl_forward(smpl_poses.detach().cpu().numpy(), \
                        smpl_trans.detach().cpu().numpy(), smpl_scaling, self.smpl) # T X 45 X 3 

        wham_floor_z = wham_verts_smpl[0, :, 2].min()
        gt_floor_z = gt_verts_smpl[0, :, 2].min()  

        wham_init_trans = wham_trans[0:1, :].clone() # 1 X 3 
        gt_init_trans = smpl_trans[0:1, :].clone() # 1 X 3 

        wham_init_trans[:, 2] = wham_floor_z 
        gt_init_trans[:, 2] = gt_floor_z 
        
        smpl_trans -= gt_init_trans 
        wham_trans -= wham_init_trans 

        wham_jnts3d_smpl, wham_verts_smpl = run_smpl_forward(wham_poses.detach().cpu().numpy(), \
                        wham_trans.detach().cpu().numpy(), smpl_scaling, self.smpl, betas=wham_betas) # T X 45 X 3 
        gt_jnts3d_smpl, gt_verts_smpl = run_smpl_forward(smpl_poses.detach().cpu().numpy(), \
                        smpl_trans.detach().cpu().numpy(), smpl_scaling, self.smpl) # T X 45 X 3 
             
        wham_jnts18 = convert_smpl_jnts_to_openpose18(wham_jnts3d_smpl.detach().cpu().numpy())
        gt_jnts18 = convert_smpl_jnts_to_openpose18(gt_jnts3d_smpl.detach().cpu().numpy())
        
        return smpl_poses, smpl_trans, gt_jnts18, wham_poses, wham_trans, wham_jnts18  

    def __getitem__(self, idx):
        curr_window_data = self.clean_window_dict[idx]

        # Apply rotation back to put human on floor z = 0 
        smpl_jnts18  = curr_window_data['smpl_jnts18'] # T X 18 X 3 
        smpl_poses = curr_window_data['smpl_poses']
        smpl_trans = curr_window_data['smpl_trans'] 

        wham_jnts18 = curr_window_data['wham_jnts18']
        wham_poses = curr_window_data['wham_poses']
        wham_trans = curr_window_data['wham_trans'] 

        local_wham_jnts18 = curr_window_data['local_wham_jnts18'] 

        # Process 2D motion data 
        motion_data = torch.from_numpy(curr_window_data['pose2d']).float() 

        # Centralize 2d pose sequence based on the firs 2D pose 
        motion_data, vis_2d_mask, scale_factors_for_input2d = move_init_pose2d_to_center(motion_data[None], for_eval=True, \
                            center_all_jnts2d=self.center_all_jnts2d) # 1 X T X 18 X 2  
        motion_data = motion_data.squeeze(0)

        actual_seq_len = cal_min_seq_len_from_mask_torch(vis_2d_mask)     
        motion_data = motion_data[:actual_seq_len]

        # Normalize 2d pose sequence 
        normalized_motion_data = normalize_pose2d(motion_data, input_normalized=True) 

        # Add padding
        if actual_seq_len < self.sample_n_frames:
            padding_motion_data = torch.zeros(self.sample_n_frames-actual_seq_len, \
                            normalized_motion_data.shape[1], normalized_motion_data.shape[2])
            normalized_motion_data = torch.cat((normalized_motion_data, padding_motion_data), dim=0) 
  
        num_epipoles = 5 
        total_epipoles = self.epipoles.shape[0] 
        random_lines_for_joints2d = []
        for e_idx in range(num_epipoles):
            curr_line, curr_epipoles = simulate_epipolar_lines(normalized_motion_data, \
                epipoles=self.epipoles[e_idx])

            random_lines_for_joints2d.append(curr_line)

        random_lines_for_joints2d = torch.stack(random_lines_for_joints2d) # K X T X 3, not used in evaluation 
       
        input_dict = {} 

        input_dict['seq_name'] = curr_window_data['seq_name'] 

        # padding WHAM and GT. 
        if wham_jnts18.shape[0] < self.sample_n_frames:
            padding_wham_jnts18 = np.zeros((self.sample_n_frames-wham_jnts18.shape[0], \
                wham_jnts18.shape[1], wham_jnts18.shape[2]))
            wham_jnts18 = np.concatenate((wham_jnts18, padding_wham_jnts18), axis=0)

            padding_local_wham_jnts18 = np.zeros((self.sample_n_frames-local_wham_jnts18.shape[0], \
                local_wham_jnts18.shape[1], local_wham_jnts18.shape[2]))
            local_wham_jnts18 = np.concatenate((local_wham_jnts18, padding_local_wham_jnts18), axis=0)

            padding_wham_poses = np.zeros((self.sample_n_frames-wham_poses.shape[0], \
                wham_poses.shape[1], wham_poses.shape[2]))
            wham_poses = np.concatenate((wham_poses, padding_wham_poses), axis=0)

            padding_wham_trans = np.zeros((self.sample_n_frames-wham_trans.shape[0], \
                wham_trans.shape[1]))
            wham_trans = np.concatenate((wham_trans, padding_wham_trans), axis=0)

        if smpl_jnts18.shape[0] < self.sample_n_frames:
            padding_smpl_jnts18 = np.zeros((self.sample_n_frames-smpl_jnts18.shape[0], \
                smpl_jnts18.shape[1], smpl_jnts18.shape[2]))
            smpl_jnts18 = np.concatenate((smpl_jnts18, padding_smpl_jnts18), axis=0)

            padding_smpl_poses = np.zeros((self.sample_n_frames-smpl_poses.shape[0], \
                smpl_poses.shape[1], smpl_poses.shape[2]))
            smpl_poses = np.concatenate((smpl_poses, padding_smpl_poses), axis=0)

            padding_smpl_trans = np.zeros((self.sample_n_frames-smpl_trans.shape[0], \
                smpl_trans.shape[1]))
            smpl_trans = np.concatenate((smpl_trans, padding_smpl_trans), axis=0) 

        input_dict['wham_jnts18'] = wham_jnts18[:self.sample_n_frames]
        input_dict['wham_poses'] = wham_poses[:self.sample_n_frames] 
        input_dict['wham_trans'] = wham_trans[:self.sample_n_frames] 

        input_dict['local_wham_jnts18'] = local_wham_jnts18[:self.sample_n_frames] 

        input_dict['wham_betas'] = curr_window_data['wham_betas'][:self.sample_n_frames]

        input_dict['smplx_jnts18'] = smpl_jnts18[:self.sample_n_frames] # W X 18 X 3 
        input_dict['smpl_poses'] = smpl_poses[:self.sample_n_frames] 
        input_dict['smpl_trans'] = smpl_trans[:self.sample_n_frames] 

        input_dict['seq_len'] = actual_seq_len 

        input_dict['normalized_jnts2d'] = normalized_motion_data[:self.sample_n_frames]  

        input_dict['lines_pass_jnts2d'] = random_lines_for_joints2d[:self.sample_n_frames]

        input_dict['scale_factors_for_input2d'] = scale_factors_for_input2d 

        if self.use_decomposed_traj_rep:
            # T X 18 X 2, decompose to T X 18 X 2, T X 1 X 2 
            l_hip_idx = 8 
            r_hip_idx = 11 
            root_trans2d = (normalized_motion_data[:, l_hip_idx:l_hip_idx+1, :]+\
                        normalized_motion_data[:, r_hip_idx:r_hip_idx+1, :])/2. 

            normalized_local_motion2d = normalized_motion_data - root_trans2d  

            random_lines_for_local_joints2d, _ = simulate_epipolar_lines(normalized_local_motion2d, \
                                            epipoles=curr_epipoles)
            
            input_dict['normalized_traj_2d'] = root_trans2d  
            input_dict['normalized_local_jnts2d'] = normalized_local_motion2d 
            input_dict['lines_pass_local_jnts2d'] = random_lines_for_local_joints2d 

        return input_dict 

if __name__ == "__main__":
    dataset = AISTPose3D()

    save_folder = "./tmp_debug_dataloader_aist_3d_init"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0,)
    for idx, batch in enumerate(dataloader): 
        print(batch["smplx_jnts18"].shape)
        jnts3d_18 = batch["smplx_jnts18"][0]

        # jnts3d_18 = jnts3d_18[0].repeat(60, 1, 1)

        dest_vid_3d_path_global = os.path.join(save_folder, \
                        str(idx)+"_jpos3d")
        plot_3d_motion(dest_vid_3d_path_global, \
            jnts3d_18.detach().cpu().numpy()) 

        if idx > 5:
            break 