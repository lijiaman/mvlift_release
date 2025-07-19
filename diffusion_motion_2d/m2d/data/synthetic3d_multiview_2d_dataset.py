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

from m2d.data.utils_2d_pose_normalize import move_init_pose2d_to_center, normalize_pose2d, de_normalize_pose2d 
from m2d.data.utils_multi_view_2d_motion import single_seq_get_multi_view_2d_motion_from_3d_torch, cal_min_seq_len_from_mask_torch, get_multi_view_cam_extrinsic

from smplx import SMPL 


class MultiViewSyntheticGenMotion2D(Dataset):
    def __init__(
            self,
            npz_folder,
            train=True,
            sample_n_frames=120,
            num_views=6, 
        ):

        self.npz_folder = npz_folder # For 3D motion sequences 

        processed_data_folder = npz_folder + "_processed_data"
        if not os.path.exists(processed_data_folder):
            os.makedirs(processed_data_folder)

        self.image_width = 1920 
        self.image_height = 1080 

        self.sample_n_frames = sample_n_frames 

        self.num_views = num_views 

        self.train = train 
        if train:
            self.processed_data_path = os.path.join(processed_data_folder, \
                    "train_synthetic_gen_motion3d_seq_new.p")
        else:
            self.processed_data_path = os.path.join(processed_data_folder, \
                    "test_synthetic_gen_motion3d_seq_new.p")
        
        if os.path.exists(self.processed_data_path):
            self.data_dict = joblib.load(self.processed_data_path) 
        else:
            self.data_dict = self.load_npy_files()  
            joblib.dump(self.data_dict, self.processed_data_path)

        print("The number of sequences: {0}".format(len(self.data_dict)))
       
    def load_npy_files(self):
        data_dict = {}

        cnt = 0
        if self.train:
            npy_folder = self.npz_folder
        else:
            npy_folder = self.npz_folder+"_val"

        npy_files = os.listdir(npy_folder) 
        for npy_file in npy_files:
            npy_file_path = os.path.join(npy_folder, npy_file)

            data = np.load(npy_file_path) # T X 18 X 3 

            # if data.shape[0] >= self.sample_n_frames:
            if data.shape[0] >= 90:
                data_dict[cnt] = {} 
                data_dict[cnt]['jnts3d_18'] = data 
                data_dict[cnt]['seq_name'] = npy_file 
                cnt += 1

        return data_dict 

    def gen_multi_view_2d_from_3d_seq(self, smplx_jnts18):
        # T X 18 X 3 

        if "cat" in self.npz_folder:
            neck_idx = 5 
            init_neck_pos = smplx_jnts18[0:1, neck_idx:neck_idx+1, :] # 1 X 1 X 3 

            smplx_jnts18 = smplx_jnts18 - init_neck_pos 
        else:
            # Move the first frame's root joint to zero (x=0, y=0, z=0) 
            l_hip_idx = 8
            r_hip_idx = 11
            root_trans = (smplx_jnts18[0:1, l_hip_idx:l_hip_idx+1, :]+smplx_jnts18[0:1, r_hip_idx:r_hip_idx+1, :])/2. # 1 X 1 X 3 

            smplx_jnts18 = smplx_jnts18 - root_trans  

        if "FineGym" in self.npz_folder:
            # Project 3D pose to different views.  
            jnts3d_aligned_xy_cam_list, jnts3d_cam_list, \
            jnts2d_list, seq_mask_list, cam_extrinsic_list = \
                        single_seq_get_multi_view_2d_motion_from_3d_torch(smplx_jnts18, \
                        return_multiview=True, num_views=self.num_views, farest=True, unified_dist=False) 
        elif "cat" in self.npz_folder:
            # Project 3D pose to different views.  
            jnts3d_aligned_xy_cam_list, jnts3d_cam_list, \
            jnts2d_list, seq_mask_list, cam_extrinsic_list = \
                        single_seq_get_multi_view_2d_motion_from_3d_torch(smplx_jnts18, \
                        return_multiview=True, num_views=self.num_views, farest=False, unified_dist=False, use_animal_data=True)
        elif "bb_data" in self.npz_folder:
            # Project 3D pose to different views.  
            jnts3d_aligned_xy_cam_list, jnts3d_cam_list, \
            jnts2d_list, seq_mask_list, cam_extrinsic_list = \
                        single_seq_get_multi_view_2d_motion_from_3d_torch(smplx_jnts18, \
                        return_multiview=True, num_views=self.num_views, farest=False, unified_dist=False, use_animal_data=False, \
                        use_baby_data=True) 
        else:
            # Project 3D pose to different views.  
            jnts3d_aligned_xy_cam_list, jnts3d_cam_list, \
            jnts2d_list, seq_mask_list, cam_extrinsic_list = \
                        single_seq_get_multi_view_2d_motion_from_3d_torch(smplx_jnts18, \
                        return_multiview=True, num_views=self.num_views, farest=False, unified_dist=True) 

        jnts2d_list = torch.stack(jnts2d_list) # in range width, height 
        seq_mask_list = torch.stack(seq_mask_list)
        cam_extrinsic_list = torch.stack(cam_extrinsic_list)

        seq_len = cal_min_seq_len_from_mask_torch(seq_mask_list)

        return jnts2d_list, seq_len, cam_extrinsic_list, smplx_jnts18   
        # K X T X J X 2, int value, K X 4 X 4, T X 18 X 3   

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        curr_window_data = self.data_dict[idx]

        jnts3d_18 = curr_window_data['jnts3d_18'] # T X 18 X 3
        jnts2d_seq, seq_len, cam_extrinsic, _ = self.gen_multi_view_2d_from_3d_seq(jnts3d_18) 
        # K X T X 18 X 2 
        jnts2d_seq = jnts2d_seq[:, :seq_len] 

        start_t_idx = 0 
        end_t_idx = start_t_idx + self.sample_n_frames 

        motion_data = jnts2d_seq[:, start_t_idx:end_t_idx] # K X sample_n_frames X 18 X 2 
        motion_data[..., 0] /= self.image_width 
        motion_data[..., 1] /= self.image_height 

        # Normalize 2d pose sequence 
        normalized_motion_data = normalize_pose2d(motion_data, input_normalized=True) # K X T X 18 X 2    
        # Normalize to range [-1, 1] 
        # Add padding
        actual_seq_len = motion_data.shape[1] 
        if actual_seq_len < self.sample_n_frames:
            padding_motion_data = torch.zeros(normalized_motion_data.shape[0], self.sample_n_frames-actual_seq_len, \
                            normalized_motion_data.shape[2], normalized_motion_data.shape[3])
            normalized_motion_data = torch.cat((normalized_motion_data, padding_motion_data), dim=1) 

        input_dict = {}
        input_dict['normalized_jnts2d'] = normalized_motion_data # K X T X 18 X 2 
        input_dict['seq_len'] = actual_seq_len 
        input_dict['cam_extrinsic'] = cam_extrinsic # K X 4 X 4 
        input_dict['seq_name'] = curr_window_data['seq_name'] 

        jnts3d_18 = torch.from_numpy(jnts3d_18).float() # T X 18 X 3 
        if jnts3d_18.shape[0] < self.sample_n_frames:
            padding_jnts3d_18 = torch.zeros(self.sample_n_frames-jnts3d_18.shape[0], \
                jnts3d_18.shape[1], jnts3d_18.shape[2])
           
            jnts3d_18 = torch.cat((jnts3d_18, padding_jnts3d_18), dim=0)

        input_dict['jnts3d_18'] = jnts3d_18 # T X 18 X 3 
       
        return input_dict 
