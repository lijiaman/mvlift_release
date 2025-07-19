import sys 
sys.path.append("../../")

import os
import random
import numpy as np

import json 
import joblib 

import torch
from torch.utils.data.dataset import Dataset

from m2d.data.utils_2d_pose_normalize import move_init_pose2d_to_center, move_init_pose2d_to_center_for_animal_data, normalize_pose2d, de_normalize_pose2d 
from m2d.data.utils_multi_view_2d_motion import cal_min_seq_len_from_mask_torch, get_multi_view_cam_extrinsic  

from smplx import SMPL 

def simulate_epipolar_lines(pose_sequence, vp_bound=3, epipoles=None):
    """
    Generate epipolar lines for a 2D pose sequence where all joints in a frame share the same point.

    Args:
        pose_sequence (torch.Tensor): A tensor of shape (T, J, 2) containing the 2D joint positions.
        vp_bound (float): The bound within which to sample vanishing points, assuming a square plot range.
    """
    T, J, _ = pose_sequence.shape
    
    # # Sample one vanishing point per frame
    # vanishing_points = (torch.rand(T, 2) - 0.5) * 2 * vp_bound

    # Sample one vanishing point per sequence 
    if epipoles is not None:
        vp_x = epipoles[0]
        vp_y = epipoles[1] 
    else:
        vanishing_points = (torch.rand(1, 2) - 0.5) * 2 * vp_bound # In range [-vp_bound, vp_bound]
        vp_x, vp_y = vanishing_points[0]

    lines = torch.zeros((T, J, 3))
    for t in range(T):
        for j in range(J):
            x1, y1 = pose_sequence[t, j]
            A = y1 - vp_y
            B = vp_x - x1
            C = x1 * vp_y - vp_x * y1
            lines[t, j] = torch.tensor([A, B, C])

    a = lines[:, :, 0]
    b = lines[:, :, 1]
    c = lines[:, :, 2] 

    # Normalize coefficients such that a^2 + b^2 = 1
    norm = torch.sqrt(a**2 + b**2) + 1e-8 
    a = a / norm
    b = b / norm
    c = c / norm
    
    # Ensure a is positive
    abc = torch.where(a.unsqueeze(-1) < 0, -torch.stack((a, b, c), dim=-1), torch.stack((a, b, c), dim=-1))
    
    return abc, [vp_x, vp_y]   # Returns T x J x 3
   
def compute_epipoles(reference_cam_extrinsic, target_cam_extrinsic): 
    target_num_views = target_cam_extrinsic.shape[0]
    
    # reference_cam_extrinsic # (K-1) X 4 X 4 
    # target_cam_extrinsic  # (K-1) X 4 X 4 

    focal_length = 1000 
    image_width = 1920
    image_height = 1080
    camera_center = torch.tensor([image_width / 2, image_height / 2], device=target_cam_extrinsic.device)

    K = torch.zeros([3, 3], device=target_cam_extrinsic.device)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[2, 2] = 1.
    K[:-1, -1] = camera_center

    epipoles = []
    for view_idx in range(target_num_views):
        T_mat = torch.matmul(target_cam_extrinsic[view_idx], \
            torch.inverse(reference_cam_extrinsic[view_idx])) # 4 X 4 

        # Computing epipoles 
        camera_center_hom = torch.tensor([0, 0, 0, 1], dtype=torch.float32, \
            device=target_cam_extrinsic.device)
        epipole_hom = torch.matmul(T_mat, camera_center_hom)
        epipole_img = torch.matmul(K, epipole_hom[:3])  # ignoring last element as it's 1
        epipole_img = epipole_img[:2] / epipole_img[2]  # normalize to image coordinates
        epipoles.append(epipole_img)

    epipoles = torch.stack(epipoles) # (K-1) X 2 
    
    # Convert pixel coordinates to normalized range [-1, 1]
    epipoles_normalized = torch.zeros_like(epipoles)
    epipoles_normalized[:, 0] = (epipoles[:, 0] - image_width / 2) / (image_width / 2)
    epipoles_normalized[:, 1] = (epipoles[:, 1] - image_height / 2) / (image_height / 2)

    return epipoles_normalized  # Returns (K-1) X 2 

class YoutubeMotion2D(Dataset):
    def __init__(
            self,
            json_path, npz_folder,
            sample_n_frames=120,
            train=True,
            for_eval=False, 
            center_all_jnts2d=False, 
            use_animal_data=False, 
        ):

        self.json_path = json_path # includes training video names and testing video names

        self.npz_folder = npz_folder
        
        self.sample_n_frames = sample_n_frames

        self.for_eval = for_eval 

        self.center_all_jnts2d = center_all_jnts2d 

        self.use_animal_data = use_animal_data 

        self.train = train 

        json_folder = "/".join(json_path.split("/")[:-1])
        
        if self.train:
            if for_eval:
                self.processed_json_path = os.path.join(json_folder, "for_eval_train_motion_2d_windows.p")
            else:
                self.processed_json_path = os.path.join(json_folder, "train_motion_2d_windows.p")
        else:
            if for_eval:
                self.processed_json_path = os.path.join(json_folder, "for_eval_motion_2d_windows.p")
            else:
                self.processed_json_path = os.path.join(json_folder, "test_motion_2d_windows.p")
        
        if os.path.exists(self.processed_json_path):
            self.data_dict = joblib.load(self.processed_json_path)
        else:
            self.data_dict = self.prep_window_motion_seq() 
            joblib.dump(self.data_dict, self.processed_json_path)
        print("Total length of the data:{0}".format(len(self.data_dict))) # Total length of the data:28972, for_eval=True 

        cam_extrinsic_list = get_multi_view_cam_extrinsic(num_views=30, farest=True)  # K X 4 X 4  
        ref_cam_extrinsic = cam_extrinsic_list[0:1] # 1 X 4 X 4 
        target_cam_extrinsic = cam_extrinsic_list[1:] # (K-1) X 4 X 4 
        ref_cam_extrinsic = ref_cam_extrinsic.repeat(target_cam_extrinsic.shape[0], 1, 1) # (K-1) X 4 X 4 
        self.epipoles = compute_epipoles(ref_cam_extrinsic, target_cam_extrinsic).cpu() # (K-1) X 2  
        # It seems all the y are 0. 
      
        smpl_dir = "/move/u/jiamanli/datasets/semantic_manip/processed_data/smpl_all_models/smpl"
        self.smpl = SMPL(model_path=smpl_dir, gender='MALE', batch_size=1)

    def prep_window_motion_seq(self):
        dest_data_dict = {}
        cnt = 0 

        if self.train:
            seq_names = json.load(open(self.json_path, 'r'))['train']
        else:
            seq_names = json.load(open(self.json_path, 'r'))['test']
       
        npz_files = os.listdir(self.npz_folder)
        for npz_name in npz_files:
            if ".npz" not in npz_name:
                continue 

            if "AIST" in self.npz_folder: 
                curr_vid_name = npz_name 
            elif "nicole" in self.npz_folder:
                curr_vid_name = npz_name 
            elif "cat" in self.npz_folder:
                curr_vid_name = npz_name.replace(".npz", "")
            elif "steezy" in self.npz_folder:
                curr_vid_name = npz_name
            else: # YouTube data 
                curr_vid_name = npz_name.split("_")[0]

            if curr_vid_name not in seq_names:
                continue 

            npz_path = os.path.join(self.npz_folder, npz_name)
            npz_data = np.load(npz_path)

            if "AIST" in self.npz_folder:
                # Downsample 60 fps to 30 fps 
                body_poses = npz_data['body_pose'][::2] # T X 18 X 2 
            else:
                body_poses = npz_data['body_pose'] # T X 18 X 2 

            if self.for_eval:
                # For evaluation, we use all the windows of each sequence to evaluate since some sequences are quite long 
                curr_seq_len = body_poses.shape[0]
                for t_idx in range(0, curr_seq_len, self.sample_n_frames):
                # for t_idx in range(0, curr_seq_len, self.sample_n_frames//2):
                    curr_seg_data = body_poses[t_idx:t_idx+self.sample_n_frames] 

                    if curr_seg_data.shape[0] >= 120:
                        dest_data_dict[cnt] = {}
                        dest_data_dict[cnt]['npz_name'] = npz_name 
                        dest_data_dict[cnt]['poses'] = curr_seg_data # T X 18 X 2 

                        dest_data_dict[cnt]['s_idx'] = t_idx 
                        dest_data_dict[cnt]['e_idx'] = t_idx+self.sample_n_frames 

                        cnt += 1 
            else:
                if body_poses.shape[0] >= 60:
                    dest_data_dict[cnt] = {}
                    dest_data_dict[cnt]['npz_name'] = npz_name 
                    dest_data_dict[cnt]['poses'] = body_poses # T X 18 X 2 

                    cnt += 1 

        return dest_data_dict 

    def get_normalized_motion2d_data(self, motion_data, start_t_idx, end_t_idx):
        motion_data = motion_data[start_t_idx:end_t_idx]
        motion_data = torch.from_numpy(motion_data).float().detach().clone() # In range [0, 1]

        if self.use_animal_data:
            motion_data, vis_2d_mask, scale_factors_for_input2d = \
                move_init_pose2d_to_center_for_animal_data(motion_data[None], \
                for_eval=self.for_eval, center_all_jnts2d=self.center_all_jnts2d) # 1 X T X 18 X 2  
        else:
            # Centralize 2d pose sequence based on the firs 2D pose 
            if "nicole" in self.npz_folder or "pamela" in self.npz_folder:
                motion_data, vis_2d_mask, scale_factors_for_input2d = move_init_pose2d_to_center(motion_data[None],\
                                    for_eval=self.for_eval, \
                                    center_all_jnts2d=self.center_all_jnts2d, rescale=False) # 1 X T X 18 X 2   
            else:
                motion_data, vis_2d_mask, scale_factors_for_input2d = move_init_pose2d_to_center(motion_data[None], \
                                    for_eval=self.for_eval, \
                                    center_all_jnts2d=self.center_all_jnts2d) # 1 X T X 18 X 2  
                                    
        motion_data = motion_data.squeeze(0) # In range [0, 1] 

        actual_seq_len = cal_min_seq_len_from_mask_torch(vis_2d_mask)     
        motion_data = motion_data[:actual_seq_len]

        # Normalize 2d pose sequence 
        normalized_motion_data = normalize_pose2d(motion_data, input_normalized=True)   
        # Normalize to range [-1, 1] 

        # Add padding
        if actual_seq_len < self.sample_n_frames:
            padding_motion_data = torch.zeros(self.sample_n_frames-actual_seq_len, \
                            normalized_motion_data.shape[1], normalized_motion_data.shape[2])
            normalized_motion_data = torch.cat((normalized_motion_data, padding_motion_data), dim=0) 

        return normalized_motion_data, actual_seq_len, scale_factors_for_input2d 

    def __len__(self):
        return len(self.data_dict)
       
    def __getitem__(self, idx):
        # idx = 0 # For debug 

        motion_data = self.data_dict[idx]['poses'].copy() # T X 18 X 2 
      
        seq_len = motion_data.shape[0]

        if self.train:
            if self.for_eval:
                start_t_idx = 0
            else:
                if seq_len <= self.sample_n_frames:
                    start_t_idx = 0 
                else:
                    start_t_idx = random.sample(list(range(seq_len-self.sample_n_frames+1)), 1)[0]
        else:
            start_t_idx = 0 

        # start_t_idx = 0 # For debug 

        end_t_idx = start_t_idx + self.sample_n_frames 

        normalized_motion_data, actual_seq_len, scale_factors_for_input2d = \
            self.get_normalized_motion2d_data(motion_data, start_t_idx, end_t_idx) 

        # actual_seq_len = motion_data.shape[0]
        normalized_motion_data = normalized_motion_data[:actual_seq_len] 
        if actual_seq_len < self.sample_n_frames:
            padding_motion_data = torch.zeros(self.sample_n_frames-actual_seq_len, \
                            normalized_motion_data.shape[1], normalized_motion_data.shape[2])
            normalized_motion_data = torch.cat((normalized_motion_data, padding_motion_data), dim=0) 

        # With a probability to sample epipoles 
        if torch.rand(1).item() < 0.5:
            # Use the camera extrinsic that will be used during testing to sample epipoles, then connect to form lines. 
            num_epipoles = self.epipoles.shape[0]
            selected_idx = random.sample(list(range(num_epipoles)), 1)[0]
            
            random_lines_for_joints2d, curr_epipoles = simulate_epipolar_lines(normalized_motion_data, \
                    epipoles=self.epipoles[selected_idx])
        else:
            # Random sample a point in [-v, v] range, and connect with each joint to simulate epipolar lines. 
            random_lines_for_joints2d, curr_epipoles = simulate_epipolar_lines(normalized_motion_data)

        input_dict = {}
        input_dict['normalized_jnts2d'] = normalized_motion_data 
        input_dict['seq_len'] = actual_seq_len 
        input_dict['lines_pass_jnts2d'] = random_lines_for_joints2d 
        
        input_dict['seq_name'] = self.data_dict[idx]['npz_name'] 
        input_dict['scale_factors_for_input2d'] = scale_factors_for_input2d

        if self.for_eval:
            if "s_idx" in self.data_dict[idx]:
                input_dict['s_idx'] = self.data_dict[idx]['s_idx']
                input_dict['e_idx'] = self.data_dict[idx]['e_idx'] 

        return input_dict 
