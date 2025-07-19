import math

import torch
import numpy as np
import cv2
import smplx
import pickle as pkl 
import trimesh
import os

import torch.nn as nn

from sklearn.preprocessing import normalize

from pytorch3d.transforms.rotation_conversions import *
import time, random

import pytorch3d.transforms as transforms 

from m2d.model.smal_model.smal_torch import SMAL

from m2d.data.utils_multi_view_2d_motion import perspective_projection

def compute_pose_loss(poses):
    poses = poses.reshape(-1, poses.shape[1]*3) 
    pose_loss = torch.linalg.norm(poses, ord=2) # The global rotation should not be constrained
    return pose_loss

def compute_time_loss(poses):
    # poses: T X 21 X 3 
    poses = poses.reshape(-1, 34*3) 
    pose_delta = poses[1:] - poses[:-1]
    time_loss = torch.linalg.norm(pose_delta, ord=2)
    return time_loss

def compute_time_loss_batch(poses):
    # poses: BS X T X 21 X 3
    # Reshape poses to handle all batches at once
    batch_size = poses.size(0)
    poses = poses.reshape(batch_size, -1, 34*3)
    
    # Compute the difference between consecutive frames for each batch
    pose_delta = poses[:, 1:] - poses[:, :-1]
    
    # Compute the L2 norm for time loss for each batch, then average over batches
    time_loss = torch.linalg.norm(pose_delta, ord=2, dim=2).mean()
    return time_loss

# Acceleration loss for smoothness
def compute_smooth_loss(poses):
    # poses: T X J X 3 (x, y, z joint positions)
    num_steps, num_jnts, _ = poses.shape 

    jnts_velocity = poses[1:] - poses[0:-1] # (T-1) X J X 3 
    jnts_accel = jnts_velocity[1:] - jnts_velocity[0:-1] # (T-2) X J X 3 

    jnts_velocity = jnts_velocity.reshape(num_steps-1, -1) # (T-1) X (J*3)
    jnts_accel = jnts_accel.reshape(num_steps-2, -1) # (T-2) X (J*3)

    vel_loss = torch.linalg.norm(jnts_velocity, ord=2)
    accel_loss = torch.linalg.norm(jnts_accel, ord=2)

    loss_smooth = vel_loss + accel_loss 

    return loss_smooth 

'''
SMAL joints:
CANONICAL_MODEL_JOINTS = [
  10, 9, 8, # upper_left [paw, middle, top]
  20, 19, 18, # lower_left [paw, middle, top]
  14, 13, 12, # upper_right [paw, middle, top]
  24, 23, 22, # lower_right [paw, middle, top]
  25, 31, # tail [start, end]
  33, 34, # ear base [left, right]
  35, 36, # nose, chin
  38, 37, # ear tip [left, right]
  39, 40, # eyes [left, right]
  15, 15, # withers, throat (TODO: Labelled same as throat for now), throat 
  28] # tail middle

joints from X-pose: 
ANIMAL_LANDMARKS = [
        "left eye",      # 0
        "right eye",     # 1
        "nose",          # 2
        "neck",          # 3
        "root of tail",  # 4
        "left shoulder", # 5
        "left elbow",    # 6
        "left front paw",# 7
        "right shoulder",# 8
        "right elbow",   # 9
        "right front paw",# 10
        "left hip",      # 11
        "left knee",     # 12
        "left back paw", # 13
        "right hip",     # 14
        "right knee",    # 15
        "right back paw" # 16
    ]

From SMAL joints to select 17 joints for computing loss with respect to X-pose's 17 joints 
xpose2smal_idx_list = [39, 40, 35, 15, 25, 8, 9, 10, 12, 13, 14, 18, 19, 20, 22, 23, 24]
'''

class SmalFitter2D():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_betas = 20 
        self.n_poses = 34 

        smal_root_folder = "/move/u/jiamanli/github/SMALify/data/SMALST/smpl_models"
        smal_file_path = os.path.join(smal_root_folder, "my_smpl_00781_4_all.pkl")
        smal_file_data_path = os.path.join(smal_root_folder, "my_smpl_data_00781_4_all.pkl")
        smal_sym_file = os.path.join(smal_root_folder, "symIdx.pkl")

        # self.smal2xpose_idx_list = [39, 40, 35, 15, 25, 8, 9, 10, 12, 13, 14, 18, 19, 20, 22, 23, 24]

        self.smal2xpose_idx_list = [39, 40, 35, 15, 25, 7, 8, 10, 11, 12, 14, 18, 19, 20, 22, 23, 24]

        self.shape_family_list = np.array([0])
        with open(smal_file_data_path, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            smal_data = u.load()

        shape_family = 0 # cat 

        model_covs = np.array(smal_data['cluster_cov'])[[shape_family]][0]

        invcov = np.linalg.inv(model_covs + 1e-5 * np.eye(model_covs.shape[0]))
        prec = np.linalg.cholesky(invcov)

        self.betas_prec = torch.FloatTensor(prec)[:self.n_betas, :self.n_betas].to(self.device)
        
        self.mean_betas = torch.FloatTensor(smal_data['cluster_means'][[shape_family]][0])[:self.n_betas].to(self.device)

        # setup SMAL skinning & differentiable renderer
        self.smal_model = SMAL(self.device, shape_family_id=shape_family)

    def init_param(self, skel3d, batch_size=None):
       
        # skel3d: (BS*T) X 18 X 3 

        num_steps = skel3d.shape[0] 

        scaling = torch.nn.Parameter(torch.ones(batch_size, 1).to(self.device), requires_grad=True) 

        cat_template_betas = np.array([[-1.201595  ,  0.15097299, -0.21082623,  0.45517337,  0.03215917,
                0.23746502, -0.58063364, -2.7282975 ,  1.613679  , -0.99766284,
                -1.5582378 ,  1.0206783 ,  0.00873854, -0.7760528 , -1.6388211 ,
                1.013249  , -1.0899303 , -0.8235719 ,  1.9492571 ,  0.41424868]])
        cat_template_betas = torch.from_numpy(cat_template_betas).float().to(self.device) 

        betas = nn.Parameter(cat_template_betas.clone().repeat(batch_size, 1), requires_grad=False) # Shape parameters (1 for the entire sequence... note expand rather than repeat)
        # betas = nn.Parameter(self.mean_betas.clone()[None].repeat(batch_size, 1), requires_grad=False) # Shape parameters (1 for the entire sequence... note expand rather than repeat)
        log_beta_scales = torch.nn.Parameter(
            torch.zeros(batch_size, 6).to(self.device), requires_grad=False) # Scale parameters for shape parameters

        trans = torch.zeros(num_steps, 3).to(self.device)
        trans = torch.nn.Parameter(trans, requires_grad=True) 

        global_rotation = torch.zeros(num_steps, 3).to(self.device)
        global_rotation = torch.nn.Parameter(global_rotation, requires_grad=True)

        default_joints = torch.zeros(num_steps, self.n_poses, 3).to(self.device)
        joint_rotations = torch.nn.Parameter(default_joints, requires_grad=False)

        smal_param = {'betas': betas, 'log_beta_scales': log_beta_scales, \
            'global_rotation': global_rotation, 'trans': trans, \
            'joint_rotations': joint_rotations, 'scaling': scaling}
        # betas: BS X 20 
        # log_beta_scales: BS X 6
        # global_rotation: (BS*T) X 3
        # trans: (BS*T) X 3
        # joint_rotations: (BS*T) X 34 X 3 
        # scaling: BS X 1
      
        return smal_param

    def smal_forward(self, smal_params, batch_size=None):
        # smpl_param {} 
        # betas: BS X 20 
        # log_beta_scales: BS X 6
        # global_rotation: (BS*T) X 3
        # trans: (BS*T) X 3
        # joint_rotations: (BS*T) X 34 X 3 

        seq_len = smal_params['joint_rotations'].shape[0]//batch_size 

        verts, joints, Rs, v_shaped = self.smal_model(
                beta=smal_params['betas'][:, None, :].repeat(1, seq_len, 1).reshape(-1, self.n_betas), 
                theta=torch.cat([
                    smal_params['global_rotation'].unsqueeze(1), 
                    smal_params['joint_rotations']], dim = 1),
                trans=smal_params['trans'],
                betas_logscale=smal_params['log_beta_scales'][:, None, :].repeat(1, seq_len, 1).reshape(-1, 6))

        verts = verts * smal_params['scaling'][:, None, :].repeat(1, seq_len, 1).reshape(-1, 1).unsqueeze(-1) # (BS*T) X 1 X 1 
        joints = joints * smal_params['scaling'][:, None, :].repeat(1, seq_len, 1).reshape(-1, 1).unsqueeze(-1) # (BS*T) X J X 3 

        verts = verts + smal_params['trans'].unsqueeze(1) # (BS*T) X N_v X 3 
        joints = joints + smal_params['trans'].unsqueeze(1) # (BS*T) X J X 3 

        global_orientation = smal_params['global_rotation'] 
        body_pose = smal_params['joint_rotations'] 
        smal_full_body_pose = torch.cat((global_orientation[:, None, :], body_pose), dim=1) # (BS*T) X 35 X 3 

        return joints, verts, smal_full_body_pose

    def calc_2d_loss(self, x1, x2):
        loss = torch.nn.functional.mse_loss(x1, x2)
        return loss

    def gen_closure(self, optimizer, smal_param, target_jnts2d, cam_rot_mat, cam_trans, batch_size=None):
        def closure():
            optimizer.zero_grad()
            skel, verts, smal_body_pose = self.smal_forward(smal_param, batch_size=batch_size) 
            
            loss = {}

            smpl_jnts3d = skel[:, self.smal2xpose_idx_list, :] # (BS*T) X 17 X 3 

            _, _, smpl_jnts2d = perspective_projection(
                    smpl_jnts3d, cam_rot_mat, \
                    cam_trans)

            width = 1920 
            height = 1080 
            smpl_jnts2d[:, :, 0] = smpl_jnts2d[:, :, 0] / width
            smpl_jnts2d[:, :, 1] = smpl_jnts2d[:, :, 1] / height 
            smpl_jnts2d = smpl_jnts2d * 2 - 1 

            loss_match = self.calc_2d_loss(smpl_jnts2d, target_jnts2d)

            # Pose loss as is
            # other_jnts_idx = []
            # all_num_joints = 35 
            # for tmp_idx in range(all_num_joints):
            #     if tmp_idx not in self.smal2xpose_idx_list and tmp_idx != 0:
            #         other_jnts_idx.append(tmp_idx)

            # pose_loss = compute_pose_loss(smal_body_pose[:, other_jnts_idx, :])

            # rigid_jnts_idx = [7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34]
            # core_jnts_idx = [1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 32, 33] 

            # rigid_jnts_idx = [1, 4, 9, 13, 16, 3, 6, 7, 8, 9, 10, 11, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
            # # core_jnts_idx = [1, 2, 4, 5, 13, 14, 16, 17] 
            # core_jnts_idx = [2, 5, 14, 17] 


            rigid_jnts_idx = [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
            core_jnts_idx = [7, 8, 11, 12, 17, 18, 19, 21, 22, 23] 

            rigid_pose_loss = compute_pose_loss(smal_body_pose[:, rigid_jnts_idx, :]) 

            pose_loss = compute_pose_loss(smal_body_pose[:, core_jnts_idx, :])

            # pose_loss = compute_pose_loss(smal_body_pose)
            
            # Time smoothness
            if smal_body_pose.shape[0] > 1:
                if batch_size is not None:
                    seq_len = smal_body_pose.shape[0]//batch_size 
                    time_loss = compute_time_loss_batch(smal_body_pose.reshape(batch_size, seq_len, -1, 3)[:, :, 1:])
                else: 
                    time_loss = compute_time_loss(smal_body_pose[:, 1:])
            else:
                time_loss = torch.zeros(1).to(smal_body_pose.device) 

            # loss = loss_match + time_loss * 1e-3 + pose_loss * 1e-4  # Old 

            # loss = loss_match + time_loss * 1e-3 + pose_loss * 1e-3  # Old 

            loss = loss_match + time_loss * 1e-4 + rigid_pose_loss * 1e-3 + pose_loss * 1e-4    

            # print(f"Joint loss: {loss_match.item():.4f}, \
            #     Pose loss: {rigid_pose_loss.item():.4f}, \
            #     Time loss: {time_loss.item():.4f}")

            loss.backward()
            return loss

        return closure

    def solve(self, smal_param, closure, optimizer, iter_max=1000, \
            iter_thresh=1e-10, loss_thresh=1e-10):
        loss_prev = float('inf')
        
        for i in range(iter_max):
            loss = optimizer.step(closure).item()
            if abs(loss - loss_prev) < iter_thresh and loss < loss_thresh:
                # print('iter ' + str(i) + ': ' + str(loss))
                break
            else:
                # print('iter ' + str(i) + ': ' + str(loss))
                loss_prev = loss

        return smal_param, self.smal_model.faces  
