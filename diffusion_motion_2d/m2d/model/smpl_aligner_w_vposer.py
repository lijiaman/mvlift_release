
"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Soyong Shin, Marilyn Keller
See https://skel.is.tue.mpg.de/license.html for licensing and contact information.
"""

import argparse
import math
import os
import pickle
import torch
import numpy as np
from tqdm import trange
import torch.nn.functional as F

from smplx import SMPL

from torch import nn

import trimesh 

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

import pytorch3d.transforms as transforms 

def compute_pose_loss(poses): # poses do not include global orientation 
    
    pose_loss = torch.linalg.norm(poses, ord=2) # The global rotation should not be constrained
    return pose_loss

def compute_time_loss(poses):
    
    pose_delta = poses[1:] - poses[:-1]
    time_loss = torch.linalg.norm(pose_delta, ord=2)
    return time_loss

class SMLPVPoserAligner(object):
    
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Init SMPL model 
        smpl_dir = "/move/u/jiamanli/datasets/semantic_manip/processed_data/smpl_all_models/smpl"
        self.smpl = SMPL(model_path=smpl_dir, gender='MALE', batch_size=1).to(self.device)

        # Init Vposer 
        expr_dir = "/move/u/jiamanli/github/mofitness/diffusion_motion_2d/V02_05"
        vposer, ps = load_model(expr_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
        self.vposer = vposer.to(self.device)
        
    def smpl_forward(self, smpl_param):
        # smpl_param['orient']: T X 3 
        # smpl_param['trans']: T X 3 
        # smpl_param['scaling]: a single scalar value   
        # smpl_param['latent']: T X 32 

        body_pose = self.vposer.decode(smpl_param['latent'])['pose_body'] # T X 21 X 3 
        padding_poses = torch.zeros(body_pose.shape[0], 2, 3).to(body_pose.device) 

        smpl_out = self.smpl.forward(
            global_orient=smpl_param['orient'][:, None, :], 
            body_pose=torch.cat((body_pose, padding_poses), dim=1), 
            transl=smpl_param['trans'],
            scaling=smpl_param['scaling'].reshape(1, 1), 
        )

        res_dict = {}

        res_dict['body_pose'] = torch.cat((body_pose, padding_poses), dim=1).detach().cpu().numpy().copy()
        res_dict['global_orient'] = smpl_param['orient'][:, None, :].detach().cpu().numpy().copy()
        res_dict['transl'] = smpl_param['trans'].detach().cpu().numpy().copy()
        res_dict['scaling'] = smpl_param['scaling'].reshape(1, 1).detach().cpu().numpy().copy()

        smpl_full_body_pose = torch.cat((smpl_param['orient'][:, None, :], body_pose), dim=1) # T X 22 X 3 

        return smpl_out, smpl_out.joints, smpl_full_body_pose, res_dict, self.smpl.faces # T X 45 X 3 
    
    def optim(self, smpl_param, 
          jnts3d_18, 
          lr=1e0,
          max_iter=5,
          num_steps=5,
          line_search_fn='strong_wolfe',
          pose_reg_factor = 1e1,
          ):
    
        # poseLoss = PoseLimitLoss().to(device)
        
        optimizer = torch.optim.LBFGS(smpl_param.values(), lr=lr, max_iter=max_iter, \
            line_search_fn=line_search_fn)
        pbar = trange(num_steps, leave=False)
        # mv = MeshViewer(keepalive=False)

        def closure():
            optimizer.zero_grad()
            
            output, smpl_jnts, smpl_body_pose, _, _ = self.smpl_forward(smpl_param)
            # model_out, joint positions, smpl joint aa representation, res_dict, faces 
           
            target_jnts18 = torch.from_numpy(jnts3d_18).float().to(output.joints.device) # T X 18 X 3 

            smpl2jnts18_idx = [24, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28]
            
            joint_loss = 1e2 * F.mse_loss(output.joints[:, smpl2jnts18_idx, :], target_jnts18)
            
            # Regularize the pose
            # scapula_loss = 1e-2 * compute_scapula_loss(poses_in)
            # spine_loss = 1e-3 * compute_spine_loss(poses_in)
            # pose_loss = 1e-4 * compute_pose_loss(poses)

            pose_loss = 1e-3 * compute_pose_loss(smpl_body_pose[:, 1:, :].reshape(-1, 21*3))
            
            # Time consistancy
            if target_jnts18.shape[0] > 1:
                time_loss = 1e-1 * compute_time_loss(smpl_body_pose[:, 1:, :].reshape(-1, 21*3))
            else:
                time_loss = torch.zeros(1).cuda() 
            
            for pl in [pose_loss]:
                pl = pose_reg_factor * pl
            
            loss = joint_loss + pose_loss + time_loss
            # loss = joint_loss 
            # make a pretty print of the losses
            print(f"Joint loss: {joint_loss.item():.4f}, \
                  Pose loss: {pose_loss.item():.4f}, \
                  Time loss: {time_loss.item():.4f}")
      
            loss.backward()
        
            return loss

        for _ in pbar:
            loss = optimizer.step(closure).item()

    def fit(self, jnts3d_18):
        """Align SKEL to a SMPL sequence."""

        # Apply a rotation to match the T-pose of SKEL model 
        align_rot_mat = np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        jnts3d_18 = np.dot(jnts3d_18.reshape(-1, 3), align_rot_mat.T).reshape(-1, 18, 3)

        l_hip_index = 8 
        r_hip_index = 11 
        hip_jpos = (jnts3d_18[:, l_hip_index, :]+jnts3d_18[:, r_hip_index, :])/2 # T X 3 

        # init_optim_params = {'lr': 1e-2, 'max_iter': 25, 'num_steps': 100}
        # seq_optim_params = {'lr': 1e-1, 'max_iter': 25, 'num_steps': 100}

        init_optim_params = {'lr': 1e0, 'max_iter': 25, 'num_steps': 20}
        # seq_optim_params = {'lr': 1e-1, 'max_iter': 25, 'num_steps': 20}
        seq_optim_params = {'lr': 1e0, 'max_iter': 25, 'num_steps': 20}
        
        nb_frames = jnts3d_18.shape[0]
        print('Fitting {} frames'.format(nb_frames))

        smpl_param = {'orient': torch.zeros((jnts3d_18.shape[0], 3), dtype=torch.float32, device=self.device, requires_grad=True),
                  'trans': torch.tensor(hip_jpos, dtype=torch.float32, device=self.device, requires_grad=True),
                  'latent': torch.zeros((jnts3d_18.shape[0], 32), dtype=torch.float32, device=self.device, requires_grad=True),
                  'scaling': torch.tensor([1.], dtype=torch.float32, device=self.device, requires_grad=True)}
                 
        # Optimize the global rotation and translation for the initial fitting
        smpl_param['latent'].requires_grad = False 
        self.optim(smpl_param, jnts3d_18)

        smpl_param['latent'].requires_grad = True 
        self.optim(smpl_param, jnts3d_18, **seq_optim_params)

        # Convert the results to stand on the floor z = 0 
        align_rot_mat = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
      
        # Apply rotation to SMPL parameters 
        smpl_global_rot_aa_rep = smpl_param['orient'] # T X 3 
        smpl_global_rot_mat = transforms.axis_angle_to_matrix(smpl_global_rot_aa_rep) # T X 3 X 3 
        smpl_global_rot_mat = torch.matmul(torch.from_numpy(align_rot_mat).float()[None].to(self.device).repeat(smpl_global_rot_mat.shape[0], 1, 1), \
                smpl_global_rot_mat) # T X 3 X 3 
        new_rot_aa_rep = transforms.matrix_to_axis_angle(smpl_global_rot_mat)

        smpl_trans = smpl_param['trans'].detach().cpu().numpy()
        aligned_smpl_trans = torch.matmul(torch.from_numpy(align_rot_mat).float()[None].to(self.device).repeat(smpl_global_rot_mat.shape[0], 1, 1), \
            torch.from_numpy(smpl_trans).float()[:, :, None].to(self.device))
        aligned_smpl_trans = aligned_smpl_trans.squeeze(-1) # T X 3 
    
        # Prepare parameters used for outter forward
        res_smpl_params = {} 
        # res_global_orient = smpl_param['orient'][:, None, :]  # T X 1 X 3 
        res_global_orient = new_rot_aa_rep[:, None, :] # T X 1 X 3 
        res_body_pose = self.vposer.decode(smpl_param['latent'])['pose_body'] # T X 21 X 3 
        paddings = torch.zeros(res_body_pose.shape[0], 2, 3).to(res_body_pose.device)
        smpl_poses = torch.cat((res_global_orient, res_body_pose, paddings), dim=1)

        # res_smpl_params['smpl_trans']  = smpl_param['trans'].detach().cpu().numpy()
        res_smpl_params['smpl_trans'] = aligned_smpl_trans.detach().cpu().numpy()
        res_smpl_params['smpl_scaling'] = smpl_param['scaling'].reshape(1, 1).detach().cpu().numpy()
        res_smpl_params['smpl_poses'] = smpl_poses.detach().cpu().numpy() 

        smpl_param['orient'] = new_rot_aa_rep 
        smpl_param['trans'] = aligned_smpl_trans 

        return smpl_param, res_smpl_params, self.smpl.faces  
            