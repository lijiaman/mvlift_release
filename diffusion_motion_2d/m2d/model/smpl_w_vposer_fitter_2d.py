import math

import torch
import numpy as np
import cv2
import smplx
import pickle
import trimesh
import os

from sklearn.preprocessing import normalize
# from human_body_prior.train.vposer_smpl import VPoser
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

from pytorch3d.transforms.rotation_conversions import *
import time, random

import pytorch3d.transforms as transforms 

from m2d.data.utils_multi_view_2d_motion import perspective_projection

from smplx import SMPL 

def compute_pose_loss(poses):
    poses = poses.reshape(-1, 22*3) 
    pose_loss = torch.linalg.norm(poses[:, 3:], ord=2) # The global rotation should not be constrained
    return pose_loss

def compute_time_loss(poses):
    # poses: T X 21 X 3 
    poses = poses.reshape(-1, 21*3) 
    pose_delta = poses[1:] - poses[:-1]
    time_loss = torch.linalg.norm(pose_delta, ord=2)
    return time_loss

def compute_time_loss_batch(poses):
    # poses: BS X T X 21 X 3
    # Reshape poses to handle all batches at once
    batch_size = poses.size(0)
    poses = poses.reshape(batch_size, -1, 21 * 3)
    
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

class SmplVPoser2DFitter():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
        smpl_dir = "/move/u/jiamanli/datasets/semantic_manip/processed_data/smpl_all_models/smpl"
        self.smpl = SMPL(model_path=smpl_dir, gender='MALE', batch_size=1).to(self.device)

        # self.vposer = VPoser(512, 32, [3, 21]).eval()
        # self.vposer.load_state_dict(torch.load('/move/u/jiamanli/github/mofitness/fit_smpl/vposer_v1_0/snapshots/TR00_E096.pt', map_location='cpu'))
        # self.vposer = self.vposer.to(self.device)
        # print(self.vposer)
        expr_dir = "/move/u/jiamanli/github/mofitness/diffusion_motion_2d/V02_05"
        vposer, ps = load_model(expr_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
        self.vposer = vposer.to(self.device)

    def smpl_forward(self, smpl_param, batch_size=None):
        # smpl_param['orient']: T X 3/(BS*T) X 3 
        # smpl_param['trans']: T X 3/(BS*T) X 3 
        # smpl_param['scaling]: a single scalar value/BS    
        # smpl_param['latent']: T X 32/(BS*T) X 32  

        body_pose = self.vposer.decode(smpl_param['latent'])['pose_body'] # T X 21 X 3/(BS*T) X 21 X 3 
        padding_poses = torch.zeros(body_pose.shape[0], 2, 3).to(body_pose.device) 

        global_orientation = smpl_param['orient']

        # global_orientation = transforms.rotation_6d_to_matrix(smpl_param['orient'])
        # global_orientation = transforms.matrix_to_axis_angle(global_orientation)

        if batch_size is None:
            smpl_out = self.smpl.forward(
                global_orient=global_orientation[:, None, :], 
                body_pose=torch.cat((body_pose, padding_poses), dim=1), 
                transl=smpl_param['trans'],
                scaling=smpl_param['scaling'].reshape(-1, 1), 
            )
        else:
            smpl_out = self.smpl.forward(
                global_orient=global_orientation[:, None, :], 
                body_pose=torch.cat((body_pose, padding_poses), dim=1), 
                transl=smpl_param['trans'],
                scaling=smpl_param['scaling'][:, None].repeat(1, body_pose.shape[0]//batch_size).reshape(-1, 1),
            )

        res_dict = {}

        res_dict['body_pose'] = torch.cat((body_pose, padding_poses), dim=1).detach().cpu().numpy().copy()
        # res_dict['global_orient'] = smpl_param['orient'][:, None, :].detach().cpu().numpy().copy()
        res_dict['global_orient'] = global_orientation[:, None, :].detach().cpu().numpy().copy()
        res_dict['transl'] = smpl_param['trans'].detach().cpu().numpy().copy()
        res_dict['scaling'] = smpl_param['scaling'].reshape(-1, 1).detach().cpu().numpy().copy()

        smpl_full_body_pose = torch.cat((global_orientation[:, None, :], body_pose), dim=1) # T X 22 X 3 

        return smpl_out, smpl_out.joints, smpl_full_body_pose, res_dict, self.smpl.faces # T X 45 X 3 

    def init_param(self, skel3d, batch_size=None):
      
        init_trans = torch.from_numpy(np.asarray([0.0030, 0.3272,  -0.0130])).to(self.device).repeat(skel3d.shape[0], 1) # (BS*T) X 3
        
        # zero_smpl_param = {'orient': torch.zeros(skel3d.shape[0], 3).to(self.device), \
        #     # 'trans': torch.zeros(skel3d.shape[0], 3).to(self.device), \
        #     'trans': init_trans, \
        #     'latent': torch.zeros(skel3d.shape[0], 32).to(self.device), \
        #     'scaling': torch.ones(batch_size).to(self.device)}
        # smpl_out, skel, smpl_body_pose, _, _ = self.smpl_forward(zero_smpl_param, batch_size=batch_size) 

        # root_jpos3d = (skel[:, 1]+skel[:, 2])/2.  # (BS*T) X 3
        # import pdb 
        # pdb.set_trace() 

        orient = np.zeros((skel3d.shape[0], 3)) # (BS*T) X 3 
        trans = init_trans.detach().cpu().numpy() # (BS*T) X 3 
        # trans = np.zeros((skel3d.shape[0], 3)) # (BS*T) X 3 
        scale = np.ones((batch_size)) # BS 

        smpl_param = {'orient': orient, 'trans': trans, 'latent': np.zeros((skel3d.shape[0], 32)), \
                    'scaling': scale}
        
        # 
        # (Pdb) skel[0, 0]
        # tensor([-0.0022, -0.2408,  0.0286], device='cuda:0', grad_fn=<SelectBackward0>)
        # (Pdb) (skel[0, 1]+skel[0, 2])/2
        # tensor([-0.0030, -0.3272,  0.0130], device='cuda:0', grad_fn=<DivBackward0>)

        # 'scale': float(scale)}
        return smpl_param

    def calc_2d_loss(self, x1, x2):
        loss = torch.nn.functional.mse_loss(x1, x2)
        return loss

    def gen_closure(self, optimizer, smpl_param, target_jnts2d, cam_rot_mat, cam_trans, batch_size=None):
        # target_jnts2d: (BS*T) X J X 2 
        # cam_rot_mat: (BS*T) X 3 X 3 
        # cam_trans: (BS*T) X 3 
        def closure():
            optimizer.zero_grad()
            smpl_out, skel, smpl_body_pose, _, _ = self.smpl_forward(smpl_param, batch_size=batch_size) 
            loss = {}
            
            smpl2jnts17_idx = [24, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28]
            target_jnts17_idx = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

            target_jnts2d_17 = target_jnts2d[:, target_jnts17_idx, :] # (BS*T) X 17 X 2 
            
            smpl_jnts3d_17 = skel[:, smpl2jnts17_idx, :] # (BS*T) X 17 X 3 

            _, _, smpl_jnts2d_17 = perspective_projection(
                    smpl_jnts3d_17, cam_rot_mat, \
                    cam_trans)
            width = 1920 
            height = 1080 
            smpl_jnts2d_17[:, :, 0] = smpl_jnts2d_17[:, :, 0] / width
            smpl_jnts2d_17[:, :, 1] = smpl_jnts2d_17[:, :, 1] / height 
            smpl_jnts2d_17 = smpl_jnts2d_17 * 2 - 1 

            loss_match = self.calc_2d_loss(smpl_jnts2d_17, target_jnts2d_17)
                                    
            # pose_loss = compute_pose_loss(smpl_body_pose)
            
            # Time smoothness
            if smpl_body_pose.shape[0] > 1:
                if batch_size is not None:
                    seq_len = smpl_body_pose.shape[0]//batch_size 
                    time_loss = compute_time_loss_batch(smpl_body_pose.reshape(batch_size, seq_len, -1, 3)[:, :, 1:])
                else: 
                    time_loss = compute_time_loss(smpl_body_pose[:, 1:])
            else:
                time_loss = torch.zeros(1).to(smpl_body_pose.device) 

            # loss = loss_match 
            loss = loss_match + time_loss * 1e-3  # Old 
            # loss = loss_match + time_loss * 1e-4 + pose_loss * 1e-4

            # print(f"Fitting 2D Joint loss: {loss_match.item():.4f}, \
            #       Time loss: {time_loss.item():.4f}")

            loss.backward()
            return loss

        return closure

    def solve(self, smpl_param, closure, optimizer, iter_max=1000, \
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

        # Prepare parameters used for outter forward
        res_smpl_params = {} 
        res_global_orient = smpl_param['orient'][:, None, :]  # (BS*T) X 1 X 3 
        res_body_pose = self.vposer.decode(smpl_param['latent'])['pose_body'] # (BS*T) X 21 X 3 

        # Use 6D representation 
        # res_global_orient = transforms.rotation_6d_to_matrix(res_global_orient)
        # res_global_orient = transforms.matrix_to_axis_angle(res_global_orient)

        paddings = torch.zeros(res_body_pose.shape[0], 2, 3).to(res_body_pose.device)
        smpl_poses = torch.cat((res_global_orient, res_body_pose, paddings), dim=1)
        
        res_smpl_params['smpl_trans']  = smpl_param['trans'].detach().cpu().numpy() # (BS*T) X 3 
        res_smpl_params['smpl_scaling'] = smpl_param['scaling'].detach().cpu().numpy() # BS 
        res_smpl_params['smpl_poses'] = smpl_poses.detach().cpu().numpy() # (BS*T) X 24 X 3 

        return smpl_param, res_smpl_params, self.smpl.faces  

    def export(self, filename, smpl_param):
        smpl_out, _ = self.smpl_forward(smpl_param)
        mesh = trimesh.Trimesh(vertices=smpl_out.vertices[0].detach().cpu().numpy(), faces=self.smpl.faces)
        mesh.export(filename)
       
        return mesh
