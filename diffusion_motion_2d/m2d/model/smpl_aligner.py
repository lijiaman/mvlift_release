
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

def compute_scapula_loss(poses):
    
    scapula_indices = [26, 27, 28, 36, 37, 38]
    
    scapula_poses = poses[:, scapula_indices]
    scapula_loss = torch.linalg.norm(scapula_poses, ord=2)
    return scapula_loss

def compute_spine_loss(poses):
    
    spine_indices = range(17, 25)
    
    spine_poses = poses[:, spine_indices]
    spine_loss = torch.linalg.norm(spine_poses, ord=2)
    return spine_loss

def compute_pose_loss(poses):
    
    pose_loss = torch.linalg.norm(poses[:, 3:], ord=2) # The global rotation should not be constrained
    return pose_loss

def compute_time_loss(poses):
    
    pose_delta = poses[1:] - poses[:-1]
    time_loss = torch.linalg.norm(pose_delta, ord=2)
    return time_loss

def optim(params, 
          jnts3d_18,
          smpl_model,
          lr=1e0,
          max_iter=5,
          num_steps=5,
          line_search_fn='strong_wolfe',
          pose_reg_factor = 1e1,
          ):
    
        # poseLoss = PoseLimitLoss().to(device)
        
        optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=max_iter, \
            line_search_fn=line_search_fn)
        pbar = trange(num_steps, leave=False)
        # mv = MeshViewer(keepalive=False)

        def closure():
            optimizer.zero_grad()
            
            output = smpl_model.forward() 
            
           
            target_jnts18 = torch.from_numpy(jnts3d_18).float().to(output.joints.device) # T X 18 X 3 

            smpl2jnts18_idx = [24, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28]
            
            joint_loss = 1e2 * F.mse_loss(output.joints[:, smpl2jnts18_idx, :], target_jnts18)
            
            # Regularize the pose
            # scapula_loss = 1e-2 * compute_scapula_loss(poses_in)
            # spine_loss = 1e-3 * compute_spine_loss(poses_in)
            # pose_loss = 1e-4 * compute_pose_loss(poses)

            pose_loss = 1e-3 * compute_pose_loss(smpl_model.body_pose)
            
            # Time consistancy
            if target_jnts18.shape[0] > 1:
                time_loss = 1e-1 * compute_time_loss(smpl_model.body_pose)
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
            # with torch.no_grad():
            #     poses[:] = torch.atan2(poses.sin(), poses.cos())
            # pbar.set_postfix_str(f"Loss {loss:.4f}")

class SMLPFitter(object):
    
    def __init__(self, smpl_model_path, smpl_model_gener='MALE') -> None:

        self.smpl_model_path = smpl_model_path
        self.smpl_model_gender = smpl_model_gener
        
    def fit(self, jnts3d_18, batch_size=120):
        """Align SKEL to a SMPL sequence."""

        # Apply a rotation to match the T-pose of SKEL model 
        align_rot_mat = np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        jnts3d_18 = np.dot(jnts3d_18.reshape(-1, 3), align_rot_mat.T).reshape(-1, 18, 3)

        l_hip_index = 8 
        r_hip_index = 11 
        hip_jpos = (jnts3d_18[:, l_hip_index, :]+jnts3d_18[:, r_hip_index, :])/2 # T X 3 
        hip_jpos = torch.from_numpy(hip_jpos).float().cuda() 

        # Plot joints3d to ball meshes 
        
        # ball_color = np.asarray([22, 173, 100]) # green 

        # ball_mesh_debug_folder = "/viscam/projects/vimotion/opt_res3d/debug_amass_mesh"
        # if not os.path.exists(ball_mesh_debug_folder):
        #     os.makedirs(ball_mesh_debug_folder)
        # ball_mesh_path = os.path.join(ball_mesh_debug_folder, "test_global_orient.ply")

        # num_mesh = jnts3d_18.shape[1]
        # for idx in range(num_mesh):
        #     ball_mesh = trimesh.primitives.Sphere(radius=0.05, center=jnts3d_18[0, idx])
            
        #     dest_ball_mesh = trimesh.Trimesh(
        #         vertices=ball_mesh.vertices,
        #         faces=ball_mesh.faces,
        #         vertex_colors=ball_color,
        #         process=False)

        #     result = trimesh.exchange.ply.export_ply(dest_ball_mesh, encoding='ascii')
        #     output_file = open(ball_mesh_path.replace(".ply", "_"+str(idx)+".ply"), "wb+")
        #     output_file.write(result)
        #     output_file.close()

        # import pdb 
        # pdb.set_trace() 

        # l_hip_to_r_hip_vec = jnts3d_18[:, r_hip_index, :] - jnts3d_18[:, l_hip_index, :] 

        
        # Optimization params

        # For fast debug 
        # init_optim_params = {'lr': 1e0, 'max_iter': 1, 'num_steps': 1}
        # seq_optim_params = {'lr': 1e-1, 'max_iter': 1, 'num_steps': 1}

        # init_optim_params = {'lr': 1e0, 'max_iter': 10, 'num_steps': 20}
        # seq_optim_params = {'lr': 1e-1, 'max_iter': 10, 'num_steps': 20}

        init_optim_params = {'lr': 1e0, 'max_iter': 25, 'num_steps': 20}
        seq_optim_params = {'lr': 1e-1, 'max_iter': 25, 'num_steps': 20}
        
        nb_frames = jnts3d_18.shape[0]
        print('Fitting {} frames'.format(nb_frames))
        
        # Init learnable smpl model
        smpl = SMPL(
            model_path=self.smpl_model_path,
            gender=self.smpl_model_gender,
            batch_size=nb_frames).cuda() 

        smpl.transl = nn.Parameter(hip_jpos) # Initialize translation 
        
        if batch_size > nb_frames:
            batch_size = nb_frames
            print('Batch size is larger than the number of frames. Setting batch size to {}'.format(batch_size))
            
        n_batch = math.ceil(nb_frames/batch_size)
        pbar = trange(n_batch, desc='Running batch optimization')
        
        # initialize the res dict with generic empty lists 
        res_dict = {
            'transl':  [],
            'scaling':  [],
            'global_orient':  [],
            'body_pose':  [],
        }
    

        for i in pbar:
            # Get mini batch
            i_start, i_end = i * batch_size, min((i+1) * batch_size, nb_frames)
            
            if i == 0:
                # Optimize the global rotation and translation for the initial fitting
                optim([smpl.transl, smpl.scaling, smpl.global_orient], jnts3d_18[i_start:i_end], \
                    smpl)
                optim([smpl.transl, smpl.scaling, smpl.global_orient, smpl.body_pose], jnts3d_18[i_start:i_end], \
                    smpl, **init_optim_params)
            else: # SHouldn't be used in current version 
                optim([smpl.transl, smpl.scaling, smpl.global_orient, smpl.body_pose], jnts3d_18[i_start:i_end], \
                    smpl, pose_reg_factor=1, **seq_optim_params)

            res_dict['body_pose'].append(smpl.body_pose.detach().cpu().numpy().copy())
            res_dict['global_orient'].append(smpl.global_orient.detach().cpu().numpy().copy())
            res_dict['transl'].append(smpl.transl.detach().cpu().numpy().copy())
            res_dict['scaling'].append(smpl.scaling.detach().cpu().numpy().copy())
            
        for key, val in res_dict.items():
            if isinstance(val, list):
                res_dict[key] = np.concatenate(val)
                
        return smpl 
            