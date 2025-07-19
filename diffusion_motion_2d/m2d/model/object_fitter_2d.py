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

from smplx import SMPL 

from m2d.data.utils_multi_view_2d_motion import perspective_projection

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

class ObjectFitter2D():
    def __init__(self, object_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load object geometry 
        rest_obj_folder = "/move/u/jiamanli/datasets/semantic_manip/processed_data/rest_object_geo"
        rest_object_geometry_path = os.path.join(rest_obj_folder, object_name+".ply")  

        mesh = trimesh.load_mesh(rest_object_geometry_path)
        self.obj_rest_verts = np.asarray(mesh.vertices) # Nv X 3
        self.obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3

        self.object2kpts_idx_dict = {
            "largebox":  np.asarray([11298, 12617, 12084, 7527, 3042]), 
            "woodchair":  np.asarray([878, 4304, 4925, 11511, 6719]), 
            "monitor": np.asarray([2393, 35456, 6686, 5386, 26582]), 
            "largetable": np.asarray([10, 418, 5728, 3846, 5695]), 
            "clothesstand": np.asarray([12379, 13347, 10372, 4007, 22]),
        }

        self.obj_kpts_idx_list = self.object2kpts_idx_dict[object_name] 

        self.rest_obj_kpts = self.obj_rest_verts[self.obj_kpts_idx_list] # 5 X 3 
        self.rest_obj_scale = self.compute_object_scale(self.rest_obj_kpts[np.newaxis])  # 1, scalar value 

    def compute_object_scale(self, obj_kpts_3d):
        # obj_kpts_3d: BS X 5 X 3 
        # Only need to compute one pair 

        kpts_pair_len = np.linalg.norm(obj_kpts_3d[:, 0] - obj_kpts_3d[:, 1], axis=1) # BS 

        return kpts_pair_len 

    def object_forward(self, obj_param, batch_size=None):
        # obj_apram['obj_rotation]: T X 3/(BS*T) X 3 
        # obj_param['obj_trans']: T X 3/(BS*T) X 3 
        # obj_param['scaling']: a single scalar value/BS    

        # Apply transformation to object mesh 
        # obj_scale: T, obj_rot: T X 3 X 3, obj_com_pos: T X 3, rest_verts: Nv X 3 
        obj_rot_aa_rep = obj_param['obj_orient'] # (BS*T) X 3
        obj_trans = obj_param['obj_trans'] # (BS*T) X 3 

        seq_len = obj_trans.shape[0]//batch_size 
        obj_scaling = obj_param['obj_scaling'].repeat(1, seq_len).reshape(-1) # (BS*T) 

        obj_rot_mat = transforms.axis_angle_to_matrix(obj_rot_aa_rep) # (BS*T) X 3 X 3

        rest_kpts = torch.from_numpy(self.rest_obj_kpts).float().to(obj_rot_mat.device)[None].repeat(obj_rot_aa_rep.shape[0], 1, 1) # (BS*T) X 5 X 3 
        transformed_obj_kpts = obj_rot_mat.bmm(rest_kpts.transpose(1, 2)) * obj_scaling[:, None, None] + obj_trans[:, :, None] # (BS*T) X 3 X 5 
        transformed_obj_kpts = transformed_obj_kpts.transpose(1, 2) # (BS*T) X 5 X 3 

        res_dict = {}
       
        res_dict['obj_orient'] = obj_rot_aa_rep.detach().cpu().numpy().copy()
        res_dict['obj_trans'] = obj_trans.detach().cpu().numpy().copy()
        res_dict['obj_scaling'] = obj_scaling.reshape(-1, 1).detach().cpu().numpy().copy()

        return transformed_obj_kpts, res_dict, self.obj_mesh_faces # T X 45 X 3 

    def get_object_verts(self, obj_param, batch_size):
        obj_rot_aa_rep = obj_param['obj_orient'] # (BS*T) X 3
        obj_trans = obj_param['obj_trans'] # (BS*T) X 3 

        seq_len = obj_trans.shape[0]//batch_size 
        obj_scaling = obj_param['obj_scaling'].repeat(1, seq_len).reshape(-1) # (BS*T) 

        obj_rot_mat = transforms.axis_angle_to_matrix(obj_rot_aa_rep) # (BS*T) X 3 X 3

        rest_kpts = torch.from_numpy(self.rest_obj_kpts).float().to(obj_rot_mat.device)[None].repeat(obj_rot_aa_rep.shape[0], 1, 1) # (BS*T) X 5 X 3 
        transformed_obj_kpts = obj_rot_mat.bmm(rest_kpts.transpose(1, 2)) * obj_scaling[:, None, None] + obj_trans[:, :, None] # (BS*T) X 3 X 5 
        transformed_obj_kpts = transformed_obj_kpts.transpose(1, 2) # (BS*T) X 5 X 3 

        rest_verts = torch.from_numpy(self.obj_rest_verts).float().to(obj_rot_mat.device)[None].repeat(obj_rot_aa_rep.shape[0], 1, 1) # (BS*T) X Nv X 3 
        transformed_obj_verts = obj_rot_mat.bmm(rest_verts.transpose(1, 2)) * obj_scaling[:, None, None] + obj_trans[:, :, None] # (BS*T) X 3 X Nv 
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # (BS*T) X Nv X 3 

        return transformed_obj_kpts, transformed_obj_verts 

    def init_param(self, skel3d, batch_size=None):
        # skel3d: (BS*T) X 5 X 3, numpy array 
      
        # scale = self.init_scale_batch(skel3d.reshape(batch_size, -1, 5, 3)[:, 0]) # BS, numpy array 

        # orient, trans = self.batch_init_RT(skel3d) # (BS*T) X 3, (BS*T) X 3 

        num_steps = skel3d.shape[0] 

        orient = torch.zeros(num_steps, 3).to(self.device)
        orient = torch.nn.Parameter(orient, requires_grad=True)

        trans = torch.zeros(num_steps, 3).to(self.device)
        trans = torch.nn.Parameter(trans, requires_grad=True) 

        scale = torch.nn.Parameter(torch.ones(batch_size, 1).to(self.device), requires_grad=True) 

        obj_param = {'obj_orient': orient, 'obj_trans': trans, 'obj_scaling': scale}
        # 'scale': float(scale)}
        return obj_param 

    def calc_2d_loss(self, x1, x2):
        loss = torch.nn.functional.mse_loss(x1, x2)
        return loss

    def gen_closure(self, optimizer, obj_param, target_jnts2d, cam_rot_mat, cam_trans, batch_size=None):
        def closure():
            optimizer.zero_grad()
            obj_kpts, _, _ = self.object_forward(obj_param, batch_size=batch_size) 

            _, _, obj_jnts2d = perspective_projection(
                obj_kpts, cam_rot_mat, \
                cam_trans)

            width = 1920 
            height = 1080 
            obj_jnts2d[:, :, 0] = obj_jnts2d[:, :, 0] / width
            obj_jnts2d[:, :, 1] = obj_jnts2d[:, :, 1] / height 
            obj_jnts2d = obj_jnts2d * 2 - 1 

            loss = {}
           
            loss_match = self.calc_2d_loss(obj_jnts2d, target_jnts2d)

            # pose_loss = compute_pose_loss(smpl_body_pose)
            
            # Time smoothness
            # seq_len = target_jnts3d.shape[0]//batch_size 
            # time_loss = compute_time_loss_batch(smpl_body_pose.reshape(batch_size, seq_len, -1, 3)[:, :, 1:])
               
            # loss_smooth = compute_smooth_loss(skel)

            loss = loss_match 
            # print(f"Joint Match loss: {loss_match.item():.4f}")

            # loss = loss_match + loss_smooth * 1e-3 

            # loss = loss_match 

            # loss = loss_match + time_loss * 1e-4 + pose_loss * 1e-4

            # print(f"Joint loss: {loss_match.item():.4f}, \
            #       Time loss: {time_loss.item():.4f}")

            # print(f"Joint loss: {loss_match.item():.4f}, \
            #       Smooth loss: {loss_smooth.item():.4f}")

            loss.backward()
            return loss

        return closure

    def solve(self, obj_param, closure, optimizer, iter_max=1000, \
            iter_thresh=1e-10, loss_thresh=1e-10, batch_size=None):
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
        res_obj_params = {}
        res_obj_params['obj_trans']  = obj_param['obj_trans'].detach().cpu().numpy() # (BS*T) X 3 
        res_obj_params['obj_scaling'] = obj_param['obj_scaling'].detach().cpu().numpy() # BS 
        res_obj_params['obj_orient'] = obj_param['obj_orient'].detach().cpu().numpy() # (BS*T) X 3 

        obj_kpts, obj_verts = self.get_object_verts(obj_param, batch_size=batch_size) # # (BS*T) X Nv X 3  

        return obj_param, res_obj_params, obj_kpts, obj_verts, self.obj_mesh_faces  
