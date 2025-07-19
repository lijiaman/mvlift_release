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

class SmplVPoserFitter():
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

    def triangulate_skel(self, cam_param, skel2d):
        # cam_param['K']: 3 X 3  
        # cam_param['R']: K X 3 X 3 
        # cam_param['T']: K X 3 X 1
        # skel2d: K X J X 2 
        view_size, joint_size = skel2d.shape[:2]
        projs = np.einsum("jk,ikl->ijl", cam_param['K'], np.concatenate([cam_param['R'], cam_param['T']], axis=2)) # K X 3 X 4 
        A = np.zeros([joint_size, view_size * 2, 4])
        for view in range(view_size):
            keypoints = skel2d[view]
            proj = projs[view]
            for jidx in range(joint_size):
                if keypoints[jidx, 2] > 1e-5:
                    A[jidx, view * 2: view * 2 + 2] = keypoints[jidx, :2, np.newaxis] * proj[2] - proj[:2]

        body_skel = np.zeros([joint_size, 3])
        for jidx in range(joint_size):
            u, s, vh = np.linalg.svd(A[jidx], full_matrices=True)
            x = (vh.T)[:, -1]
            body_skel[jidx] = x[:3] / x[-1]

        return body_skel # J X 3 

    def init_scale_batch(self, body_skel):
        # body_skel: BS x J x 3 (J is the number of joints)
        # Define the joints for left and right legs using zero-based indexing
        left_up_leg = 11
        left_leg = 12
        left_foot = 13
        right_up_leg = 8
        right_leg = 9
        right_foot = 10

        # Compute left and right leg lengths for the batch
        left_leg_len = np.linalg.norm(body_skel[:, left_up_leg] - body_skel[:, left_leg], axis=1) + \
                    np.linalg.norm(body_skel[:, left_leg] - body_skel[:, left_foot], axis=1)
        right_leg_len = np.linalg.norm(body_skel[:, right_up_leg] - body_skel[:, right_leg], axis=1) + \
                        np.linalg.norm(body_skel[:, right_leg] - body_skel[:, right_foot], axis=1)

        # Calculate the scale for each sample in the batch
        scale = (left_leg_len + right_leg_len) / 2 / 1.0

        return scale # BS 

    def init_scale(self, body_skel):
        # leftUpLeg 11, leftLeg 12, leftFoot 13
        # rightUpleg 8, rightLeg 9, rightFoot 10  
        left_leg_len = np.linalg.norm(body_skel[11] - body_skel[12]) + \
            np.linalg.norm(body_skel[12] - body_skel[13])
        right_leg_len = np.linalg.norm(body_skel[8] - body_skel[9]) + \
            np.linalg.norm(body_skel[9] - body_skel[10])
        
        scale = ((left_leg_len + right_leg_len) / 2) / 1.0
        return scale

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

    def init_RT(self, body_skel):
        smpl_out = self.smpl()
        joints = smpl_out.joints[0].detach().cpu().numpy() # 45 X 3 
        src = joints[np.array([4, 5, 16, 17])].transpose() # leftLeg, rightLeg, leftArm, rightArm
        tar = body_skel[np.array([12, 9, 5, 2])].transpose()

        # solve RT
        mu1, mu2 = src.mean(axis=1, keepdims=True), tar.mean(axis=1, keepdims=True)
        X1, X2 = src - mu1, tar - mu2

        K = X1.dot(X2.T)
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        R = V.dot(Z.dot(U.T))
        t = mu2 - R.dot(mu1)

        orient, _ = cv2.Rodrigues(R)
        return orient.reshape(3), t.reshape(3)

    def convert_to_axis_angle(self, facing_directions, body_axes):
        # Ensure input is in torch tensor format
        facing_directions = torch.tensor(facing_directions, dtype=torch.float32)
        body_axes = torch.tensor(body_axes, dtype=torch.float32)
            
        # Normalize vectors
        facing_directions = facing_directions / torch.norm(facing_directions, dim=1, keepdim=True)
        body_axes = body_axes / torch.norm(body_axes, dim=1, keepdim=True)
        
        # Compute the right (lateral) direction as cross product of body_axis and facing_direction
        right_directions = torch.cross(facing_directions, body_axes, dim=1)
        right_directions = right_directions / torch.norm(right_directions, dim=1, keepdim=True)
        
        smpl_rest_pose_right = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32)
        smpl_rest_pose_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32) 
        smpl_rest_pose_facing = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32) 
        smpl_rest_mat = torch.stack([smpl_rest_pose_right, smpl_rest_pose_up, smpl_rest_pose_facing], dim=-1)
        smpl_rest_mat = smpl_rest_mat[None].repeat(body_axes.shape[0], 1, 1) # T X 3 X 3 

        # Construct the rotation matrix from right, body (up), and forward vectors
        rotation_matrices = torch.stack([right_directions, body_axes, facing_directions], dim=-1)
        # rotation_matrices = torch.stack([body_axes, right_directions, facing_directions], dim=-1)

        res_rot_mat = torch.matmul(rotation_matrices, smpl_rest_mat.inverse())

        # Convert rotation matrices to axis-angle
        axis_angles = transforms.matrix_to_axis_angle(res_rot_mat)
        
        return axis_angles

    def initialize_orientation(self, jnts3d_18):
        jnts3d_18 = torch.tensor(jnts3d_18, dtype=torch.float32)

        # jnts3d_18: T X 18 X 3 
        left_shoulder = jnts3d_18[:, 5, :]
        right_shoulder = jnts3d_18[:, 2, :]
        left_hip = jnts3d_18[:, 11, :]
        right_hip = jnts3d_18[:, 8, :] 
        
        # Calculate midpoints
        M_shoulder = (left_shoulder + right_shoulder) / 2.0
        M_hip = (left_hip + right_hip) / 2.0
        
        # Calculate vectors
        body_axis = M_shoulder - M_hip
        # hip_span = right_hip - left_hip 
        hip_span = right_shoulder - left_shoulder  
        
        # Calculate the forward direction as perpendicular to the body axis and hip span
        facing_direction = torch.cross(body_axis, hip_span, dim=1)
        
        # Normalize the facing direction and body axis
        facing_direction = facing_direction / torch.norm(facing_direction, dim=1, keepdim=True)
        body_axis = body_axis / torch.norm(body_axis, dim=1, keepdim=True)
        
        # Compute axis-angle representation
        init_aa_rep = self.convert_to_axis_angle(facing_direction, body_axis)
        
        return init_aa_rep

    def batch_init_RT(self, body_skel_sequence):
        # Assuming self.smpl() has been adapted or precomputed to provide a batched output
        smpl_out = self.smpl()  # Assuming output shape is T x 45 x 3
        joints_sequence = smpl_out.joints.repeat(body_skel_sequence.shape[0], 1, 1).detach().cpu().numpy()
        
        # Extract relevant joints for source and target
        # src_indices = np.array([4, 5, 16, 17])
        # tar_indices = np.array([12, 9, 5, 2])
        # tar_indices = np.array([12, 9, 6, 3])

        src_indices = np.array([13, 14, 1, 2]) # Left shoulder, right shoulder, leftUpleg, rightUpleg 
        tar_indices = np.array([5, 2, 11, 8])
        src = joints_sequence[:, src_indices, :].transpose(0, 2, 1)  # T x 3 x 4
        tar = body_skel_sequence[:, tar_indices, :].transpose(0, 2, 1)  # T x 3 x 4

        # Compute centroids
        mu1 = src.mean(axis=2, keepdims=True)
        mu2 = tar.mean(axis=2, keepdims=True)

        # Center the points
        X1 = src - mu1
        X2 = tar - mu2

        # Compute correlation matrix for each batch
        K = np.matmul(X1, X2.transpose(0, 2, 1))

        # Batch SVD for rotation matrices
        U, s, Vh = np.linalg.svd(K, full_matrices=False)
        V = Vh.transpose(0, 2, 1)

        # Ensure proper rotations (handle reflections)
        Z = np.eye(3)[np.newaxis, :, :]
        det = np.linalg.det(np.matmul(U, V.transpose(0, 2, 1)))
        signs = np.sign(det)
        Z = np.tile(Z, (len(signs), 1, 1)) 
        Z[:, -1, -1] = signs 

        # Compute rotations
        R = np.matmul(np.matmul(V, Z), U.transpose(0, 2, 1))

        # Ensure mu1_squeezed is reshaped to [T, 3, 1]
        mu1_squeezed = mu1.squeeze(axis=2)[:, :, np.newaxis]
        mu2_squeezed = mu2.squeeze(axis=2)

        # Correct matrix multiplication
        t = mu2_squeezed - np.matmul(R, mu1_squeezed)[:, :, 0]

        orientations = np.array([cv2.Rodrigues(r)[0] for r in R])
        translations = t.transpose()

        return orientations.reshape(-1, 3), translations.reshape(-1, 3)

    def init_param(self, skel3d, center_all_jnts2d=False, batch_size=None):
        if center_all_jnts2d:
            scale = self.init_scale(skel3d[0])
            scale = np.asarray([scale])
        
            num_steps = skel3d.shape[0]
            fixed_scale_seq = np.zeros(num_steps)
            for t_idx in range(num_steps):
                curr_scale = self.init_scale(skel3d[t_idx])
                fixed_scale_seq[t_idx] = curr_scale 

            fixed_scale_seq /= fixed_scale_seq[0] # Make the first frame to be 1.0 
            
        else:
            # skel3d: T X 18 X 3/ (BS*T) X 18 X 3 

            if batch_size is None:
                scale = self.init_scale(skel3d[0])

                scale = np.asarray([scale])
            else:
                scale = self.init_scale_batch(skel3d.reshape(batch_size, -1, 18, 3)[:, 0]) # BS, numpy array 

            fixed_scale_seq = None 

        # orient, trans = self.init_RT(skel3d[0])  
        orient, trans = self.batch_init_RT(skel3d) 

        # orient = self.initialize_orientation(skel3d) 

        l_hip_index = 8 
        r_hip_index = 11 
        trans = (skel3d[:, l_hip_index, :]+skel3d[:, r_hip_index, :])/2. # T X 3 

        # Convert global orientation to 6D 
        # orient = torch.from_numpy(orient).float() # T X 3 
        # orient = transforms.axis_angle_to_matrix(orient) # T X 3 X 3 
        # orient = transforms.matrix_to_rotation_6d(orient).detach().cpu().numpy() # T X 6 

        smpl_param = {'orient': orient, 'trans': trans, 'latent': np.zeros((skel3d.shape[0], 32)), \
                    'scaling': scale}
        # 'scale': float(scale)}
        # return smpl_param, fixed_scale_seq 
        return smpl_param 

    def calc_3d_loss(self, x1, x2):
        loss = torch.nn.functional.mse_loss(x1, x2)
        return loss

    def gen_closure(self, optimizer, smpl_param, target_jnts3d, batch_size=None, small_time_loss=False):
        def closure():
            optimizer.zero_grad()
            smpl_out, skel, smpl_body_pose, _, _ = self.smpl_forward(smpl_param, batch_size=batch_size) 
            loss = {}
           
            # We should remove the neck joint since this joint is not consistent with SMPL. 
            # smpl2jnts18_idx = [24, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28]
            # loss_match = self.calc_3d_loss(skel[:, smpl2jnts18_idx, :],
            #                         target_jnts3d)
            
            smpl2jnts17_idx = [24, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28]
            target_jnts17_idx = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            loss_match = self.calc_3d_loss(skel[:, smpl2jnts17_idx, :],
                                    target_jnts3d[:, target_jnts17_idx, :])

            # debug_loss = torch.nn.functional.mse_loss(skel[:, smpl2jnts18_idx, :], \
            # target_jnts3d, reduction="none")

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

            # loss_smooth = compute_smooth_loss(skel)

            if small_time_loss:
                loss = loss_match 
            else:
                loss = loss_match + time_loss * 1e-3  # Old 

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
