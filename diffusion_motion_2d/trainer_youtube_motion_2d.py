import argparse
import os
import numpy as np
import yaml
import random
import json 

import trimesh 

from tqdm import tqdm
from pathlib import Path

import wandb

import time 

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

import torch.nn.functional as F

import pytorch3d.transforms as transforms 

from ema_pytorch import EMA
from multiprocessing import cpu_count

from m2d.data.youtube_motion2d_dataset import YoutubeMotion2D, simulate_epipolar_lines
from m2d.data.omomo_2d_dataset import OMOMODataset 
from m2d.data.aist_motion3d_dataset import AISTPose3D, run_smpl_forward, convert_smpl_jnts_to_openpose18   

from m2d.data.utils_2d_pose_normalize import normalize_pose2d, normalize_pose2d_w_grad, de_normalize_pose2d 
from m2d.data.utils_multi_view_2d_motion import extract_smplx2openpose_jnts19, single_seq_get_multi_view_2d_motion_from_3d_torch, cal_min_seq_len_from_mask_torch, get_multi_view_cam_extrinsic

from m2d.model.transformer_motion2d_diffusion_model import CondGaussianDiffusion

from m2d.vis.vis_jnts import plot_3d_motion, gen_2d_motion_vis, plot_3d_motion_new, plot_pose_2d_for_paper, plot_pose_2d_omomo_for_paper, plot_pose_2d_cat_for_paper, visualize_pose_and_lines_for_paper 
from m2d.vis.vis_jnts import visualize_root_trajectory_for_figure, plot_trajectory_components_for_figure, plot_multiple_trajectories

from m2d.lafan1.utils import normalize, quat_normalize, quat_between, quat_mul, quat_inv, quat_mul_vec 

from related_epipolar_lines import calc_fundamental_matrix 

from human_body_prior.body_model.body_model import BodyModel

from matplotlib import pyplot as plt

import pickle as pkl 

import clip 

from evaluation_metrics import compute_metrics_for_jpos, calculate_bone_lengths

from m2d.vis.vis_camera_motion import vis_multiple_head_pose_traj 

from m2d.data.utils_multi_view_2d_motion import perspective_projection

torch.manual_seed(1)
random.seed(1)

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=10000000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=20000,
        results_folder='./results',
        use_wandb=True,      
    ):
        super().__init__()

        self.use_wandb = use_wandb           
        if self.use_wandb:
            # Loggers
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, \
            name=opt.exp_name, dir=opt.save_dir)

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.test_2d_diffusion_w_line_cond = opt.test_2d_diffusion_w_line_cond 
       
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.youtube_train_val_json_path = opt.youtube_train_val_json_path 
        self.youtube_data_npz_folder = opt.youtube_data_npz_folder 

        self.gen_synthetic_3d_data_w_our_model = opt.gen_synthetic_3d_data_w_our_model 

        self.use_animal_data = opt.use_animal_data 

        self.use_omomo_data = opt.use_omomo_data 

        self.omomo_object_name = opt.omomo_object_name 

        self.opt = opt 

        self.window = opt.window

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.optimize_3d_w_trained_2d_diffusion = opt.optimize_3d_w_trained_2d_diffusion 

        if "AIST" in self.youtube_train_val_json_path and self.test_2d_diffusion_w_line_cond:
            self.aist_optimize_3d_w_trained_2d_diffusion = True 
        else:
            self.aist_optimize_3d_w_trained_2d_diffusion = False  

        self.train_2d_diffusion_w_line_cond = opt.train_2d_diffusion_w_line_cond 
       
        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)

        if self.aist_optimize_3d_w_trained_2d_diffusion and \
            self.test_2d_diffusion_w_line_cond:
            if self.gen_synthetic_3d_data_w_our_model:
                self.prep_dataloader()
            else:
                self.prep_aist_3d_dataloader()
        elif self.aist_optimize_3d_w_trained_2d_diffusion:
            self.prep_aist_3d_dataloader() 
        else:
            if self.use_omomo_data:
                self.prep_omomo_dataloader() 
            else:
                self.prep_dataloader()

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.test_on_train = self.opt.test_sample_res_on_train 

        self.eval_w_best_mpjpe = self.opt.eval_w_best_mpjpe 

        self.gen_synthetic_3d_data_for_training = self.opt.gen_synthetic_3d_data_for_training

    def prep_aist_3d_dataloader(self):
        # Define dataset
        val_dataset = AISTPose3D(sample_n_frames=self.window, center_all_jnts2d=False, \
                use_decomposed_traj_rep=False)

        self.val_ds = val_dataset
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, \
            shuffle=False, pin_memory=True, num_workers=4))

    def prep_dataloader(self):
        # Define dataset
        train_youtube_dataset = YoutubeMotion2D(json_path=self.youtube_train_val_json_path, \
            npz_folder=self.youtube_data_npz_folder, \
            sample_n_frames=self.window, train=True, \
            center_all_jnts2d=False, \
            for_eval=self.gen_synthetic_3d_data_w_our_model, use_animal_data=self.use_animal_data)
        
        val_youtube_dataset = YoutubeMotion2D(json_path=self.youtube_train_val_json_path, \
            npz_folder=self.youtube_data_npz_folder, \
            sample_n_frames=self.window, train=False, \
            # for_eval=self.test_2d_diffusion_w_line_cond, \
            center_all_jnts2d=False, \
            for_eval=self.gen_synthetic_3d_data_w_our_model or self.test_2d_diffusion_w_line_cond, \
            use_animal_data=self.use_animal_data)

        self.ds = train_youtube_dataset 
        self.val_ds = val_youtube_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, \
            shuffle=True, pin_memory=True, num_workers=4))

        # if self.test_2d_diffusion_w_line_cond:
        #     self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=1, \
        #         shuffle=False, pin_memory=True, num_workers=4))
        # else:
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, \
            shuffle=False, pin_memory=True, num_workers=4))

    def prep_omomo_dataloader(self):
        # Define dataset
        train_omomo_dataset = OMOMODataset(window=self.window, train=True, object_name=self.omomo_object_name)
        
        val_omomo_dataset = OMOMODataset(window=self.window, train=False, object_name=self.omomo_object_name)

        self.ds = train_omomo_dataset 
        self.val_ds = val_omomo_dataset

        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, \
            shuffle=True, pin_memory=True, num_workers=4))
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, \
            shuffle=False, pin_memory=True, num_workers=4))

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

    def load(self, milestone, pretrained_path=None):
        if pretrained_path is None:
            data = torch.load(os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))
        else:
            data = torch.load(pretrained_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

    def prep_temporal_condition_mask(self, data, t_idx=0):
        # Missing regions are ones, the condition regions are zeros. 
        mask = torch.ones_like(data).to(data.device) # BS X T X D 
        mask[:, t_idx, :] = torch.zeros(data.shape[0], data.shape[2]).to(data.device) # BS X D  

        return mask 

    def prep_multiview_cam_extrinsic(self, num_views=6, add_elevation=False):
        curr_cam_extrinsic_list = get_multi_view_cam_extrinsic(num_views=num_views, \
                                farest=True, add_elevation=add_elevation) 
           
        return curr_cam_extrinsic_list[None]  
        # BS(1) X K X 4 X 4 

    def compute_epi_lines(self, reference_seq_2d, reference_cam_extrinsic, \
        target_cam_extrinsic): 
        target_num_views = target_cam_extrinsic.shape[0]
        seq_len = reference_seq_2d.shape[1]

        if self.use_animal_data:
            num_joints = 17
        elif self.use_omomo_data:
            num_joints = 18 + 5 
        else:
            num_joints = 18 
        reference_seq_2d = reference_seq_2d.reshape(target_num_views, -1, num_joints, 2)
       
        x = de_normalize_pose2d(reference_seq_2d) # (K-1) X T X 18 X 2 

        # Convert 2D points to homogeneous coordinates
        ones = torch.ones(target_num_views, seq_len, num_joints, 1, device=x.device)
        x_hom = torch.cat([x, ones], dim=3)  # [K-1, T, 18, 3]

        # reference_cam_extrinsic # (K-1) X 4 X 4 
        # target_cam_extrinsic  # (K-1) X 4 X 4 

        focal_length = 1000 
        image_width = 1920
        image_height = 1080
        camera_center = torch.tensor([image_width / 2, image_height / 2], device=x.device)

        K = torch.zeros([3, 3], device=x.device)
        K[0, 0] = focal_length
        K[1, 1] = focal_length
        K[2, 2] = 1.
        K[:-1, -1] = camera_center

        abc_list = []
        epipoles = []
        for view_idx in range(target_num_views):
            T_mat = torch.matmul(target_cam_extrinsic[view_idx], \
                torch.inverse(reference_cam_extrinsic[view_idx])) # 4 X 4 

            F_mat = calc_fundamental_matrix(T_mat.detach().cpu().numpy(), \
                    K.detach().cpu().numpy(), \
                    K.detach().cpu().numpy())
            # 3 X 3 

            coord1 = x_hom[view_idx] # T X 18 X 3
            
            F_mat = torch.from_numpy(F_mat).float()[None, \
                None, :, :].repeat(seq_len, num_joints, 1, 1).to(coord1.device) 
            # T X 18 X 3 X 3
            
            abc = torch.matmul(F_mat, coord1[:, :, :, None]).squeeze(-1) # T X 18 X 3

            abc_list.append(abc) 

            # Computing epipoles 
            camera_center_hom = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=x.device)
            epipole_hom = torch.matmul(T_mat, camera_center_hom)
            epipole_img = torch.matmul(K, epipole_hom[:3])  # ignoring last element as it's 1
            epipole_img = epipole_img[:2] / epipole_img[2]  # normalize to image coordinates
            epipoles.append(epipole_img)

        abc_list = torch.stack(abc_list) # (K-1) X T X 18 X 3 

        epipoles = torch.stack(epipoles) # (K-1) X 2 
       
        # # Normalize coefficients such that a^2 + b^2 = 1
        a, b, c = abc_list[..., 0], abc_list[..., 1], abc_list[..., 2]

        # The comuted line is for the image size 1920 X 1080, we need to convert the line parameters for pose in [-1, 1] 
        new_a = (a*image_width)/2. 
        new_b = (b*image_height)/2.
        new_c = (a*image_width)/2. + (b*image_height)/2. + c 

        norm = torch.sqrt(new_a**2 + new_b**2) + 1e-8 
        new_a, new_b, new_c = new_a / norm, new_b / norm, new_c / norm

        # Ensure a is positive
        new_abc = torch.where(new_a.unsqueeze(-1) < 0, -torch.stack((new_a, new_b, new_c), dim=-1), \
            torch.stack((new_a, new_b, new_c), dim=-1))

        # Convert pixel coordinates to normalized range [-1, 1]
        epipoles_normalized = torch.zeros_like(epipoles)
        epipoles_normalized[:, 0] = (epipoles[:, 0] - image_width / 2) / (image_width / 2)
        epipoles_normalized[:, 1] = (epipoles[:, 1] - image_height / 2) / (image_height / 2)

        return new_abc, epipoles_normalized  # Returns (K-1) x T x J x 3, (K-1) X 2 

    def prep_evaluation_metric_list(self):
        self.mpjpe_list = []
        self.pa_mpjpe_list = []
        self.trans_list = []
        self.mpjpe_w_trans_list = [] # Directly global joint position error. 

        self.jpos2d_err_list = [] 
        self.centered_jpos2d_err_list = [] 

        self.bone_consistency_list = [] 

        # Prepare for WHAM evaluation 
        self.wham_mpjpe_list = []
        self.wham_pa_mpjpe_list = []
        self.wham_trans_list = []
        self.wham_mpjpe_w_trans_list = [] # Directly global joint position error. 

        self.wham_jpos2d_err_list = [] 
        self.wham_centered_jpos2d_err_list = [] 

        self.wham_bone_consistency_list = [] 

        if self.opt.add_motionbert_for_eval:
            # Prepare for MotionBERT evaluation 
            self.mbert_mpjpe_list = []
            self.mbert_pa_mpjpe_list = []
            self.mbert_trans_list = []
            self.mbert_mpjpe_w_trans_list = [] # Directly global joint position error. 

            self.mbert_jpos2d_err_list = [] 
            self.mbert_centered_jpos2d_err_list = [] 

            self.mbert_bone_consistency_list = [] 

    def add_new_evaluation_to_list(self, mpjpe, pa_mpjpe, trans_err, jpos_err, jpos2d_err, centered_jpos2d_err, bone_consistency, \
        mpjpe_wham=0, pa_mpjpe_wham=0, trans_err_wham=0, jpos_err_wham=0, jpos2d_err_wham=0, centered_jpos2d_err_wham=0, bone_consistency_wham=0, \
        mpjpe_mbert=0, pa_mpjpe_mbert=0, trans_err_mbert=0, jpos_err_mbert=0, jpos2d_err_mbert=0, centered_jpos2d_err_mbert=0, bone_consistency_mbert=0, \
        mpjpe_elepose=0, pa_mpjpe_elepose=0, trans_err_elepose=0, jpos_err_elepose=0, jpos2d_err_elepose=0, centered_jpos2d_err_elepose=0, bone_consistency_elepose=0, \
        mpjpe_smplify=0, pa_mpjpe_smplify=0, trans_err_smplify=0, jpos_err_smplify=0, jpos2d_err_smplify=0, centered_jpos2d_err_smplify=0, bone_consistency_smplify=0, \
        mpjpe_mas=0, pa_mpjpe_mas=0, trans_err_mas=0, jpos_err_mas=0, jpos2d_err_mas=0, centered_jpos2d_err_mas=0, bone_consistency_mas=0, \
        ):
        
        self.mpjpe_list.append(mpjpe)
        self.pa_mpjpe_list.append(pa_mpjpe) 
        self.trans_list.append(trans_err)
        self.mpjpe_w_trans_list.append(jpos_err)

        self.jpos2d_err_list.append(jpos2d_err)
        self.centered_jpos2d_err_list.append(centered_jpos2d_err)
       
        self.bone_consistency_list.append(bone_consistency) 

        # Prepare for WHAM evaluation
        self.wham_mpjpe_list.append(mpjpe_wham)
        self.wham_pa_mpjpe_list.append(pa_mpjpe_wham)
        self.wham_trans_list.append(trans_err_wham)
        self.wham_mpjpe_w_trans_list.append(jpos_err_wham)

        self.wham_jpos2d_err_list.append(jpos2d_err_wham)
        self.wham_centered_jpos2d_err_list.append(centered_jpos2d_err_wham) 

        self.wham_bone_consistency_list.append(bone_consistency_wham) 

    def print_metrics_for_each_method(self, mpjpe_list, pa_mpjpe_list, trans_list, mpjpe_w_trans_list, \
            jpos2d_err_list, centered_jpos2d_err_list, bone_consistency_list, method_name):
        mpjpe_arr = np.asarray(mpjpe_list)
        pa_mpjpe_arr = np.asarray(pa_mpjpe_list)
        trans_arr = np.asarray(trans_list)
        mpjpe_w_trans_arr = np.asarray(mpjpe_w_trans_list)

        jpos2d_err_arr = np.asarray(jpos2d_err_list)
        centered_jpos2d_err_arr = np.asarray(centered_jpos2d_err_list)

        bone_consistency_arr = np.asarray(bone_consistency_list) 

        print("**************"+method_name+" Evaluation******************")
        print("The number of sequences: {0}".format(len(mpjpe_list)))
        print("MPJPE: {0}".format(mpjpe_arr.mean()))
        print("PA-MPJPE: {0}".format(pa_mpjpe_arr.mean()))
        print("Root Trans Err: {0}".format(trans_arr.mean()))
        print("MPJPE w Trans Err: {0}".format(mpjpe_w_trans_arr.mean()))
        print("Jpos2D Err: {0}".format(jpos2d_err_arr.mean()))
        print("Centered Jpos2D Err: {0}".format(centered_jpos2d_err_arr.mean()))
        print("Bone Consistency: {0}".format(bone_consistency_arr.mean()))

    def print_mean_metrics(self):
        self.print_metrics_for_each_method(self.mpjpe_list, self.pa_mpjpe_list, self.trans_list, self.mpjpe_w_trans_list, \
            self.jpos2d_err_list, self.centered_jpos2d_err_list, self.bone_consistency_list, "Ours")

    def align_init_facing_direction(self, gt_smpl_poses, pred_smpl_poses, \
        pred_smpl_trans):
        # gt_smpl_poses: T X 24 X 3
        # pred_smpl_poses: T X 72 
        # pred_smpl_trans: T X 3 

        # gt_smpl_poses = gt_smpl_poses.squeeze(0)

        if not torch.is_tensor(gt_smpl_poses):
            gt_smpl_poses = torch.from_numpy(gt_smpl_poses).float() 

        if gt_smpl_poses.shape[-1] == 72:
            gt_smpl_poses = gt_smpl_poses.reshape(-1, 24, 3) 

        if not torch.is_tensor(pred_smpl_poses):
            pred_smpl_poses = torch.from_numpy(pred_smpl_poses).float()
            pred_smpl_trans = torch.from_numpy(pred_smpl_trans).float() 

        if pred_smpl_poses.shape[-1] == 72:
            pred_smpl_poses = pred_smpl_poses.reshape(-1, 24, 3) # T X 24 X 3 

        pred_global_rot_aa_rep = pred_smpl_poses[:, 0, :] # T X 3 
        pred_global_rot_mat = transforms.axis_angle_to_matrix(pred_global_rot_aa_rep) # T X 3 X 3 
        
        # Then, apply rotation to make the facing diretion of the first frame consistent 
        pred_global_rot_quat = transforms.matrix_to_quaternion(pred_global_rot_mat) # T X 4 
        key_glob_Q = pred_global_rot_quat[0:1][:, None, None, :].detach().cpu().numpy() # 1 X 1 X 1 X 4 

        pred_forward = np.array([1, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :] * quat_mul_vec(
            key_glob_Q, np.array([1, 0, 0])[np.newaxis, np.newaxis, np.newaxis, :]
        ) # In rest pose, x direction is the body left direction, root joint point to left hip joint.  

        pred_forward = normalize(pred_forward)

        # Get GT facing direction 
        gt_smpl_global_rot_aa_rep = gt_smpl_poses[:, 0, :] 
        gt_smpl_global_rot_mat = transforms.axis_angle_to_matrix(gt_smpl_global_rot_aa_rep) # T X 3 X 3 
       
        gt_smpl_global_rot_quat = transforms.matrix_to_quaternion(gt_smpl_global_rot_mat) # T X 4 
        gt_key_glob_Q = gt_smpl_global_rot_quat[0:1][:, None, None, :].detach().cpu().numpy() # 1 X 1 X 1 X 4 

        gt_forward = np.array([1, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :] * quat_mul_vec(
            gt_key_glob_Q, np.array([1, 0, 0])[np.newaxis, np.newaxis, np.newaxis, :]
        ) # In rest pose, x direction is the body left direction, root joint point to left hip joint.  

        gt_forward = normalize(gt_forward)

        yrot = quat_normalize(quat_between(gt_forward, pred_forward))[0, 0] # 1 X 4 

        pred_global_rot_quat = quat_mul(quat_inv(yrot), pred_global_rot_quat) # T X 4 
        pred_smpl_trans = quat_mul_vec(quat_inv(yrot), pred_smpl_trans) # T X 3
       
        # Replace the global rotation with new global orientation in WHAM 
        pred_global_rot_quat = torch.from_numpy(pred_global_rot_quat).float() 
        new_pred_rot_mat = transforms.quaternion_to_matrix(pred_global_rot_quat) 
        new_pred_rot_aa_rep = transforms.matrix_to_axis_angle(new_pred_rot_mat)

        # Update wham result 
        pred_smpl_poses[:, 0, :] = new_pred_rot_aa_rep 

        # Run forward to get SMPL joints again. 
        smpl_scaling = np.asarray([1.])
        pred_jnts3d_smpl, pred_verts_smpl = run_smpl_forward(pred_smpl_poses.detach().cpu().numpy(), \
                        pred_smpl_trans.detach().cpu().numpy(), smpl_scaling, self.val_ds.smpl) # T X 45 X 3 

        pred_jnts3d_18 = convert_smpl_jnts_to_openpose18(pred_jnts3d_smpl) # T X 18 X 3 

        pred_floor_z = pred_verts_smpl[0, :, 2].min()  
        pred_init_trans = pred_smpl_trans[0:1, :].clone() # 1 X 3 
        pred_init_trans[:, 2] = pred_floor_z 
        
        pred_smpl_trans -= pred_init_trans 

        return pred_smpl_poses, pred_smpl_trans

    def move_smpl_seq_to_floor(self, pred_smpl_poses, pred_smpl_trans):
        smpl_scaling = np.asarray([1.])

        pred_jnts3d_smpl, pred_verts_smpl = run_smpl_forward(pred_smpl_poses, \
                        pred_smpl_trans, smpl_scaling, self.val_ds.smpl) # T X 45 X 3 

        pred_smpl_poses = torch.from_numpy(pred_smpl_poses).float().to(pred_verts_smpl.device)
        pred_smpl_trans = torch.from_numpy(pred_smpl_trans).float().to(pred_verts_smpl.device)

        pred_floor_z = pred_verts_smpl[0, :, 2].min()  
        pred_init_trans = pred_smpl_trans[0:1, :].clone() # 1 X 3 
        pred_init_trans[:, 2] = pred_floor_z 
        
        pred_smpl_trans -= pred_init_trans 

        return pred_smpl_poses.detach().cpu().numpy(), pred_smpl_trans.detach().cpu().numpy() 
    
    def get_all_line_conditions_from_val_data(self, val_data, cam_extrinsic): 
        # If use decomposed rep, val_data: BS X K X T X (18+1) X 2 

        # val_data: BS X K X T X 18 X 2 
        # cam_extrinsic: BS X K X 4 X 4 

        batch, num_views, num_steps, num_joints, _ = val_data.shape 

        simulated_lines_for_input2d = [] 
        for view_idx in range(1):
        # for view_idx in range(val_data.shape[1]):
            curr_simulated_line, _ = simulate_epipolar_lines(val_data[:, 0].reshape(batch*num_steps, num_joints, 2), vp_bound=1) # (BS*T) X 18 X 3
            
            simulated_lines_for_input2d.append(curr_simulated_line)

        simulated_lines_for_input2d = torch.stack(simulated_lines_for_input2d) # 1 X (BS*T) X 18 X 3 

        # K' X T X J X 3 
        num_lines_for_ref = simulated_lines_for_input2d.shape[0] 

        # Padding val_data 
        val_data = torch.cat((val_data[:, 0:1].repeat(1, num_lines_for_ref, 1, 1, 1), \
            val_data[:, 1:]), dim=1) # BS X (K'+K-1) X T X J X 2 
        val_data = val_data.cuda() 
                    
        val_bs, val_num_views, val_num_steps, _, _ = val_data.shape 

        # Padding cam
        cam_extrinsic = torch.cat((cam_extrinsic[:, 0:1, :, :].repeat(1, num_lines_for_ref, 1, 1), \
            cam_extrinsic[:, 1:, :, :]), dim=1) # BS X (K'+K-1) X 4 X 4 
        cam_extrinsic = cam_extrinsic.reshape(val_bs, val_num_views, 4, 4).to(val_data.device)
        gen_num_views = val_num_views - num_lines_for_ref 
        
        reference_cam_extrinsic = cam_extrinsic[:, 0:1].repeat(1, gen_num_views, 1, 1) # BS X K' X 4 X 4
        target_cam_extrinsic = cam_extrinsic[:, num_lines_for_ref:] # BS X (K-1) X 4 X 4 

        val_data = val_data.reshape(val_bs, val_num_views, val_num_steps, -1) 
        # BS X K X T X D(18*2)

        reference_seq_2d = val_data[:, 0:1].repeat(1, gen_num_views, 1, 1) # BS X (K-1) X T X (18*2)

        # Compute epipolar lines based on the reference 2d sequence and all the cam extrinsics 
        epi_line_conditions, epipoles = self.compute_epi_lines(reference_seq_2d.reshape(-1, num_steps, num_joints*2), \
                reference_cam_extrinsic.reshape(-1, 4, 4), target_cam_extrinsic.reshape(-1, 4, 4))
        # (BS*(K-1)) X T X J X 3 

        all_line_conditions = torch.cat((simulated_lines_for_input2d.to(epi_line_conditions.device).reshape(-1, batch, \
                        num_steps, num_joints, 3).transpose(0, 1), \
                        epi_line_conditions.reshape(batch, -1, num_steps, num_joints, 3)), dim=1) # BS X (K'+(K-1)) X T X J X 3 
    
        
        return val_data, all_line_conditions, val_num_views, val_bs, num_lines_for_ref, val_num_steps, cam_extrinsic   
        # val_data: BS X K X T X (18*2), all_line_conditions: BS X K X T X 18 X 3, cam_extrinsic: BS X K X 4 X 4 

    def move_init3d_to_center(self, jpos3d_18):
        # jpos3d_18: T X 18 X 3 
        l_hip_idx = 8 
        r_hip_idx = 11

        init_jpos3d = (jpos3d_18[0:1, l_hip_idx:l_hip_idx+1, :] + jpos3d_18[0:1, r_hip_idx:r_hip_idx+1, :])/2. # 1 X 1 X 3 
        jpos3d_18 -= init_jpos3d 

        return jpos3d_18 

    def center_jnts3d_seq(self, jpos3d_18):
        # jpos3d_18: T X 18 X 3 
        l_hip_idx = 8 
        r_hip_idx = 11

        if torch.is_tensor(jpos3d_18):
            new_jpos3d_18 = jpos3d_18.clone()
        else:
            new_jpos3d_18 = jpos3d_18.copy() 

        init_jpos3d = (jpos3d_18[:, l_hip_idx:l_hip_idx+1, :] + jpos3d_18[:, r_hip_idx:r_hip_idx+1, :])/2. # T X 1 X 3 
        new_jpos3d_18 -= init_jpos3d 

        return new_jpos3d_18 

    def run_optimization_w_learned_model(self, val_data_dict, ema_model):
        # For ablation study on AIST 
        if self.aist_optimize_3d_w_trained_2d_diffusion:
            gt_3d_jnts3d = val_data_dict['smplx_jnts18'].float().cuda() # BS X T X 18 X 3 

            batch = gt_3d_jnts3d.shape[0]
          
            cam_extrinsic = self.prep_multiview_cam_extrinsic(num_views=6) # 1 X K X 4 X 4 
            cam_extrinsic = cam_extrinsic.repeat(batch, 1, 1, 1) # BS X K X 4 X 4 

            ori_val_data = val_data_dict['normalized_jnts2d'].float().cuda() # BS X T X 18 X 2 
            ori_val_data = ori_val_data[:, None, :, :, :].repeat(1, cam_extrinsic.shape[1], 1, 1, 1) # BS X K X T X 18 X 2 

        val_data, all_line_conditions, val_num_views, val_bs, num_lines_for_ref, val_num_steps, cam_extrinsic = \
            self.get_all_line_conditions_from_val_data(ori_val_data, cam_extrinsic)
        # val_data: BS X K X T X (18*2), all_line_conditions: BS X K X T X 18 X 3, cam_extrinsic: BS X K X 4 X 4 
    
        cond_mask = None  

        # Generate padding mask 
        actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
        actual_seq_len = actual_seq_len[:, None].repeat(1, val_num_views) # BS X K 
        actual_seq_len = actual_seq_len.reshape(-1) # (BS*K)

        tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0]*val_data.shape[1], \
        self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
        # (BS*K) X T 
        padding_mask = tmp_mask[:, None, :].to(val_data.device) # (BS*K) X 1 X T 

        language_input = None 

        curr_seq_len = int(actual_seq_len.detach().cpu().numpy()[0])-1
        if cond_mask is None:
            curr_cond_mask = None 
        else:
            curr_cond_mask = cond_mask 

        ref_2d_seq_cond = None 
       
        if self.opt.gen_more_consistent_2d_seq: # Ablation 1 
            # Run SDS optimization to get more consistent 2D pose sequences 
            start_time = time.time()
            opt_paired_jnts2d_seq = ema_model.opt_paired_2d_w_sds_loss_and_jnts2line_dist_loss(val_data, \
                                cam_extrinsic, cond_mask=curr_cond_mask, \
                                padding_mask=padding_mask, \
                                input_line_cond=all_line_conditions, \
                                compute_epi_lines_func=self.compute_epi_lines, \
                                init_jnts2d_seq=None)  
                                # init_jnts2d_seq=init_epi_line_cond_seq_2d.detach()[:, 1:])  
            # BS X (K-1) X T X (18*2) 

            # print("Multi-View 2D Optimization time:{0}".format(time.time()-start_time))

            direct_sampled_epi_seq_2d = torch.cat((val_data[:, 0:1], opt_paired_jnts2d_seq.clone()), dim=1) # BS X K X T X (18*2)
            epi_line_cond_seq_2d = torch.cat((val_data[:, 0:1], opt_paired_jnts2d_seq.clone()), dim=1) # BS X K X T X (18*2)
        else: # Ablation 2 
            epi_line_cond_seq_2d = ema_model.sample(val_data.reshape(val_bs*val_num_views, val_num_steps, -1), \
                        curr_cond_mask, padding_mask, \
                        input_line_cond=all_line_conditions.reshape(val_bs*val_num_views, val_num_steps, -1))
            # (BS*K) X T X 36 

            epi_line_cond_seq_2d = epi_line_cond_seq_2d.reshape(val_bs, val_num_views, val_num_steps, -1) # BS X K X T X 36 

            direct_sampled_epi_seq_2d = torch.cat((val_data[:, 0:1], epi_line_cond_seq_2d.clone()[:, 1:]), dim=1)
            epi_line_cond_seq_2d = torch.cat((val_data[:, 0:1], epi_line_cond_seq_2d.clone()[:, 1:]), dim=1) 
       
        curr_epi_line_cond_seq_2d = epi_line_cond_seq_2d # BS X K X T X 36 
       
        epi_line_cond_seq_2d, opt_jpos3d_18, view_mask = \
                    ema_model.opt_3d_w_multi_view_2d_res(curr_epi_line_cond_seq_2d, \
                    cam_extrinsic, padding_mask=padding_mask, actual_seq_len=curr_seq_len, \
                    input_line_cond=all_line_conditions, \
                    x_start=val_data, cond_mask=curr_cond_mask, \
                    language_input=language_input)

        # BS X K X T X (18*2), BS X T X 18 X 3, BS X T X 18 X 3 
            
        pred_smpl_jnts18_list = [] 
        gt_smpl_jnts18_list = [] 

        pred_verts_smpl_list = []
        gt_verts_smpl_list = [] 
        
        scale_val_list = []
        
        ori_opt_skel_seq_list, opt_skel_seq_list, opt_jpos3d_18_list, skel_faces, scale_val_list = \
                    ema_model.opt_3d_w_vposer_w_joints3d_input(opt_jpos3d_18.detach().cpu().numpy())

        for bs_idx in range(batch):

            curr_opt_skel_seq = opt_skel_seq_list[bs_idx] 
            curr_opt_jpos3d_18 = opt_jpos3d_18_list[bs_idx] 
            curr_scale_val = scale_val_list[bs_idx] 

            if self.aist_optimize_3d_w_trained_2d_diffusion:
                smpl_scaling = torch.ones(1).detach().cpu().numpy() # 1

                gt_smpl_poses = val_data_dict['smpl_poses'][bs_idx] # T X 24 X 3 
                gt_smpl_trans = val_data_dict['smpl_trans'][bs_idx] # T X 3 

                pred_smpl_scaling = curr_opt_skel_seq['smpl_scaling'] # 1 
                pred_smpl_poses = curr_opt_skel_seq['smpl_poses'] # T X 24 X 3 
                pred_smpl_trans = curr_opt_skel_seq['smpl_trans'] # T X 3 

                # Align GT results to our orientation 
                gt_smpl_poses, gt_smpl_trans = \
                    self.align_init_facing_direction(pred_smpl_poses, gt_smpl_poses, \
                    gt_smpl_trans) 
                gt_jnts3d_smpl, gt_verts_smpl = run_smpl_forward(gt_smpl_poses.detach().cpu().numpy(), \
                        gt_smpl_trans.detach().cpu().numpy(), smpl_scaling, self.val_ds.smpl) 
                # T X 45 X 3, T X Nv X 3  
                gt_smpl_jnts18 = convert_smpl_jnts_to_openpose18(gt_jnts3d_smpl)

                pred_smpl_poses, pred_smpl_trans = self.move_smpl_seq_to_floor(pred_smpl_poses, pred_smpl_trans)
                pred_jnts3d_smpl, pred_verts_smpl = run_smpl_forward(pred_smpl_poses, \
                        pred_smpl_trans, pred_smpl_scaling, self.val_ds.smpl) 
                # T X 45 X 3, T X Nv X 3  
                pred_smpl_jnts18 = convert_smpl_jnts_to_openpose18(pred_jnts3d_smpl)

                # Move the first frame's joints3D root to the center
                gt_smpl_jnts18 = self.move_init3d_to_center(gt_smpl_jnts18)
                pred_smpl_jnts18 = self.move_init3d_to_center(pred_smpl_jnts18) 

            pred_smpl_jnts18_list.append(pred_smpl_jnts18)
            gt_smpl_jnts18_list.append(gt_smpl_jnts18)

            pred_verts_smpl_list.append(pred_verts_smpl)
            gt_verts_smpl_list.append(gt_verts_smpl)
            
            scale_val_list.append(curr_scale_val) # The scale of optimized 3D joint positions compared to vposer fitting results 

        return direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, \
            val_bs, val_num_views, val_num_steps, val_data, actual_seq_len, num_lines_for_ref, \
            pred_smpl_jnts18_list, gt_smpl_jnts18_list, curr_seq_len, opt_jpos3d_18_list, \
            pred_verts_smpl_list, gt_verts_smpl_list, \
            skel_faces, cam_extrinsic, scale_val_list, all_line_conditions, \
            ori_opt_skel_seq_list

    def run_optimization_w_learned_model_ablation(self, val_data_dict, ema_model):
        # For ablation study on AIST 
        if self.aist_optimize_3d_w_trained_2d_diffusion:
            gt_3d_jnts3d = val_data_dict['smplx_jnts18'].float().cuda() # BS X T X 18 X 3 

            batch = gt_3d_jnts3d.shape[0]
          
            cam_extrinsic = self.prep_multiview_cam_extrinsic(num_views=6) # 1 X K X 4 X 4 
            cam_extrinsic = cam_extrinsic.repeat(batch, 1, 1, 1) # BS X K X 4 X 4 

            ori_val_data = val_data_dict['normalized_jnts2d'].float().cuda() # BS X T X 18 X 2 
            ori_val_data = ori_val_data[:, None, :, :, :].repeat(1, cam_extrinsic.shape[1], 1, 1, 1) # BS X K X T X 18 X 2 

        val_data, all_line_conditions, val_num_views, val_bs, num_lines_for_ref, val_num_steps, cam_extrinsic = \
            self.get_all_line_conditions_from_val_data(ori_val_data, cam_extrinsic)
        # val_data: BS X K X T X (18*2), all_line_conditions: BS X K X T X 18 X 3, cam_extrinsic: BS X K X 4 X 4 
    
        cond_mask = None  

        # Generate padding mask 
        actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
        actual_seq_len = actual_seq_len[:, None].repeat(1, val_num_views) # BS X K 
        actual_seq_len = actual_seq_len.reshape(-1) # (BS*K)

        tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0]*val_data.shape[1], \
        self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
        # (BS*K) X T 
        padding_mask = tmp_mask[:, None, :].to(val_data.device) # (BS*K) X 1 X T 

        language_input = None 

        curr_seq_len = int(actual_seq_len.detach().cpu().numpy()[0])-1
        if cond_mask is None:
            curr_cond_mask = None 
        else:
            curr_cond_mask = cond_mask 

        ref_2d_seq_cond = None 

        if self.opt.test_2d_diffusion_w_line_cond:
            epi_line_cond_seq_2d, opt_jpos3d_18 = ema_model.opt_3d_by_multi_step_recon_loss(val_data.squeeze(0), cam_extrinsic.squeeze(0), \
            input_line_cond=all_line_conditions.squeeze(0), cond_mask=curr_cond_mask, padding_mask=padding_mask, use_sds_loss=self.opt.ablation_sds_loss_only)
        else:
            epi_line_cond_seq_2d, opt_jpos3d_18 = ema_model.opt_3d_by_multi_step_recon_loss(val_data.squeeze(0), cam_extrinsic.squeeze(0), \
            input_line_cond=None, cond_mask=curr_cond_mask, padding_mask=padding_mask, use_sds_loss=self.opt.ablation_sds_loss_only)

        curr_epi_line_cond_seq_2d = epi_line_cond_seq_2d # BS X K X T X 36 

        # BS X K X T X (18*2), BS X T X 18 X 3, BS X T X 18 X 3 
            
        pred_smpl_jnts18_list = [] 
        gt_smpl_jnts18_list = [] 

        pred_verts_smpl_list = []
        gt_verts_smpl_list = [] 
        
        scale_val_list = []
        
        ori_opt_skel_seq_list, opt_skel_seq_list, opt_jpos3d_18_list, skel_faces, scale_val_list = \
                    ema_model.opt_3d_w_vposer_w_joints3d_input(opt_jpos3d_18[None].detach().cpu().numpy())

        for bs_idx in range(batch):

            curr_opt_skel_seq = opt_skel_seq_list[bs_idx] 
            curr_opt_jpos3d_18 = opt_jpos3d_18_list[bs_idx] 
            curr_scale_val = scale_val_list[bs_idx] 

            if self.aist_optimize_3d_w_trained_2d_diffusion:
                smpl_scaling = torch.ones(1).detach().cpu().numpy() # 1

                gt_smpl_poses = val_data_dict['smpl_poses'][bs_idx] # T X 24 X 3 
                gt_smpl_trans = val_data_dict['smpl_trans'][bs_idx] # T X 3 

                pred_smpl_scaling = curr_opt_skel_seq['smpl_scaling'] # 1 
                pred_smpl_poses = curr_opt_skel_seq['smpl_poses'] # T X 24 X 3 
                pred_smpl_trans = curr_opt_skel_seq['smpl_trans'] # T X 3 

                # Align GT results to our orientation 
                gt_smpl_poses, gt_smpl_trans = \
                    self.align_init_facing_direction(pred_smpl_poses, gt_smpl_poses, \
                    gt_smpl_trans) 
                gt_jnts3d_smpl, gt_verts_smpl = run_smpl_forward(gt_smpl_poses.detach().cpu().numpy(), \
                        gt_smpl_trans.detach().cpu().numpy(), smpl_scaling, self.val_ds.smpl) 
                # T X 45 X 3, T X Nv X 3  
                gt_smpl_jnts18 = convert_smpl_jnts_to_openpose18(gt_jnts3d_smpl)

                pred_smpl_poses, pred_smpl_trans = self.move_smpl_seq_to_floor(pred_smpl_poses, pred_smpl_trans)
                pred_jnts3d_smpl, pred_verts_smpl = run_smpl_forward(pred_smpl_poses, \
                        pred_smpl_trans, pred_smpl_scaling, self.val_ds.smpl) 
                # T X 45 X 3, T X Nv X 3  
                pred_smpl_jnts18 = convert_smpl_jnts_to_openpose18(pred_jnts3d_smpl)

                # Move the first frame's joints3D root to the center
                gt_smpl_jnts18 = self.move_init3d_to_center(gt_smpl_jnts18)
                pred_smpl_jnts18 = self.move_init3d_to_center(pred_smpl_jnts18) 

            pred_smpl_jnts18_list.append(pred_smpl_jnts18)
            gt_smpl_jnts18_list.append(gt_smpl_jnts18)

            pred_verts_smpl_list.append(pred_verts_smpl)
            gt_verts_smpl_list.append(gt_verts_smpl)
            
            scale_val_list.append(curr_scale_val) # The scale of optimized 3D joint positions compared to vposer fitting results 

        direct_sampled_epi_seq_2d = epi_line_cond_seq_2d 

        return direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, \
            val_bs, val_num_views, val_num_steps, val_data, actual_seq_len, num_lines_for_ref, \
            pred_smpl_jnts18_list, gt_smpl_jnts18_list, curr_seq_len, opt_jpos3d_18_list, \
            pred_verts_smpl_list, gt_verts_smpl_list, \
            skel_faces, cam_extrinsic, scale_val_list, all_line_conditions, \
            ori_opt_skel_seq_list
    
    def run_optimization_w_learned_model_for_synthetic_data_gen(self, val_data_dict, ema_model, input_first_human_pose=False, \
                                    init_3d_jnts3d=None):
        ori_val_data = val_data_dict['normalized_jnts2d'].float().cuda() # BS X T X 18 X 2 
        
        batch = ori_val_data.shape[0] 

        cam_extrinsic = self.prep_multiview_cam_extrinsic(num_views=6) # 1 X K X 4 X 4 
        cam_extrinsic = cam_extrinsic.repeat(batch, 1, 1, 1) # BS X K X 4 X 4 

        ori_val_data = ori_val_data[:, None, :, :, :].repeat(1, cam_extrinsic.shape[1], 1, 1, 1) # BS X K X T X 18 X 2 

        val_data, all_line_conditions, val_num_views, val_bs, num_lines_for_ref, val_num_steps, cam_extrinsic = \
            self.get_all_line_conditions_from_val_data(ori_val_data, cam_extrinsic)
        # val_data: BS X K X T X (18*2), all_line_conditions: BS X K X T X 18 X 3, cam_extrinsic: BS X K X 4 X 4 
    
        cond_mask = None  
       
        # Generate padding mask 
        actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
        actual_seq_len = actual_seq_len[:, None].repeat(1, val_num_views) # BS X K 
        actual_seq_len = actual_seq_len.reshape(-1) # (BS*K)

        tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0]*val_data.shape[1], \
        self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
        # (BS*K) X T 
        padding_mask = tmp_mask[:, None, :].to(val_data.device) # (BS*K) X 1 X T 

        language_input = None 

        curr_seq_len = int(actual_seq_len.detach().cpu().numpy()[0])-1
        if cond_mask is None:
            curr_cond_mask = None 
        else:
            curr_cond_mask = cond_mask 
       
        if self.opt.gen_more_consistent_2d_seq:
            # Run SDS optimization to get more consistent 2D pose sequences 
            opt_paired_jnts2d_seq = ema_model.opt_paired_2d_w_sds_loss_and_jnts2line_dist_loss(val_data, \
                                cam_extrinsic, cond_mask=curr_cond_mask, \
                                padding_mask=padding_mask, \
                                input_line_cond=all_line_conditions, \
                                compute_epi_lines_func=self.compute_epi_lines, \
                                init_jnts2d_seq=None)  
                                # init_jnts2d_seq=init_epi_line_cond_seq_2d.detach()[:, 1:])  
            # BS X (K-1) X T X (18*2) 

            direct_sampled_epi_seq_2d = torch.cat((val_data[:, 0:1], opt_paired_jnts2d_seq.clone()), dim=1) # BS X K X T X (18*2)
            epi_line_cond_seq_2d = torch.cat((val_data[:, 0:1], opt_paired_jnts2d_seq.clone()), dim=1) # BS X K X T X (18*2)
        else:
            # val_data: BS X K X T X 36, all_line_conditions: BS X K X T X 18 X 3 
            epi_line_cond_seq_2d = ema_model.sample(val_data.reshape(val_bs*val_num_views, val_num_steps, -1), \
                        curr_cond_mask, padding_mask, \
                        input_line_cond=all_line_conditions.reshape(val_bs*val_num_views, val_num_steps, -1))
            # (BS*K) X T X 36 

            epi_line_cond_seq_2d = epi_line_cond_seq_2d.reshape(val_bs, val_num_views, val_num_steps, -1) # BS X K X T X 36 

            direct_sampled_epi_seq_2d = torch.cat((val_data[:, 0:1], epi_line_cond_seq_2d.clone()[:, 1:]), dim=1)
            epi_line_cond_seq_2d = torch.cat((val_data[:, 0:1], epi_line_cond_seq_2d.clone()[:, 1:]), dim=1) 

        good_views_idx_list = None # Seems not help.  
       
        epi_line_cond_seq_2d, opt_jpos3d_18, view_mask = \
                    ema_model.opt_3d_w_multi_view_2d_res(epi_line_cond_seq_2d, \
                    cam_extrinsic, padding_mask=padding_mask, actual_seq_len=curr_seq_len, \
                    input_line_cond=all_line_conditions, \
                    x_start=val_data, cond_mask=curr_cond_mask, \
                    language_input=language_input)
        # BS X K X T X (18*2), BS X T X 18 X 3, BS X T X 18 X 3   

        pred_smpl_jnts18_list = [] 
        opt_jpos3d_18_list = [] # Before using vposer optimization 

        pred_verts_smpl_list = []
        
        scale_val_list = []
        
        ori_opt_skel_seq_list = [] 

        if self.use_animal_data:
            opt_skel_seq_list, skel_faces, smal_scale_val_list, smal_opt_jnts3d, smal_opt_verts  = \
                    ema_model.opt_3d_w_smal_animal_w_joints3d_input(opt_jpos3d_18.detach().cpu().numpy())

            return direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, opt_jpos3d_18, cam_extrinsic, \
                val_bs, val_num_views, val_num_steps, val_data, actual_seq_len, num_lines_for_ref, \
                opt_skel_seq_list, skel_faces, smal_scale_val_list, smal_opt_jnts3d, smal_opt_verts  

        ori_opt_skel_seq_list, opt_skel_seq_list, opt_jpos3d_18_list, skel_faces, scale_val_list = \
                        ema_model.opt_3d_w_vposer_w_joints3d_input(opt_jpos3d_18.detach().cpu().numpy())

        for bs_idx in range(batch):
            ori_opt_skel_seq = ori_opt_skel_seq_list[bs_idx]
            curr_opt_skel_seq = opt_skel_seq_list[bs_idx] 
            curr_opt_jpos3d_18 = opt_jpos3d_18_list[bs_idx] 
            curr_scale_val = scale_val_list[bs_idx] 

            smpl_scaling = torch.ones(1).detach().cpu().numpy() # 1

            pred_smpl_scaling = curr_opt_skel_seq['smpl_scaling'] # 1 
            pred_smpl_poses = curr_opt_skel_seq['smpl_poses'] # T X 24 X 3 
            pred_smpl_trans = curr_opt_skel_seq['smpl_trans'] # T X 3 

            pred_smpl_poses, pred_smpl_trans = self.move_smpl_seq_to_floor(pred_smpl_poses, pred_smpl_trans)
            pred_jnts3d_smpl, pred_verts_smpl = run_smpl_forward(pred_smpl_poses, \
                    pred_smpl_trans, pred_smpl_scaling, self.val_ds.smpl) 
            # T X 45 X 3, T X Nv X 3  
            pred_smpl_jnts18 = convert_smpl_jnts_to_openpose18(pred_jnts3d_smpl)

            # Move the first frame's joints3D root to the center
            pred_smpl_jnts18 = self.move_init3d_to_center(pred_smpl_jnts18) 
            curr_opt_jpos3d_18 = self.move_init3d_to_center(curr_opt_jpos3d_18) 
            
            pred_smpl_jnts18_list.append(pred_smpl_jnts18)
            opt_jpos3d_18_list.append(curr_opt_jpos3d_18)

            pred_verts_smpl_list.append(pred_verts_smpl)
            
            scale_val_list.append(curr_scale_val) # The scale of optimized 3D joint positions compared to vposer fitting results 
            
            ori_opt_skel_seq_list.append(ori_opt_skel_seq)

        return direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, \
            val_bs, val_num_views, val_num_steps, val_data, \
            actual_seq_len, num_lines_for_ref, \
            pred_smpl_jnts18_list, curr_seq_len, opt_jpos3d_18_list, \
            pred_verts_smpl_list, skel_faces, cam_extrinsic, \
            scale_val_list, all_line_conditions, ori_opt_skel_seq_list

    def run_optimization_w_learned_model_for_synthetic_data_gen_interaction(self, val_data_dict, ema_model, input_first_human_pose=False, \
                                    init_3d_jnts3d=None):
        ori_val_data = val_data_dict['normalized_jnts2d'].float().cuda() # BS X T X 18 X 2 
        
        batch = ori_val_data.shape[0] 

        cam_extrinsic = self.prep_multiview_cam_extrinsic(num_views=6) # 1 X K X 4 X 4 
        cam_extrinsic = cam_extrinsic.repeat(batch, 1, 1, 1) # BS X K X 4 X 4 

        ori_val_data = ori_val_data[:, None, :, :, :].repeat(1, cam_extrinsic.shape[1], 1, 1, 1) # BS X K X T X 18 X 2 

        val_data, all_line_conditions, val_num_views, val_bs, num_lines_for_ref, val_num_steps, cam_extrinsic = \
            self.get_all_line_conditions_from_val_data(ori_val_data, cam_extrinsic)
        # val_data: BS X K X T X (18*2), all_line_conditions: BS X K X T X 18 X 3, cam_extrinsic: BS X K X 4 X 4 
    
        cond_mask = None  
       
        # Generate padding mask 
        actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
        actual_seq_len = actual_seq_len[:, None].repeat(1, val_num_views) # BS X K 
        actual_seq_len = actual_seq_len.reshape(-1) # (BS*K)

        tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0]*val_data.shape[1], \
        self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
        # (BS*K) X T 
        padding_mask = tmp_mask[:, None, :].to(val_data.device) # (BS*K) X 1 X T 

        language_input = None 

        curr_seq_len = int(actual_seq_len.detach().cpu().numpy()[0])-1
        if cond_mask is None:
            curr_cond_mask = None 
        else:
            curr_cond_mask = cond_mask 
       
        if self.opt.gen_more_consistent_2d_seq:
            # Run SDS optimization to get more consistent 2D pose sequences 
            opt_paired_jnts2d_seq = ema_model.opt_paired_2d_w_sds_loss_and_jnts2line_dist_loss(val_data, \
                                cam_extrinsic, cond_mask=curr_cond_mask, \
                                padding_mask=padding_mask, \
                                input_line_cond=all_line_conditions, \
                                compute_epi_lines_func=self.compute_epi_lines, \
                                init_jnts2d_seq=None)  
                                # init_jnts2d_seq=init_epi_line_cond_seq_2d.detach()[:, 1:])  
            # BS X (K-1) X T X (18*2) 

            direct_sampled_epi_seq_2d = torch.cat((val_data[:, 0:1], opt_paired_jnts2d_seq.clone()), dim=1) # BS X K X T X (18*2)
            epi_line_cond_seq_2d = torch.cat((val_data[:, 0:1], opt_paired_jnts2d_seq.clone()), dim=1) # BS X K X T X (18*2)
        else:
            # val_data: BS X K X T X 36, all_line_conditions: BS X K X T X 18 X 3 
            epi_line_cond_seq_2d = ema_model.sample(val_data.reshape(val_bs*val_num_views, val_num_steps, -1), \
                        curr_cond_mask, padding_mask, \
                        input_line_cond=all_line_conditions.reshape(val_bs*val_num_views, val_num_steps, -1))
            # (BS*K) X T X 36 

            epi_line_cond_seq_2d = epi_line_cond_seq_2d.reshape(val_bs, val_num_views, val_num_steps, -1) # BS X K X T X 36 

            direct_sampled_epi_seq_2d = torch.cat((val_data[:, 0:1], epi_line_cond_seq_2d.clone()[:, 1:]), dim=1)
            epi_line_cond_seq_2d = torch.cat((val_data[:, 0:1], epi_line_cond_seq_2d.clone()[:, 1:]), dim=1) 
       
        epi_line_cond_seq_2d, opt_jpos3d_18, view_mask = \
                    ema_model.opt_3d_w_multi_view_2d_res(epi_line_cond_seq_2d, \
                    cam_extrinsic, padding_mask=padding_mask, actual_seq_len=curr_seq_len, \
                    input_line_cond=all_line_conditions, \
                    x_start=val_data, cond_mask=curr_cond_mask, \
                    language_input=language_input)
        # BS X K X T X (18*2), BS X T X 18 X 3, BS X T X 18 X 3   

        pred_smpl_jnts18_list = [] 
        pred_verts_smpl_list = []

        # opt_jpos3d_18_list = [] # Before using vposer optimization 
        # scale_val_list = []
        # ori_opt_skel_seq_list = [] 

        # For object 
        ori_opt_obj_param_list, opt_obj_param_list, opt_obj_jpos3d_list, opt_obj_verts_list, obj_faces, obj_scale_val_list = \
                    ema_model.opt_3d_object_w_joints3d_input(opt_jpos3d_18.detach().cpu().numpy()[:, :, -5:, :]) 
        # rest_obj_verts(real world scale) * scale ==> optimized 3d object positions 

        # For human 
        ori_opt_skel_seq_list, opt_skel_seq_list, opt_jpos3d_18_list, skel_faces, scale_val_list = \
                    ema_model.opt_3d_w_vposer_w_joints3d_input(opt_jpos3d_18.detach().cpu().numpy()[:, :, :-5, :])
        # mean_human_verts * human_scale ==> optimized 3d human positions 
        # opt_skel_seq_list, opt_jpos3d_18_list, skel_faces, scale_val_list has been converted in real-world scale. 

        # mean_human_verts * new_human_scale ==> real-world scale human vertices 

        # we need to have human vertices in real world scale. 

        # Adjust human's scaling value based on object's scale value since object is fixed while human can have different body shape in this dataset 

        new_human_opt_smpl_seq_list = []
        new_human_smpl_jnts18_list = [] 
        new_human_smpl_verts_list = [] 

        for bs_idx in range(batch):
            opt_skel_seq = opt_skel_seq_list[bs_idx]
           
            new_smpl_param = {} 

            pred_smpl_scaling = opt_skel_seq['smpl_scaling'] # 1 
            pred_smpl_poses = opt_skel_seq['smpl_poses'] # T X 24 X 3 
            pred_smpl_trans = opt_skel_seq['smpl_trans'] # T X 3 

           
            # pred_smpl_poses, pred_smpl_trans = self.move_smpl_seq_to_floor(pred_smpl_poses, pred_smpl_trans)
            pred_jnts3d_smpl, pred_verts_smpl = run_smpl_forward(pred_smpl_poses, \
                    pred_smpl_trans, pred_smpl_scaling, self.val_ds.smpl) 
            # # T X 45 X 3, T X Nv X 3, directly oiptimized positions   
            pred_smpl_jnts18 = convert_smpl_jnts_to_openpose18(pred_jnts3d_smpl)

            smpl_scaling = obj_scale_val_list[bs_idx] # Use object's sdcale to adjust human's scale 
            pred_smpl_jnts18 = pred_smpl_jnts18 * scale_val_list[bs_idx] / smpl_scaling
            pred_verts_smpl = pred_verts_smpl * scale_val_list[bs_idx] / smpl_scaling 

            new_human_smpl_jnts18_list.append(pred_smpl_jnts18)
            new_human_smpl_verts_list.append(pred_verts_smpl) 

            new_smpl_param['smpl_scaling'] = pred_smpl_scaling * scale_val_list[bs_idx] / smpl_scaling
            new_smpl_param['smpl_poses'] = pred_smpl_poses
            new_smpl_param['smpl_trans'] = pred_smpl_trans * scale_val_list[bs_idx] / smpl_scaling

            new_human_opt_smpl_seq_list.append(new_smpl_param)

        return direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, \
            val_bs, val_num_views, val_num_steps, val_data, \
            actual_seq_len, num_lines_for_ref, \
            new_human_smpl_jnts18_list, curr_seq_len, new_human_smpl_jnts18_list, \
            new_human_smpl_verts_list, skel_faces, cam_extrinsic, \
            obj_scale_val_list, all_line_conditions, new_human_opt_smpl_seq_list, \
            ori_opt_obj_param_list, opt_obj_param_list, opt_obj_jpos3d_list, opt_obj_verts_list, obj_faces, obj_scale_val_list

    def prep_vis_res_folder(self):
        dest_res3d_root_folder = "/viscam/projects/mofitness/AIST_ablations_cvpr25"

        if "AIST" in self.youtube_data_npz_folder:
            dest_res3d_npy_folder = os.path.join(dest_res3d_root_folder, "AIST")
        elif "nicole" in self.youtube_data_npz_folder:
            dest_res3d_npy_folder = os.path.join(dest_res3d_root_folder, "nicole")
        elif "steezy" in self.youtube_data_npz_folder:
            dest_res3d_npy_folder = os.path.join(dest_res3d_root_folder, "steezy")
        elif "cat" in self.youtube_data_npz_folder:
            dest_res3d_npy_folder = os.path.join(dest_res3d_root_folder, "cat") 

        if self.opt.gen_more_consistent_2d_seq:
            dest_res3d_npy_folder = dest_res3d_npy_folder + "_opt2d_first"
        elif self.opt.ablation_sds_loss_only:
            dest_res3d_npy_folder = dest_res3d_npy_folder + "_sds_loss_only"
        elif self.opt.ablation_multi_step_recon_loss_only:
            dest_res3d_npy_folder = dest_res3d_npy_folder + "_multi_step_recon_loss_only" 
        else:
            dest_res3d_npy_folder = dest_res3d_npy_folder + "_direct_opt_mv2d"

        if not os.path.exists(dest_res3d_npy_folder):
            os.makedirs(dest_res3d_npy_folder)

        return dest_res3d_npy_folder 

    def replace_neck_jnt_w_avg(self, jpos3d_18):
        # jpos3d_18: T X 18 X 3 
        l_shoulder_idx = 5
        r_shoulder_idx = 2 
        new_neck_jnts = (jpos3d_18[:, l_shoulder_idx:l_shoulder_idx+1, :] + \
                jpos3d_18[:, r_shoulder_idx:r_shoulder_idx+1, :]) / 2. # T X 1 X 3
        
        new_jpos3d_18 = jpos3d_18.clone()
        new_jpos3d_18[:, 1:2, :] = new_neck_jnts

        return new_jpos3d_18 # T X 18 X 3 

    def reproject_wham_local_3D_to_2D(self, local_wham_jpos3d_18, scale_factors):
        # local_wham_jpos3d_18: T X 18 X 3 

        width = 1920 
        height = 1080 
        focal_length = (width ** 2 + height ** 2) ** 0.5 

        cam_rot_mat = torch.eye(3)[None].repeat(local_wham_jpos3d_18.shape[0], 1, 1).to(local_wham_jpos3d_18.device) # T X 3 X 3 
        cam_trans = torch.zeros(local_wham_jpos3d_18.shape[0], 3).to(local_wham_jpos3d_18.device) # T X 3 
    
        _, _, reprojected_ori_jnts2d = perspective_projection(
                        local_wham_jpos3d_18, cam_rot_mat, \
                        cam_trans, focal_length=focal_length, \
                        image_width=width, image_height=height) # T X 18 X 2 

        # Normalize to [0, 1] for applying scale_factors 
        reprojected_ori_jnts2d[:, :, 0] /= width 
        reprojected_ori_jnts2d[:, :, 1] /= height 

        reprojected_ori_jnts2d *= scale_factors.item() 

        lhip_idx = 8 
        rhip_idx = 11 
        pred_root_trans = (reprojected_ori_jnts2d[0:1, lhip_idx, :] + reprojected_ori_jnts2d[0:1, rhip_idx, :]) / 2. # 1 X 2 

        # Central position in normalized coordinates
        center = torch.tensor([0.5, 0.5], device=reprojected_ori_jnts2d.device, dtype=reprojected_ori_jnts2d.dtype)

        # Compute translation needed to move initial root position to the center (K x 2)
        move2center = center[None, :] - pred_root_trans # 1 X 2 

        # Apply the translation to all positions in all sequences (broadcasting)
        reprojected_ori_jnts2d += move2center[:, None, :]
        reprojected_ori_jnts2d = self.replace_neck_jnt_w_avg(reprojected_ori_jnts2d)

        reprojected_jnts2d = normalize_pose2d(reprojected_ori_jnts2d, input_normalized=True)
            
        return reprojected_jnts2d # T X 18 X 2 

    def load_motionbert_res(self, npz_name):
        seq_name = npz_name.replace(".npz", "")

        motionbert_res_folder = "/viscam/projects/vimotion/datasets/AIST/MotionBert_results_mesh"
        motionbert_jnts3d_data_path = os.path.join(motionbert_res_folder, seq_name+".npy")
        motionbert_jnts3d_h36m = np.load(motionbert_jnts3d_data_path) # T X 49 X 3 

        # Convert SMPL to COCO 18 
        motionbert_jnts18 = convert_smpl_jnts_to_openpose18(motionbert_jnts3d_h36m) # T X 18 X 3 

        # dest_res3d_npy_folder = "./tmp_debug_motionbert_res_check"
        # if not os.path.exists(dest_res3d_npy_folder):
        #     os.makedirs(dest_res3d_npy_folder) 
        # dest_vid_3d_path = os.path.join(dest_res3d_npy_folder, seq_name+".mp4")
        # plot_3d_motion(dest_vid_3d_path, \
        #     motionbert_jnts18[:120]) 

        return motionbert_jnts18 

    def compute_forward_direction(self, jnts3d_18):
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
        hip_span = right_hip - left_hip 
        # hip_span = right_shoulder - left_shoulder  
        
        # Calculate the forward direction as perpendicular to the body axis and hip span
        facing_direction = torch.cross(body_axis, hip_span, dim=1)
        
        # Normalize the facing direction and body axis
        facing_direction = facing_direction / torch.norm(facing_direction, dim=1, keepdim=True)

        return facing_direction # T X 3 

    def align_motionbert_to_ours(self, motionbert_jnts18, gt_jnts18):
        # motionbert_jnts18 = torch.from_numpy(motionbert_jnts18).float().to(gt_jnts18.device) # T X 18 X 3

        # motionbert_facing_dir = self.compute_forward_direction(motionbert_jnts18)[0:1] # 1 X 3
        # gt_facing_dir = self.compute_forward_direction(gt_jnts18)[0:1] # 1 X 3 

        # gt_facing_dir = gt_facing_dir.detach().cpu().numpy()
        # motionbert_facing_dir = motionbert_facing_dir.detach().cpu().numpy()

        # yrot = quat_normalize(quat_between(gt_facing_dir, motionbert_facing_dir)) # 1 X 4 

        # aligned_jnts18 = quat_mul_vec(quat_inv(yrot), motionbert_jnts18.reshape(-1, 3)) # (T*J) X 3

        # aligned_jnts18 = aligned_jnts18.reshape(motionbert_jnts18.shape) # T X J X 3 

        motionbert_jnts18 = motionbert_jnts18[:gt_jnts18.shape[0]]
        gt_jnts18 = gt_jnts18.detach().cpu().numpy()

        src = motionbert_jnts18.transpose(0, 2, 1)  # T x 3 x J
        tar = gt_jnts18.transpose(0, 2, 1)  # T x 3 x J

        # Compute centroids
        mu1 = src.mean(axis=2, keepdims=True) # T X 3 X 1
        mu2 = tar.mean(axis=2, keepdims=True) # T X 3 X 1

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
        R = np.matmul(np.matmul(V, Z), U.transpose(0, 2, 1)) # T X 3 X 3 

        # Ensure mu1_squeezed is reshaped to [T, 3, 1]
        mu1_squeezed = mu1.squeeze(axis=2)[:, :, np.newaxis]
        mu2_squeezed = mu2.squeeze(axis=2)

        # Correct matrix multiplication
        # t = mu2_squeezed - np.matmul(R, mu1_squeezed)[:, :, 0] # T X 3 

        align_rot_mat = torch.from_numpy(R).float()[0:1] # 1 X 3 X 3 
        align_rot_mat = align_rot_mat.repeat(motionbert_jnts18.shape[0], 1, 1) # T X 3 X 3
        motionbert_jnts18 = torch.from_numpy(motionbert_jnts18).float() # T X 18 X 3

        aligned_jnts18 = torch.matmul(align_rot_mat[:, None].repeat(1, 18, 1, 1), motionbert_jnts18[:, :, :, None])  # T X 18 X 3 X 1
    
        aligned_jnts18 = aligned_jnts18.squeeze(-1) # T X 18 X 3 

        return aligned_jnts18

    def sample_2d_motion_w_line_conditioned_diffusion(self):
        # Load line-conditioned model. 
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        # milestone = "5"

        self.load(milestone)
        self.ema.ema_model.eval()
       
        # Prepare testing data loader 
        if self.opt.ablation_sds_loss_only or self.opt.ablation_multi_step_recon_loss_only:
            test_loader = torch.utils.data.DataLoader(
                self.val_ds, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=False) 
        else:
            test_loader = torch.utils.data.DataLoader(
                self.val_ds, batch_size=32, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=False) 

        # Prepare empty list for evaluation metrics 
        self.prep_evaluation_metric_list() 
        
        for s_idx, val_data_dict in enumerate(test_loader): 
            # if s_idx != 7:
            #     continue 
            if self.opt.ablation_sds_loss_only or self.opt.ablation_multi_step_recon_loss_only:
                direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, \
                val_bs, val_num_views, val_num_steps, val_data, actual_seq_len, num_lines_for_ref, \
                pred_smpl_jnts18_list, gt_smpl_jnts18_list, curr_seq_len, opt_jpos3d_18_list, \
                pred_verts_smpl_list, gt_verts_smpl_list, \
                skel_faces, cam_extrinsic, scale_val_list, all_line_conditions, \
                ori_opt_skel_seq_list = self.run_optimization_w_learned_model_ablation(val_data_dict, self.ema.ema_model)
            else:
                direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, \
                val_bs, val_num_views, val_num_steps, val_data, actual_seq_len, num_lines_for_ref, \
                pred_smpl_jnts18_list, gt_smpl_jnts18_list, curr_seq_len, opt_jpos3d_18_list, \
                pred_verts_smpl_list, gt_verts_smpl_list, \
                skel_faces, cam_extrinsic, scale_val_list, all_line_conditions, \
                ori_opt_skel_seq_list = self.run_optimization_w_learned_model(val_data_dict, self.ema.ema_model)

            dest_res3d_npy_folder = self.prep_vis_res_folder() 

            # The direct sampled 2D pose sequences of the trained 2D motion diffusion model, the first view is the input GT 2D poses  
            direct_sampled_seq_2d_for_vis = direct_sampled_epi_seq_2d.reshape(val_bs, val_num_views, \
                        val_num_steps, -1) # BS X K X T X D
                    
            # The input GT 2D pose sequences. 
            gt_for_vis = val_data.reshape(val_bs, val_num_views, val_num_steps, -1)
            
            cam_rot_mat = cam_extrinsic[:, :, :3, :3].unsqueeze(2).repeat(1, 1, val_num_steps, 1, 1) # BS X K X T X 3 X 3 
            cam_trans = cam_extrinsic[:, :, :3, -1].unsqueeze(2).repeat(1, 1, val_num_steps, 1) # BS X K X T X 3 
        
            actual_seq_len = actual_seq_len.reshape(val_bs, -1) # BS X K 

            for bs_idx in range(val_bs):
                vis_folder_tag = "epi_guided_2d"+"_"+str(milestone)+"_batch_"+str(s_idx)+"_seq_"+str(bs_idx)

                # Get reprojected our results 
                pred_smpl_jnts18 = self.replace_neck_jnt_w_avg(pred_smpl_jnts18_list[bs_idx]) 
                reprojected_pred_jnts2d_list, reprojected_pred_ori_jnts2d_list, _ = \
                                self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                pred_smpl_jnts18.to(cam_rot_mat.device)*scale_val_list[bs_idx]) # K X T X (18*2)
                pred_for_vis = reprojected_pred_jnts2d_list.reshape(1, val_num_views, -1, 18*2)   

                # Visualization for 2D joint positions. 
                de_direct_sampled_pred_2d_for_vis = self.gen_multiview_vis_res(direct_sampled_seq_2d_for_vis[bs_idx:bs_idx+1], \
                    vis_folder_tag, actual_seq_len[bs_idx:bs_idx+1], \
                    dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_direct_sampled")
                
                de_gt_2d_for_vis = self.gen_multiview_vis_res(gt_for_vis[bs_idx:bs_idx+1], vis_folder_tag, \
                    actual_seq_len[bs_idx:bs_idx+1], vis_gt=True, \
                    dest_vis_data_folder=dest_res3d_npy_folder)
                de_pred_2d_for_vis = self.gen_multiview_vis_res(pred_for_vis, vis_folder_tag, \
                    actual_seq_len[bs_idx:bs_idx+1], \
                    dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_ours")
              
                tmp_idx = 0
                v_tmp_idx = 0 
                
                mpjpe, pa_mpjpe, trans_err, jpos3d_err, jpos2d_err, centered_jpos2d_err, bone_consistency = \
                compute_metrics_for_jpos(pred_smpl_jnts18_list[bs_idx], \
                    de_pred_2d_for_vis[tmp_idx, v_tmp_idx], \
                    gt_smpl_jnts18_list[bs_idx], \
                    de_gt_2d_for_vis[tmp_idx, v_tmp_idx], \
                    int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1, opt_jpos3d_18_list[bs_idx])

                self.add_new_evaluation_to_list(mpjpe, pa_mpjpe, trans_err, \
                        jpos3d_err, jpos2d_err, centered_jpos2d_err, bone_consistency) 

                self.print_mean_metrics() 

                curr_actual_seq_len = int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1 
        
                if not self.eval_w_best_mpjpe:
              
                    dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d")
                    plot_3d_motion(dest_vid_3d_path_global, \
                        opt_jpos3d_18_list[bs_idx].detach().cpu().numpy()[:curr_actual_seq_len]) 

                    dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_w_smpl_fitting")
                    plot_3d_motion(dest_vid_3d_path_global, \
                        pred_smpl_jnts18.detach().cpu().numpy()[:curr_actual_seq_len]) 

                    if self.aist_optimize_3d_w_trained_2d_diffusion:
                        gt_dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_gt")
                        plot_3d_motion(gt_dest_vid_3d_path_global, \
                            gt_smpl_jnts18_list[bs_idx].detach().cpu().numpy()[:curr_actual_seq_len]) 

                    if self.aist_optimize_3d_w_trained_2d_diffusion:
                        dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_objs_gt")
                        if not os.path.exists(dest_mesh_folder):
                            os.makedirs(dest_mesh_folder)
                        for t_idx in range(curr_actual_seq_len):
                            skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_gt.obj")

                            curr_mesh = trimesh.Trimesh(vertices=gt_verts_smpl_list[bs_idx][t_idx].detach().cpu().numpy(),
                                faces=skel_faces) 
            
                            curr_mesh.export(skin_mesh_out)

                    dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_objs")
                    if not os.path.exists(dest_mesh_folder):
                        os.makedirs(dest_mesh_folder)
                    for t_idx in range(curr_actual_seq_len):
                        skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_ours.obj")

                        curr_mesh = trimesh.Trimesh(vertices=pred_verts_smpl_list[bs_idx][t_idx].detach().cpu().numpy(),
                            faces=skel_faces) 

                        curr_mesh.export(skin_mesh_out)

    def gen_synthetic_3d_data_w_trained_2d_motion_diffusion(self):
        # Load line-conditioned model. 
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        # milestone = "23"
        # milestone = "10"

        self.load(milestone)
        self.ema.ema_model.eval()
       
        # Prepare testing data loader 
        if not self.eval_w_best_mpjpe:
            curr_bs = 1
        else:
            curr_bs = 64

        if self.gen_synthetic_3d_data_for_training:
            test_loader = torch.utils.data.DataLoader(
                self.ds, batch_size=curr_bs, shuffle=False,
                num_workers=4, pin_memory=True, drop_last=False) 
        else:
            test_loader = torch.utils.data.DataLoader(
                self.val_ds, batch_size=curr_bs, shuffle=False,
                num_workers=4, pin_memory=True, drop_last=False) 
            
        num_epochs = 10 
        for epoch_idx in range(1):
            for s_idx, val_data_dict in enumerate(test_loader): 
              
                direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, \
                val_bs, val_num_views, val_num_steps, val_data, \
                actual_seq_len, num_lines_for_ref, \
                pred_smpl_jnts18_list, curr_seq_len, opt_jpos3d_18_list, \
                pred_verts_smpl_list, skel_faces, cam_extrinsic, \
                scale_val_list, all_line_conditions, ori_opt_smpl_param_list = \
                    self.run_optimization_w_learned_model_for_synthetic_data_gen(val_data_dict, self.ema.ema_model, \
                    input_first_human_pose=False)

                # The input GT 2D pose sequences. 
                gt_for_vis = val_data.reshape(val_bs, val_num_views, val_num_steps, -1)

                # The direct sampled 2D pose sequences of the trained 2D motion diffusion model, the first view is the input GT 2D poses  
                direct_sampled_seq_2d_for_vis = direct_sampled_epi_seq_2d.reshape(val_bs, val_num_views, \
                            val_num_steps, -1) # BS X K X T X D
                        
                cam_rot_mat = cam_extrinsic[:, :, :3, :3].unsqueeze(2).repeat(1, 1, val_num_steps, 1, 1) # BS X K X T X 3 X 3 
                cam_trans = cam_extrinsic[:, :, :3, -1].unsqueeze(2).repeat(1, 1, val_num_steps, 1) # BS X K X T X 3 
            
                actual_seq_len = actual_seq_len.reshape(val_bs, -1) # BS X K 

                for bs_idx in range(val_bs):
                    vis_folder_tag = "epi_guided_2d"+"_"+str(milestone)+"_epoch_"+str(epoch_idx)+"_batch_"+str(s_idx)+"_seq_"+str(bs_idx)

                    pred_smpl_jnts18 = self.replace_neck_jnt_w_avg(pred_smpl_jnts18_list[bs_idx]) 

                    if self.gen_synthetic_3d_data_for_training:
                        dest_res3d_npy_folder = os.path.join("/".join(self.opt.youtube_data_npz_folder.split("/")[:-1]), "synthetic_gen_3d_data_vis_check")
                        dest_jnts3d_npy_folder = os.path.join("/".join(self.opt.youtube_data_npz_folder.split("/")[:-1]), "synthetic_gen_3d_data")
                    else:
                        dest_res3d_npy_folder = os.path.join("/".join(self.opt.youtube_data_npz_folder.split("/")[:-1]), "synthetic_gen_3d_data_vis_check_val")
                        dest_jnts3d_npy_folder = os.path.join("/".join(self.opt.youtube_data_npz_folder.split("/")[:-1]), "synthetic_gen_3d_data_val")

                    if not os.path.exists(dest_res3d_npy_folder):
                        os.makedirs(dest_res3d_npy_folder)
                    if not os.path.exists(dest_jnts3d_npy_folder):
                        os.makedirs(dest_jnts3d_npy_folder) 
                    dest_jnts3d_npy_path = os.path.join(dest_jnts3d_npy_folder, \
                            "epoch_"+str(epoch_idx)+"_batch_"+str(s_idx)+"_seq_"+str(bs_idx)+".npy")
                    
                    curr_actual_seq_len = int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1 
                    np.save(dest_jnts3d_npy_path, \
                        pred_smpl_jnts18.detach().cpu().numpy()[:curr_actual_seq_len])

                    if not self.eval_w_best_mpjpe:
                        # Get reprojected our results 
                        
                        reprojected_pred_jnts2d_list, reprojected_pred_ori_jnts2d_list, _ = \
                                        self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                        cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                        pred_smpl_jnts18.to(cam_rot_mat.device)*scale_val_list[bs_idx]) # K X T X (18*2)

                        pred_for_vis = reprojected_pred_jnts2d_list.reshape(1, val_num_views, 120, -1)   

                        # Visualization for 2D joint positions. 
                        de_direct_sampled_pred_2d_for_vis = self.gen_multiview_vis_res(direct_sampled_seq_2d_for_vis[bs_idx:bs_idx+1], \
                            vis_folder_tag, actual_seq_len[bs_idx:bs_idx+1], \
                            dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_direct_sampled")

                        vis_for_paper = False 
                        if vis_for_paper:
                            dest_for_paper_vis_folder = "./for_cvpr25_paper_method_figure_rough_consistent_2d"
                            if not os.path.exists(dest_for_paper_vis_folder):
                                os.makedirs(dest_for_paper_vis_folder) 

                            # Plot for multi-view 2D sequences 
                            num_views = direct_sampled_seq_2d_for_vis[0].shape[0] 
                            num_steps = direct_sampled_seq_2d_for_vis[0].shape[1] 
                            for view_idx in range(num_views):
                                for t_idx in range(0, num_steps, 20):
                                    dest_pose2d_fig_path = os.path.join(dest_for_paper_vis_folder, \
                                        "s_idx_"+str(s_idx)+"_bs_idx_"+str(bs_idx)+"_pose2d_"+"view_"+str(view_idx)+"_t_"+str(t_idx)+".png") 
                                    plot_pose_2d_for_paper(direct_sampled_seq_2d_for_vis[0][view_idx, t_idx].reshape(-1, 2).detach().cpu().numpy(), \
                                    dest_pose2d_fig_path) 

                            import pdb 
                            pdb.set_trace() 
                        
                        de_pred_2d_for_vis = self.gen_multiview_vis_res(pred_for_vis, vis_folder_tag, \
                            actual_seq_len[bs_idx:bs_idx+1], \
                            dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_ours")

                        dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d")
                        plot_3d_motion(dest_vid_3d_path_global, \
                            opt_jpos3d_18_list[bs_idx].detach().cpu().numpy()[:curr_actual_seq_len]) 

                        dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_w_smpl_fitting")
                        plot_3d_motion(dest_vid_3d_path_global, \
                            pred_smpl_jnts18.detach().cpu().numpy()[:curr_actual_seq_len]) 

                        dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_objs")
                        if not os.path.exists(dest_mesh_folder):
                            os.makedirs(dest_mesh_folder)
                        for t_idx in range(curr_actual_seq_len):
                            skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_ours.obj")

                            curr_mesh = trimesh.Trimesh(vertices=pred_verts_smpl_list[bs_idx][t_idx].detach().cpu().numpy(),
                                faces=skel_faces) 

                            curr_mesh.export(skin_mesh_out)

    def gen_synthetic_3d_data_w_trained_2d_motion_diffusion_for_interaction(self):
        # Load line-conditioned model. 
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        # milestone = "23"
        # milestone = "10"

        self.load(milestone)
        self.ema.ema_model.eval()

        if not self.eval_w_best_mpjpe:
            curr_bs = 1
        else:
            curr_bs = 64
       
        # Prepare testing data loader 
        if self.gen_synthetic_3d_data_for_training:
            test_loader = torch.utils.data.DataLoader(
                self.ds, batch_size=curr_bs, shuffle=False,
                num_workers=4, pin_memory=True, drop_last=False) 
        else:
            test_loader = torch.utils.data.DataLoader(
                self.val_ds, batch_size=curr_bs, shuffle=False,
                num_workers=4, pin_memory=True, drop_last=False) 
            
        num_epochs = 10 
        for epoch_idx in range(1):
            for s_idx, val_data_dict in enumerate(test_loader): 
              
                direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, \
                val_bs, val_num_views, val_num_steps, val_data, \
                actual_seq_len, num_lines_for_ref, \
                pred_smpl_jnts18_list, curr_seq_len, _, \
                pred_smpl_verts_list, skel_faces, cam_extrinsic, \
                obj_scale_val_list, all_line_conditions, new_human_opt_smpl_seq_list, \
                ori_opt_obj_param_list, opt_obj_param_list, opt_obj_jpos3d_list, opt_obj_verts_list, obj_faces, obj_scale_val_list = \
                    self.run_optimization_w_learned_model_for_synthetic_data_gen_interaction(val_data_dict, self.ema.ema_model, \
                    input_first_human_pose=False)

                # The input GT 2D pose sequences. 
                gt_for_vis = val_data.reshape(val_bs, val_num_views, val_num_steps, -1)

                # The direct sampled 2D pose sequences of the trained 2D motion diffusion model, the first view is the input GT 2D poses  
                direct_sampled_seq_2d_for_vis = direct_sampled_epi_seq_2d.reshape(val_bs, val_num_views, \
                            val_num_steps, -1) # BS X K X T X D
                        
                cam_rot_mat = cam_extrinsic[:, :, :3, :3].unsqueeze(2).repeat(1, 1, val_num_steps, 1, 1) # BS X K X T X 3 X 3 
                cam_trans = cam_extrinsic[:, :, :3, -1].unsqueeze(2).repeat(1, 1, val_num_steps, 1) # BS X K X T X 3 
            
                actual_seq_len = actual_seq_len.reshape(val_bs, -1) # BS X K 

                for bs_idx in range(val_bs):
                    vis_folder_tag = "epi_guided_2d"+"_"+str(milestone)+"_epoch_"+str(epoch_idx)+"_batch_"+str(s_idx)+"_seq_"+str(bs_idx)

                    pred_smpl_jnts18 = self.replace_neck_jnt_w_avg(pred_smpl_jnts18_list[bs_idx]) 

                    if self.gen_synthetic_3d_data_for_training:
                        dest_res3d_npy_folder = os.path.join("/".join(self.opt.youtube_data_npz_folder.split("/")[:-1]), "synthetic_gen_3d_data_vis_check")
                        dest_jnts3d_npy_folder = os.path.join("/".join(self.opt.youtube_data_npz_folder.split("/")[:-1]), "synthetic_gen_3d_data")
                    else:
                        dest_res3d_npy_folder = os.path.join("/".join(self.opt.youtube_data_npz_folder.split("/")[:-1]), "synthetic_gen_3d_data_vis_check_val")
                        dest_jnts3d_npy_folder = os.path.join("/".join(self.opt.youtube_data_npz_folder.split("/")[:-1]), "synthetic_gen_3d_data_val")
                 
                    if not os.path.exists(dest_res3d_npy_folder):
                        os.makedirs(dest_res3d_npy_folder)
                    if not os.path.exists(dest_jnts3d_npy_folder):
                        os.makedirs(dest_jnts3d_npy_folder) 
                    dest_jnts3d_npy_path = os.path.join(dest_jnts3d_npy_folder, \
                            "epoch_"+str(epoch_idx)+"_batch_"+str(s_idx)+"_seq_"+str(bs_idx)+".npy")

                    combined_jnts3d = torch.cat((pred_smpl_jnts18.to(opt_obj_jpos3d_list[bs_idx].device), \
                            opt_obj_jpos3d_list[bs_idx]), dim=1) # T X 23 X 3 
                    
                    curr_actual_seq_len = int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1 
                    np.save(dest_jnts3d_npy_path, \
                        combined_jnts3d.detach().cpu().numpy()[:curr_actual_seq_len])

                    if not self.eval_w_best_mpjpe:
                        # Get reprojected our results 
                        
                        reprojected_pred_jnts2d_list, reprojected_pred_ori_jnts2d_list, _ = \
                                        self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                        cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                        combined_jnts3d.to(cam_rot_mat.device)*obj_scale_val_list[bs_idx]) # K X T X (18*2)

                        pred_for_vis = reprojected_pred_jnts2d_list.reshape(1, val_num_views, 120, -1)   

                        # Visualization for 2D joint positions. 
                        # de_direct_sampled_pred_2d_for_vis = self.gen_multiview_vis_res(direct_sampled_seq_2d_for_vis[bs_idx:bs_idx+1], \
                        #     vis_folder_tag, actual_seq_len[bs_idx:bs_idx+1], \
                        #     dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_direct_sampled")
                        
                        de_pred_2d_for_vis = self.gen_multiview_vis_res(pred_for_vis, vis_folder_tag, \
                            actual_seq_len[bs_idx:bs_idx+1], \
                            dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_ours")

                        de_gt_2d_for_vis = self.gen_multiview_vis_res(gt_for_vis, vis_folder_tag, \
                            actual_seq_len[bs_idx:bs_idx+1], \
                            dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_gt")

                        # dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                        #         str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d")
                        # plot_3d_motion(dest_vid_3d_path_global, \
                        #     opt_jpos3d_18_list[bs_idx].detach().cpu().numpy()[:curr_actual_seq_len]) 

                        dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_w_smpl_fitting")
                        plot_3d_motion(dest_vid_3d_path_global, \
                            pred_smpl_jnts18.detach().cpu().numpy()[:curr_actual_seq_len]) 

                        dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_objs")
                        if not os.path.exists(dest_mesh_folder):
                            os.makedirs(dest_mesh_folder)
                        for t_idx in range(curr_actual_seq_len):
                            skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_ours.obj")

                            curr_mesh = trimesh.Trimesh(vertices=pred_smpl_verts_list[bs_idx][t_idx].detach().cpu().numpy(),
                                faces=skel_faces) 
                            curr_mesh.export(skin_mesh_out)

                            # Save object mesh 
                            obj_skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_ours_obj.obj")

                            curr_obj_mesh = trimesh.Trimesh(vertices=opt_obj_verts_list[bs_idx][t_idx].detach().cpu().numpy(),
                                faces=obj_faces) 
                            curr_obj_mesh.export(obj_skin_mesh_out)

    def gen_synthetic_3d_data_w_trained_2d_motion_diffusion_for_animal(self):
        # Load line-conditioned model. 
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        # milestone = "23"
        # milestone = "10"

        self.load(milestone)
        self.ema.ema_model.eval()

        if not self.eval_w_best_mpjpe:
            curr_bs = 1
        else:
            curr_bs = 64
       
        # Prepare testing data loader 
        if self.gen_synthetic_3d_data_for_training:
            test_loader = torch.utils.data.DataLoader(
                self.ds, batch_size=curr_bs, shuffle=False,
                num_workers=4, pin_memory=True, drop_last=False) 
        else:
            test_loader = torch.utils.data.DataLoader(
                self.val_ds, batch_size=curr_bs, shuffle=False,
                num_workers=4, pin_memory=True, drop_last=False) 
            
        num_epochs = 10 
        for epoch_idx in range(1):
            for s_idx, val_data_dict in enumerate(test_loader): 
                direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, opt_jpos3d_18_list, cam_extrinsic, \
                val_bs, val_num_views, val_num_steps, val_data, actual_seq_len, num_lines_for_ref, \
                opt_skel_seq_list, skel_faces, smal_scale_val_list, smal_opt_jnts3d, smal_opt_verts = \
                    self.run_optimization_w_learned_model_for_synthetic_data_gen(val_data_dict, self.ema.ema_model, \
                    input_first_human_pose=False)

             
                # The input GT 2D pose sequences. 
                gt_for_vis = val_data.reshape(val_bs, val_num_views, val_num_steps, -1)

                # The direct sampled 2D pose sequences of the trained 2D motion diffusion model, the first view is the input GT 2D poses  
                direct_sampled_seq_2d_for_vis = direct_sampled_epi_seq_2d.reshape(val_bs, val_num_views, \
                            val_num_steps, -1) # BS X K X T X D
                        
                cam_rot_mat = cam_extrinsic[:, :, :3, :3].unsqueeze(2).repeat(1, 1, val_num_steps, 1, 1) # BS X K X T X 3 X 3 
                cam_trans = cam_extrinsic[:, :, :3, -1].unsqueeze(2).repeat(1, 1, val_num_steps, 1) # BS X K X T X 3 
            
                actual_seq_len = actual_seq_len.reshape(val_bs, -1) # BS X K 

                for bs_idx in range(val_bs):
                    vis_folder_tag = "epi_guided_2d"+"_"+str(milestone)+"_epoch_"+str(epoch_idx)+"_batch_"+str(s_idx)+"_seq_"+str(bs_idx)

                    if self.gen_synthetic_3d_data_for_training:
                        dest_res3d_npy_folder = os.path.join("/".join(self.opt.youtube_data_npz_folder.split("/")[:-1]), "synthetic_gen_3d_data_vis_check")
                        dest_jnts3d_npy_folder = os.path.join("/".join(self.opt.youtube_data_npz_folder.split("/")[:-1]), "synthetic_gen_3d_data")
                    else:
                        dest_res3d_npy_folder = os.path.join("/".join(self.opt.youtube_data_npz_folder.split("/")[:-1]), "synthetic_gen_3d_data_vis_check_val")
                        dest_jnts3d_npy_folder = os.path.join("/".join(self.opt.youtube_data_npz_folder.split("/")[:-1]), "synthetic_gen_3d_data_val")
                   
                    if not os.path.exists(dest_res3d_npy_folder):
                        os.makedirs(dest_res3d_npy_folder)
                    if not os.path.exists(dest_jnts3d_npy_folder):
                        os.makedirs(dest_jnts3d_npy_folder) 
                    dest_jnts3d_npy_path = os.path.join(dest_jnts3d_npy_folder, \
                            "epoch_"+str(epoch_idx)+"_batch_"+str(s_idx)+"_seq_"+str(bs_idx)+".npy")
                    
                    curr_actual_seq_len = int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1 
                    np.save(dest_jnts3d_npy_path, \
                        smal_opt_jnts3d[bs_idx].detach().cpu().numpy()[:curr_actual_seq_len])

                    if not self.eval_w_best_mpjpe:
                        # Get reprojected our results 
                        
                        # Visualization for 2D joint positions. 
                        de_direct_sampled_pred_2d_for_vis = self.gen_multiview_vis_res(direct_sampled_seq_2d_for_vis[bs_idx:bs_idx+1], \
                            vis_folder_tag, actual_seq_len[bs_idx:bs_idx+1], \
                            dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_direct_sampled")

                        reprojected_pred_jnts2d_list, reprojected_pred_ori_jnts2d_list, _ = \
                                        self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                        cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                        opt_jpos3d_18_list[bs_idx]*smal_scale_val_list[bs_idx]) # K X T X (18*2)
                        pred_for_vis = reprojected_pred_jnts2d_list.reshape(1, val_num_views, -1, 17*2)   
                        
                        de_pred_2d_for_vis = self.gen_multiview_vis_res(pred_for_vis, vis_folder_tag, \
                            actual_seq_len[bs_idx:bs_idx+1], \
                            dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_ours")

                        de_gt_2d_for_vis = self.gen_multiview_vis_res(gt_for_vis[bs_idx:bs_idx+1], vis_folder_tag, \
                            actual_seq_len[bs_idx:bs_idx+1], vis_gt=True, \
                            dest_vis_data_folder=dest_res3d_npy_folder)
                        
                        dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d")
                        plot_3d_motion(dest_vid_3d_path_global, \
                            opt_jpos3d_18_list[bs_idx].detach().cpu().numpy()[:curr_actual_seq_len], \
                            use_animal_pose17=True) 

                        # Visualize smal fitted 3d joint positions 
                        dest_vid_3d_path_opt_smal = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_smal_fitting")
                        plot_3d_motion(dest_vid_3d_path_opt_smal, \
                            smal_opt_jnts3d[bs_idx].detach().cpu().numpy()[:curr_actual_seq_len], \
                            use_animal_pose17=True) 

                        # Save smal fitted mesh to .obj 
                        dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_objs")
                        if not os.path.exists(dest_mesh_folder):
                            os.makedirs(dest_mesh_folder)
                        for t_idx in range(curr_actual_seq_len):
                            skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_ours.obj")

                            curr_mesh = trimesh.Trimesh(vertices=smal_opt_verts[bs_idx][t_idx].cpu().detach().cpu().numpy(),
                                faces=skel_faces.cpu().detach().cpu().numpy()) 

                            curr_mesh.export(skin_mesh_out)


    def train(self):
        init_step = self.step 
        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()

            nan_exists = False # If met nan in loss or gradient, need to skip to next data. 
            for i in range(self.gradient_accumulate_every):
                data_dict = next(self.dl)
              
                data = data_dict['normalized_jnts2d'].float().cuda() # BS X T X 18 X 2 
               
                bs, num_steps, num_joints, _ = data.shape 
             
                data = data.reshape(bs, num_steps, -1) # BS X T X D(18*2)

                motion_mask = torch.ones(bs, num_steps, num_joints*2).float().cuda() # BS X T X 18 

                cond_mask = None 

                # Generate padding mask 
                actual_seq_len = data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                tmp_mask = torch.arange(self.window+1).expand(data.shape[0], \
                self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(data.device)

                language_input = None 

                with autocast(enabled = self.amp):   
                    if self.train_2d_diffusion_w_line_cond:
                        line_conditions = data_dict['lines_pass_jnts2d'].float().cuda() 

                        ref_2d_seq_cond = None 

                        loss_diffusion, loss_line = self.model(data, cond_mask=cond_mask, \
                            padding_mask=padding_mask, motion_mask=motion_mask, \
                            language_input=language_input, input_line_cond=line_conditions) 
                        
                        loss = loss_diffusion + loss_line 
                    else: 
                        loss_diffusion = self.model(data, cond_mask=cond_mask, \
                            padding_mask=padding_mask, motion_mask=motion_mask, \
                            language_input=language_input)
                        
                        loss = loss_diffusion

                    if torch.isnan(loss).item():
                        print('WARNING: NaN loss. Skipping to next data...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    # check gradients
                    parameters = [p for p in self.model.parameters() if p.grad is not None]
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(data.device) for p in parameters]), 2.0)
                    if torch.isnan(total_norm):
                        print('WARNING: NaN gradients. Skipping to next data...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    if self.use_wandb:
                        if self.train_2d_diffusion_w_line_cond:
                            log_dict = {
                                "Train/Loss/Total Loss": loss.item(),
                                "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                                "Train/Loss/Line Loss": loss_line.item(),
                            }
                        else:
                            log_dict = {
                                "Train/Loss/Total Loss": loss.item(),
                                "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                            }
                        wandb.log(log_dict)

                    if idx % 50 == 0 and i == 0:
                        print("Step: {0}".format(idx))
                        print("Loss: %.4f" % (loss.item()))
                        print("Loss Diffusion: %.4f" % (loss_diffusion.item()))
                        if self.train_2d_diffusion_w_line_cond:
                            print("Loss Line: %.4f" % (loss_line.item()))

            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update()

            if self.step != 0 and self.step % 10 == 0:
                self.ema.ema_model.eval()

                with torch.no_grad():
                    val_data_dict = next(self.val_dl)

                    val_data = val_data_dict['normalized_jnts2d'].float().cuda()

                    val_bs, val_num_steps, num_joints, _ = val_data.shape 
                    val_data = val_data.reshape(val_bs, val_num_steps, -1) # BS X T X D(18*2)

                    val_motion_mask = torch.ones(val_bs, val_num_steps, num_joints*2).float().cuda() # BS X T X 18 

                    cond_mask = None  

                    # Generate padding mask 
                    actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                    tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
                    self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                    # BS X max_timesteps
                    padding_mask = tmp_mask[:, None, :].to(val_data.device)

                    language_input = None 

                    # Get validation loss 
                    if self.train_2d_diffusion_w_line_cond:
                        val_line_conditions = val_data_dict['lines_pass_jnts2d'].float().cuda() 

                        val_ref_2d_seq_cond = None 

                        val_loss_diffusion, val_loss_line = self.model(val_data, cond_mask=cond_mask, \
                            padding_mask=padding_mask, motion_mask=val_motion_mask, \
                            language_input=language_input, input_line_cond=val_line_conditions) 
                        val_loss = val_loss_diffusion + val_loss_line 
                    else:
                        val_line_conditions = None 
                        val_loss_diffusion = self.model(val_data, cond_mask=cond_mask, \
                            padding_mask=padding_mask, motion_mask=val_motion_mask, \
                            language_input=language_input)
                        val_loss = val_loss_diffusion 
                    
                    if self.use_wandb:
                        if self.train_2d_diffusion_w_line_cond:
                            val_log_dict = {
                                "Validation/Loss/Total Loss": val_loss.item(),
                                "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
                                "Validation/Loss/Line Loss": val_loss_line.item(),
                            }
                        else:
                            val_log_dict = {
                                "Validation/Loss/Total Loss": val_loss.item(),
                                "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
                            }
                        wandb.log(val_log_dict)

                    milestone = self.step // self.save_and_sample_every

                    if self.step % self.save_and_sample_every == 0:
                        self.save(milestone)

                        all_res_list = self.ema.ema_model.sample(val_data, \
                                    cond_mask, padding_mask, language_input=language_input, \
                                    input_line_cond=val_line_conditions)

                        if val_line_conditions is not None:
                            line_vis_folder_tag = str(self.step) + "_" + "line_vis_res"
                            line_dest_vis_folder_path = os.path.join(self.vis_folder, line_vis_folder_tag)
                            if not os.path.exists(line_dest_vis_folder_path):
                                os.makedirs(line_dest_vis_folder_path)

                            self.visualize_pose_and_lines(val_data[0].detach().cpu().numpy()[::4], \
                                val_line_conditions[0].detach().cpu().numpy()[::4], line_dest_vis_folder_path, vis_gt=True)
                            self.visualize_pose_and_lines(all_res_list[0].detach().cpu().numpy()[::4], \
                                val_line_conditions[0].detach().cpu().numpy()[::4], line_dest_vis_folder_path)


                        # Visualization
                        vis_folder_tag = str(self.step) + "_" + "motion2d_res"
                        num_vis = 8
                      
                        self.gen_vis_res(val_data[:num_vis], vis_folder_tag, actual_seq_len[:num_vis], vis_gt=True)
                        self.gen_vis_res(all_res_list[:num_vis], vis_folder_tag, actual_seq_len[:num_vis])

            self.step += 1

        print('training complete')

        if self.use_wandb:
            wandb.run.finish()

    def visualize_pose_and_lines(self, pose_sequence, line_coeffs, output_folder, \
        vis_gt=False, epipoles=None):
        """
        Visualizes 2D poses and corresponding lines for each frame, saving images to a folder.

        Parameters:
        - pose_sequence (numpy.ndarray): The pose sequence with shape (T, J, 2).
        - line_coeffs (numpy.ndarray): The line coefficients with shape (T, J, 3).
        - output_folder (str): Path to the folder where images will be saved.
        """
        if pose_sequence.shape[-1] != 2:
            T = pose_sequence.shape[0]
            pose_sequence = pose_sequence.reshape(T, -1, 2)

        T, J, _ = pose_sequence.shape

        colors = [
            [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], 
            [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], 
            [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], 
            [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], 
            [255, 0, 170], [255, 0, 85], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], 
            [170, 255, 0],
        ]
        colors = np.array(colors) / 255  # Scale RGB values to [0, 1]

        for t in range(T):
            plt.figure(figsize=(8, 6))
            plt.title(f"Frame {t+1}")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.grid(True)

            # plt.plot(epipoles[0], epipoles[1], 'rx') 

            # Plot each joint and its corresponding line
            for j in range(J):
                x0, y0 = pose_sequence[t, j]
                a, b, c = line_coeffs[t, j]

                # # Plot the joint
                # plt.plot(x0, y0, 'ro')  # Red dot for the joint

                # Plot the joint
                plt.plot(x0, y0, 'o', color=colors[j]) 

                # Generate line for plotting
                if b != 0:
                    # Generate x values
                    x_values = np.linspace(-10, 10, 400)
                    # Compute corresponding y values from the line equation
                    y_values = (-a * x_values - c) / b
                    # plt.plot(x_values, y_values, 'b-')  # Blue line for the corresponding line
                    plt.plot(x_values, y_values, '-', color=colors[j])  # Blue line for the corresponding line
                else:
                    # If b is 0, it's a vertical line
                    # plt.axvline(-c/a, color='b')

                    plt.axvline(-c/a, color=colors[j])

            # Save the plot to an image file
            if vis_gt:
                plt.savefig(f"{output_folder}/frame_{t+1}_gt.png")
            else:
                plt.savefig(f"{output_folder}/frame_{t+1}.png")
            plt.close()

    def gen_vis_res(self, motion_2d_seq, vis_folder_tag, actual_seq_len, dest_vis_folder_path=None, vis_gt=False):
        # motion_2d_seq: BS X T X D (18 * 2) 
        # actual_seq_len: BS 

        if dest_vis_folder_path is None:
            dest_vis_folder_path = os.path.join(self.vis_folder, vis_folder_tag)
        else:
            dest_vis_folder_path = os.path.join(dest_vis_folder_path, vis_folder_tag)

        if not os.path.exists(dest_vis_folder_path):
            os.makedirs(dest_vis_folder_path)

        num_seq, num_steps, _ = motion_2d_seq.shape
        motion_2d_seq = motion_2d_seq.reshape(num_seq, num_steps, -1, 2) # BS X T X 18 X 2 

        de_motion_2d_seq = de_normalize_pose2d(motion_2d_seq)

        for seq_idx in range(num_seq):
            if vis_gt:
                dest_vid_path = os.path.join(dest_vis_folder_path, "seq_"+str(seq_idx)+"_gt.mp4")
            else:
                dest_vid_path = os.path.join(dest_vis_folder_path, "seq_"+str(seq_idx)+".mp4")

            curr_seq_actual_len = actual_seq_len[seq_idx] - 1 
            gen_2d_motion_vis(de_motion_2d_seq[seq_idx, \
                        :curr_seq_actual_len].detach().cpu().numpy(), dest_vid_path, \
                        use_smpl_jnts13=False, use_animal_pose17=self.use_animal_data, \
                        use_interaction_pose23=self.use_omomo_data) 

    def gen_multiview_vis_res(self, motion_2d_seq, vis_folder_tag, actual_seq_len, \
        dest_vis_data_folder=None, vis_gt=False, file_name_tag=""):
        # motion_2d_seq: BS X K X T X D (18 * 2) 
        # actual_seq_len: BS 

        if dest_vis_data_folder is None:
            dest_vis_folder_path = os.path.join(self.vis_folder, vis_folder_tag)
        else:
            dest_vis_folder_path = os.path.join(dest_vis_data_folder, vis_folder_tag) 
        
        if not os.path.exists(dest_vis_folder_path):
            os.makedirs(dest_vis_folder_path)

        num_seq, num_views, num_steps, _ = motion_2d_seq.shape
        motion_2d_seq = motion_2d_seq.reshape(num_seq, num_views, num_steps, -1, 2) # BS X K X T X 18 X 2 

        de_motion_2d_seq = de_normalize_pose2d(motion_2d_seq.reshape(num_seq*num_views, \
                num_steps, -1, 2)) # (BS*K) X T X 18 X 2 
        de_motion_2d_seq = de_motion_2d_seq.reshape(num_seq, num_views, num_steps, -1, 2) # BS X K X T X 18 X 2 

        for seq_idx in range(num_seq):
            for view_idx in range(num_views):
                if vis_gt:
                    dest_vid_path = os.path.join(dest_vis_folder_path, \
                    "seq_"+str(seq_idx)+"_view"+str(view_idx)+"_gt.mp4")
                else:
                    dest_vid_path = os.path.join(dest_vis_folder_path, \
                    "seq_"+str(seq_idx)+"_view"+str(view_idx)+file_name_tag+".mp4")

                curr_seq_actual_len = int(actual_seq_len[seq_idx, view_idx] - 1)
                if not self.eval_w_best_mpjpe:
                    gen_2d_motion_vis(de_motion_2d_seq[seq_idx, view_idx, \
                                :curr_seq_actual_len].detach().cpu().numpy(), dest_vid_path, \
                                use_smpl_jnts13=False, use_animal_pose17=self.use_animal_data, \
                                use_interaction_pose23=self.use_omomo_data)  

        return de_motion_2d_seq 

def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Define model  
    if opt.use_animal_data:
        repr_dim = 17 * 2
    elif opt.use_omomo_data:
        repr_dim = 18 * 2 + 5 * 2 # For omomo data, we have 5 additional joints for object keypoints.
    else:
        repr_dim = 18 * 2
    
    loss_type = "l1"
    
    diffusion_model = CondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, \
                # batch_size=opt.batch_size, \
                train_2d_diffusion_w_line_cond=opt.train_2d_diffusion_w_line_cond, \
                use_animal_data=opt.use_animal_data, use_omomo_data=opt.use_omomo_data, omomo_object_name=opt.omomo_object_name)
   
    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=600000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=False,                        # turn on mixed precision
        results_folder=str(wdir),
    )

    trainer.train()

    torch.cuda.empty_cache()

def run_sample(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'

    # Define model     
    if opt.use_animal_data:
        repr_dim = 17 * 2
    elif opt.use_omomo_data:
        repr_dim = 18 * 2 + 5 * 2 # For omomo data, we have 5 additional joints for object keypoints.
    else:
        repr_dim = 18 * 2
   
    loss_type = "l1"
    
    diffusion_model = CondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, \
                train_2d_diffusion_w_line_cond=opt.train_2d_diffusion_w_line_cond, \
                use_animal_data=opt.use_animal_data, use_omomo_data=opt.use_omomo_data, omomo_object_name=opt.omomo_object_name) 

    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=200000,         # total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=False,                        # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False, 
    )
   
    if opt.gen_synthetic_3d_data_w_our_model:
        if opt.use_animal_data:
            trainer.gen_synthetic_3d_data_w_trained_2d_motion_diffusion_for_animal() 
        elif opt.use_omomo_data:
            trainer.gen_synthetic_3d_data_w_trained_2d_motion_diffusion_for_interaction()
        else:
            trainer.gen_synthetic_3d_data_w_trained_2d_motion_diffusion() 
    else:
        trainer.sample_2d_motion_w_line_conditioned_diffusion() 

    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--wandb_pj_name', type=str, default='', help='project name')
    parser.add_argument('--entity', default='wandb_account_name', help='W&B entity')
    parser.add_argument('--exp_name', default='', help='save to project/name')
    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--window', type=int, default=120, help='horizon')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')

    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    # For testing sampled results 
    parser.add_argument("--test_sample_res", action="store_true")
    # For testing sampled results on training dataset 
    parser.add_argument("--test_sample_res_on_train", action="store_true")

    parser.add_argument('--youtube_data_npz_folder', default='data', help='root folder for dataset')
    parser.add_argument('--youtube_train_val_json_path', default='data', help='json file for train/val split')

    parser.add_argument("--optimize_3d_w_trained_2d_diffusion", action="store_true") 
    parser.add_argument("--aist_optimize_3d_w_trained_2d_diffusion", action="store_true")

    parser.add_argument("--train_2d_diffusion_w_line_cond", action="store_true")
    parser.add_argument("--test_2d_diffusion_w_line_cond", action="store_true")

    parser.add_argument("--eval_w_best_mpjpe", action="store_true")

    parser.add_argument("--use_init_pose_only_model_for_sds", action="store_true")

    parser.add_argument("--center_all_jnts2d", action="store_true")

    parser.add_argument("--gen_more_consistent_2d_seq", action="store_true")

    parser.add_argument("--add_motionbert_for_eval", action="store_true")

    parser.add_argument("--gen_synthetic_3d_data_w_our_model", action="store_true")

    parser.add_argument("--gen_synthetic_3d_data_for_training", action="store_true")

    parser.add_argument("--use_animal_data", action="store_true")

    parser.add_argument("--use_omomo_data", action="store_true")

    parser.add_argument('--add_vposer_opt_w_2d', action="store_true")

    parser.add_argument('--omomo_object_name', type=str, default='largebox', help='project name')

    parser.add_argument('--ablation_sds_loss_only', action="store_true")

    parser.add_argument('--ablation_multi_step_recon_loss_only', action="store_true")

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device)
    else:
        run_train(opt, device)
