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

from m2d.data.youtube_motion2d_dataset import YoutubeMotion2D
from m2d.data.aist_motion3d_dataset import AISTPose3D, run_smpl_forward, convert_smpl_jnts_to_openpose18, convert_h36m17_to_openpose18, convert_coco17_to_openpose18 

from m2d.data.omomo_2d_dataset import OMOMODataset 

from m2d.data.synthetic3d_multiview_2d_dataset import MultiViewSyntheticGenMotion2D

from m2d.data.utils_2d_pose_normalize import normalize_pose2d, de_normalize_pose2d 
from m2d.data.utils_multi_view_2d_motion import get_multi_view_cam_extrinsic

from m2d.model.transformer_motion2d_diffusion_model import MultiViewCondGaussianDiffusion 

from m2d.vis.vis_jnts import plot_3d_motion, gen_2d_motion_vis, plot_pose_2d_for_paper, plot_pose_2d_omomo_for_paper, plot_pose_2d_cat_for_paper
from m2d.vis.vis_jnts import plot_multiple_trajectories, visualize_pose_sequence_to_video_for_demo, visualize_pose_sequence_to_video_for_demo_interaction 

from m2d.lafan1.utils import normalize, quat_normalize, quat_between, quat_mul, quat_inv, quat_mul_vec 

from evaluation_metrics import compute_metrics_for_jpos, calculate_bone_lengths, compute_metrics_for_jpos_wo_root, compute_metrics_for_interaction_contact 

from m2d.vis.vis_camera_motion import vis_multiple_head_pose_traj 

from m2d.data.utils_multi_view_2d_motion import perspective_projection

import time 

import joblib 

import glob 

import cv2 

import shutil 
# torch.manual_seed(1)
# random.seed(1)

# For debugging 
# Generate a dynamic seed using current time
seed = int(time.time())  # Or use any other dynamic value

# Set the seeds for reproducibility across runs
torch.manual_seed(seed)
random.seed(seed)

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
        save_and_sample_every=10000,
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

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.youtube_data_npz_folder = opt.youtube_data_npz_folder 
        self.youtube_train_val_json_path = opt.youtube_train_val_json_path 

        self.train_2d_diffusion_w_line_cond = opt.train_2d_diffusion_w_line_cond 
        self.test_2d_diffusion_w_line_cond = opt.test_2d_diffusion_w_line_cond 

        self.use_animal_data = opt.use_animal_data 

        self.use_omomo_data = opt.use_omomo_data 
        self.omomo_object_name = opt.omomo_object_name 

        self.opt = opt 

        self.window = opt.window

        self.train_num_views = opt.train_num_views  

        self.add_vposer_opt_w_2d = opt.add_vposer_opt_w_2d 

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.use_cfg = opt.use_cfg 
        self.cfg_scale = opt.cfg_scale 
       
        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)

        self.eval_on_aist_3d = opt.eval_on_aist_3d 

        self.test_sample_res = opt.test_sample_res 

        if self.eval_on_aist_3d:
            self.prep_aist_3d_dataloader() 
        else:
            if self.opt.test_sample_res:
                if self.use_omomo_data:
                    self.prep_omomo_dataloader()
                else:
                    self.prep_dataloader() 
            else:
                self.prep_multiview_dataloader()

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.test_on_train = self.opt.test_sample_res_on_train 

        self.eval_w_best_mpjpe = self.opt.eval_w_best_mpjpe 

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
            sample_n_frames=self.window, train=True, use_smpl_jnts13=False, \
            center_all_jnts2d=False, \
            use_fixed_epipoles=False, \
            use_decomposed_traj_rep=False, \
            input_ref_2d_seq_condition=False, use_animal_data=self.use_animal_data)
        
        val_youtube_dataset = YoutubeMotion2D(json_path=self.youtube_train_val_json_path, \
            npz_folder=self.youtube_data_npz_folder, \
            sample_n_frames=self.window, train=False, use_smpl_jnts13=False, \
            for_eval=self.test_sample_res, \
            center_all_jnts2d=False, \
            use_fixed_epipoles=False, \
            use_decomposed_traj_rep=False, \
            input_ref_2d_seq_condition=False, use_animal_data=self.use_animal_data)

        self.ds = train_youtube_dataset 
        self.val_ds = val_youtube_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, \
            shuffle=True, pin_memory=True, num_workers=4))

        if self.test_2d_diffusion_w_line_cond:
            self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=1, \
                shuffle=False, pin_memory=True, num_workers=4))
        else:
            self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, \
                shuffle=False, pin_memory=True, num_workers=4))

    def prep_multiview_dataloader(self):
        # Define dataset
        train_multiview_dataset = MultiViewSyntheticGenMotion2D(npz_folder=self.youtube_data_npz_folder, \
            sample_n_frames=self.window, train=True, num_views=self.train_num_views)
        
        val_multiview_dataset = MultiViewSyntheticGenMotion2D(npz_folder=self.youtube_data_npz_folder, \
            sample_n_frames=self.window, train=False, num_views=self.train_num_views)  

        self.ds = train_multiview_dataset 
        self.val_ds = val_multiview_dataset

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

        if self.opt.add_elepose_for_eval:
            self.elepose_mpjpe_list = []
            self.elepose_pa_mpjpe_list = []
            self.elepose_trans_list = []
            self.elepose_mpjpe_w_trans_list = [] 

            self.elepose_jpos2d_err_list = []
            self.elepose_centered_jpos2d_err_list = []

            self.elepose_bone_consistency_list = []

        if self.opt.add_vposer_opt_w_2d:
            self.smplify_mpjpe_list = []
            self.smplify_pa_mpjpe_list = []
            self.smplify_trans_list = []
            self.smplify_mpjpe_w_trans_list = [] 

            self.smplify_jpos2d_err_list = []
            self.smplify_centered_jpos2d_err_list = []

            self.smplify_bone_consistency_list = []

        if self.opt.add_mas_for_eval:
            self.mas_mpjpe_list = []
            self.mas_pa_mpjpe_list = []
            self.mas_trans_list = []
            self.mas_mpjpe_w_trans_list = [] 

            self.mas_jpos2d_err_list = []
            self.mas_centered_jpos2d_err_list = []

            self.mas_bone_consistency_list = []

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

        if self.opt.add_motionbert_for_eval:
            # Prepare for MotionBERT evaluation
            self.mbert_mpjpe_list.append(mpjpe_mbert)
            self.mbert_pa_mpjpe_list.append(pa_mpjpe_mbert)
            self.mbert_trans_list.append(trans_err_mbert)
            self.mbert_mpjpe_w_trans_list.append(jpos_err_mbert)

            self.mbert_jpos2d_err_list.append(jpos2d_err_mbert)
            self.mbert_centered_jpos2d_err_list.append(centered_jpos2d_err_mbert) 

            self.mbert_bone_consistency_list.append(bone_consistency_mbert) 

        if self.opt.add_elepose_for_eval:
            self.elepose_mpjpe_list.append(mpjpe_elepose)
            self.elepose_pa_mpjpe_list.append(pa_mpjpe_elepose)
            self.elepose_trans_list.append(trans_err_elepose)
            self.elepose_mpjpe_w_trans_list.append(jpos_err_elepose) 

            self.elepose_jpos2d_err_list.append(jpos2d_err_elepose)
            self.elepose_centered_jpos2d_err_list.append(centered_jpos2d_err_elepose)

            self.elepose_bone_consistency_list.append(bone_consistency_elepose)

        if self.opt.add_vposer_opt_w_2d:
            self.smplify_mpjpe_list.append(mpjpe_smplify)
            self.smplify_pa_mpjpe_list.append(pa_mpjpe_smplify)
            self.smplify_trans_list.append(trans_err_smplify)
            self.smplify_mpjpe_w_trans_list.append(jpos_err_smplify) 

            self.smplify_jpos2d_err_list.append(jpos2d_err_smplify)
            self.smplify_centered_jpos2d_err_list.append(centered_jpos2d_err_smplify)

            self.smplify_bone_consistency_list.append(bone_consistency_smplify)

        if self.opt.add_mas_for_eval:
            self.mas_mpjpe_list.append(mpjpe_mas)
            self.mas_pa_mpjpe_list.append(pa_mpjpe_mas)
            self.mas_trans_list.append(trans_err_mas)
            self.mas_mpjpe_w_trans_list.append(jpos_err_mas) 

            self.mas_jpos2d_err_list.append(jpos2d_err_mas)
            self.mas_centered_jpos2d_err_list.append(centered_jpos2d_err_mas)

            self.mas_bone_consistency_list.append(bone_consistency_mas) 

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

        self.print_metrics_for_each_method(self.wham_mpjpe_list, self.wham_pa_mpjpe_list, self.wham_trans_list, self.wham_mpjpe_w_trans_list, \
            self.wham_jpos2d_err_list, self.wham_centered_jpos2d_err_list, self.wham_bone_consistency_list, "WHAM")

        if self.opt.add_motionbert_for_eval:
            self.print_metrics_for_each_method(self.mbert_mpjpe_list, self.mbert_pa_mpjpe_list, self.mbert_trans_list, self.mbert_mpjpe_w_trans_list, \
                self.mbert_jpos2d_err_list, self.mbert_centered_jpos2d_err_list, self.mbert_bone_consistency_list, "MotionBERT")

        if self.opt.add_elepose_for_eval:
            self.print_metrics_for_each_method(self.elepose_mpjpe_list, self.elepose_pa_mpjpe_list, self.elepose_trans_list, self.elepose_mpjpe_w_trans_list, \
                self.elepose_jpos2d_err_list, self.elepose_centered_jpos2d_err_list, self.elepose_bone_consistency_list, "ElePose")

        if self.opt.add_vposer_opt_w_2d:
            self.print_metrics_for_each_method(self.smplify_mpjpe_list, self.smplify_pa_mpjpe_list, self.smplify_trans_list, self.smplify_mpjpe_w_trans_list, \
                self.smplify_jpos2d_err_list, self.smplify_centered_jpos2d_err_list, self.smplify_bone_consistency_list, "SMPLify") 

        if self.opt.add_mas_for_eval:
            self.print_metrics_for_each_method(self.mas_mpjpe_list, self.mas_pa_mpjpe_list, self.mas_trans_list, self.mas_mpjpe_w_trans_list, \
                self.mas_jpos2d_err_list, self.mas_centered_jpos2d_err_list, self.mas_bone_consistency_list, "MAS")

    def prep_evaluation_metric_list_for_interaction(self):
        self.mpjpe_list = []
        self.pa_mpjpe_list = []
        self.trans_list = []
        self.mpjpe_w_trans_list = [] # Directly global joint position error. 

        self.jpos2d_err_list = [] 
        self.centered_jpos2d_err_list = [] 

        self.object_mpjpe_list = [] 
        self.object_trans_err_list = []

        self.bone_consistency_list = []

        if self.opt.add_vposer_opt_w_2d:
            self.smplify_mpjpe_list = []
            self.smplify_pa_mpjpe_list = []
            self.smplify_trans_list = []
            self.smplify_mpjpe_w_trans_list = [] 

            self.smplify_jpos2d_err_list = []
            self.smplify_centered_jpos2d_err_list = []

            self.smplify_bone_consistency_list = []

            self.smplify_object_mpjpe_list = [] 
            self.smplify_object_trans_err_list = [] 

    def add_new_evaluation_to_list_for_interaction(self, mpjpe, pa_mpjpe, trans_err, jpos_err, jpos2d_err, \
        centered_jpos2d_err, bone_consistency, obj_mpjpe, obj_trans_err, \
        mpjpe_smplify=0, pa_mpjpe_smplify=0, trans_err_smplify=0, jpos_err_smplify=0, \
        jpos2d_err_smplify=0, centered_jpos2d_err_smplify=0, bone_consistency_smplify=0, \
        obj_mpjpe_smplify=0, obj_trans_err_smplify=0):
        
        self.mpjpe_list.append(mpjpe)
        self.pa_mpjpe_list.append(pa_mpjpe) 
        self.trans_list.append(trans_err)
        self.mpjpe_w_trans_list.append(jpos_err)

        self.jpos2d_err_list.append(jpos2d_err)
        self.centered_jpos2d_err_list.append(centered_jpos2d_err)
       
        self.bone_consistency_list.append(bone_consistency) 

        self.object_mpjpe_list.append(obj_mpjpe)
        self.object_trans_err_list.append(obj_trans_err)

        if self.opt.add_vposer_opt_w_2d:
            self.smplify_mpjpe_list.append(mpjpe_smplify)
            self.smplify_pa_mpjpe_list.append(pa_mpjpe_smplify)
            self.smplify_trans_list.append(trans_err_smplify)
            self.smplify_mpjpe_w_trans_list.append(jpos_err_smplify) 

            self.smplify_jpos2d_err_list.append(jpos2d_err_smplify)
            self.smplify_centered_jpos2d_err_list.append(centered_jpos2d_err_smplify)

            self.smplify_bone_consistency_list.append(bone_consistency_smplify)

            self.smplify_object_mpjpe_list.append(obj_mpjpe_smplify) 
            self.smplify_object_trans_err_list.append(obj_trans_err_smplify) 

    def print_metrics_for_each_method_for_interaction(self, mpjpe_list, pa_mpjpe_list, trans_list, mpjpe_w_trans_list, \
            jpos2d_err_list, centered_jpos2d_err_list, bone_consistency_list, obj_mpjpe_list, obj_trans_err_list, method_name):
        mpjpe_arr = np.asarray(mpjpe_list)
        pa_mpjpe_arr = np.asarray(pa_mpjpe_list)
        trans_arr = np.asarray(trans_list)
        mpjpe_w_trans_arr = np.asarray(mpjpe_w_trans_list)

        jpos2d_err_arr = np.asarray(jpos2d_err_list)
        centered_jpos2d_err_arr = np.asarray(centered_jpos2d_err_list)

        bone_consistency_arr = np.asarray(bone_consistency_list) 

        obj_mpjpe = np.asarray(obj_mpjpe_list)
        obj_trans_err = np.asarray(obj_trans_err_list)

        print("**************"+method_name+" Evaluation******************")
        print("The number of sequences: {0}".format(len(mpjpe_list)))
        print("MPJPE: {0}".format(mpjpe_arr.mean()))
        print("PA-MPJPE: {0}".format(pa_mpjpe_arr.mean()))
        print("Root Trans Err: {0}".format(trans_arr.mean()))
        print("MPJPE w Trans Err: {0}".format(mpjpe_w_trans_arr.mean()))
        print("Jpos2D Err: {0}".format(jpos2d_err_arr.mean()))
        print("Centered Jpos2D Err: {0}".format(centered_jpos2d_err_arr.mean()))
        print("Bone Consistency: {0}".format(bone_consistency_arr.mean()))
        print("Object kpts wo trans: {0}".format(obj_mpjpe.mean()))
        print("Object kpts trans err: {0}".format(obj_trans_err.mean()))

    def print_mean_metrics_for_interaction(self):
        self.print_metrics_for_each_method_for_interaction(self.mpjpe_list, self.pa_mpjpe_list, self.trans_list, self.mpjpe_w_trans_list, \
            self.jpos2d_err_list, self.centered_jpos2d_err_list, self.bone_consistency_list, \
            self.object_mpjpe_list, self.object_trans_err_list, "Ours")

        if self.add_vposer_opt_w_2d:
            self.print_metrics_for_each_method_for_interaction(self.smplify_mpjpe_list, self.smplify_pa_mpjpe_list, \
                self.smplify_trans_list, self.smplify_mpjpe_w_trans_list, \
                self.smplify_jpos2d_err_list, self.smplify_centered_jpos2d_err_list, \
                self.smplify_bone_consistency_list, self.smplify_object_mpjpe_list, self.smplify_object_trans_err_list, "SMPLify") 

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

    def prep_vis_res_folder(self):
        # dest_res3d_root_folder = "/viscam/projects/mofitness/final_cvpr25_opt_3d_w_multiview_diffusion"

        dest_res3d_root_folder = "/move/u/jiamanli/final_cvpr25_opt_3d_w_multiview_diffusion"

        if "AIST" in self.youtube_data_npz_folder:
            dest_res3d_npy_folder = os.path.join(dest_res3d_root_folder, "AIST")
        elif "nicole" in self.youtube_data_npz_folder:
            dest_res3d_npy_folder = os.path.join(dest_res3d_root_folder, "nicole")
        elif "steezy" in self.youtube_data_npz_folder:
            dest_res3d_npy_folder = os.path.join(dest_res3d_root_folder, "steezy")
        elif "cat" in self.youtube_data_npz_folder:
            dest_res3d_npy_folder = os.path.join(dest_res3d_root_folder, "cat_data_final_debug")

        if self.opt.eval_on_aist_3d:
            dest_res3d_npy_folder = os.path.join(dest_res3d_root_folder, "AIST") 

        if self.use_omomo_data:
            dest_res3d_npy_folder = os.path.join(dest_res3d_root_folder, "omomo_data")

        if not self.eval_w_best_mpjpe:
            dest_res3d_npy_folder += "_for_vis"

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

        # Project WHAM 3D to the same view's 2D for comparison, may need to apply a scale? 
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

    def load_wham_res(self, npz_name, start_idx, end_idx):
        if "nicole" in self.youtube_data_npz_folder:
            wham_res_folder = "/viscam/projects/mofitness/datasets/nicole_move/clean_test_clips_wham_res" 
            npz_name = npz_name.replace(".npz", "") 

            ori_video_name = npz_name 
            ori_wham_res_path = os.path.join(wham_res_folder, ori_video_name, "wham_output.pkl")
            ori_seq_wham_data = joblib.load(ori_wham_res_path)

            sub_idx = 0 
        elif "steezy" in self.youtube_data_npz_folder:
            wham_res_folder = "/viscam/projects/mofitness/datasets/steezy_new/ori_clips_wham_res_complete"
            npz_name = npz_name.replace(".npz", "")

            ori_video_name = npz_name.split("_sub")[0]
            ori_wham_res_path = os.path.join(wham_res_folder, ori_video_name, "wham_output.pkl")
            ori_seq_wham_data = joblib.load(ori_wham_res_path)

            sub_idx = int(npz_name.split("_sub_")[1].split("_")[0])

        else:
            if "bb_data" in self.youtube_data_npz_folder:
                wham_res_folder = "/viscam/projects/mofitness/datasets/bb_data/ori_video_wham_res_complete"
            elif "FineGym" in self.youtube_data_npz_folder:
                wham_res_folder = "/viscam/projects/vimotion/datasets/FineGym/manual_selected_clips_wham_res_complete"

            ori_video_name = npz_name.split("_sub")[0]
            ori_wham_res_path = os.path.join(wham_res_folder, ori_video_name, "wham_output.pkl")
            ori_seq_wham_data = joblib.load(ori_wham_res_path)

            sub_idx = int(npz_name.split("_sub_")[1].split("_")[0])

        curr_sub_wham_data = ori_seq_wham_data[sub_idx]
        # dict_keys(['pose', 'trans', 'pose_world', 'trans_world', 'betas', 'verts', 'frame_ids'])

        # wham_local_poses = curr_sub_wham_data["pose"][start_idx:end_idx] 
        # wham_local_trans = curr_sub_wham_data['trans'][start_idx:end_idx]

        wham_world_poses = curr_sub_wham_data["pose_world"][start_idx:end_idx]
        if "bb_data" in self.youtube_data_npz_folder:
            wham_world_trans = curr_sub_wham_data["trans_world"][start_idx:end_idx]/5.5 # Heuristic scaling from adult to baby, based on scaling 1 scale
        else:
            wham_world_trans = curr_sub_wham_data["trans_world"][start_idx:end_idx]

        wham_world_poses = torch.from_numpy(wham_world_poses).float()
        wham_world_trans = torch.from_numpy(wham_world_trans).float()

        # wham_betas = curr_sub_wham_data["betas"][start_idx:end_idx]

        # First, apply rotation back to put human on floor z = 0 
        align_rot_mat = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        seq_len, _ = wham_world_poses.shape 
        align_rot_mat = torch.from_numpy(align_rot_mat).float()[None].repeat(seq_len,\
            1, 1) # T X 3 X 3 

        wham_world_trans = torch.matmul(align_rot_mat, \
            wham_world_trans[:, :, None])
        wham_world_trans = wham_world_trans.squeeze(-1) # T X 3 
      
        wham_global_rot_aa_rep = wham_world_poses[:, :3] # T X 3 
        wham_global_rot_mat = transforms.axis_angle_to_matrix(wham_global_rot_aa_rep) # T X 3 X 3 
        wham_global_rot_mat = torch.matmul(align_rot_mat, wham_global_rot_mat) # T X 3 X 3 

        wham_world_poses[:, :3] = transforms.matrix_to_axis_angle(wham_global_rot_mat) # T X 3 

        return wham_world_poses.detach().cpu().numpy(), wham_world_trans.detach().cpu().numpy()

    def load_motionbert_res(self, npz_name, start_idx=None, end_idx=None):
        # Not for bb_data 

        seq_name = npz_name.replace(".npz", "")

        if "AIST" in self.youtube_data_npz_folder:
            motionbert_res_folder = "/viscam/projects/vimotion/datasets/AIST/MotionBert_results_mesh"
            motionbert_jnts3d_data_path = os.path.join(motionbert_res_folder, seq_name+".npy")

            motionbert_jnts3d_h36m = np.load(motionbert_jnts3d_data_path) # T X 49 X 3 

            motionbert_jnts3d_h36m = motionbert_jnts3d_h36m[0:120]

            # Convert SMPL to COCO 18 
            motionbert_jnts18 = convert_smpl_jnts_to_openpose18(motionbert_jnts3d_h36m) # T X 18 X 3 

        elif "FineGym" in self.youtube_data_npz_folder:
            motionbert_res_folder = "/viscam/projects/vimotion/datasets/FineGym/MotionBert_results_mesh"
            motionbert_jnts3d_data_path = os.path.join(motionbert_res_folder, seq_name+".npy")
        elif "nicole" in self.youtube_data_npz_folder:
            motionbert_res_folder = "/viscam/projects/mofitness/datasets/nicole_move/MotionBert_results_mesh" 
            motionbert_jnts3d_data_path = os.path.join(motionbert_res_folder, seq_name+".npy")

            motionbert_jnts3d_h36m = np.load(motionbert_jnts3d_data_path) # T X 49 X 3 
            motionbert_jnts3d_h36m = motionbert_jnts3d_h36m[start_idx:end_idx]  

            motionbert_jnts18 = convert_smpl_jnts_to_openpose18(motionbert_jnts3d_h36m) # T X 18 X 3 
        elif "steezy" in self.youtube_data_npz_folder:
            motionbert_res_folder = "/viscam/projects/mofitness/datasets/steezy_new/MotionBert_results_mesh"
            motionbert_jnts3d_data_path = os.path.join(motionbert_res_folder, seq_name+"_"+str(start_idx)+"_"+str(end_idx)+".npy")

            motionbert_jnts3d_h36m = np.load(motionbert_jnts3d_data_path) # T X 49 X 3 

            motionbert_jnts18 = convert_smpl_jnts_to_openpose18(motionbert_jnts3d_h36m) # T X 18 X 3 

        # motionbert_jnts18 = convert_h36m17_to_openpose18(motionbert_jnts3d_h36m) # T X 18 X 3 

        return motionbert_jnts18 

    def load_elepose_res(self, npz_name, start_idx=None, end_idx=None):
        seq_name = npz_name.replace(".npz", "")

        if "AIST" in self.youtube_data_npz_folder:
            elepose_res_folder = "/viscam/projects/vimotion/datasets/AIST/elepose_results"
        elif "nicole" in self.youtube_data_npz_folder:
            elepose_res_folder = "/viscam/projects/mofitness/datasets/nicole_move/elepose_results" 
        elif "steezy" in self.youtube_data_npz_folder:
            elepose_res_folder = "/viscam/projects/mofitness/datasets/steezy_new/elepose_results"
    
        if "steezy" in self.youtube_data_npz_folder or "nicole" in self.youtube_data_npz_folder:
            elepose_jnts3d_data_path = os.path.join(elepose_res_folder, seq_name+"_sidx_"+str(start_idx)+"_eidx_"+str(end_idx)+".npy")
        
            elepose_jnts3d_h36m = np.load(elepose_jnts3d_data_path) # T X 17 X 3 
        else:
            elepose_jnts3d_data_path = os.path.join(elepose_res_folder, seq_name+".npy")
        
            elepose_jnts3d_h36m = np.load(elepose_jnts3d_data_path) # T X 17 X 3 

            if start_idx is not None and end_idx is not None:
                elepose_jnts3d_h36m = elepose_jnts3d_h36m[start_idx:end_idx] 

        # Convert h36m joints17 to COCO 18 
        elepose_jnts18 = convert_h36m17_to_openpose18(elepose_jnts3d_h36m) # T X 18 X 3 

        return elepose_jnts18 

    def load_mas_res(self, npz_name, start_idx=None, end_idx=None):
        seq_name = npz_name.replace(".npz", "")

        if "AIST" in self.youtube_data_npz_folder:
            mas_res_folder = "/viscam/projects/vimotion/datasets/AIST/MAS_results"
        elif "steezy" in self.youtube_data_npz_folder:
            mas_res_folder = "/viscam/projects/mofitness/datasets/steezy_new/MAS_results" 
        elif "nicole" in self.youtube_data_npz_folder:
            mas_res_folder = "/viscam/projects/mofitness/datasets/nicole_move/MAS_results" 
        elif "cat" in self.youtube_data_npz_folder:
            mas_res_folder = "/viscam/projects/mofitness/datasets/cat_data/MAS_results" 


        if "cat" in self.youtube_data_npz_folder:
            mas_jnts3d_data_path = os.path.join(mas_res_folder, seq_name+".npz"+"_"+str(start_idx)+"_"+str(end_idx)+".npy") 

            mas_jnts3d_h36m = np.load(mas_jnts3d_data_path) # T X 17 X 3 

            return mas_jnts3d_h36m 
        else:
            if start_idx is not None and end_idx is not None:
                mas_jnts3d_data_path = os.path.join(mas_res_folder, seq_name+".npy"+"_"+str(start_idx)+"_"+str(end_idx)+".npy") 
            else:
                mas_jnts3d_data_path = os.path.join(mas_res_folder, seq_name+".npy") 

            mas_jnts3d_h36m = np.load(mas_jnts3d_data_path) # T X 17 X 3 

            # Convert h36m joints17 to COCO 18 
            mas_jnts18 = convert_coco17_to_openpose18(mas_jnts3d_h36m) # T X 18 X 3 

        return mas_jnts18 

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

    def align_motionbert_to_ours(self, motionbert_jnts18, gt_jnts18, motionbert_verts, input_obj_kpts=False): 
        num_joints = motionbert_jnts18.shape[1] 

        motionbert_jnts18 = motionbert_jnts18[:gt_jnts18.shape[0]]
        motionbert_verts = motionbert_verts[:gt_jnts18.shape[0]]

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

        # Calculate the scaling factor
        # The scale factor is the ratio of the sum of variances (or norms) of the centered points
        scale_factor = np.sum(np.linalg.norm(X2, axis=1)) / np.sum(np.linalg.norm(X1, axis=1))

        # Ensure mu1_squeezed is reshaped to [T, 3, 1]
        mu1_squeezed = mu1.squeeze(axis=2)[:, :, np.newaxis]
        mu2_squeezed = mu2.squeeze(axis=2)

        # Correct matrix multiplication
        # t = mu2_squeezed - np.matmul(R, mu1_squeezed)[:, :, 0] # T X 3 

        align_rot_mat = torch.from_numpy(R).float()[0:1] # 1 X 3 X 3 
        align_rot_mat = align_rot_mat.repeat(motionbert_jnts18.shape[0], 1, 1) # T X 3 X 3
        motionbert_jnts18 = torch.from_numpy(motionbert_jnts18).float() # T X 18 X 3
 
        if input_obj_kpts:
            # Apply rotation and scaling
            aligned_jnts18 = torch.matmul(
                align_rot_mat[:, None].repeat(1, num_joints, 1, 1), 
                (motionbert_jnts18 * scale_factor)[:, :, :, None]
            )  # T X 18 X 3 X 1

        else:
            # Apply rotation and scaling
            aligned_jnts18 = torch.matmul(
                align_rot_mat[:, None].repeat(1, num_joints, 1, 1), 
                (motionbert_jnts18 * scale_factor)[:, :, :, None]
            )  # T X 18 X 3 X 1

        aligned_jnts18 = aligned_jnts18.squeeze(-1)  # T X 18 X 3

        # Apply rotation and scaling
        if not torch.is_tensor(motionbert_verts):
            motionbert_verts = torch.from_numpy(motionbert_verts).float().to(align_rot_mat.device)

       
        if input_obj_kpts:
            aligned_verts= torch.matmul(
                align_rot_mat[:, None].repeat(1, motionbert_verts.shape[1], 1, 1), 
                (motionbert_verts * scale_factor)[:, :, :, None]
            )  # T X Nv X 3 X 1
        else:
            aligned_verts= torch.matmul(
                align_rot_mat[:, None].repeat(1, motionbert_verts.shape[1], 1, 1), 
                motionbert_verts[:, :, :, None]
            )  # T X Nv X 3 X 1

        aligned_verts = aligned_verts.squeeze(-1)  # T X Nv X 3

        if input_obj_kpts:
            return aligned_jnts18, aligned_verts, scale_factor 
        # return aligned_jnts18, aligned_verts.detach().cpu().numpy() 
        return aligned_jnts18, aligned_verts  

    def train(self):
        init_step = self.step 
        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()

            nan_exists = False # If met nan in loss or gradient, need to skip to next data. 
            for i in range(self.gradient_accumulate_every):
                data_dict = next(self.dl)
              
                data = data_dict['normalized_jnts2d'].float().cuda() # BS X K X T X 18 X 2 
               
                bs, num_views, num_steps, num_joints, _ = data.shape 

                # Use the first camera view's 2D pose sequence as input condition 
                cond_data = data[:, 0:1] # BS X 1 X T X 18 X 2 
                cond_data = cond_data.repeat(1, num_views, 1, 1, 1) # BS X K X T X 18 X 2 

                # Generate padding mask 
                actual_seq_len = data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                tmp_mask = torch.arange(self.window+1).expand(data.shape[0], \
                self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1) # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(data.device).repeat(1, num_views, 1) # BS X K X T 
                padding_mask = padding_mask.reshape(bs*num_views, 1, -1) # (BS*K) X 1 X T

                data = data.reshape(bs*num_views, num_steps, -1) # (BS*K) X T X D(18*2)

                cond_data = cond_data.reshape(bs*num_views, num_steps, -1) # (BS*K) X T X D(18*2) 
                
                if self.train_2d_diffusion_w_line_cond:
                    all_line_conditions = all_line_conditions.reshape(bs*num_views, num_steps, -1) # (BS*K) X T X (18*3)

                    cond_data = torch.cat((cond_data, all_line_conditions), dim=-1) # (BS*K) X T X (18*2+18*3)

                language_input = None 

                with autocast(enabled = self.amp):
                    if self.train_2d_diffusion_w_line_cond:   
                        loss_diffusion, loss_line = self.model(data, cond_data=cond_data, \
                            padding_mask=padding_mask, \
                            language_input=language_input, \
                            use_cfg=self.use_cfg)
                        loss = loss_diffusion + loss_line 
                    else:
                        loss_diffusion = self.model(data, cond_data=cond_data, \
                            padding_mask=padding_mask, \
                            language_input=language_input, \
                            use_cfg=self.use_cfg)
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
                                "Train/Loss/Line Loss": loss_line.item(),
                                "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                            }
                        else:
                            log_dict = {
                                "Train/Loss/Total Loss": loss.item(),
                                "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                            }
                        wandb.log(log_dict)

                    if idx % 20 == 0 and i == 0:
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

                    val_bs, val_num_views, val_num_steps, num_joints, _ = val_data.shape 

                    val_cond_data = val_data[:, 0:1] # BS X 1 X T X 18 X 2
                    val_cond_data = val_cond_data.repeat(1, val_num_views, 1, 1, 1) # BS X K X T X 18 X 2

                    # Generate padding mask 
                    actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                    tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
                    self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1) # BS X max_timesteps
                    padding_mask = tmp_mask[:, None, :].to(val_data.device) # BS X 1 X T 
                    padding_mask = padding_mask.repeat(1, val_num_views, 1).reshape(val_bs*val_num_views, \
                        1, -1) # (BS*K) X 1 X T 

                    val_data = val_data.reshape(val_bs*val_num_views, val_num_steps, -1) # (BS*K) X T X D(18*2)

                    # Use the first camera view's 2D pose sequence as input condition
                    val_cond_data = val_cond_data.reshape(val_bs*val_num_views, val_num_steps, -1) # (BS*K) X T X D(18*2) 
                    
                    if self.train_2d_diffusion_w_line_cond:
                        val_all_line_conditions = val_all_line_conditions.reshape(val_bs*val_num_views, val_num_steps, -1) # (BS*K) X T X (18*3)

                        val_cond_data = torch.cat((val_cond_data, val_all_line_conditions), dim=-1) # (BS*K) X T X (18*2+18*3)

                    language_input = None 

                    # Get validation loss 
                    if self.train_2d_diffusion_w_line_cond:
                        val_loss_diffusion, val_loss_line = self.model(val_data, cond_data=val_cond_data, \
                            padding_mask=padding_mask, \
                            language_input=language_input, use_cfg=self.use_cfg)
                        val_loss = val_loss_diffusion + val_loss_line
                    else:
                        val_loss_diffusion = self.model(val_data, cond_data=val_cond_data, \
                            padding_mask=padding_mask, \
                            language_input=language_input, use_cfg=self.use_cfg)
                        val_loss = val_loss_diffusion 
                    
                    if self.use_wandb:
                        if self.train_2d_diffusion_w_line_cond:
                            val_log_dict = {
                                "Validation/Loss/Total Loss": val_loss.item(),
                                "Validation/Loss/Line Loss": val_loss_line.item(),
                                "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
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
                                val_cond_data, padding_mask, language_input=language_input) # (BS*K) X T X D(18*2) 

                        val_data_for_vis = val_data.reshape(val_bs, val_num_views, val_num_steps, -1) # BS X K X T X D(18*2) 
                        all_res_list_for_vis = all_res_list.reshape(val_bs, val_num_views, val_num_steps, -1) # BS X K X T X D(18*2) 
                       
                        # Visualization
                        vis_folder_tag = str(self.step) + "_" + "motion2d_res"
                        num_vis = 4
                      
                        self.gen_multiview_vis_res(val_data_for_vis[:num_vis], vis_folder_tag, actual_seq_len[:num_vis], vis_gt=True)
                        self.gen_multiview_vis_res(all_res_list_for_vis[:num_vis], vis_folder_tag, actual_seq_len[:num_vis])

            self.step += 1

        print('training complete')

        if self.use_wandb:
            wandb.run.finish()

    def get_jnts3d_w_smpl_params(self, ori_opt_skel_seq_list, opt_skel_seq_list, scale_val_list, force_head_zero=False):
        pred_smpl_jnts18_list = []
        pred_smpl_verts_list = [] 

        batch = len(ori_opt_skel_seq_list) 
        for bs_idx in range(batch):
            ori_opt_skel_seq = ori_opt_skel_seq_list[bs_idx]
            curr_opt_skel_seq = opt_skel_seq_list[bs_idx] 
            curr_scale_val = scale_val_list[bs_idx] 

            pred_smpl_scaling = curr_opt_skel_seq['smpl_scaling'] # 1 
            pred_smpl_poses = curr_opt_skel_seq['smpl_poses'] # T X 24 X 3 
            pred_smpl_trans = curr_opt_skel_seq['smpl_trans'] # T X 3 

            if force_head_zero:
                pred_smpl_poses[:, 12] = 0.0 
                pred_smpl_poses[:, 15] = 0.0 

            pred_smpl_poses, pred_smpl_trans = self.move_smpl_seq_to_floor(pred_smpl_poses, pred_smpl_trans)
            pred_jnts3d_smpl, pred_verts_smpl = run_smpl_forward(pred_smpl_poses, \
                    pred_smpl_trans, pred_smpl_scaling, self.val_ds.smpl) 
            # T X 45 X 3, T X Nv X 3  
            pred_smpl_jnts18 = convert_smpl_jnts_to_openpose18(pred_jnts3d_smpl)

            pred_smpl_jnts18 = self.move_init3d_to_center(pred_smpl_jnts18)

            pred_smpl_jnts18_list.append(pred_smpl_jnts18)
            pred_smpl_verts_list.append(pred_verts_smpl) 

        return pred_smpl_jnts18_list, pred_smpl_verts_list 

    def get_jnts3d_w_smpl_params_for_interaction(self, opt_skel_seq_list):
        pred_smpl_jnts18_list = []
        pred_smpl_verts_list = [] 

        batch = len(opt_skel_seq_list) 
        for bs_idx in range(batch):
            # ori_opt_skel_seq = ori_opt_skel_seq_list[bs_idx]
            curr_opt_skel_seq = opt_skel_seq_list[bs_idx] 
            # curr_scale_val = scale_val_list[bs_idx] 

            pred_smpl_scaling = curr_opt_skel_seq['smpl_scaling'] # 1 
            pred_smpl_poses = curr_opt_skel_seq['smpl_poses'] # T X 24 X 3 
            pred_smpl_trans = curr_opt_skel_seq['smpl_trans'] # T X 3 

            # pred_smpl_poses, pred_smpl_trans = self.move_smpl_seq_to_floor(pred_smpl_poses, pred_smpl_trans)
            pred_jnts3d_smpl, pred_verts_smpl = run_smpl_forward(pred_smpl_poses, \
                    pred_smpl_trans, pred_smpl_scaling, self.val_ds.smpl) 
            # T X 45 X 3, T X Nv X 3  
            pred_smpl_jnts18 = convert_smpl_jnts_to_openpose18(pred_jnts3d_smpl)

            # pred_smpl_jnts18 = self.move_init3d_to_center(pred_smpl_jnts18)

            pred_smpl_jnts18_list.append(pred_smpl_jnts18)
            pred_smpl_verts_list.append(pred_verts_smpl) 

        return pred_smpl_jnts18_list, pred_smpl_verts_list 

    def opt_3d_w_learned_model(self, val_data_dict):
        val_data = val_data_dict['normalized_jnts2d'].float().cuda() # BS X T X 18 X 2 
        batch = val_data.shape[0]

        cam_extrinsic = self.prep_multiview_cam_extrinsic(num_views=self.opt.train_num_views) # 1 X K X 4 X 4 
        cam_extrinsic = cam_extrinsic.repeat(batch, 1, 1, 1) # BS X K X 4 X 4 
        
        val_data = val_data[:, None, :, :, :].repeat(1, cam_extrinsic.shape[1], 1, 1, 1) # BS X K X T X 18 X 2 

        val_bs, val_num_views, val_num_steps, num_joints, _ = val_data.shape 

        val_cond_data = val_data[:, 0:1] # BS X 1 X T X 18 X 2
        val_cond_data = val_cond_data.repeat(1, val_num_views, 1, 1, 1) # BS X K X T X 18 X 2

        # Generate padding mask 
        actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
        tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
        self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1) # BS X max_timesteps
        padding_mask = tmp_mask[:, None, :].to(val_data.device) # BS X 1 X T 
        padding_mask = padding_mask.repeat(1, val_num_views, 1).reshape(val_bs*val_num_views, \
            1, -1) # (BS*K) X 1 X T 

        val_data = val_data.reshape(val_bs*val_num_views, val_num_steps, -1) # (BS*K) X T X D(18*2)

        # Use the first camera view's 2D pose sequence as input condition
        val_cond_data = val_cond_data.reshape(val_bs*val_num_views, val_num_steps, -1) # (BS*K) X T X D(18*2) 

        language_input = None 

        direct_sampled_epi_seq_2d = self.ema.ema_model.sample(val_data, \
                    val_cond_data, padding_mask, language_input=language_input) # (BS*K) X T X D(18*2) 
        direct_sampled_epi_seq_2d = direct_sampled_epi_seq_2d.reshape(val_bs, val_num_views, \
                    val_num_steps, -1) # BS X K X T X D(18*2) 

        # Replace the first camera view's 2D with the inpit 2D pose sequence 
        direct_sampled_epi_seq_2d[:, 0] = val_data.reshape(val_bs, val_num_views, val_num_steps, -1)[:, 0] # BS X K X T X D(18*2) 

        epi_line_cond_seq_2d = direct_sampled_epi_seq_2d # BS X K X T X D(18*2) 

        good_views_idx_list = None 
        epi_line_cond_seq_2d, opt_jpos3d_18 = \
                    self.ema.ema_model.opt_3d_w_multi_view_2d_res(epi_line_cond_seq_2d, \
                    cam_extrinsic, padding_mask=padding_mask)
        # BS X K X T X (18*2), BS X T X 18 X 3, BS X T X 18 X 3 

        pred_smpl_jnts18_list = [] 
        gt_smpl_jnts18_list = [] 
        wham_jpos3d_18_list = []

        pred_verts_smpl_list = []
        gt_verts_smpl_list = [] 
        wham_verts_smpl_list = [] 
        
        centered_wham_jnts3d_18_list = [] 
        centered_gt_smpl_jnts18_list = [] 
        centered_pred_smpl_jnts18_list = [] 
        centered_opt_jpos3d_18_list = []
        
        local_wham_jnts3d_18_list = [] # Not in global coorinates. 

        if self.use_animal_data:
            opt_skel_seq_list, skel_faces, scale_val_list, smal_opt_jnts3d, smal_opt_verts  = \
                    self.ema.ema_model.opt_3d_w_smal_animal_w_joints3d_input(opt_jpos3d_18.detach().cpu().numpy())

            return direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, opt_jpos3d_18, cam_extrinsic, \
                val_bs, val_num_views, val_num_steps, val_data, actual_seq_len, \
                opt_skel_seq_list, skel_faces, scale_val_list, smal_opt_jnts3d, smal_opt_verts  

        ori_opt_skel_seq_list, opt_skel_seq_list, opt_jpos3d_18_list, skel_faces, scale_val_list = \
                        self.ema.ema_model.opt_3d_w_vposer_w_joints3d_input(opt_jpos3d_18.detach().cpu().numpy()) 

        for bs_idx in range(batch):

            curr_opt_skel_seq = opt_skel_seq_list[bs_idx] 
            curr_opt_jpos3d_18 = opt_jpos3d_18_list[bs_idx] 
            curr_scale_val = scale_val_list[bs_idx] 

            pred_smpl_scaling = curr_opt_skel_seq['smpl_scaling'] # 1 
            pred_smpl_poses = curr_opt_skel_seq['smpl_poses'] # T X 24 X 3 
            pred_smpl_trans = curr_opt_skel_seq['smpl_trans'] # T X 3 

            # pred_smpl_poses, pred_smpl_trans = self.move_smpl_seq_to_floor(pred_smpl_poses, pred_smpl_trans)
            pred_jnts3d_smpl, pred_verts_smpl = run_smpl_forward(pred_smpl_poses, \
                    pred_smpl_trans, pred_smpl_scaling, self.val_ds.smpl) 
            # T X 45 X 3, T X Nv X 3  
            pred_smpl_jnts18 = convert_smpl_jnts_to_openpose18(pred_jnts3d_smpl)

            if "wham_poses" in val_data_dict: # Only for AIST data 
                # Get WHAM results 
                wham_poses_world = val_data_dict['wham_poses'][bs_idx].detach().cpu().numpy() # T X 72 
                wham_trans_world = val_data_dict['wham_trans'][bs_idx].detach().cpu().numpy() # T X 3 
                wham_betas = val_data_dict['wham_betas'][bs_idx] # T X 10 
                smpl_scaling = torch.ones(1).detach().cpu().numpy() # 1

                local_wham_jnts3d_18 = val_data_dict['local_wham_jnts18'][bs_idx] # T X 18 X 3 

                gt_smpl_poses = val_data_dict['smpl_poses'][bs_idx] # T X 24 X 3 
                gt_smpl_trans = val_data_dict['smpl_trans'][bs_idx] # T X 3 

                # Align wham results to our orientation 
                wham_poses_world, wham_trans_world = \
                    self.align_init_facing_direction(pred_smpl_poses, wham_poses_world, \
                    wham_trans_world) 
                wham_jnts3d_smpl, wham_verts_smpl = run_smpl_forward(wham_poses_world.detach().cpu().numpy(), \
                        wham_trans_world.detach().cpu().numpy(), smpl_scaling, self.val_ds.smpl, betas=wham_betas) 
                # T X 45 X 3, T X Nv X 3  
                wham_jpos3d_18 = convert_smpl_jnts_to_openpose18(wham_jnts3d_smpl) 

                # Align GT results to our orientation 
                gt_smpl_poses, gt_smpl_trans = \
                    self.align_init_facing_direction(pred_smpl_poses, gt_smpl_poses, \
                    gt_smpl_trans) 
                gt_jnts3d_smpl, gt_verts_smpl = run_smpl_forward(gt_smpl_poses.detach().cpu().numpy(), \
                        gt_smpl_trans.detach().cpu().numpy(), smpl_scaling, self.val_ds.smpl) 
                # T X 45 X 3, T X Nv X 3  
                gt_smpl_jnts18 = convert_smpl_jnts_to_openpose18(gt_jnts3d_smpl)
            else:
                # For bb_data 
               
                wham_poses_world, wham_trans_world = \
                    self.load_wham_res(val_data_dict['seq_name'][bs_idx], start_idx=val_data_dict['s_idx'][bs_idx], \
                    end_idx=val_data_dict['e_idx'][bs_idx])

                smpl_scaling = torch.ones(1).detach().cpu().numpy() # 1

                # Align wham results to our orientation 
                wham_poses_world, wham_trans_world = \
                    self.align_init_facing_direction(pred_smpl_poses, wham_poses_world, \
                    wham_trans_world) 
                wham_jnts3d_smpl, wham_verts_smpl = run_smpl_forward(wham_poses_world.detach().cpu().numpy(), \
                    wham_trans_world.detach().cpu().numpy(), smpl_scaling, self.val_ds.smpl) 
                # T X 45 X 3 
                wham_jpos3d_18 = convert_smpl_jnts_to_openpose18(wham_jnts3d_smpl) 

                # wham_jpos3d_18 = pred_smpl_jnts18.clone() 
                # wham_verts_smpl = pred_verts_smpl.clone()

                gt_smpl_jnts18 = pred_smpl_jnts18.clone() 
                gt_verts_smpl = pred_verts_smpl.clone() 
                local_wham_jnts3d_18 = pred_smpl_jnts18.clone() # Shouldn't be used. 

            # # Move the first frame's joints3D root to the center
            wham_jpos3d_18 = self.move_init3d_to_center(wham_jpos3d_18)
            gt_smpl_jnts18 = self.move_init3d_to_center(gt_smpl_jnts18)
            # pred_smpl_jnts18 = self.move_init3d_to_center(pred_smpl_jnts18) 
            # curr_opt_jpos3d_18 = self.move_init3d_to_center(curr_opt_jpos3d_18) 

            # # Center each frame's joints3D root to the center for comparing reprojected 2D errors 
            centered_wham_jnts3d_18 = self.center_jnts3d_seq(wham_jpos3d_18)
            centered_gt_smpl_jnts18 = self.center_jnts3d_seq(gt_smpl_jnts18) 
            centered_pred_smpl_jnts18 = self.center_jnts3d_seq(pred_smpl_jnts18)
            centered_opt_jpos3d_18 = self.center_jnts3d_seq(curr_opt_jpos3d_18) 
           
            pred_smpl_jnts18_list.append(pred_smpl_jnts18)
            gt_smpl_jnts18_list.append(gt_smpl_jnts18)
            wham_jpos3d_18_list.append(wham_jpos3d_18) 

            pred_verts_smpl_list.append(pred_verts_smpl)
            gt_verts_smpl_list.append(gt_verts_smpl)
            wham_verts_smpl_list.append(wham_verts_smpl)
            
            centered_wham_jnts3d_18_list.append(centered_wham_jnts3d_18)
            centered_gt_smpl_jnts18_list.append(centered_gt_smpl_jnts18)
            centered_pred_smpl_jnts18_list.append(centered_pred_smpl_jnts18)
            centered_opt_jpos3d_18_list.append(centered_opt_jpos3d_18)
            
            local_wham_jnts3d_18_list.append(local_wham_jnts3d_18)

        return direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, \
            val_bs, val_num_views, val_num_steps, val_data, actual_seq_len, \
            pred_smpl_jnts18_list, gt_smpl_jnts18_list, wham_jpos3d_18_list, opt_jpos3d_18_list, \
            pred_verts_smpl_list, gt_verts_smpl_list, wham_verts_smpl_list, \
            skel_faces, cam_extrinsic, scale_val_list, \
            centered_wham_jnts3d_18_list, centered_gt_smpl_jnts18_list, centered_pred_smpl_jnts18_list, centered_opt_jpos3d_18_list, \
            ori_opt_skel_seq_list, local_wham_jnts3d_18_list 

    def opt_3d_w_learned_model_for_interaction(self, val_data_dict):
        val_data = val_data_dict['normalized_jnts2d'].float().cuda() # BS X T X 18 X 2 
        batch = val_data.shape[0]

        cam_extrinsic = self.prep_multiview_cam_extrinsic(num_views=self.opt.train_num_views) # 1 X K X 4 X 4 
        cam_extrinsic = cam_extrinsic.repeat(batch, 1, 1, 1) # BS X K X 4 X 4 
        
        val_data = val_data[:, None, :, :, :].repeat(1, cam_extrinsic.shape[1], 1, 1, 1) # BS X K X T X 18 X 2 

        val_bs, val_num_views, val_num_steps, num_joints, _ = val_data.shape 

        val_cond_data = val_data[:, 0:1] # BS X 1 X T X 18 X 2
        val_cond_data = val_cond_data.repeat(1, val_num_views, 1, 1, 1) # BS X K X T X 18 X 2

        # Generate padding mask 
        actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
        tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
        self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1) # BS X max_timesteps
        padding_mask = tmp_mask[:, None, :].to(val_data.device) # BS X 1 X T 
        padding_mask = padding_mask.repeat(1, val_num_views, 1).reshape(val_bs*val_num_views, \
            1, -1) # (BS*K) X 1 X T 

        val_data = val_data.reshape(val_bs*val_num_views, val_num_steps, -1) # (BS*K) X T X D(18*2)

        # Use the first camera view's 2D pose sequence as input condition
        val_cond_data = val_cond_data.reshape(val_bs*val_num_views, val_num_steps, -1) # (BS*K) X T X D(18*2) 

        language_input = None 

        direct_sampled_epi_seq_2d = self.ema.ema_model.sample(val_data, \
                    val_cond_data, padding_mask, language_input=language_input) # (BS*K) X T X D(18*2) 
        direct_sampled_epi_seq_2d = direct_sampled_epi_seq_2d.reshape(val_bs, val_num_views, \
                    val_num_steps, -1) # BS X K X T X D(18*2) 

        # Replace the first camera view's 2D with the inpit 2D pose sequence 
        direct_sampled_epi_seq_2d[:, 0] = val_data.reshape(val_bs, val_num_views, val_num_steps, -1)[:, 0] # BS X K X T X D(18*2) 

        epi_line_cond_seq_2d = direct_sampled_epi_seq_2d # BS X K X T X D(18*2) 

        good_views_idx_list = None 
        epi_line_cond_seq_2d, opt_jpos3d_18 = \
                    self.ema.ema_model.opt_3d_w_multi_view_2d_res(epi_line_cond_seq_2d, \
                    cam_extrinsic, padding_mask=padding_mask)
        # BS X K X T X (18*2), BS X T X 18 X 3, BS X T X 18 X 3 

        # For object 
        ori_opt_obj_param_list, opt_obj_param_list, opt_obj_jpos3d_list, opt_obj_verts_list, obj_faces, obj_scale_val_list = \
                    self.ema.ema_model.opt_3d_object_w_joints3d_input(opt_jpos3d_18.detach().cpu().numpy()[:, :, -5:, :]) 
        # rest_obj_verts(real world scale) * scale ==> optimized 3d object positions 

        # opt_obj_verts_list represent object vertices with scale = 1 

        # For human 
        ori_opt_skel_seq_list, opt_skel_seq_list, opt_jpos3d_18_list, skel_faces, scale_val_list = \
                    self.ema.ema_model.opt_3d_w_vposer_w_joints3d_input(opt_jpos3d_18.detach().cpu().numpy()[:, :, :-5, :], for_interaction=True)

        # opt_skel_seq_list, opt_jpos3d_18_list represent w scale that majkkes the reprojection match 2D input 

        # opt_jpos3d_18_list represent human vertices with scale = 1 



        pred_smpl_jnts18_list, pred_smpl_verts_list = \
                    self.get_jnts3d_w_smpl_params_for_interaction(opt_skel_seq_list)
        # results with scaling == 1 for human joints and human vertices 

        # pred_smpl_jnts18 = pred_smpl_jnts18 * scale_val_list[bs_idx] / smpl_scaling
        # pred_verts_smpl = pred_verts_smpl * scale_val_list[bs_idx] / smpl_scaling 

        return direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, \
            val_bs, val_num_views, val_num_steps, val_data, \
            actual_seq_len, opt_jpos3d_18, \
            pred_smpl_jnts18_list, \
            pred_smpl_verts_list, skel_faces, cam_extrinsic, \
            scale_val_list, opt_skel_seq_list, \
            ori_opt_obj_param_list, opt_obj_param_list, opt_obj_jpos3d_list, opt_obj_verts_list, obj_faces, obj_scale_val_list
           
    def gen_multi_view_2d_seq_for_eval(self, smpl_jnts18, scale_val):
        # smpl_jnts18: T X J X 3 
        val_num_views = 6 
        cam_extrinsic = self.prep_multiview_cam_extrinsic(num_views=val_num_views) # 1 X K X 4 X 4 
        cam_extrinsic = cam_extrinsic.squeeze(0)[:, None].repeat(1, smpl_jnts18.shape[0], 1, 1) # K X T X 4 X 4 

        cam_rot_mat = cam_extrinsic[:, :, :3, :3] # K X T X 3 X 3 
        cam_trans = cam_extrinsic[:, :, :3, -1] # K X T X 3 

        reprojected_pred_jnts2d_list, reprojected_pred_ori_jnts2d_list, _ = \
                                self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[0], \
                                cam_rot_mat, cam_trans, \
                                smpl_jnts18.to(cam_rot_mat.device)*scale_val) # K X T X (18*2)

        return reprojected_pred_ori_jnts2d_list 

    def adjust_2d_seq_scale_for_eval(self, ori_jpos2d_pred, ori_jpos2d_gt):
        # ori_jpos2d_pred: BS X K X T X (J*2) 
        # ori_jpos2d_gt: BS X K X T X (J*2) 

        num_bs = ori_jpos2d_pred.shape[0] 

        new_jpos2d_list = [] 
        scale_list = []
        for bs_idx in range(num_bs):
            # 2D Pose Error with Scale Adjustment for the whole sequence, compte the scale based on the first time step and aply for all timesteps 
            init_pred_2d = ori_jpos2d_pred[bs_idx, 0, 0] 
            init_gt_2d = ori_jpos2d_gt[bs_idx, 0, 0] 
            global_scale = torch.sum(init_pred_2d * init_gt_2d) / torch.sum(init_pred_2d ** 2)
            
            scaled_jpos2d_pred = global_scale * ori_jpos2d_pred[bs_idx]
            new_jpos2d_list.append(scaled_jpos2d_pred)

            scale_list.append(global_scale)
        
        new_jpos2d_list = torch.stack(new_jpos2d_list, dim=0) # BS X K X T X (J*2) 
        
        return new_jpos2d_list, scale_list  

    def compute_yaw_rotation_and_rotated_3d_pose(self, pose_3d_seq, pose_2d_seq, cam_rot_mat, cam_trans, verts_seq, lr=0.01, iterations=200):
        # pose_3d_seq: (T, J, 3) 
        # pose_2d_seq: (T, J, 2) 
        # camera_rot_mat: (3, 3)
        # cam_trans = (3,) 
        # verts_seq: (T, Nv, 3) 

        # Extract the first frame's 3D and 2D poses
        first_frame_3d_pose = pose_3d_seq[0].detach()  # shape: (J, 3)
        first_frame_2d_pose = pose_2d_seq[0].to(pose_3d_seq.device)  # shape: (J, 2)

        # Initialize the yaw angle as a learnable parameter
        theta = torch.zeros(1).to(first_frame_3d_pose.device)
        scale = torch.ones(1).to(first_frame_3d_pose.device) 
        theta.requires_grad_(True) 
        scale.requires_grad_(True) 

        # Optimizer for the yaw angle
        optimizer = torch.optim.Adam([theta, scale], lr=lr)

        for _ in range(iterations):
            optimizer.zero_grad()

            # Compute the yaw rotation matrix around the z-axis from theta
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            R_yaw = torch.stack([
                torch.stack([cos_theta, -sin_theta, torch.zeros(1, device=theta.device)]),
                torch.stack([sin_theta,  cos_theta, torch.zeros(1, device=theta.device)]),
                torch.stack([torch.zeros(1, device=theta.device), torch.zeros(1, device=theta.device), torch.ones(1, device=theta.device)])
            ]).squeeze(2)  # shape: (3, 3)

            # Apply the yaw rotation to the 3D pose
            rotated_3d_pose = first_frame_3d_pose @ R_yaw.T * scale   # (J, 3)


            reprojected_norm_2d_pose, reprojected_2d_pose, _ = \
                                self.ema.ema_model.get_projection_from_motion3d(1, \
                                cam_rot_mat[None, None].to(rotated_3d_pose.device).detach(), cam_trans[None, None].to(rotated_3d_pose.device).detach(), \
                                rotated_3d_pose[None]) # K(1) X T(1) X (18*2), need cam_rot_mat: K X T X 3 X 3, cam_trans: K X T X 3 
            # 1 X 1 X 36 

            reprojected_norm_2d_pose = reprojected_norm_2d_pose.squeeze(0).squeeze(0).reshape(-1, 2) # 18 X 2 

            # Compute the reprojection loss (L2 distance)
            loss = F.mse_loss(reprojected_norm_2d_pose, first_frame_2d_pose) # 18 X 2 

            print("Optimizing R to align 3D reprojection loss:{0}".format(loss))

            # Backpropagate and update the yaw angle
            loss.backward()
            optimizer.step()

        # Compute the final yaw rotation matrix from the optimized theta
        final_R_yaw = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta),  torch.cos(theta), 0],
            [0,                0,               1]
        ], device=first_frame_3d_pose.device)

        # Apply the final yaw rotation to the entire 3D pose sequence
        rotated_3d_pose_seq = pose_3d_seq @ final_R_yaw.T 
        scaled_rotated_3d_pose_seq = rotated_3d_pose_seq * scale   # (T, J, 3)

        # Apply the final yaw rotation to the entire 3D vertices sequence 
        rotated_3d_verts_seq = verts_seq @ final_R_yaw.T 
        scaled_rotated_3d_verts_seq = rotated_3d_verts_seq * scale   # (T, J, 3)

        return rotated_3d_pose_seq, scaled_rotated_3d_pose_seq, rotated_3d_verts_seq, scaled_rotated_3d_verts_seq  

    def process_verts_for_vis_cmp(self, ori_verts):
        # ori_verts: T X Nv X 3 

        # Move the whole sequence to be on the floor z = 0. 
        floor_z = ori_verts[0, :, 2].min()
        ori_verts[:, :, 2] -= floor_z 

        # Move the first frame's center to be at the origin 
        first_frame_center = ori_verts[0].mean(dim=0).clone() # 3 
        first_frame_center[2] = 0.0 
        ori_verts -= first_frame_center[None, None]

        return ori_verts   

    def process_verts_for_vis_cmp_animal(self, ori_verts, ori_jnts3d):
        # ori_verts: T X Nv X 3 
        # ori_jnts3d: T X J X 3 

        # Move the whole sequence to be on the floor z = 0. 
        floor_z = ori_jnts3d[0, :, 2].min()
        ori_verts[:, :, 2] -= floor_z 

        # Move the first frame's center to be at the origin 
        first_frame_center = ori_verts[0].mean(dim=0).clone() # 3 
        first_frame_center[2] = 0.0 
        ori_verts -= first_frame_center[None, None]

        return ori_verts   

    def process_verts_for_vis_cmp_interaction(self, ori_verts, obj_verts):
        # ori_verts: T X Nv X 3 
        # obj_verts: T X Nv' X 3 

        # Move the whole sequence to be on the floor z = 0. 
        floor_z = ori_verts[0, :, 2].min()
        ori_verts[:, :, 2] -= floor_z 
        obj_verts[:, :, 2] -= floor_z 

        # Move the first frame's center to be at the origin 
        first_frame_center = ori_verts[0].mean(dim=0).clone() # 3 
        first_frame_center[2] = 0.0 
        ori_verts -= first_frame_center[None, None]
        obj_verts -= first_frame_center[None, None]

        return ori_verts, obj_verts    

    def load_ori_rgb_images(self, val_data_dict, dest_img_folder):
        if "AIST" in self.opt.youtube_data_npz_folder:
            rgb_folder = "/viscam/projects/vimotion/datasets/AIST/clip_mp4_files"
            
            res_tag = "AIST"

            seq_name = val_data_dict['seq_name'][0]

            s_idx = 0 
            e_idx = 120 * 2 # For AIST, 60 fps data. 

            ori_video_path = os.path.join(rgb_folder, seq_name.replace(".npz", "") + ".mp4") 
      
        elif "nicole" in self.opt.youtube_data_npz_folder:
            rgb_folder = "/viscam/projects/mofitness/datasets/nicole_move/clip_mp4_files"

            res_tag = "nicole"

            seq_name = val_data_dict['seq_name'][0]

            s_idx = val_data_dict['s_idx'][0]
            e_idx = val_data_dict['e_idx'][0]

            if torch.is_tensor(s_idx):
                s_idx = s_idx.item()
                e_idx = e_idx.item() 

            ori_video_path = os.path.join(rgb_folder, seq_name.replace(".npz", "") + ".mp4") 
        elif "steezy" in self.opt.youtube_data_npz_folder:
            rgb_folder = "/viscam/projects/mofitness/datasets/steezy_new/ori_mp4_files"

            res_tag = "steezy" 

            seq_name = val_data_dict['seq_name'][0]

            ori_start_idx = int(seq_name.split("_s_")[1].split("_")[0])

            s_idx = int(val_data_dict['s_idx'][0]) + ori_start_idx
            e_idx = int(val_data_dict['e_idx'][0]) + ori_start_idx 

            ori_video_path = os.path.join(rgb_folder, seq_name.split("_sub")[0] + ".mp4") 
        elif "cat" in self.opt.youtube_data_npz_folder:
            rgb_folder = "/viscam/projects/mofitness/datasets/cat_data/test_clip_mp4_files"

            res_tag = "cat" 

            seq_name = val_data_dict['seq_name'][0]

            s_idx = val_data_dict['s_idx'][0]
            e_idx = val_data_dict['e_idx'][0]

            if torch.is_tensor(s_idx):
                s_idx = s_idx.item()
                e_idx = e_idx.item() 

            ori_video_path = os.path.join(rgb_folder, seq_name.replace(".npz", "") + ".mp4") 


        # dest_video_folder = os.path.join(dest_img_folder, res_tag)
        dest_video_folder = dest_img_folder 
        os.makedirs(dest_video_folder, exist_ok=True)
        
        output_video_path = os.path.join(dest_video_folder, f"{seq_name}_s{s_idx}_e{e_idx}.mp4")

        # Open the video file
        cap = cv2.VideoCapture(ori_video_path)
        if not cap.isOpened():
            print(f"Failed to open video file {ori_video_path}")
            return

        # Set the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, s_idx)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        current_frame_index = s_idx
        while True:
            ret, frame = cap.read()
            if not ret or current_frame_index > e_idx:
                break

            # Write the frame to the output video
            video_writer.write(frame)
            current_frame_index += 1

        # Release resources
        cap.release()
        video_writer.release()

        print(f"Video from frames {s_idx} to {e_idx} saved to {output_video_path}")

    def prep_global_traj_cmp_vis(self, pred_smpl_jnts18, gt_smpl_jnts18, wham_pred_smpl_jnts18, aligned_motionbert_jnts18, \
                vposer2d_pred_smpl_jnts18, s_idx, bs_idx, dest_for_paper_vis_folder):
        l_hip_idx = 11
        r_hip_idx = 8 

        # # Visualize for our results 
        pred_root_traj = (pred_smpl_jnts18[:, l_hip_idx] + pred_smpl_jnts18[:, r_hip_idx]) / 2.0 # T X 3 
        # dest_3d_traj_path = os.path.join(dest_for_paper_vis_folder, "s_idx_"+str(s_idx)+"_bs_idx_"+str(bs_idx)+"_3d_traj_ours.png") 
        # # visualize_root_trajectory_for_figure(pred_root_traj.detach().cpu().numpy(), dest_3d_traj_path)
        # plot_trajectory_components_for_figure(pred_root_traj.detach().cpu().numpy(), dest_3d_traj_path)

        # # Visualize for GT results 
        gt_root_traj = (gt_smpl_jnts18[:, l_hip_idx] + gt_smpl_jnts18[:, r_hip_idx]) / 2.0 # T X 3 
        # dest_3d_traj_path = os.path.join(dest_for_paper_vis_folder, "s_idx_"+str(s_idx)+"_bs_idx_"+str(bs_idx)+"_3d_traj_gt.png") 
        # # visualize_root_trajectory_for_figure(gt_root_traj.detach().cpu().numpy(), dest_3d_traj_path)
        # plot_trajectory_components_for_figure(gt_root_traj.detach().cpu().numpy(), dest_3d_traj_path)
        

        # # Visualize for WHAM results 
        wham_root_traj = (wham_pred_smpl_jnts18[:, l_hip_idx] + wham_pred_smpl_jnts18[:, r_hip_idx]) / 2.0 # T X 3 
        # dest_3d_traj_path = os.path.join(dest_for_paper_vis_folder, "s_idx_"+str(s_idx)+"_bs_idx_"+str(bs_idx)+"_3d_traj_wham.png") 
        # # visualize_root_trajectory_for_figure(wham_root_traj.detach().cpu().numpy(), dest_3d_traj_path)
        # plot_trajectory_components_for_figure(wham_root_traj.detach().cpu().numpy(), dest_3d_traj_path)

        # # Visualize for MotionBERT results 
        mbert_root_traj = (aligned_motionbert_jnts18[:, l_hip_idx] + aligned_motionbert_jnts18[:, r_hip_idx]) / 2.0 # T X 3 
        # dest_3d_traj_path = os.path.join(dest_for_paper_vis_folder, "s_idx_"+str(s_idx)+"_bs_idx_"+str(bs_idx)+"_3d_traj_mbert.png") 
        # # visualize_root_trajectory_for_figure(mbert_root_traj.detach().cpu().numpy(), dest_3d_traj_path)
        # plot_trajectory_components_for_figure(mbert_root_traj.detach().cpu().numpy(), dest_3d_traj_path)

        # # Visualize for SMPLify results 
        smplify_root_traj = (vposer2d_pred_smpl_jnts18[:, l_hip_idx] + vposer2d_pred_smpl_jnts18[:, r_hip_idx]) / 2.0 # T X 3 
        # dest_3d_traj_path = os.path.join(dest_for_paper_vis_folder, "s_idx_"+str(s_idx)+"_bs_idx_"+str(bs_idx)+"_3d_traj_smplify.png") 
        # # visualize_root_trajectory_for_figure(smplify_root_traj.detach().cpu().numpy(), dest_3d_traj_path)
        # plot_trajectory_components_for_figure(smplify_root_traj.detach().cpu().numpy(), dest_3d_traj_path)

        if "AIST" in self.youtube_data_npz_folder:
            # Move ours initial frame to x=0,y=0
            # Visualzie all the components in a single figure 
            pred_root_traj = pred_root_traj - pred_root_traj[0:1]  
            dest_3d_traj_path = os.path.join(dest_for_paper_vis_folder, "s_idx_"+str(s_idx)+"_bs_idx_"+str(bs_idx)+"_all_cmp_xyz/") 
            plot_multiple_trajectories([pred_root_traj.detach().cpu().numpy(), wham_root_traj.detach().cpu().numpy(), \
                mbert_root_traj.detach().cpu().numpy(), smplify_root_traj.detach().cpu().numpy(), \
                gt_root_traj.detach().cpu().numpy()], \
                labels=["Ours", "WHAM", "MotionBERT", "SMPLify", "GT"], save_dir=dest_3d_traj_path) 
        else:
            pred_root_traj = pred_root_traj - pred_root_traj[0:1]  
            # Visualzie all the components in a single figure 
            dest_3d_traj_path = os.path.join(dest_for_paper_vis_folder, "s_idx_"+str(s_idx)+"_bs_idx_"+str(bs_idx)+"_all_cmp_xyz/") 
            plot_multiple_trajectories([pred_root_traj.detach().cpu().numpy(), wham_root_traj.detach().cpu().numpy(), \
                mbert_root_traj.detach().cpu().numpy(), smplify_root_traj.detach().cpu().numpy()], \
                labels=["Ours", "WHAM", "MotionBERT", "SMPLify"], save_dir=dest_3d_traj_path) 

    def eval_3d_w_multiview_diffusion(self):
        # Load line-conditioned model. 
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        milestone = "3"
        # milestone = "4"
        # milestone = "2" # MPJPE: 119.9 
        # milestone = "1"
        # if self.eval_on_aist_3d:
        #     milestone = "3" 

        self.load(milestone)
        self.ema.ema_model.eval()
       
        # Prepare testing data loader 
        if self.eval_w_best_mpjpe:
            batch = 32 # 64 
        else:
            batch = 1 
        test_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=batch, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False) 

        dest_res3d_npy_folder = self.prep_vis_res_folder() 

        # Prepare empty list for evaluation metrics 
        self.prep_evaluation_metric_list() 
        
        for s_idx, val_data_dict in enumerate(test_loader): 

            direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, \
            val_bs, val_num_views, val_num_steps, val_data, actual_seq_len, \
            pred_smpl_jnts18_list, gt_smpl_jnts18_list, wham_jpos3d_18_list, opt_jpos3d_18_list, \
            pred_verts_smpl_list, gt_verts_smpl_list, wham_verts_smpl_list, skel_faces, cam_extrinsic, scale_val_list, \
            centered_wham_jnts3d_18_list, centered_gt_smpl_jnts18_list, \
            centered_pred_smpl_jnts18_list, centered_opt_jpos3d_18_list, \
            ori_opt_smpl_param_list, local_wham_jpos3d_18_list = \
                self.opt_3d_w_learned_model(val_data_dict)

            # The direct sampled 2D pose sequences of the trained 2D motion diffusion model, the first view is the input GT 2D poses  
            direct_sampled_seq_2d_for_vis = direct_sampled_epi_seq_2d.reshape(val_bs, val_num_views, \
                        val_num_steps, -1) # BS X K X T X D
                    
            # The input GT 2D pose sequences. 
            gt_for_vis = val_data.reshape(val_bs, val_num_views, val_num_steps, -1)
            
            cam_rot_mat = cam_extrinsic[:, :, :3, :3].unsqueeze(2).repeat(1, 1, val_num_steps, 1, 1) # BS X K X T X 3 X 3 
            cam_trans = cam_extrinsic[:, :, :3, -1].unsqueeze(2).repeat(1, 1, val_num_steps, 1) # BS X K X T X 3 

            convert_wham_skeleton = False 
            if convert_wham_skeleton:
                wham_ori_opt_skel_seq_list, wham_opt_skel_seq_list, wham_opt_jpos3d_18_list, wham_skel_faces, wham_scale_val_list = \
                    self.ema.ema_model.opt_3d_w_vposer_w_joints3d_input(torch.stack(wham_jpos3d_18_list).detach().cpu().numpy())
            
                wham_smpl_jnts18_list, wham_smpl_verts_list = \
                self.get_jnts3d_w_smpl_params(wham_ori_opt_skel_seq_list, wham_opt_skel_seq_list, wham_scale_val_list, force_head_zero=False)

            # For SMPLify baseline
            if self.add_vposer_opt_w_2d:
                vposer_2d_ori_opt_skel_seq_list, vposer_2d_opt_skel_seq_list, skel_faces, vposer_2d_scale_val_list = \
                    self.ema.ema_model.opt_3d_w_vposer_w_joints2d_reprojection(gt_for_vis[:, 0], cam_rot_mat[:, 0], cam_trans[:, 0])

                vposer2d_smpl_jnts18_list, vposer2d_smpl_verts_list = \
                    self.get_jnts3d_w_smpl_params(vposer_2d_ori_opt_skel_seq_list, vposer_2d_opt_skel_seq_list, vposer_2d_scale_val_list)
    
            # For Elepose baseline 
            if self.opt.add_elepose_for_eval:
                elepose_jnts18_list = [] 
                for tmp_idx in range(val_bs):
                    if "steezy" in self.youtube_data_npz_folder or "nicole" in self.youtube_data_npz_folder:
                        elepose_jnts18 = self.load_elepose_res(val_data_dict['seq_name'][tmp_idx], \
                            start_idx=val_data_dict['s_idx'].detach().cpu().numpy()[tmp_idx], \
                            end_idx=val_data_dict['e_idx'].detach().cpu().numpy()[tmp_idx])
                    else:
                        elepose_jnts18 = self.load_elepose_res(val_data_dict['seq_name'][tmp_idx]) # For AIST , T X 18 X 3 

                    elepose_jnts18_list.append(elepose_jnts18)
                
                elepose_jnts18_list = np.asarray(elepose_jnts18_list) * 0.5 # BS X T X 18 X 3 

                # Use vposer to process the original joints3D 
                elepose_ori_opt_skel_seq_list, elepose_opt_skel_seq_list, elepose_opt_jpos3d_18_list, elepose_skel_faces, elepose_scale_val_list = \
                    self.ema.ema_model.opt_3d_w_vposer_w_joints3d_input(elepose_jnts18_list) 
                
                elepose_smpl_jnts18_list, elepose_smpl_verts_list = \
                    self.get_jnts3d_w_smpl_params(elepose_ori_opt_skel_seq_list, elepose_opt_skel_seq_list, elepose_scale_val_list, force_head_zero=True)

            # For MAS baseline 
            if self.opt.add_mas_for_eval:
                mas_jnts18_list = [] 
                for tmp_idx in range(val_bs):
                    if "s_idx" in val_data_dict:
                        mas_jnts18 = self.load_mas_res(val_data_dict['seq_name'][tmp_idx], \
                            start_idx=val_data_dict['s_idx'].detach().cpu().numpy()[tmp_idx], \
                            end_idx=val_data_dict['e_idx'].detach().cpu().numpy()[tmp_idx]) # For AIST , T X 18 X 3 
                    else:
                        # if "gBR_sBM_c09_d06_mBR5_ch08" in val_data_dict['seq_name'][tmp_idx]:
                        #     import pdb; pdb.set_trace() 

                        mas_jnts18 = self.load_mas_res(val_data_dict['seq_name'][tmp_idx]) # For AIST , T X 18 X 3 

                    mas_jnts18_list.append(mas_jnts18)
                
                mas_jnts18_list = np.asarray(mas_jnts18_list) * 10  # BS X T X 18 X 3 

                # Use vposer to process the original joints3D 
                mas_ori_opt_skel_seq_list, mas_opt_skel_seq_list, mas_opt_jpos3d_18_list, mas_skel_faces, mas_scale_val_list = \
                    self.ema.ema_model.opt_3d_w_vposer_w_joints3d_input(mas_jnts18_list) 
              
                mas_smpl_jnts18_list, mas_smpl_verts_list = \
                    self.get_jnts3d_w_smpl_params(mas_ori_opt_skel_seq_list, mas_opt_skel_seq_list, mas_scale_val_list, force_head_zero=False)

            # For MotionBERT baseline 
            if self.opt.add_motionbert_for_eval:
                motionbert_jnts18_list = []
                for tmp_idx in range(val_bs):
                    if "steezy" in self.youtube_data_npz_folder:
                        motionbert_jnts18 = self.load_motionbert_res(val_data_dict['seq_name'][tmp_idx], \
                                start_idx=val_data_dict['s_idx'].cpu().numpy()[tmp_idx], end_idx=val_data_dict['e_idx'].cpu().numpy()[tmp_idx]) # T X 18 X 3 
                    elif "nicole" in self.youtube_data_npz_folder:
                        motionbert_jnts18 = self.load_motionbert_res(val_data_dict['seq_name'][tmp_idx], \
                                start_idx=val_data_dict['s_idx'].cpu().numpy()[tmp_idx], end_idx=val_data_dict['e_idx'].cpu().numpy()[tmp_idx]) # T X 18 X 3 
                    else:
                        motionbert_jnts18 = self.load_motionbert_res(val_data_dict['seq_name'][tmp_idx]) # T X 18 X 3 

                    motionbert_jnts18_list.append(motionbert_jnts18)

                motionbert_jnts18_list = np.asarray(motionbert_jnts18_list)  # BS X T X 18 X 3

                # Use vposer to process the original joints3D
                motionbert_ori_opt_skel_seq_list, motionbert_opt_skel_seq_list, motionbert_opt_jpos3d_18_list, motionbert_skel_faces, motionbert_scale_val_list = \
                        self.ema.ema_model.opt_3d_w_vposer_w_joints3d_input(motionbert_jnts18_list)

                motionbert_smpl_jnts18_list, motionbert_smpl_verts_list = \
                    self.get_jnts3d_w_smpl_params(motionbert_ori_opt_skel_seq_list, motionbert_opt_skel_seq_list, motionbert_scale_val_list, force_head_zero=False)

            actual_seq_len = actual_seq_len.reshape(val_bs, -1) # BS X K 

            for bs_idx in range(val_bs):
                if val_data_dict['seq_len'][bs_idx] < 120:
                    continue 

                vis_folder_tag = "gen_multiview_2d_batch_"+str(s_idx)+"_seq_"+str(bs_idx)

                # 1. For ours, Get reprojected our results 
                # aligned_ours_jnts18, aligned_ours_verts = self.align_motionbert_to_ours(pred_smpl_jnts18_list[bs_idx].detach().cpu().numpy(), \
                #                             gt_smpl_jnts18_list[bs_idx], pred_verts_smpl_list[bs_idx].detach().cpu().numpy())
                # pred_verts_smpl_list[bs_idx] = torch.from_numpy(aligned_ours_verts).to(pred_verts_smpl_list[bs_idx].device)

                gt_smpl_jnts18 = self.replace_neck_jnt_w_avg(gt_smpl_jnts18_list[bs_idx]) 
                # new_gt_reprojected_jnts2d_list, _, _ = \
                #                 self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                #                 cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                #                 gt_smpl_jnts18.to(cam_rot_mat.device)*scale_val_list[bs_idx]) # K X T X (18*2)
                # new_gt_for_vis = new_gt_reprojected_jnts2d_list.reshape(1, val_num_views, -1, 18*2)  

                pred_smpl_jnts18 = self.replace_neck_jnt_w_avg(pred_smpl_jnts18_list[bs_idx]) 
                # spred_smpl_jnts18 = self.replace_neck_jnt_w_avg(aligned_ours_jnts18)
                reprojected_pred_jnts2d_list, reprojected_pred_ori_jnts2d_list, _ = \
                                self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                pred_smpl_jnts18.to(cam_rot_mat.device)*scale_val_list[bs_idx]) # K X T X (18*2)
                pred_for_vis = reprojected_pred_jnts2d_list.reshape(1, val_num_views, -1, 18*2)  

                vis_for_paper = False 
                if vis_for_paper:
                    dest_for_paper_vis_folder = "./final_day_for_cvpr25_paper_res_figure"
                    if not os.path.exists(dest_for_paper_vis_folder):
                        os.makedirs(dest_for_paper_vis_folder) 

                    # pose_sequence = epi_line_cond_seq_2d 
                    # line_coeffs =  
                    # visualize_pose_and_lines_for_paper(pose_sequence, line_coeffs, output_folder, vis_gt=False, epipoles=None)
                    
                    # Plot for single pose 
                    # num_steps = reprojected_pred_jnts2d_list.shape[1] 
                    # for t_idx in range(0, num_steps, 1):
                    #     dest_pose2d_fig_path = os.path.join(dest_for_paper_vis_folder, "s_idx_"+str(s_idx)+"_bs_idx_"+str(bs_idx)+"_pose2d_"+str(t_idx)+".png") 
                    #     plot_pose_2d_for_paper(reprojected_pred_jnts2d_list[0, t_idx].reshape(-1, 2).detach().cpu().numpy(), dest_pose2d_fig_path) 


                    # Plot for multi-view 2D sequences 
                    num_views = reprojected_pred_jnts2d_list.shape[0] 
                    num_steps = reprojected_pred_jnts2d_list.shape[1] 
                    for view_idx in range(num_views):
                        for t_idx in range(0, num_steps, 1):
                            dest_pose2d_fig_path = os.path.join(dest_for_paper_vis_folder, \
                                "s_idx_"+str(s_idx)+"_bs_idx_"+str(bs_idx)+"_pose2d_"+"view_"+str(view_idx)+"_t_"+str(t_idx)+".png") 
                            plot_pose_2d_for_paper(reprojected_pred_jnts2d_list[view_idx, t_idx].reshape(-1, 2).detach().cpu().numpy(), dest_pose2d_fig_path) 

                    import pdb 
                    pdb.set_trace() 
                 

                # ours_mv_2d_for_eval = self.gen_multi_view_2d_seq_for_eval(pred_smpl_jnts18, scale_val_list[bs_idx]) # K X T X (18*2) 
                # ours_mv_2d_for_eval = torch.cat((pred_for_vis[0, 1:2], pred_for_vis[0, 3:4]), dim=0) # 2 X T X (18*2) 
                ours_mv_2d_for_eval = pred_for_vis.clone()[0, 1:] 


                # if self.eval_on_aist_3d:
                #     # Use WHAM's local pose to do reprojection. (WHAM uses identity rotation and zero translation for each local pose)
                #     reprojected_wham_jnts2d_list = self.reproject_wham_local_3D_to_2D(local_wham_jpos3d_18_list[bs_idx], \
                #                             val_data_dict['scale_factors_for_input2d'][bs_idx:bs_idx+1]) # T X 18 X 2 
                #     wham_2d_for_vis = reprojected_wham_jnts2d_list[None].repeat(val_num_views, 1, 1, 1) # K X T X 18 X 2 
                #     wham_2d_for_vis = wham_2d_for_vis.reshape(1, val_num_views, -1, 18*2) 
                # else: 
                #     # wham_2d_for_vis = pred_for_vis.clone() 
                    
                #     reprojected_wham_jnts2d_list, reprojected_wham_ori_jnts2d_list, _ = \
                #                 self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                #                 cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                #                 wham_jpos3d_18_list[bs_idx].to(cam_rot_mat.device)*scale_val_list[bs_idx]) # K X T X (18*2)
                #     wham_2d_for_vis = reprojected_wham_jnts2d_list.reshape(1, val_num_views, -1, 18*2)  

                # 2. For WHAM 
                # aligned_wham_jnts18 = self.align_motionbert_to_ours(wham_jpos3d_18_list[bs_idx].detach().cpu().numpy(), \
                #                             gt_smpl_jnts18_list[bs_idx], wham_verts_smpl_list[bs_idx].detach().cpu().numpy())
                # wham_verts_smpl_list[bs_idx] = torch.from_numpy(aligned_wham_verts).to(wham_verts_smpl_list[bs_idx].device)

                # wham_pred_smpl_jnts18 = self.replace_neck_jnt_w_avg(aligned_wham_jnts18) 

                if convert_wham_skeleton:
                    wham_pred_smpl_jnts18, scaled_aligned_wham_pred_smpl_jnts18, \
                    wham_pred_verts_aligned, wham_pred_verts_scaled_aligned = self.compute_yaw_rotation_and_rotated_3d_pose(self.replace_neck_jnt_w_avg(wham_smpl_jnts18_list[bs_idx]), \
                                    gt_for_vis[bs_idx, 0].reshape(gt_for_vis.shape[2], -1, 2), cam_rot_mat[bs_idx, 0, 0], cam_trans[bs_idx, 0, 0], wham_smpl_verts_list[bs_idx])
    
                else:
                    wham_pred_smpl_jnts18, scaled_aligned_wham_pred_smpl_jnts18, \
                    wham_pred_verts_aligned, wham_pred_verts_scaled_aligned = self.compute_yaw_rotation_and_rotated_3d_pose(self.replace_neck_jnt_w_avg(wham_jpos3d_18_list[bs_idx]), \
                                    gt_for_vis[bs_idx, 0].reshape(gt_for_vis.shape[2], -1, 2), cam_rot_mat[bs_idx, 0, 0], cam_trans[bs_idx, 0, 0], wham_verts_smpl_list[bs_idx])
    
                # wham_pred_smpl_jnts18 = self.replace_neck_jnt_w_avg(wham_jpos3d_18_list[bs_idx]) 
                reprojected_wham_jnts2d_list, reprojected_wham_ori_jnts2d_list, _ = \
                                self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                scaled_aligned_wham_pred_smpl_jnts18.to(cam_rot_mat.device))
                                # wham_pred_smpl_jnts18.to(cam_rot_mat.device)*val_data_dict['scale_factors_for_input2d'][bs_idx].to(cam_rot_mat.device)) # K X T X (18*2)
                
                # reprojected_wham_jnts2d_list, scale_list = self.adjust_2d_seq_scale_for_eval(reprojected_wham_jnts2d_list[None], gt_for_vis[bs_idx:bs_idx+1]) # K X T X (18*2)
                wham_2d_for_vis = reprojected_wham_jnts2d_list.reshape(1, val_num_views, -1, 18*2)  

                # Adjust the scale based on the first 2d pose difference for the whole WHAM sequence 

                # wham_mv_2d_for_eval = self.gen_multi_view_2d_seq_for_eval(wham_pred_smpl_jnts18, \
                #                 val_data_dict['scale_factors_for_input2d'][bs_idx]) # K X T X (18*2) 

                # debug_mv_vis = False 
                # wham_mv_2d_for_eval = self.gen_multi_view_2d_seq_for_eval(wham_pred_smpl_jnts18, 1) # K X T X (18*2) 

                # wham_mv_2d_for_eval = torch.cat((wham_2d_for_vis[0, 1:2], wham_2d_for_vis[0, 3:4]), dim=0) # 2 X T X (18*2) 
                wham_mv_2d_for_eval = wham_2d_for_vis.clone()[0, 1:] 

                # Visualization for 2D joint positions. 
                de_direct_sampled_pred_2d_for_vis = self.gen_multiview_vis_res(direct_sampled_seq_2d_for_vis[bs_idx:bs_idx+1], \
                    vis_folder_tag, actual_seq_len[bs_idx:bs_idx+1], \
                    dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_direct_sampled")
                
                de_gt_2d_for_vis = self.gen_multiview_vis_res(gt_for_vis[bs_idx:bs_idx+1], vis_folder_tag, \
                    actual_seq_len[bs_idx:bs_idx+1], vis_gt=True, \
                    dest_vis_data_folder=dest_res3d_npy_folder)

                # de_new_gt_2d_for_vis = self.gen_multiview_vis_res(new_gt_for_vis, vis_folder_tag, \
                #     actual_seq_len[bs_idx:bs_idx+1], \
                #     dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_new_gt")

                de_pred_2d_for_vis = self.gen_multiview_vis_res(pred_for_vis, vis_folder_tag, \
                    actual_seq_len[bs_idx:bs_idx+1], \
                    dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_ours")
                de_wham_2d_for_vis = self.gen_multiview_vis_res(wham_2d_for_vis, vis_folder_tag, \
                    actual_seq_len[bs_idx:bs_idx+1], \
                    dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_wham")

                tmp_idx = 0
                v_tmp_idx = 0 
                
                mpjpe, pa_mpjpe, trans_err, jpos3d_err, jpos2d_err, centered_jpos2d_err, bone_consistency = \
                compute_metrics_for_jpos(pred_smpl_jnts18, \
                    de_pred_2d_for_vis[tmp_idx, v_tmp_idx], \
                    gt_smpl_jnts18, \
                    de_gt_2d_for_vis[tmp_idx, v_tmp_idx], \
                    int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1, opt_jpos3d_18_list[bs_idx])

                wham_mpjpe, wham_pa_mpjpe, wham_trans_err, wham_jpos3d_err, wham_jpos2d_err, \
                wham_centered_jpos2d_err, wham_bone_consistency = \
                compute_metrics_for_jpos(wham_pred_smpl_jnts18.to(gt_smpl_jnts18_list[bs_idx].device), \
                    de_wham_2d_for_vis[tmp_idx, v_tmp_idx].to(gt_smpl_jnts18_list[bs_idx].device), \
                    gt_smpl_jnts18, \
                    de_gt_2d_for_vis[tmp_idx, v_tmp_idx], \
                    int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1, \
                    wham_jpos3d_18_list[bs_idx])

                # 3. For SMPLify 
                if self.add_vposer_opt_w_2d:
                    # Get reprojected our results 
                    # aligned_vposer2d_jnts18, aligned_vposer2d_verts = self.align_motionbert_to_ours(vposer2d_smpl_jnts18_list[bs_idx].detach().cpu().numpy(), \
                    #                         gt_smpl_jnts18_list[bs_idx], vposer2d_smpl_verts_list[bs_idx].detach().cpu().numpy())
                    # vposer2d_smpl_verts_list[bs_idx] = torch.from_numpy(aligned_vposer2d_verts).to(vposer2d_smpl_verts_list[bs_idx].device) 

                    # vposer2d_pred_smpl_jnts18 = self.replace_neck_jnt_w_avg(aligned_vposer2d_jnts18) 

                    vposer2d_pred_smpl_jnts18 = self.replace_neck_jnt_w_avg(vposer2d_smpl_jnts18_list[bs_idx]) 
                    vposer2d_reprojected_pred_jnts2d_list, vposer2d_reprojected_pred_ori_jnts2d_list, _ = \
                                    self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                    cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                    vposer2d_pred_smpl_jnts18.to(cam_rot_mat.device)*vposer_2d_scale_val_list[bs_idx]) # K X T X (18*2)
                    vposer2d_pred_for_vis = vposer2d_reprojected_pred_jnts2d_list.reshape(1, val_num_views, -1, 18*2)  

                    de_vposer2d_pred_2d_for_vis = self.gen_multiview_vis_res(vposer2d_pred_for_vis, vis_folder_tag, \
                        actual_seq_len[bs_idx:bs_idx+1], \
                        dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_smplify")

                    # smplify_mv_2d_for_eval = self.gen_multi_view_2d_seq_for_eval(vposer2d_pred_smpl_jnts18, vposer_2d_scale_val_list[bs_idx]) # K X T X (18*2) 
                    # smplify_mv_2d_for_eval = torch.cat((vposer2d_pred_for_vis[0, 1:2], vposer2d_pred_for_vis[0, 3:4]), dim=0) # 2 X T X (18*2) 
                    smplify_mv_2d_for_eval = vposer2d_pred_for_vis.clone()[0, 1:] 

                    smplify_mpjpe, smplify_pa_mpjpe, smplify_trans_err, smplify_jpos3d_err, \
                    smplify_jpos2d_err, smplify_centered_jpos2d_err, smplify_bone_consistency = \
                        compute_metrics_for_jpos(vposer2d_pred_smpl_jnts18, \
                        de_vposer2d_pred_2d_for_vis[tmp_idx, v_tmp_idx], \
                        gt_smpl_jnts18, \
                        de_gt_2d_for_vis[tmp_idx, v_tmp_idx], \
                        int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1, vposer2d_smpl_jnts18_list[bs_idx])
                else:
                    smplify_mpjpe, smplify_pa_mpjpe, smplify_trans_err, smplify_jpos3d_err, \
                    smplify_jpos2d_err, smplify_centered_jpos2d_err, smplify_bone_consistency = 0, 0, 0, 0, 0, 0, 0 

                # 4. For MotionBERT 
                if self.opt.add_motionbert_for_eval:
                    aligned_motionbert_jnts18, aligned_motionbert_verts = self.align_motionbert_to_ours(motionbert_smpl_jnts18_list[bs_idx].detach().cpu().numpy(), \
                                            gt_smpl_jnts18_list[bs_idx], motionbert_smpl_verts_list[bs_idx].detach().cpu().numpy())
                    aligned_motionbert_jnts18, scaled_aligned_motionbert_jnts18, \
                    aligned_motionbert_verts, scaled_aligned_motionbert_verts = \
                                self.compute_yaw_rotation_and_rotated_3d_pose(self.replace_neck_jnt_w_avg(aligned_motionbert_jnts18), \
                                gt_for_vis[bs_idx, 0].reshape(gt_for_vis.shape[2], -1, 2), cam_rot_mat[bs_idx, 0, 0], cam_trans[bs_idx, 0, 0], aligned_motionbert_verts)
  
                    # motionbert_smpl_verts_list[bs_idx] = torch.from_numpy(aligned_motionbert_verts).to(motionbert_smpl_verts_list[bs_idx].device)

                    # mbert_smpl_jnts18 = self.replace_neck_jnt_w_avg(aligned_motionbert_jnts18) 
                    reprojected_mbert_jnts2d_list, reprojected_mbert_ori_jnts2d_list, _ = \
                                    self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                    cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                    scaled_aligned_motionbert_jnts18.to(cam_rot_mat.device)) # K X T X (18*2)
                    mbert_for_vis = reprojected_mbert_jnts2d_list.reshape(1, val_num_views, -1, 18*2)   
                    de_mbert_2d_for_vis = self.gen_multiview_vis_res(mbert_for_vis, vis_folder_tag, \
                        actual_seq_len[bs_idx:bs_idx+1], \
                        dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_mbert")

                    # mbert_mv_2d_for_eval = self.gen_multi_view_2d_seq_for_eval(mbert_smpl_jnts18, scale_val_list[bs_idx]) # K X T X (18*2) 
                    # mbert_mv_2d_for_eval = torch.cat((mbert_for_vis[0, 1:2], mbert_for_vis[0, 3:4]), dim=0) # 2 X T X (18*2) 
                    mbert_mv_2d_for_eval = mbert_for_vis.clone()[0, 1:] 

                    mbert_mpjpe, mbert_pa_mpjpe, mbert_trans_err, mbert_jpos3d_err, mbert_jpos2d_err, mbert_centered_jpos2d_err, mbert_bone_consistency = \
                        compute_metrics_for_jpos(aligned_motionbert_jnts18, \
                            de_mbert_2d_for_vis[tmp_idx, v_tmp_idx], \
                            gt_smpl_jnts18, \
                            de_gt_2d_for_vis[tmp_idx, v_tmp_idx], \
                            int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1, aligned_motionbert_jnts18)
                else:
                    mbert_mpjpe, mbert_pa_mpjpe, mbert_trans_err, mbert_jpos3d_err, mbert_jpos2d_err, \
                    mbert_centered_jpos2d_err, mbert_bone_consistency = 0, 0, 0, 0, 0, 0, 0

                if self.opt.add_elepose_for_eval:
                    aligned_elepose_jnts18, aligned_elepose_verts = self.align_motionbert_to_ours(elepose_smpl_jnts18_list[bs_idx].detach().cpu().numpy(), \
                                            gt_smpl_jnts18_list[bs_idx], elepose_smpl_verts_list[bs_idx].detach().cpu().numpy())    
                    # elepose_smpl_verts_list[bs_idx] = torch.from_numpy(aligned_elepose_verts).to(elepose_smpl_verts_list[bs_idx].device) 
                    aligned_elepose_jnts18, scaled_aligned_elepose_jnts18, \
                    aligned_elepose_verts, scaled_aligned_elepose_verts = \
                                self.compute_yaw_rotation_and_rotated_3d_pose(self.replace_neck_jnt_w_avg(aligned_elepose_jnts18), \
                                gt_for_vis[bs_idx, 0].reshape(gt_for_vis.shape[2], -1, 2), cam_rot_mat[bs_idx, 0, 0], cam_trans[bs_idx, 0, 0], \
                                aligned_elepose_verts)
  

                    # elepose_smpl_jnts18 = self.replace_neck_jnt_w_avg(aligned_elepose_jnts18) 
                    reprojected_elepose_jnts2d_list, reprojected_elepose_ori_jnts2d_list, _ = \
                                    self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                    cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                    scaled_aligned_elepose_jnts18.to(cam_rot_mat.device)) # K X T X (18*2)
                    elepose_for_vis = reprojected_elepose_jnts2d_list.reshape(1, val_num_views, -1, 18*2)   
                    de_elepose_2d_for_vis = self.gen_multiview_vis_res(elepose_for_vis, vis_folder_tag, \
                        actual_seq_len[bs_idx:bs_idx+1], \
                        dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_elepose")

                    # elepose_mv_2d_for_eval = self.gen_multi_view_2d_seq_for_eval(elepose_smpl_jnts18, scale_val_list[bs_idx]) # K X T X (18*2) 
                    # elepose_mv_2d_for_eval = torch.cat((elepose_for_vis[0, 1:2], elepose_for_vis[0, 3:4]), dim=0) # 2 X T X (18*2) 
                    elepose_mv_2d_for_eval = elepose_for_vis.clone()[0, 1:]

                    # Elepose cannot predict root translation, so we should compare the results without root translation 
                    # tmp_gt_smpl_jnts18 = gt_smpl_jnts18_list[bs_idx].detach().cpu().numpy() 

                    elepose_mpjpe, elepose_pa_mpjpe, elepose_trans_err, elepose_jpos3d_err, elepose_jpos2d_err, \
                    elepose_centered_jpos2d_err, elepose_bone_consistency = \
                        compute_metrics_for_jpos(aligned_elepose_jnts18, \
                            de_elepose_2d_for_vis[tmp_idx, v_tmp_idx], \
                            gt_smpl_jnts18, \
                            de_gt_2d_for_vis[tmp_idx, v_tmp_idx], \
                            int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1, aligned_elepose_jnts18)
                else:
                    elepose_mpjpe, elepose_pa_mpjpe, elepose_trans_err, elepose_jpos3d_err, elepose_jpos2d_err, \
                    elepose_centered_jpos2d_err, elepose_bone_consistency = 0, 0, 0, 0, 0, 0, 0 

                if self.opt.add_mas_for_eval:
                    # aligned_mas_jnts18, aligned_mas_verts = self.align_motionbert_to_ours(mas_jnts18_list[bs_idx], \
                    #                         gt_smpl_jnts18_list[bs_idx], mas_smpl_verts_list[bs_idx].detach().cpu().numpy())

                    aligned_mas_jnts18, aligned_mas_verts = self.align_motionbert_to_ours(mas_smpl_jnts18_list[bs_idx].detach().cpu().numpy(), \
                                            gt_smpl_jnts18_list[bs_idx], mas_smpl_verts_list[bs_idx].detach().cpu().numpy())
                    # mas_smpl_verts_list[bs_idx] = torch.from_numpy(aligned_mas_verts).to(mas_smpl_verts_list[bs_idx].device) 
                    aligned_mas_jnts18, scaled_aligned_mas_jnts18, \
                    aligned_mas_verts, scaled_aligned_mas_verts = \
                                self.compute_yaw_rotation_and_rotated_3d_pose(self.replace_neck_jnt_w_avg(aligned_mas_jnts18), \
                                gt_for_vis[bs_idx, 0].reshape(gt_for_vis.shape[2], -1, 2), cam_rot_mat[bs_idx, 0, 0], cam_trans[bs_idx, 0, 0], \
                                aligned_mas_verts)

                    # mas_smpl_jnts18 = self.replace_neck_jnt_w_avg(aligned_mas_jnts18) 
                    reprojected_mas_jnts2d_list, reprojected_mas_ori_jnts2d_list, _ = \
                                    self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                    cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                    scaled_aligned_mas_jnts18.to(cam_rot_mat.device)) # K X T X (18*2)
                    mas_for_vis = reprojected_mas_jnts2d_list.reshape(1, val_num_views, -1, 18*2)   
                    de_mas_2d_for_vis = self.gen_multiview_vis_res(mas_for_vis, vis_folder_tag, \
                        actual_seq_len[bs_idx:bs_idx+1], \
                        dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_mas")

                    # mas_mv_2d_for_eval = self.gen_multi_view_2d_seq_for_eval(mas_smpl_jnts18, scale_val_list[bs_idx]) # K X T X (18*2) 
                    # mas_mv_2d_for_eval = torch.cat((mas_for_vis[0, 1:2], mas_for_vis[0, 3:4]), dim=0) # 2 X T X (18*2) \
                    mas_mv_2d_for_eval = mas_for_vis.clone()[0, 1:]

                    # Elepose cannot predict root translation, so we should compare the results without root translation 
                    # tmp_gt_smpl_jnts18 = gt_smpl_jnts18_list[bs_idx].detach().cpu().numpy() 

                    mas_mpjpe, mas_pa_mpjpe, mas_trans_err, mas_jpos3d_err, mas_jpos2d_err, \
                    mas_centered_jpos2d_err, mas_bone_consistency = \
                        compute_metrics_for_jpos(aligned_mas_jnts18, \
                            de_mas_2d_for_vis[tmp_idx, v_tmp_idx], \
                            gt_smpl_jnts18, \
                            de_gt_2d_for_vis[tmp_idx, v_tmp_idx], \
                            int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1, aligned_mas_jnts18)
                else:
                    mas_mpjpe, mas_pa_mpjpe, mas_trans_err, mas_jpos3d_err, mas_jpos2d_err, \
                    mas_centered_jpos2d_err, mas_bone_consistency = 0, 0, 0, 0, 0, 0, 0 

                self.add_new_evaluation_to_list(mpjpe, pa_mpjpe, trans_err, \
                        jpos3d_err, jpos2d_err, centered_jpos2d_err, bone_consistency, \
                        mpjpe_wham=wham_mpjpe, pa_mpjpe_wham=wham_pa_mpjpe, trans_err_wham=wham_trans_err, jpos_err_wham=wham_jpos3d_err, \
                        jpos2d_err_wham=wham_jpos2d_err, centered_jpos2d_err_wham=wham_centered_jpos2d_err, bone_consistency_wham=wham_bone_consistency, \
                        mpjpe_mbert=mbert_mpjpe, pa_mpjpe_mbert=mbert_pa_mpjpe, trans_err_mbert=mbert_trans_err, jpos_err_mbert=mbert_jpos3d_err, \
                        jpos2d_err_mbert=mbert_jpos2d_err, centered_jpos2d_err_mbert=mbert_centered_jpos2d_err, bone_consistency_mbert=mbert_bone_consistency, \
                        mpjpe_elepose=elepose_mpjpe, pa_mpjpe_elepose=elepose_pa_mpjpe, trans_err_elepose=elepose_trans_err, \
                        jpos_err_elepose=elepose_jpos3d_err, jpos2d_err_elepose=elepose_jpos2d_err, \
                        centered_jpos2d_err_elepose=elepose_centered_jpos2d_err, bone_consistency_elepose=elepose_bone_consistency, \
                        mpjpe_smplify=smplify_mpjpe, pa_mpjpe_smplify=smplify_pa_mpjpe, trans_err_smplify=smplify_trans_err, \
                        jpos_err_smplify=smplify_jpos3d_err, jpos2d_err_smplify=smplify_jpos2d_err, \
                        centered_jpos2d_err_smplify=smplify_centered_jpos2d_err, bone_consistency_smplify=smplify_bone_consistency, \
                        mpjpe_mas=mas_mpjpe, pa_mpjpe_mas=mas_pa_mpjpe, trans_err_mas=mas_trans_err, jpos_err_mas=mas_jpos3d_err, \
                        jpos2d_err_mas=mas_jpos2d_err, centered_jpos2d_err_mas=mas_centered_jpos2d_err, bone_consistency_mas=mas_bone_consistency) 

                self.print_mean_metrics() 

                curr_seq_len = int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1 
                # dest_jnts3d_npy_path = os.path.join(dest_res3d_npy_folder, \
                #         str(s_idx)+"_sample_"+str(bs_idx)+".npy")
                # np.save(dest_jnts3d_npy_path, pred_smpl_jnts18.detach().cpu().numpy()[:curr_seq_len])

                if self.eval_w_best_mpjpe:
                    dest_jnts3d_json_path = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_eval.json")
                    eval_dict = {}
                    eval_dict['mpjpe'] = str(mpjpe)
                    eval_dict['pa_mpjpe'] = str(pa_mpjpe)            
                    eval_dict['trans_err'] = str(trans_err)
                    eval_dict['jpos3d_err'] = str(jpos3d_err)
                    eval_dict['jpos2d_err'] = str(jpos2d_err)
                    eval_dict['centered_jpos2d_err'] = str(centered_jpos2d_err) 
                
                    eval_dict['wham_mpjpe'] = str(wham_mpjpe)
                    eval_dict['wham_pa_mpjpe'] = str(wham_pa_mpjpe)
                    eval_dict['wham_trans_err'] = str(wham_trans_err) 
                    eval_dict['wham_jpos3d_err'] = str(wham_jpos3d_err) 
                    eval_dict['wham_jpos2d_err'] = str(wham_jpos2d_err) 
                    eval_dict['wham_centered_jpos2d_err'] = str(wham_centered_jpos2d_err)

                    # MotionBERT
                    eval_dict['mbert_mpjpe'] = str(mbert_mpjpe)
                    eval_dict['mbert_pa_mpjpe'] = str(mbert_pa_mpjpe)
                    eval_dict['mbert_trans_err'] = str(mbert_trans_err)
                    eval_dict['mbert_jpos2d_err'] = str(mbert_jpos2d_err)
                    eval_dict['mbert_jpos3d_err'] = str(mbert_jpos3d_err) 
                    eval_dict['mbert_centered_jpos2d_err'] = str(mbert_centered_jpos2d_err)

                    # Elepose
                    eval_dict['elepose_mpjpe'] = str(elepose_mpjpe)
                    eval_dict['elepose_pa_mpjpe'] = str(elepose_pa_mpjpe)
                    eval_dict['elepose_trans_err'] = str(elepose_trans_err)
                    eval_dict['elepose_jpos2d_err'] = str(elepose_jpos2d_err)
                    eval_dict['elepose_jpos3d_err'] = str(elepose_jpos3d_err) 
                    eval_dict['elepose_centered_jpos2d_err'] = str(elepose_centered_jpos2d_err)

                    # MAS
                    eval_dict['mas_mpjpe'] = str(mas_mpjpe)
                    eval_dict['mas_pa_mpjpe'] = str(mas_pa_mpjpe)
                    eval_dict['mas_trans_err'] = str(mas_trans_err)
                    eval_dict['mas_jpos2d_err'] = str(mas_jpos2d_err)
                    eval_dict['mas_jpos3d_err'] = str(mas_jpos3d_err) 
                    eval_dict['mas_centered_jpos2d_err'] = str(mas_centered_jpos2d_err)

                    # SMPLify
                    eval_dict['smplify_mpjpe'] = str(smplify_mpjpe)
                    eval_dict['smplify_pa_mpjpe'] = str(smplify_pa_mpjpe)
                    eval_dict['smplify_trans_err'] = str(smplify_trans_err)
                    eval_dict['smplify_jpos2d_err'] = str(smplify_jpos2d_err)
                    eval_dict['smplify_jpos3d_err'] = str(smplify_jpos3d_err) 
                    eval_dict['smplify_centered_jpos2d_err'] = str(smplify_centered_jpos2d_err)

                    json.dump(eval_dict, open(dest_jnts3d_json_path, 'w')) 

                    # Save multi-view 2D sequences 
                    dest_ours_2d_folder = os.path.join(dest_res3d_npy_folder, "ours_2d")
                    dest_wham_2d_folder = os.path.join(dest_res3d_npy_folder, "wham_2d")
                    dest_motionbert_2d_folder = os.path.join(dest_res3d_npy_folder, "motionbert_2d") 
                    dest_smplify_2d_folder = os.path.join(dest_res3d_npy_folder, "smplify_2d") 
                    dest_elepose_2d_folder = os.path.join(dest_res3d_npy_folder, "elepose_2d") 
                    dest_mas_2d_folder = os.path.join(dest_res3d_npy_folder, "mas_2d") 
                    if not os.path.exists(dest_ours_2d_folder):
                        os.makedirs(dest_ours_2d_folder)
                    if not os.path.exists(dest_wham_2d_folder):
                        os.makedirs(dest_wham_2d_folder)
                    if not os.path.exists(dest_smplify_2d_folder):
                        os.makedirs(dest_smplify_2d_folder)
                    if not os.path.exists(dest_motionbert_2d_folder):
                        os.makedirs(dest_motionbert_2d_folder)
                    if not os.path.exists(dest_elepose_2d_folder):
                        os.makedirs(dest_elepose_2d_folder)
                    if not os.path.exists(dest_mas_2d_folder):
                        os.makedirs(dest_mas_2d_folder)

                    dest_wham_2d_res_path = os.path.join(dest_wham_2d_folder, str(s_idx)+"_sample_"+str(bs_idx)+"_eval.npy")
                    np.save(dest_wham_2d_res_path, wham_mv_2d_for_eval.detach().cpu().numpy()) # (K-1) X T X 18 X 2 

                    dest_ours_2d_res_path = os.path.join(dest_ours_2d_folder, str(s_idx)+"_sample_"+str(bs_idx)+"_eval.npy")
                    np.save(dest_ours_2d_res_path, ours_mv_2d_for_eval.detach().cpu().numpy()) # (K-1) X T X 18 X 2 

                    if self.add_vposer_opt_w_2d:
                        dest_vposer2d_2d_res_path = os.path.join(dest_smplify_2d_folder, str(s_idx)+"_sample"+str(bs_idx)+"_eval.npy")
                        np.save(dest_vposer2d_2d_res_path, smplify_mv_2d_for_eval.detach().cpu().numpy()) # (K-1) X T X 18 X 2

                    if self.opt.add_motionbert_for_eval:
                        dest_motionbert_2d_res_path = os.path.join(dest_motionbert_2d_folder, str(s_idx)+"_sample"+str(bs_idx)+"_eval.npy")
                        np.save(dest_motionbert_2d_res_path, mbert_mv_2d_for_eval.detach().cpu().numpy()) # (K-1) X T X 18 X 2 

                    if self.opt.add_elepose_for_eval:
                        dest_elepose_2d_res_path = os.path.join(dest_elepose_2d_folder, str(s_idx)+"_sample"+str(bs_idx)+"_eval.npy")
                        np.save(dest_elepose_2d_res_path, elepose_mv_2d_for_eval.detach().cpu().numpy()) # (K-1) X T X 18 X 2

                    if self.opt.add_mas_for_eval:
                        dest_mas_2d_res_path = os.path.join(dest_mas_2d_folder, str(s_idx)+"_sample"+str(bs_idx)+"_eval.npy")
                        np.save(dest_mas_2d_res_path, mas_mv_2d_for_eval.detach().cpu().numpy())

                if self.opt.gen_vis_res_for_demo:
                    # Prepare videos for comparisons. 
                    if "nicole" in self.youtube_data_npz_folder:
                        dest_for_demo_res_folder = "/move/u/jiamanli/for_cvpr25_lift_demo/nicole"

                        ori_rendered_vid_folder = "/move/u/jiamanli/final_cvpr25_for_demo/nicole_for_vis_fast_blender_video_res"
                    elif "AIST" in self.youtube_data_npz_folder:
                        dest_for_demo_res_folder = "/move/u/jiamanli/for_cvpr25_lift_demo/AIST"

                        ori_rendered_vid_folder = "/move/u/jiamanli/final_cvpr25_for_demo/AIST_for_vis_blender_video_res"
                    elif "steezy" in self.youtube_data_npz_folder:
                        dest_for_demo_res_folder = "/move/u/jiamanli/for_cvpr25_lift_demo/steezy"

                        ori_rendered_vid_folder = "/move/u/jiamanli/final_cvpr25_for_demo/nicole_for_vis_fast_blender_video_res"

                    if "s_idx" in val_data_dict:
                        start_idx = val_data_dict['s_idx'][bs_idx]
                        end_idx = val_data_dict['e_idx'][bs_idx]

                        if torch.is_tensor(start_idx):
                            start_idx = start_idx.item()
                            end_idx = end_idx.item() 
                    else:
                        start_idx = 0 
                        end_idx = 120 * 2 # For AIST, 60 fps data. 

                    # Gather original RGB images 
                    dest_for_ori_img_frames_folder = os.path.join(dest_for_demo_res_folder, "ori_img_frames")
                    # self.load_ori_rgb_images(val_data_dict, dest_for_ori_img_frames_folder)

                    # Gather 2D pose sequences visualizations aligned with original image resolution 
                    dest_motion_2d_vid_folder = os.path.join(dest_for_demo_res_folder, "input_2d_pose_videos") 
                    if not os.path.exists(dest_motion_2d_vid_folder):
                        os.makedirs(dest_motion_2d_vid_folder) 
                    dest_motion2d_vid_path = os.path.join(dest_motion_2d_vid_folder,  \
                        str(s_idx)+"_sample_"+str(bs_idx)+"_start_"+str(start_idx)+"_end_"+str(end_idx)+val_data_dict['seq_name'][0].replace(".npz", "_pose2d.mp4"))
                    visualize_pose_sequence_to_video_for_demo(de_gt_2d_for_vis[bs_idx, 0].reshape(-1, 18, 2).detach().cpu().numpy(), dest_motion2d_vid_path) # T X 120 X (J*3)

                    # Gather multi-view 2D pose sequences visualizations aligned with original image resolution 
                    dest_motion_2d_vid_folder = os.path.join(dest_for_demo_res_folder, "output_mv_2d_pose_videos") 
                    if not os.path.exists(dest_motion_2d_vid_folder):
                        os.makedirs(dest_motion_2d_vid_folder) 
                    
                    num_views_for_vis = de_pred_2d_for_vis.shape[1]
                 
                    for tmp_v_idx in range(num_views_for_vis):
                        dest_motion2d_vid_path = os.path.join(dest_motion_2d_vid_folder,  \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_start_"+str(start_idx)+"_end_"+str(end_idx)+val_data_dict['seq_name'][0].replace(".npz", \
                                "_view_"+str(tmp_v_idx)+"_out_mv_pose2d.mp4"))
                        visualize_pose_sequence_to_video_for_demo(de_pred_2d_for_vis[bs_idx, tmp_v_idx].reshape(-1, 18, 2).detach().cpu().numpy(), dest_motion2d_vid_path) # T X 120 X (J*3)

                    # Move rendered video to the same folder for comparisons 
                    # ori_rendered_vid_folder = "/move/u/jiamanli/final_cvpr25_opt_3d_w_multiview_diffusion"
                    
                    # dest_rendered_vid_folder = os.path.join(dest_for_demo_res_folder, "rendered_videos")
                    # if not os.path.exists(dest_rendered_vid_folder):
                    #     os.makedirs(dest_rendered_vid_folder) 

                    # ori_vid_path = os.path.join(ori_rendered_vid_folder, str(s_idx)+"_sample_"+str(bs_idx)+"_objs.mp4")
                    # pattern = os.path.join(ori_rendered_vid_folder, f"{s_idx}_sample_{bs_idx}_objs*.mp4")
                    # # Use glob to find matching files
                    # ori_matching_files = glob.glob(pattern)
                   
                    # for ori_vid_path in ori_matching_files:
                    #     tag = ori_vid_path.split("/")[-1].split("_objs_")[1]
                    #     dest_vid_path = os.path.join(dest_rendered_vid_folder, \
                    #         str(s_idx)+"_sample_"+str(bs_idx)+"_start_"+str(start_idx)+"_end_"+str(end_idx)+val_data_dict['seq_name'][0].replace(".npz", "")+tag)
                    #     shutil.copy(ori_vid_path, dest_vid_path) 

                    # Visualize root trajectories for comparisons 
                    # dest_for_root_traj_vis_folder = os.path.join(dest_for_demo_res_folder, "root_traj3D_vis")
                    # self.prep_global_traj_cmp_vis(pred_smpl_jnts18, gt_smpl_jnts18, wham_pred_smpl_jnts18, aligned_motionbert_jnts18, \
                    #     vposer2d_pred_smpl_jnts18, s_idx, bs_idx, dest_for_root_traj_vis_folder)



                if not self.eval_w_best_mpjpe:
                    # dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                    #         str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_ours")
                    # plot_3d_motion(dest_vid_3d_path_global, \
                    #     opt_jpos3d_18_list[bs_idx].detach().cpu().numpy()[:curr_seq_len]) 

                    dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_ours")
                    plot_3d_motion(dest_vid_3d_path_global, \
                        pred_smpl_jnts18.detach().cpu().numpy()[:curr_seq_len]) 
              
                    # Visualize WHAM results 
                    dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_wham")
                    plot_3d_motion(dest_vid_3d_path_global, \
                        wham_pred_smpl_jnts18.detach().cpu().numpy()[:curr_seq_len]) 
                    
                    if self.opt.add_motionbert_for_eval:
                        dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_mbert")
                        plot_3d_motion(dest_vid_3d_path_global, \
                            aligned_motionbert_jnts18.detach().cpu().numpy()[:curr_seq_len]) 

                    if self.opt.add_elepose_for_eval:
                        dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_elepose")
                        plot_3d_motion(dest_vid_3d_path_global, \
                            aligned_elepose_jnts18.detach().cpu().numpy()[:curr_seq_len]) 

                    if self.opt.add_mas_for_eval:
                        dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_mas")
                        plot_3d_motion(dest_vid_3d_path_global, \
                            aligned_mas_jnts18.detach().cpu().numpy()[:curr_seq_len]) 
                        
                    if self.add_vposer_opt_w_2d:
                        dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_smplify")
                        plot_3d_motion(dest_vid_3d_path_global, \
                            vposer2d_pred_smpl_jnts18.detach().cpu().numpy()[:curr_seq_len]) 
                   
                    gt_dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_gt")
                    plot_3d_motion(gt_dest_vid_3d_path_global, \
                        gt_smpl_jnts18_list[bs_idx].detach().cpu().numpy()[:curr_seq_len]) 

                    # Save WHAM mesh 
                    dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_objs_wham")
                    if not os.path.exists(dest_mesh_folder):
                        os.makedirs(dest_mesh_folder)
                    for_vis_wham_mesh_seq = self.process_verts_for_vis_cmp(wham_pred_verts_aligned)
                    for t_idx in range(curr_seq_len):
                        skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_wham.obj")

                        curr_mesh = trimesh.Trimesh(vertices=for_vis_wham_mesh_seq[t_idx].detach().cpu().numpy(),
                            faces=skel_faces) 
        
                        curr_mesh.export(skin_mesh_out)

                    # Save GT mesh
                    dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_objs_gt")
                    for_vis_gt_mesh_seq = self.process_verts_for_vis_cmp(gt_verts_smpl_list[bs_idx])
                    if not os.path.exists(dest_mesh_folder):
                        os.makedirs(dest_mesh_folder)
                    for t_idx in range(curr_seq_len):
                        skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_gt.obj")

                        curr_mesh = trimesh.Trimesh(vertices=for_vis_gt_mesh_seq[t_idx].detach().cpu().numpy(),
                            faces=skel_faces) 
        
                        curr_mesh.export(skin_mesh_out)

                    # Save ours 
                    dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_objs")
                    if not os.path.exists(dest_mesh_folder):
                        os.makedirs(dest_mesh_folder)
                    for_vis_ours_mesh_seq = self.process_verts_for_vis_cmp(pred_verts_smpl_list[bs_idx]) 
                    for t_idx in range(curr_seq_len):
                        skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_ours.obj")

                        curr_mesh = trimesh.Trimesh(vertices=for_vis_ours_mesh_seq[t_idx].detach().cpu().numpy(),
                            faces=skel_faces) 

                        curr_mesh.export(skin_mesh_out)

                    # Save SMPLify 
                    if self.add_vposer_opt_w_2d:
                        dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_objs_smplify")
                        if not os.path.exists(dest_mesh_folder):
                            os.makedirs(dest_mesh_folder)
                        for_vis_smplify_mesh_seq = self.process_verts_for_vis_cmp(vposer2d_smpl_verts_list[bs_idx]) 
                        for t_idx in range(curr_seq_len):
                            skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_smplify.obj")

                            curr_mesh = trimesh.Trimesh(vertices=for_vis_smplify_mesh_seq[t_idx].detach().cpu().numpy(),
                                faces=skel_faces) 

                            curr_mesh.export(skin_mesh_out)

                    # Save MotionBERT 
                    if self.opt.add_motionbert_for_eval:
                        dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_objs_mbert")
                        if not os.path.exists(dest_mesh_folder):
                            os.makedirs(dest_mesh_folder)
                        for_vis_motionbert_mesh_seq = self.process_verts_for_vis_cmp(aligned_motionbert_verts)
                        for t_idx in range(curr_seq_len):
                            skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_mbert.obj")

                            curr_mesh = trimesh.Trimesh(vertices=for_vis_motionbert_mesh_seq[t_idx].detach().cpu().numpy(),
                                faces=skel_faces) 

                            curr_mesh.export(skin_mesh_out)

                    if self.opt.add_elepose_for_eval:
                        dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_objs_elepose")
                        if not os.path.exists(dest_mesh_folder):
                            os.makedirs(dest_mesh_folder)
                        for_vis_elepose_mesh_seq = self.process_verts_for_vis_cmp(aligned_elepose_verts)
                        for t_idx in range(curr_seq_len):
                            skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_elepose.obj")

                            curr_mesh = trimesh.Trimesh(vertices=for_vis_elepose_mesh_seq[t_idx].detach().cpu().numpy(),
                                faces=skel_faces) 

                            curr_mesh.export(skin_mesh_out)

                    if self.opt.add_mas_for_eval:
                        dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_objs_mas")
                        if not os.path.exists(dest_mesh_folder):
                            os.makedirs(dest_mesh_folder)
                        for_vis_mas_mesh_seq = self.process_verts_for_vis_cmp(aligned_mas_verts)
                        for t_idx in range(curr_seq_len):
                            skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_mas.obj")

                            curr_mesh = trimesh.Trimesh(vertices=for_vis_mas_mesh_seq[t_idx].detach().cpu().numpy(),
                                faces=skel_faces) 

                            curr_mesh.export(skin_mesh_out)

                # if self.opt.gen_vis_res_for_demo:
                #     # Prepare videos for comparisons. 
                #     dest_for_demo_res_folder = "/move/u/jiamanli/for_cvpr25_lift_demo/"

                #     # Gather original RGB images 
                #     dest_for_ori_img_frames_folder = os.path.join(dest_for_demo_res_folder, "ori_img_frames")
                #     # self.load_ori_rgb_images(val_data_dict, dest_for_demo_res_folder)

                #     # Gather 2D pose sequences visualizations aligned with original image resolution 
                #     dest_motion_2d_vid_folder = os.path.join(dest_for_demo_res_folder, "motion_2d_videos") 

                #     # Move rendered video to the same folder for comparisons 
                #     ori_rendered_vid_folder = "/move/u/jiamanli/final_cvpr25_opt_3d_w_multiview_diffusion"
                #     dest_rendered_vid_folder = os.path.join(dest_for_demo_res_folder, "rendered_videos")

                #     # Visualize root trajectories for comparisons 
                #     dest_for_root_traj_vis_folder = os.path.join(dest_for_demo_res_folder, "root_traj3D_vis")
                #     self.prep_global_traj_cmp_vis(pred_smpl_jnts18, gt_smpl_jnts18, wham_pred_smpl_jnts18, aligned_motionbert_jnts18, \
                #         vposer2d_pred_smpl_jnts18, s_idx, bs_idx, dest_for_root_traj_vis_folder)


                    # import pdb 
                    # pdb.set_trace()  
            # break 
    
    def eval_3d_w_multiview_diffusion_for_interaction(self):
        # Load line-conditioned model. 
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        # milestone = "3"
        # milestone = "4"
        # milestone = "2"
      
        self.load(milestone)
        self.ema.ema_model.eval()
       
        # Prepare testing data loader 
        if self.eval_w_best_mpjpe:
            batch = 32 # 64 
        else:
            batch = 1 
        test_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=batch, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False) 

        dest_res3d_npy_folder = self.prep_vis_res_folder() 

        # Prepare empty list for evaluation metrics 
        self.prep_evaluation_metric_list_for_interaction() 
        
        for s_idx, val_data_dict in enumerate(test_loader): 

            # if s_idx not in [3, 21, 25]:
            #     continue 

            # if s_idx % 4 != 0:
            #     continue 

            if s_idx not in [40]:
                continue 

            # if s_idx not in [4, 16, 20, 48]:
            #     continue 

          
            direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, \
            val_bs, val_num_views, val_num_steps, val_data, \
            actual_seq_len, opt_combined_jnts3d_23, \
            pred_smpl_jnts18_list, \
            pred_smpl_verts_list, skel_faces, cam_extrinsic, \
            scale_val_list, opt_smpl_seq_list, \
            ori_opt_obj_param_list, opt_obj_param_list, opt_obj_jpos3d_list, opt_obj_verts_list, obj_faces, obj_scale_val_list = \
                self.opt_3d_w_learned_model_for_interaction(val_data_dict)

            # Human vertices are for human scale = 1, use scale_val_list to get reprojection right
            # object vertices are for object scale = 1, use obj_scale_val_list to get reprojection right 

            # The direct sampled 2D pose sequences of the trained 2D motion diffusion model, the first view is the input GT 2D poses  
            direct_sampled_seq_2d_for_vis = direct_sampled_epi_seq_2d.reshape(val_bs, val_num_views, \
                        val_num_steps, -1) # BS X K X T X D
                    
            # The input GT 2D pose sequences. 
            gt_for_vis = val_data.reshape(val_bs, val_num_views, val_num_steps, -1)
            
            cam_rot_mat = cam_extrinsic[:, :, :3, :3].unsqueeze(2).repeat(1, 1, val_num_steps, 1, 1) # BS X K X T X 3 X 3 
            cam_trans = cam_extrinsic[:, :, :3, -1].unsqueeze(2).repeat(1, 1, val_num_steps, 1) # BS X K X T X 3 

            # For SMPLify baseline
            if self.add_vposer_opt_w_2d:
                smplify_ori_opt_obj_param_list, smplify_opt_obj_param_list, smplify_opt_obj_jpos3d_list, \
                smplify_opt_obj_verts_list, obj_faces, smplify_obj_scale_val_list = \
                    self.ema.ema_model.opt_3d_object_w_joints2d_input(gt_for_vis[:, 0, :, -5*2:].reshape(val_bs, -1, 5, 2).detach().cpu().numpy(), \
                    cam_rot_mat[:, 0], cam_trans[:, 0])

                vposer_2d_ori_opt_skel_seq_list, vposer_2d_opt_skel_seq_list, skel_faces, vposer_2d_scale_val_list = \
                    self.ema.ema_model.opt_3d_w_vposer_w_joints2d_reprojection(gt_for_vis[:, 0, :, :-5*2], cam_rot_mat[:, 0], cam_trans[:, 0])
                # In vposer_2d_opt_skel_seq_list, scaling = 1 

                vposer2d_smpl_jnts18_list, vposer2d_smpl_verts_list = \
                    self.get_jnts3d_w_smpl_params_for_interaction(vposer_2d_opt_skel_seq_list)


            actual_seq_len = actual_seq_len.reshape(val_bs, -1) # BS X K 

            for bs_idx in range(val_bs):
                if val_data_dict['seq_len'][bs_idx] < 120:
                    continue 

                vis_folder_tag = "gen_multiview_2d_batch_"+str(s_idx)+"_seq_"+str(bs_idx)

                gt_combined_jnts3d = val_data_dict['jnts3d_18'][bs_idx] # T X 23 X 3 
                gt_smpl_jnts18 = gt_combined_jnts3d[:, :18] # T X 18 X 3 
                gt_obj_jpos3d = gt_combined_jnts3d[:, 18:] # T X 5 X 3 

                # pred_smpl_jnts18 = self.replace_neck_jnt_w_avg(pred_smpl_jnts18_list[bs_idx]) 
                # pred_smpl_jnts18 = pred_smpl_jnts18_list[bs_idx]

                # pred_smpl_jnts18 = pred_smpl_jnts18_list[bs_idx] * scale_val_list[bs_idx] / obj_scale_val_list[bs_idx]
                # pred_smpl_verts = pred_smpl_verts_list[bs_idx] * scale_val_list[bs_idx] / obj_scale_val_list[bs_idx]

                pred_smpl_jnts18 = pred_smpl_jnts18_list[bs_idx] / obj_scale_val_list[bs_idx]
                pred_smpl_verts = pred_smpl_verts_list[bs_idx] / obj_scale_val_list[bs_idx]

                # import pdb 
                # pdb.set_trace() 

                pred_obj_kpts = opt_obj_jpos3d_list[bs_idx] # For object_scale = 1 

                combined_jnts3d = torch.cat((pred_smpl_jnts18.to(opt_obj_jpos3d_list[bs_idx].device), \
                            pred_obj_kpts.to(opt_obj_jpos3d_list[bs_idx].device)), dim=1) # T X 23 X 3 

                reprojected_pred_jnts2d_list, reprojected_pred_ori_jnts2d_list, _ = \
                                        self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                        cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                        combined_jnts3d.to(cam_rot_mat.device)*obj_scale_val_list[bs_idx]) # K X T X (18*2)

                pred_for_vis = reprojected_pred_jnts2d_list.reshape(1, val_num_views, 120, -1)   

                vis_for_paper = False 
                if vis_for_paper:
                    dest_for_paper_vis_folder = "./for_cvpr25_paper_vis_2d_sequences_omomo_sample3_sample21_sample25"
                    if not os.path.exists(dest_for_paper_vis_folder):
                        os.makedirs(dest_for_paper_vis_folder) 

                    # pose_sequence = epi_line_cond_seq_2d 
                    # line_coeffs =  
                    # visualize_pose_and_lines_for_paper(pose_sequence, line_coeffs, output_folder, vis_gt=False, epipoles=None)
                    
                    # Plot for single pose 
                    num_steps = reprojected_pred_jnts2d_list.shape[1] 
                    for t_idx in range(0, num_steps, 1):
                        dest_pose2d_fig_path = os.path.join(dest_for_paper_vis_folder, "s_idx_"+str(s_idx)+"_bs_idx_"+str(bs_idx)+"_pose2d_"+str(t_idx)+".png") 
                        plot_pose_2d_omomo_for_paper(reprojected_pred_jnts2d_list[0, t_idx].reshape(-1, 2).detach().cpu().numpy(), dest_pose2d_fig_path) 

                    # import pdb 
                    # pdb.set_trace() 

                # Align our results to GT 
                pred_smpl_jnts18, pred_obj_kpts, pred_scale_factor_for_human = self.align_motionbert_to_ours(pred_smpl_jnts18.detach().cpu().numpy(), \
                                            gt_smpl_jnts18, pred_obj_kpts.detach().cpu().numpy(), input_obj_kpts=True)   
                combined_jnts3d = torch.cat((pred_smpl_jnts18.to(opt_obj_jpos3d_list[bs_idx].device), \
                            pred_obj_kpts.to(opt_obj_jpos3d_list[bs_idx].device)), dim=1) # T X 23 X 3 


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

                # Visualize the original optimized 3D joint results 
                init_reprojected_pred_jnts2d_list, init_reprojected_pred_ori_jnts2d_list, _ = \
                                        self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                        cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                        opt_combined_jnts3d_23[bs_idx].to(cam_rot_mat.device)) # K X T X (18*2)

                init_pred_for_vis = init_reprojected_pred_jnts2d_list.reshape(1, val_num_views, 120, -1)   
                de_init_pred_2d_for_vis = self.gen_multiview_vis_res(init_pred_for_vis, vis_folder_tag, \
                    actual_seq_len[bs_idx:bs_idx+1], \
                    dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_init_opt3d")     


                tmp_idx = 0
                v_tmp_idx = 0 

                # Compute metrics for human pose estimation    
                mpjpe, pa_mpjpe, trans_err, jpos3d_err, jpos2d_err, centered_jpos2d_err, bone_consistency, obj_mpjpe, obj_trans_err = \
                compute_metrics_for_jpos(pred_smpl_jnts18, \
                    de_pred_2d_for_vis[tmp_idx, v_tmp_idx], \
                    gt_smpl_jnts18, \
                    de_gt_2d_for_vis[tmp_idx, v_tmp_idx], \
                    int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1, pred_smpl_jnts18, \
                    obj_kpts_pred=pred_obj_kpts, obj_kpts_gt=gt_obj_jpos3d)

             

                # # 3. For SMPLify 
                if self.add_vposer_opt_w_2d:
                    # Make scale consistent for human and object 
                    smplify_smpl_scaling = smplify_obj_scale_val_list[bs_idx] # Use object's sdcale to adjust human's scale 
                    # smplify_smpl_jnts18 = vposer2d_smpl_jnts18_list[bs_idx] * vposer_2d_scale_val_list[bs_idx] 
                    # smplify_verts_smpl = vposer2d_smpl_verts_list[bs_idx] * vposer_2d_scale_val_list[bs_idx]

                    smplify_smpl_jnts18 = vposer2d_smpl_jnts18_list[bs_idx] * vposer_2d_scale_val_list[bs_idx] / smplify_smpl_scaling
                    smplify_verts_smpl = vposer2d_smpl_verts_list[bs_idx] * vposer_2d_scale_val_list[bs_idx] / smplify_smpl_scaling 

                    # smplify_smpl_jnts18 = vposer2d_smpl_jnts18_list[bs_idx] / smplify_smpl_scaling
                    # smplify_verts_smpl = vposer2d_smpl_verts_list[bs_idx] / smplify_smpl_scaling 

                    # Get reprojected our results 
                    aligned_vposer2d_jnts18, aligned_vposer2d_obj_kpts, smplify_scale_factor_for_human = self.align_motionbert_to_ours(smplify_smpl_jnts18.detach().cpu().numpy(), \
                                            gt_smpl_jnts18, smplify_opt_obj_jpos3d_list[bs_idx].detach().cpu().numpy(), input_obj_kpts=True) 

                    # _, aligned_vposer2d_verts = self.align_motionbert_to_ours(smplify_smpl_jnts18.detach().cpu().numpy(), \
                    #                         gt_smpl_jnts18, smplify_verts_smpl[bs_idx].detach().cpu().numpy(), input_obj_kpts=False) 

                    smplify_combined_jnts3d_for_quant = torch.cat((aligned_vposer2d_jnts18.to(smplify_opt_obj_jpos3d_list[bs_idx].device), \
                            aligned_vposer2d_obj_kpts.to(smplify_opt_obj_jpos3d_list[bs_idx].device)), dim=1) # T X 23 X 3 
                  

                    smplify_combined_jnts3d = torch.cat((smplify_smpl_jnts18.to(smplify_opt_obj_jpos3d_list[bs_idx].device), \
                        smplify_opt_obj_jpos3d_list[bs_idx]), dim=1) # T X 23 X 3 

                    vposer2d_reprojected_pred_jnts2d_list, vposer2d_reprojected_pred_ori_jnts2d_list, _ = \
                                    self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                    cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                    smplify_combined_jnts3d.to(cam_rot_mat.device)*smplify_obj_scale_val_list[bs_idx]) # K X T X (18*2)
                    vposer2d_pred_for_vis = vposer2d_reprojected_pred_jnts2d_list.reshape(1, val_num_views, -1, 23*2)  

                    de_vposer2d_pred_2d_for_vis = self.gen_multiview_vis_res(vposer2d_pred_for_vis, vis_folder_tag, \
                        actual_seq_len[bs_idx:bs_idx+1], \
                        dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_smplify")

                    # de_gt_2d_for_vis = self.gen_multiview_vis_res(gt_for_vis[bs_idx:bs_idx+1], vis_folder_tag, \
                    #     actual_seq_len[bs_idx:bs_idx+1], vis_gt=True, \
                    #     dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_smplify_gt") 

                    smplify_mpjpe, smplify_pa_mpjpe, smplify_trans_err, smplify_jpos3d_err, \
                    smplify_jpos2d_err, smplify_centered_jpos2d_err, smplify_bone_consistency, smplify_obj_mpjpe, smplify_obj_trans_err = \
                        compute_metrics_for_jpos(aligned_vposer2d_jnts18, \
                        de_vposer2d_pred_2d_for_vis[tmp_idx, v_tmp_idx], \
                        gt_smpl_jnts18, \
                        de_gt_2d_for_vis[tmp_idx, v_tmp_idx], \
                        int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1, aligned_vposer2d_jnts18, \
                        obj_kpts_pred=aligned_vposer2d_obj_kpts, obj_kpts_gt=gt_obj_jpos3d)

                  
                else:
                    smplify_mpjpe, smplify_pa_mpjpe, smplify_trans_err, smplify_jpos3d_err, \
                    smplify_jpos2d_err, smplify_centered_jpos2d_err, smplify_bone_consistency, \
                    smplify_obj_mpjpe, smplify_obj_trans_err = 0, 0, 0, 0, 0, 0, 0, 0, 0

                self.add_new_evaluation_to_list_for_interaction(mpjpe, pa_mpjpe, trans_err, \
                        jpos3d_err, jpos2d_err, centered_jpos2d_err, bone_consistency, obj_mpjpe, obj_trans_err, \
                        mpjpe_smplify=smplify_mpjpe, pa_mpjpe_smplify=smplify_pa_mpjpe, trans_err_smplify=smplify_trans_err, \
                        jpos_err_smplify=smplify_jpos3d_err, jpos2d_err_smplify=smplify_jpos2d_err, \
                        centered_jpos2d_err_smplify=smplify_centered_jpos2d_err, \
                        bone_consistency_smplify=smplify_bone_consistency, \
                        obj_mpjpe_smplify=smplify_obj_mpjpe, obj_trans_err_smplify=smplify_obj_trans_err) 

                self.print_mean_metrics_for_interaction() 

                if self.opt.gen_vis_res_for_demo:
                    # Prepare videos for comparisons. 
                   
                    dest_for_demo_res_folder = "/move/u/jiamanli/for_cvpr25_lift_demo/omomo"

                        # ori_rendered_vid_folder = "/move/u/jiamanli/final_cvpr25_for_demo/nicole_for_vis_fast_blender_video_res"
                  

                    if "s_idx" in val_data_dict:
                        start_idx = val_data_dict['s_idx'][bs_idx]
                        end_idx = val_data_dict['e_idx'][bs_idx]

                        if torch.is_tensor(start_idx):
                            start_idx = start_idx.item()
                            end_idx = end_idx.item() 
                    else:
                        start_idx = 0 
                        end_idx = 120 

                    # Gather 2D pose sequences visualizations aligned with original image resolution 
                    dest_motion_2d_vid_folder = os.path.join(dest_for_demo_res_folder, "input_2d_pose_videos") 
                    if not os.path.exists(dest_motion_2d_vid_folder):
                        os.makedirs(dest_motion_2d_vid_folder) 
                    dest_motion2d_vid_path = os.path.join(dest_motion_2d_vid_folder,  \
                        str(s_idx)+"_sample_"+str(bs_idx)+"_start_"+str(start_idx)+"_end_"+str(end_idx)+val_data_dict['seq_name'][0]+"_pose2d.mp4")
                    visualize_pose_sequence_to_video_for_demo_interaction(de_gt_2d_for_vis[bs_idx, 0].reshape(-1, 23, 2).detach().cpu().numpy(), dest_motion2d_vid_path) # T X 120 X (J*3)


                    # import pdb 
                    # pdb.set_trace() 
                    # Move rendered video to the same folder for comparisons 
                    # ori_rendered_vid_folder = "/move/u/jiamanli/final_cvpr25_opt_3d_w_multiview_diffusion"
                    
                    # dest_rendered_vid_folder = os.path.join(dest_for_demo_res_folder, "rendered_videos")
                    # if not os.path.exists(dest_rendered_vid_folder):
                    #     os.makedirs(dest_rendered_vid_folder) 

                    # ori_vid_path = os.path.join(ori_rendered_vid_folder, str(s_idx)+"_sample_"+str(bs_idx)+"_objs.mp4")
                    # pattern = os.path.join(ori_rendered_vid_folder, f"{s_idx}_sample_{bs_idx}_objs*.mp4")
                    # # Use glob to find matching files
                    # ori_matching_files = glob.glob(pattern)
                   
                    # for ori_vid_path in ori_matching_files:
                    #     tag = ori_vid_path.split("/")[-1].split("_objs_")[1]
                    #     dest_vid_path = os.path.join(dest_rendered_vid_folder, \
                    #         str(s_idx)+"_sample_"+str(bs_idx)+"_start_"+str(start_idx)+"_end_"+str(end_idx)+val_data_dict['seq_name'][0].replace(".npz", "")+tag)
                    #     shutil.copy(ori_vid_path, dest_vid_path) 

                    # # Visualize root trajectories for comparisons 
                    # dest_for_root_traj_vis_folder = os.path.join(dest_for_demo_res_folder, "root_traj3D_vis")
                    # self.prep_global_traj_cmp_vis(pred_smpl_jnts18, gt_smpl_jnts18, wham_pred_smpl_jnts18, aligned_motionbert_jnts18, \
                    #     vposer2d_pred_smpl_jnts18, s_idx, bs_idx, dest_for_root_traj_vis_folder)


              

                # curr_seq_len = int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1 
                curr_actual_seq_len = int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1 

                if not self.eval_w_best_mpjpe:
                    dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_gt")
                    # plot_3d_motion(dest_vid_3d_path_global, \
                    #     gt_smpl_jnts18.detach().cpu().numpy()[:curr_actual_seq_len], use_omomo_jnts23=True) 

                    # cmp_gt_ours = torch.cat((gt_combined_jnts3d[None].to(combined_jnts3d.device), combined_jnts3d[None]), dim=0) # 2 X T X J X 3 
                    # dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                    #             str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_cmp_gt_ours")
                    # plot_3d_motion_new(dest_vid_3d_path_global, cmp_gt_ours.detach().cpu().numpy(), use_omomo_jnts23=True)


                    plot_3d_motion(dest_vid_3d_path_global, \
                        gt_combined_jnts3d.detach().cpu().numpy()[:curr_actual_seq_len], use_omomo_jnts23=True) 


                    dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_init")
                    # plot_3d_motion(dest_vid_3d_path_global, \
                    #     pred_smpl_jnts18.detach().cpu().numpy()[:curr_actual_seq_len]) 
                    plot_3d_motion(dest_vid_3d_path_global, \
                        opt_combined_jnts3d_23[bs_idx].detach().cpu().numpy()[:curr_actual_seq_len], use_omomo_jnts23=True) 

                    dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_w_smpl_fitting")
                    # plot_3d_motion(dest_vid_3d_path_global, \
                    #     pred_smpl_jnts18.detach().cpu().numpy()[:curr_actual_seq_len]) 
                    plot_3d_motion(dest_vid_3d_path_global, \
                        combined_jnts3d.detach().cpu().numpy()[:curr_actual_seq_len], use_omomo_jnts23=True) 

                    if self.add_vposer_opt_w_2d:
                        dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_jpos3d_smplify")
                   
                        plot_3d_motion(dest_vid_3d_path_global, \
                            smplify_combined_jnts3d_for_quant.detach().cpu().numpy()[:curr_actual_seq_len], use_omomo_jnts23=True) 

                    dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_objs")
                    if not os.path.exists(dest_mesh_folder):
                        os.makedirs(dest_mesh_folder)
                    # for_vis_human_mesh_seq, for_vis_obj_mesh_seq = \
                    #     self.process_verts_for_vis_cmp_interaction(pred_smpl_verts_list[bs_idx], opt_obj_verts_list[bs_idx].to(pred_smpl_verts_list[bs_idx].device))
                    for_vis_human_mesh_seq, for_vis_obj_mesh_seq = \
                        self.process_verts_for_vis_cmp_interaction(pred_smpl_verts*pred_scale_factor_for_human, \
                        opt_obj_verts_list[bs_idx].to(pred_smpl_verts.device)*pred_scale_factor_for_human)
                    
                    for t_idx in range(curr_actual_seq_len):
                        skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_ours.obj")
                        

                        curr_mesh = trimesh.Trimesh(vertices=for_vis_human_mesh_seq[t_idx].detach().cpu().numpy(),
                            faces=skel_faces) 
                        curr_mesh.export(skin_mesh_out)

                        # Save object mesh 
                        obj_skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_ours_object.obj")

                        curr_obj_mesh = trimesh.Trimesh(vertices=for_vis_obj_mesh_seq[t_idx].detach().cpu().numpy(),
                            faces=obj_faces) 
                        curr_obj_mesh.export(obj_skin_mesh_out)

                    if self.add_vposer_opt_w_2d:
                        dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_objs_smplify")
                        if not os.path.exists(dest_mesh_folder):
                            os.makedirs(dest_mesh_folder)
                        for_vis_human_mesh_seq, for_vis_obj_mesh_seq = \
                            self.process_verts_for_vis_cmp_interaction(smplify_verts_smpl*smplify_scale_factor_for_human, \
                                smplify_opt_obj_verts_list[bs_idx].to(smplify_verts_smpl.device)*smplify_scale_factor_for_human) # vposer2d_smpl_verts_list[bs_idx]
                        # for_vis_human_mesh_seq, for_vis_obj_mesh_seq = \
                        #     self.process_verts_for_vis_cmp_interaction(vposer2d_smpl_verts_list[bs_idx], \
                        #         smplify_opt_obj_verts_list[bs_idx].to(vposer2d_smpl_verts_list[bs_idx].device))
                        for t_idx in range(curr_actual_seq_len):
                            skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_smplify.obj")
                            
                            curr_mesh = trimesh.Trimesh(vertices=for_vis_human_mesh_seq[t_idx].detach().cpu().numpy(),
                                faces=skel_faces) 
                            curr_mesh.export(skin_mesh_out)

                            # Save object mesh 
                            obj_skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_smplify_object.obj")

                            curr_obj_mesh = trimesh.Trimesh(vertices=for_vis_obj_mesh_seq[t_idx].detach().cpu().numpy(),
                                faces=obj_faces) 
                            curr_obj_mesh.export(obj_skin_mesh_out)

    def eval_3d_w_multiview_diffusion_for_animal(self):
        # Load line-conditioned model. 
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        # milestone = "1"
        # milestone = "4"
        # milestone = "2"
        # if self.eval_on_aist_3d:
        #     milestone = "3" 

        self.load(milestone)
        self.ema.ema_model.eval()
       
        # Prepare testing data loader 
        if self.eval_w_best_mpjpe:
            batch = 16 # 64 
        else:
            batch = 1 
        test_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=batch, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False) 

        dest_res3d_npy_folder = self.prep_vis_res_folder() 

        # Prepare empty list for evaluation metrics 
        self.prep_evaluation_metric_list() 
        
        for s_idx, val_data_dict in enumerate(test_loader): 
            if self.opt.gen_vis_res_for_demo:
                val_data = val_data_dict['normalized_jnts2d'].float() # BS X T X 18 X 2 
                
                de_gt_2d_for_vis = de_normalize_pose2d(val_data) # BSX T X 18 X 2 
       
                # Prepare videos for comparisons. 
               
                dest_for_demo_res_folder = "/move/u/jiamanli/for_cvpr25_lift_demo/cat_input_videos"

                bs_idx = 0 
                if "s_idx" in val_data_dict:
                    start_idx = val_data_dict['s_idx'][bs_idx]
                    end_idx = val_data_dict['e_idx'][bs_idx]

                    if torch.is_tensor(start_idx):
                        start_idx = start_idx.item()
                        end_idx = end_idx.item() 
                else:
                    start_idx = 0 
                    end_idx = 120  # For AIST, 60 fps data. 

                # Gather original RGB images 
                # dest_for_ori_img_frames_folder = os.path.join(dest_for_demo_res_folder, "ori_img_frames")
                # self.load_ori_rgb_images(val_data_dict, dest_for_ori_img_frames_folder)

                bs_idx = 0 
                # Gather 2D pose sequences visualizations aligned with original image resolution 
                dest_motion_2d_vid_folder = os.path.join(dest_for_demo_res_folder, "input_2d_pose_videos") 
                if not os.path.exists(dest_motion_2d_vid_folder):
                    os.makedirs(dest_motion_2d_vid_folder) 
                dest_motion2d_vid_path = os.path.join(dest_motion_2d_vid_folder,  \
                    str(s_idx)+"_sample_"+str(bs_idx)+"_start_"+str(start_idx)+"_end_"+str(end_idx)+val_data_dict['seq_name'][0].replace(".npz", "_pose2d.mp4"))
                visualize_pose_sequence_to_video_for_demo(de_gt_2d_for_vis[bs_idx].reshape(-1, 17, 2).detach().cpu().numpy(), \
                    dest_motion2d_vid_path, for_animal=True) # T X 120 X (J*3)
           
                continue 
           
            direct_sampled_epi_seq_2d, epi_line_cond_seq_2d, opt_jpos3d_18_list, cam_extrinsic, \
            val_bs, val_num_views, val_num_steps, val_data, actual_seq_len, \
            opt_skel_seq_list, skel_faces, scale_val_list, smal_opt_jnts3d, smal_opt_verts = \
                self.opt_3d_w_learned_model(val_data_dict)

            # The direct sampled 2D pose sequences of the trained 2D motion diffusion model, the first view is the input GT 2D poses  
            direct_sampled_seq_2d_for_vis = direct_sampled_epi_seq_2d.reshape(val_bs, val_num_views, \
                        val_num_steps, -1) # BS X K X T X D
                    
            # The input GT 2D pose sequences. 
            gt_for_vis = val_data.reshape(val_bs, val_num_views, val_num_steps, -1)
            
            cam_rot_mat = cam_extrinsic[:, :, :3, :3].unsqueeze(2).repeat(1, 1, val_num_steps, 1, 1) # BS X K X T X 3 X 3 
            cam_trans = cam_extrinsic[:, :, :3, -1].unsqueeze(2).repeat(1, 1, val_num_steps, 1) # BS X K X T X 3 

            actual_seq_len = actual_seq_len.reshape(val_bs, -1) # BS X K 

            if self.opt.add_vposer_opt_w_2d:
                vposer_2d_opt_skel_seq_list, \
                skel_faces, vposer_2d_scale_val_list, \
                vposer2d_smpl_jnts18_list, vposer2d_smpl_verts_list = \
                    self.ema.ema_model.opt_3d_w_smal_animal_w_joints2d_reprojection(gt_for_vis[:, 0], \
                    cam_rot_mat[:, 0], cam_trans[:, 0])

            # For MAS baseline 
            if self.opt.add_mas_for_eval:
                mas_jnts18_list = [] 
                for tmp_idx in range(val_bs):
                    if "s_idx" in val_data_dict:
                        mas_jnts18 = self.load_mas_res(val_data_dict['seq_name'][tmp_idx], \
                            start_idx=val_data_dict['s_idx'].detach().cpu().numpy()[tmp_idx], \
                            end_idx=val_data_dict['e_idx'].detach().cpu().numpy()[tmp_idx]) # For AIST , T X 18 X 3 
                    else:
                        # if "gBR_sBM_c09_d06_mBR5_ch08" in val_data_dict['seq_name'][tmp_idx]:
                        #     import pdb; pdb.set_trace() 

                        mas_jnts18 = self.load_mas_res(val_data_dict['seq_name'][tmp_idx]) # For AIST , T X 18 X 3 

                    mas_jnts18_list.append(mas_jnts18)
                
                mas_jnts18_list = np.asarray(mas_jnts18_list) * 10  # BS X T X 18 X 3 

                mas_opt_skel_seq_list, mas_skel_faces, mas_scale_val_list, mas_smal_opt_jnts3d, mas_smal_opt_verts  = \
                    self.ema.ema_model.opt_3d_w_smal_animal_w_joints3d_input(mas_jnts18_list)

            for bs_idx in range(val_bs):
                if val_data_dict['seq_len'][bs_idx] < 120:
                    continue 

                vis_folder_tag = "gen_multiview_2d_batch_"+str(s_idx)+"_seq_"+str(bs_idx)

                # Get reprojected our results 
                reprojected_pred_jnts2d_list, reprojected_pred_ori_jnts2d_list, _ = \
                                self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                smal_opt_jnts3d[bs_idx].to(cam_rot_mat.device)*scale_val_list[bs_idx]) # K X T X (18*2)
                pred_for_vis = reprojected_pred_jnts2d_list.reshape(1, val_num_views, -1, 17*2)  

                vis_for_paper = False 
                if vis_for_paper:
                    dest_for_paper_vis_folder = "./for_cvpr25_paper_vis_2d_sequences_cat_sample28"
                    if not os.path.exists(dest_for_paper_vis_folder):
                        os.makedirs(dest_for_paper_vis_folder) 

                    # pose_sequence = epi_line_cond_seq_2d 
                    # line_coeffs =  
                    # visualize_pose_and_lines_for_paper(pose_sequence, line_coeffs, output_folder, vis_gt=False, epipoles=None)
                    
                    # Plot for single pose 
                    num_steps = reprojected_pred_jnts2d_list.shape[1] 
                    for t_idx in range(0, num_steps, 1):
                        dest_pose2d_fig_path = os.path.join(dest_for_paper_vis_folder, "s_idx_"+str(s_idx)+"_bs_idx_"+str(bs_idx)+"_pose2d_"+str(t_idx)+".png") 
                        plot_pose_2d_cat_for_paper(reprojected_pred_jnts2d_list[0, t_idx].reshape(-1, 2).detach().cpu().numpy(), dest_pose2d_fig_path) 

                    # import pdb 
                    # pdb.set_trace() 

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

                if self.opt.add_vposer_opt_w_2d:
                    # Get reprojected our results 
                    vposer2d_pred_smpl_jnts18 = vposer2d_smpl_jnts18_list[bs_idx]
                    vposer2d_reprojected_pred_jnts2d_list, vposer2d_reprojected_pred_ori_jnts2d_list, _ = \
                                    self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                    cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                    vposer2d_pred_smpl_jnts18.to(cam_rot_mat.device)*vposer_2d_scale_val_list[bs_idx]) # K X T X (18*2)
                    vposer2d_pred_for_vis = vposer2d_reprojected_pred_jnts2d_list.reshape(1, val_num_views, -1, 17*2)  

                    de_vposer2d_pred_2d_for_vis = self.gen_multiview_vis_res(vposer2d_pred_for_vis, vis_folder_tag, \
                        actual_seq_len[bs_idx:bs_idx+1], \
                        dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_smplify")

                if self.opt.add_mas_for_eval:
                    mas_pred_smal_jnts18 = mas_smal_opt_jnts3d[bs_idx]

                    aligned_mas_jnts18, aligned_mas_verts = self.align_motionbert_to_ours(mas_pred_smal_jnts18.detach().cpu().numpy(), \
                                            smal_opt_jnts3d[bs_idx], mas_smal_opt_verts[bs_idx].detach().cpu().numpy())
                    # mas_smpl_verts_list[bs_idx] = torch.from_numpy(aligned_mas_verts).to(mas_smpl_verts_list[bs_idx].device) 
                    aligned_mas_jnts18, scaled_aligned_mas_jnts18, \
                    aligned_mas_verts, scaled_aligned_mas_verts = \
                                self.compute_yaw_rotation_and_rotated_3d_pose(aligned_mas_jnts18, \
                                gt_for_vis[bs_idx, 0].reshape(gt_for_vis.shape[2], -1, 2), cam_rot_mat[bs_idx, 0, 0], cam_trans[bs_idx, 0, 0], \
                                aligned_mas_verts)

                    # Get reprojected our results 
                    mas_reprojected_pred_jnts2d_list, mas_reprojected_pred_ori_jnts2d_list, _ = \
                                    self.ema.ema_model.get_projection_from_motion3d(cam_rot_mat.shape[1], \
                                    cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                                    scaled_aligned_mas_jnts18.to(cam_rot_mat.device))
                                    # mas_pred_smal_jnts18.to(cam_rot_mat.device)*mas_scale_val_list[bs_idx]) # K X T X (18*2)
                    mas_pred_for_vis = mas_reprojected_pred_jnts2d_list.reshape(1, val_num_views, -1, 17*2)  

                    de_mas_pred_2d_for_vis = self.gen_multiview_vis_res(mas_pred_for_vis, vis_folder_tag, \
                        actual_seq_len[bs_idx:bs_idx+1], \
                        dest_vis_data_folder=dest_res3d_npy_folder, file_name_tag="_mas")

                tmp_idx = 0
                v_tmp_idx = 0 
                
                mpjpe, pa_mpjpe, trans_err, jpos3d_err, jpos2d_err, centered_jpos2d_err, bone_consistency = \
                compute_metrics_for_jpos(smal_opt_jnts3d[bs_idx], \
                    de_pred_2d_for_vis[tmp_idx, v_tmp_idx], \
                    smal_opt_jnts3d[bs_idx], \
                    de_gt_2d_for_vis[tmp_idx, v_tmp_idx], \
                    int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1, smal_opt_jnts3d[bs_idx], eval_animal_data=True)

                if self.add_vposer_opt_w_2d:
                    smplify_mpjpe, smplify_pa_mpjpe, smplify_trans_err, smplify_jpos3d_err, \
                    smplify_jpos2d_err, smplify_centered_jpos2d_err, smplify_bone_consistency = \
                        compute_metrics_for_jpos(vposer2d_smpl_jnts18_list[bs_idx], \
                        de_vposer2d_pred_2d_for_vis[tmp_idx, v_tmp_idx], \
                        smal_opt_jnts3d[bs_idx], \
                        de_gt_2d_for_vis[tmp_idx, v_tmp_idx], \
                        int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1, vposer2d_smpl_jnts18_list[bs_idx], eval_animal_data=True)
                else:
                    smplify_mpjpe, smplify_pa_mpjpe, smplify_trans_err, smplify_jpos3d_err, \
                    smplify_jpos2d_err, smplify_centered_jpos2d_err, smplify_bone_consistency = 0, 0, 0, 0, 0, 0, 0 

                if self.opt.add_mas_for_eval:
                    mas_mpjpe, mas_pa_mpjpe, mas_trans_err, mas_jpos3d_err, \
                    mas_jpos2d_err, mas_centered_jpos2d_err, mas_bone_consistency = \
                        compute_metrics_for_jpos(mas_pred_smal_jnts18, \
                        de_mas_pred_2d_for_vis[tmp_idx, v_tmp_idx], \
                        smal_opt_jnts3d[bs_idx], \
                        de_gt_2d_for_vis[tmp_idx, v_tmp_idx], \
                        int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1, mas_pred_smal_jnts18, eval_animal_data=True)
                else:
                    mas_mpjpe, mas_pa_mpjpe, mas_trans_err, mas_jpos3d_err, \
                    mas_jpos2d_err, mas_centered_jpos2d_err, mas_bone_consistency = 0, 0, 0, 0, 0, 0, 0 

                self.add_new_evaluation_to_list(mpjpe, pa_mpjpe, trans_err, \
                    jpos3d_err, jpos2d_err, centered_jpos2d_err, bone_consistency, \
                    mpjpe_smplify=smplify_mpjpe, pa_mpjpe_smplify=smplify_pa_mpjpe, trans_err_smplify=smplify_trans_err, jpos_err_smplify=smplify_jpos3d_err, \
                    jpos2d_err_smplify=smplify_jpos2d_err, centered_jpos2d_err_smplify=smplify_centered_jpos2d_err, \
                    bone_consistency_smplify=smplify_bone_consistency, \
                    mpjpe_mas=mas_mpjpe, pa_mpjpe_mas=mas_pa_mpjpe, trans_err_mas=mas_trans_err, \
                    jpos_err_mas=mas_jpos3d_err, jpos2d_err_mas=mas_jpos2d_err, 
                    centered_jpos2d_err_mas=mas_centered_jpos2d_err, bone_consistency_mas=mas_bone_consistency) 
                    
                self.print_mean_metrics() 

                curr_seq_len = int(actual_seq_len[bs_idx].detach().cpu().numpy()[0])-1 
                # dest_jnts3d_npy_path = os.path.join(dest_res3d_npy_folder, \
                #         str(s_idx)+"_sample_"+str(bs_idx)+".npy")
                # np.save(dest_jnts3d_npy_path, pred_smpl_jnts18.detach().cpu().numpy()[:curr_seq_len])
                if self.eval_w_best_mpjpe:
                    dest_jnts3d_json_path = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_eval.json")
                    eval_dict = {}
                    eval_dict['mpjpe'] = str(mpjpe)
                    eval_dict['pa_mpjpe'] = str(pa_mpjpe)            
                    eval_dict['trans_err'] = str(trans_err)
                    eval_dict['jpos3d_err'] = str(jpos3d_err)
                    eval_dict['jpos2d_err'] = str(jpos2d_err)
                    eval_dict['centered_jpos2d_err'] = str(centered_jpos2d_err) 
                
                    # # Elepose
                    # eval_dict['elepose_mpjpe'] = str(elepose_mpjpe)
                    # eval_dict['elepose_pa_mpjpe'] = str(elepose_pa_mpjpe)
                    # eval_dict['elepose_trans_err'] = str(elepose_trans_err)
                    # eval_dict['elepose_jpos2d_err'] = str(elepose_jpos2d_err)
                    # eval_dict['elepose_jpos3d_err'] = str(elepose_jpos3d_err) 
                    # eval_dict['elepose_centered_jpos2d_err'] = str(elepose_centered_jpos2d_err)

                    # # MAS
                    eval_dict['mas_mpjpe'] = str(mas_mpjpe)
                    eval_dict['mas_pa_mpjpe'] = str(mas_pa_mpjpe)
                    eval_dict['mas_trans_err'] = str(mas_trans_err)
                    eval_dict['mas_jpos2d_err'] = str(mas_jpos2d_err)
                    eval_dict['mas_jpos3d_err'] = str(mas_jpos3d_err) 
                    eval_dict['mas_centered_jpos2d_err'] = str(mas_centered_jpos2d_err)

                    # SMPLify
                    eval_dict['smplify_mpjpe'] = str(smplify_mpjpe)
                    eval_dict['smplify_pa_mpjpe'] = str(smplify_pa_mpjpe)
                    eval_dict['smplify_trans_err'] = str(smplify_trans_err)
                    eval_dict['smplify_jpos2d_err'] = str(smplify_jpos2d_err)
                    eval_dict['smplify_jpos3d_err'] = str(smplify_jpos3d_err) 
                    eval_dict['smplify_centered_jpos2d_err'] = str(smplify_centered_jpos2d_err)

                    json.dump(eval_dict, open(dest_jnts3d_json_path, 'w')) 

                    # Save multi-view 2D sequences 
                    dest_ours_2d_folder = os.path.join(dest_res3d_npy_folder, "ours_2d")
                    dest_smplify_2d_folder = os.path.join(dest_res3d_npy_folder, "smplify_2d") 
                    dest_mas_2d_folder = os.path.join(dest_res3d_npy_folder, "mas_2d") 
                    if not os.path.exists(dest_ours_2d_folder):
                        os.makedirs(dest_ours_2d_folder)
                    if not os.path.exists(dest_smplify_2d_folder):
                        os.makedirs(dest_smplify_2d_folder)
                    if not os.path.exists(dest_mas_2d_folder):
                        os.makedirs(dest_mas_2d_folder)

                    dest_ours_2d_res_path = os.path.join(dest_ours_2d_folder, str(s_idx)+"_sample_"+str(bs_idx)+"_eval.npy")
                    np.save(dest_ours_2d_res_path, pred_for_vis.squeeze(0)[1:].detach().cpu().numpy()) # (K-1) X T X 18 X 2 

                    if self.add_vposer_opt_w_2d:
                        dest_vposer2d_2d_res_path = os.path.join(dest_smplify_2d_folder, str(s_idx)+"_sample"+str(bs_idx)+"_eval.npy")
                        np.save(dest_vposer2d_2d_res_path, vposer2d_pred_for_vis.squeeze(0)[1:].detach().cpu().numpy()) # (K-1) X T X 18 X 2

                    if self.opt.add_mas_for_eval:
                        dest_mas_2d_res_path = os.path.join(dest_mas_2d_folder, str(s_idx)+"_sample"+str(bs_idx)+"_eval.npy")
                        np.save(dest_mas_2d_res_path, mas_pred_for_vis.squeeze(0)[1:].detach().cpu().numpy())

                if not self.eval_w_best_mpjpe:
                    # Visualize 3D pose sequence 
                    dest_vid_3d_path_global = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_ours_jpos3d")
                    plot_3d_motion(dest_vid_3d_path_global, \
                        opt_jpos3d_18_list[bs_idx].detach().cpu().numpy()[:curr_seq_len], \
                        use_animal_pose17=True) 

                    # Visualize smal fitted 3d joint positions 
                    dest_vid_3d_path_opt_smal = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_ours_smal")
                    plot_3d_motion(dest_vid_3d_path_opt_smal, \
                        smal_opt_jnts3d[bs_idx].detach().cpu().numpy()[:curr_seq_len], \
                        use_animal_pose17=True) 

                    if self.opt.add_vposer_opt_w_2d:
                        # Visualize smal fitted 3d joint positions 
                        dest_vid_3d_path_opt_smal = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_smplify")
                        plot_3d_motion(dest_vid_3d_path_opt_smal, \
                            vposer2d_pred_smpl_jnts18.detach().cpu().numpy()[:curr_seq_len], \
                            use_animal_pose17=True) 

                    if self.opt.add_mas_for_eval:
                        # Visualize smal fitted 3d joint positions 
                        dest_vid_3d_path_opt_smal = os.path.join(dest_res3d_npy_folder, \
                                str(s_idx)+"_sample_"+str(bs_idx)+"_mas")
                        plot_3d_motion(dest_vid_3d_path_opt_smal, \
                            aligned_mas_jnts18.detach().cpu().numpy()[:curr_seq_len], \
                            use_animal_pose17=True) 

                    # Save smal fitted mesh to .obj 
                    dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_objs")
                    if not os.path.exists(dest_mesh_folder):
                        os.makedirs(dest_mesh_folder)
                    for_vis_ours_smal_mesh_seq = self.process_verts_for_vis_cmp_animal(smal_opt_verts[bs_idx], smal_opt_jnts3d[bs_idx])
                    for t_idx in range(curr_seq_len):
                        skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_ours.obj")

                        curr_mesh = trimesh.Trimesh(vertices=for_vis_ours_smal_mesh_seq[t_idx].cpu().detach().cpu().numpy(),
                            faces=skel_faces.detach().cpu().numpy()) 

                        curr_mesh.export(skin_mesh_out)

                    if self.add_vposer_opt_w_2d:
                        dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_objs_smplify")
                        if not os.path.exists(dest_mesh_folder):
                            os.makedirs(dest_mesh_folder)
                        for_vis_smplify_smal_mesh_seq = self.process_verts_for_vis_cmp_animal(vposer2d_smpl_verts_list[bs_idx], \
                            vposer2d_pred_smpl_jnts18)
                        for t_idx in range(curr_seq_len):
                            skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_smplify.obj")

                            curr_mesh = trimesh.Trimesh(vertices=for_vis_smplify_smal_mesh_seq[t_idx].detach().cpu().numpy(),
                                faces=skel_faces.detach().cpu().numpy()) 

                            curr_mesh.export(skin_mesh_out)

                    if self.opt.add_mas_for_eval:
                        dest_mesh_folder = os.path.join(dest_res3d_npy_folder, \
                            str(s_idx)+"_sample_"+str(bs_idx)+"_objs_mas")
                        if not os.path.exists(dest_mesh_folder):
                            os.makedirs(dest_mesh_folder)
                        for_vis_mas_smal_mesh_seq = self.process_verts_for_vis_cmp_animal(aligned_mas_verts, aligned_mas_jnts18)
                        for t_idx in range(curr_seq_len):
                            skin_mesh_out = os.path.join(dest_mesh_folder, "%05d"%(t_idx)+"_mas.obj")

                            curr_mesh = trimesh.Trimesh(vertices=for_vis_mas_smal_mesh_seq[t_idx].detach().cpu().numpy(),
                                faces=skel_faces.detach().cpu().numpy()) 

                            curr_mesh.export(skin_mesh_out)
        
    def cond_sample_res(self):
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        
        self.load(milestone)
        self.ema.ema_model.eval()
      
        if self.test_on_train:
            test_loader = torch.utils.data.DataLoader(
                self.ds, batch_size=8, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=False) 
        else:
            test_loader = torch.utils.data.DataLoader(
                self.val_ds, batch_size=8, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=False) 
        
        with torch.no_grad():
            for s_idx, val_data_dict in enumerate(test_loader):
                val_data = val_data_dict['normalized_jnts2d'].float().cuda()

                val_bs, val_num_steps, num_joints, _ = val_data.shape 
                val_data = val_data.reshape(val_bs, val_num_steps, -1) # BS X T X D(18*2)

                cond_mask = None 

                if self.opt.input_first_human_pose:
                    cond_mask = self.prep_temporal_condition_mask(val_data)

                # Generate padding mask 
                actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
                self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(val_data.device)

                if self.add_language_condition:
                    text_anno_data = val_data_dict['text']
                    language_input = self.encode_text(text_anno_data) # BS X 512 
                    language_input = language_input.to(data.device)
                else:
                    language_input = None 

                dest_vis_folder_path = "/viscam/projects/vimotion/hope_final_work_opt3d_res/AIST_only_cond_init_pose_2d_res" 

                num_samples_per_seq = 10
                for sample_idx in range(num_samples_per_seq):
                    all_res_list = self.ema.ema_model.sample(val_data, \
                    cond_mask=cond_mask, padding_mask=padding_mask, \
                    language_input=language_input) # BS X T X D 

                    # all_res_list = self.ema.ema_model.ddim_sample(val_data, \
                    # cond_mask=cond_mask, padding_mask=padding_mask, \
                    # language_input=language_input) # BS X T X D 
                
                    vis_folder_tag = str(milestone)+"_bidx_"+str(s_idx)+"_sample_cnt_"+str(sample_idx)
                 
                    if self.test_on_train:
                        vis_folder_tag = vis_folder_tag + "_on_train"

                    # vis_folder_tag = "ddim_sample_" + vis_folder_tag 
                    
                    self.gen_vis_res(all_res_list, vis_folder_tag, actual_seq_len, \
                            dest_vis_folder_path=dest_vis_folder_path) 

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
                        use_smpl_jnts13=False) 

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
            
            curr_seq_actual_len = int(actual_seq_len[seq_idx] - 1)

            for view_idx in range(num_views):
                if vis_gt:
                    dest_vid_path = os.path.join(dest_vis_folder_path, \
                    "seq_"+str(seq_idx)+"_view"+str(view_idx)+"_gt.mp4")
                else:
                    dest_vid_path = os.path.join(dest_vis_folder_path, \
                    "seq_"+str(seq_idx)+"_view"+str(view_idx)+file_name_tag+".mp4")

                
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
    
    use_bb_data = False 
    
    diffusion_model = MultiViewCondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, input_line_cond=opt.train_2d_diffusion_w_line_cond, \
                num_views=opt.train_num_views, use_animal_data=opt.use_animal_data, use_bb_data=use_bb_data, \
                use_omomo_data=opt.use_omomo_data, omomo_object_name=opt.omomo_object_name)  
   
    diffusion_model.to(device)

    train_num_steps = 500000 

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=train_num_steps,         # 700000, total training steps
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

    use_bb_data = False 
    
    diffusion_model = MultiViewCondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, input_line_cond=opt.train_2d_diffusion_w_line_cond, \
                num_views=opt.train_num_views, use_animal_data=opt.use_animal_data, use_bb_data=use_bb_data, \
                use_omomo_data=opt.use_omomo_data, omomo_object_name=opt.omomo_object_name)

    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=2000000,         # total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=False,                        # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False, 
    )
   
    if opt.use_animal_data:
        trainer.eval_3d_w_multiview_diffusion_for_animal() 
    elif opt.use_omomo_data:
        trainer.eval_3d_w_multiview_diffusion_for_interaction()
    else:
        trainer.eval_3d_w_multiview_diffusion()

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

    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument('--cfg_scale', type=float, default=10, help='the scale for classifier-free guidance')

    parser.add_argument("--eval_w_best_mpjpe", action="store_true")

    parser.add_argument("--add_motionbert_for_eval", action="store_true")

    parser.add_argument("--add_elepose_for_eval", action="store_true")

    parser.add_argument("--add_mas_for_eval", action="store_true")

    parser.add_argument("--eval_on_aist_3d", action="store_true")

    parser.add_argument("--train_2d_diffusion_w_line_cond", action="store_true")
    parser.add_argument("--test_2d_diffusion_w_line_cond", action="store_true")

    parser.add_argument('--train_num_views', type=int, default=6, help='the number of decoder layers')

    parser.add_argument('--train_on_amass', action="store_true")

    parser.add_argument('--train_on_both_amass_and_syn3d', action="store_true")

    parser.add_argument('--add_vposer_opt_w_2d', action="store_true")
    
    parser.add_argument("--use_animal_data", action="store_true")

    parser.add_argument("--use_omomo_data", action="store_true")

    parser.add_argument('--omomo_object_name', type=str, default='largebox', help='project name')

    parser.add_argument("--gen_vis_res_for_demo", action="store_true")

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
