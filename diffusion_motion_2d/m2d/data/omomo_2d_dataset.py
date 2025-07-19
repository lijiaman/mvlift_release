import sys
sys.path.append("../../")

import os
import numpy as np
import joblib 
import trimesh  
import json 

import random 

import torch
from torch.utils.data import Dataset

import pytorch3d.transforms as transforms 

from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform

from human_body_prior.body_model.body_model import BodyModel

from m2d.data.utils_multi_view_2d_motion import single_seq_get_multi_view_2d_motion_from_3d_torch

from m2d.data.utils_multi_view_2d_motion import cal_min_seq_len_from_mask_torch, get_multi_view_cam_extrinsic  

from m2d.data.utils_2d_pose_normalize import move_init_pose2d_to_center, move_init_pose2d_to_center_for_animal_data, normalize_pose2d, de_normalize_pose2d 

from m2d.lafan1.utils import rotate_at_frame_w_obj 

from smplx import SMPL 

SMPLH_PATH = "/viscam/u/jiamanli/github/hm_interaction/smpl_all_models/smplh_amass"


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

def convert_smpl_jnts_to_openpose18(smpl_jnts, smpl_verts):
    # smpl_jnts: T X 45 X 3 

    # SMPLX 
    #  'nose':		    9120,
    #     'reye':		    9929,
    #     'leye':		    9448,
    #     'rear':		    616,
    #     'lear':		    6,

    selected_vertex_idx = [9120, 9929, 9448, 616, 6] 
    selected_verts = smpl_verts[:, selected_vertex_idx, :] # T X 5 X 3 

    smpl2openpose_idx = [22, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, \
                23, 24, 25, 26] 

    smplx_jnts27 = np.concatenate((smpl_jnts[:, :22, :], selected_verts), \
                        axis=1) # T X (22+5) X 3 
    smplx_for_openpose_jnts19 = smplx_jnts27[:, smpl2openpose_idx, :] # T X 19 X 3  

    # Remove pelvis joint to be aligned with DWPose results T X 18 X 3 
    smplx_jnts18 = np.concatenate((smplx_for_openpose_jnts19[:, :8], \
        smplx_for_openpose_jnts19[:, 9:]), axis=1) # T X 18 X 3 

    return smplx_jnts18 

    # # This is the correct version that align with the 2D pose model. 
    # smpl2jnts18_idx = [24, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28]

    # # new version, but using this version leads to worse results, why? 
    # # smpl2jnts18_idx = [24, 12, 16, 18, 20, 17, 19, 21, 1, 4, 7, 2, 5, 8, 26, 25, 28, 27]
    # smpl_jnts18 = smpl_jnts[:, smpl2jnts18_idx, :] # T X 18 X 3 

    #$ return smpl_jnts18 

def get_smpl_parents(use_joints24=False):
    bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 

    if use_joints24:
        parents = ori_kintree_table[0, :23] # 23 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.

        parents_list = parents.tolist()
        parents_list.append(ori_kintree_table[0][37])
        parents = np.asarray(parents_list) # 24 
    else:
        parents = ori_kintree_table[0, :22] # 22 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.
    
    return parents

def local2global_pose(local_pose):
    # local_pose: T X J X 3 X 3 
    kintree = get_smpl_parents(use_joints24=False) 

    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose # T X J X 3 X 3 

def quat_ik_torch(grot_mat):
    # grot: T X J X 3 X 3 
    parents = get_smpl_parents(use_joints24=False) 

    grot = transforms.matrix_to_quaternion(grot_mat) # T X J X 4 

    res = torch.cat(
            [
                grot[..., :1, :],
                transforms.quaternion_multiply(transforms.quaternion_invert(grot[..., parents[1:], :]), \
                grot[..., 1:, :]),
            ],
            dim=-2) # T X J X 4 

    res_mat = transforms.quaternion_to_matrix(res) # T X J X 3 X 3 

    return res_mat 

def quat_fk_torch(lrot_mat, lpos, use_joints24=False):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J/(J+2) X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    if use_joints24:
        parents = get_smpl_parents(use_joints24=True)
    else:
        parents = get_smpl_parents() 

    lrot = transforms.matrix_to_quaternion(lrot_mat)

    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
        )
        if i < lrot.shape[-2]:
            gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

    return res

def run_smplx_model(root_trans, aa_rot_rep, betas, gender, bm_dict):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3 
    # betas: BS X 16
    # gender: BS 
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(aa_rot_rep.device) # BS X T X 30 X 3 
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2) # BS X T X 52 X 3 

    aa_rot_rep = aa_rot_rep.reshape(bs*num_steps, -1, 3) # (BS*T) X n_joints X 3 
    betas = betas[:, None, :].repeat(1, num_steps, 1).reshape(bs*num_steps, -1) # (BS*T) X 16 
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist() # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3) # (BS*T) X 3  
    smpl_betas = betas # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :] # (BS*T) X 3 
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63) # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90) # (BS*T) X 90 

    B = smpl_trans.shape[0] # (BS*T) 

    smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body, smpl_pose_hand]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ['male', 'female', "neutral"]
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=np.int64)*-1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=np.int64)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue
        
        # reconstruct SMPL
        cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose, cur_pred_pose_hand = gender_smpl_vals
        bm = bm_dict[gender_name]

        pred_body = bm(pose_body=cur_pred_pose, pose_hand=cur_pred_pose_hand, \
                betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)
        
        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)

    # cat all genders and reorder to original batch ordering
    # if return_joints24:
    #     x_pred_smpl_joints_all = torch.cat(pred_joints, axis=0) # () X 52 X 3 
    #     lmiddle_index= 28 
    #     rmiddle_index = 43 
    #     x_pred_smpl_joints = torch.cat((x_pred_smpl_joints_all[:, :22, :], \
    #         x_pred_smpl_joints_all[:, lmiddle_index:lmiddle_index+1, :], \
    #         x_pred_smpl_joints_all[:, rmiddle_index:rmiddle_index+1, :]), dim=1) 
    # else:
    #     x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]

    x_pred_smpl_joints = torch.cat(pred_joints, axis=0)

    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map] # (BS*T) X 22 X 3 

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map] # (BS*T) X 6890 X 3 

    
    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3) # BS X T X 22 X 3/BS X T X 24 X 3  
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3) # BS X T X 6890 X 3 

    mesh_faces = pred_body.f 

    #  # This is the correct version that align with the 2D pose model. 
    # smpl2jnts18_idx = [24, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28]

    # # new version, but using this version leads to worse results, why? 
    # # smpl2jnts18_idx = [24, 12, 16, 18, 20, 17, 19, 21, 1, 4, 7, 2, 5, 8, 26, 25, 28, 27]
    # smpl_jnts18 = smpl_jnts[:, smpl2jnts18_idx, :] # T X 18 X 3 
    
    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces 

class OMOMODataset(Dataset):
    def __init__(
        self,
        train,
        window=120,
        object_name="largebox", 
        for_eval=False, 
    ):
        self.data_root_folder = "/move/u/jiamanli/datasets/semantic_manip/processed_data"

        self.for_eval = for_eval 

        # Prepare SMPLX model 
        soma_work_base_dir = os.path.join(self.data_root_folder, 'smpl_all_models')
        support_base_dir = soma_work_base_dir 
        surface_model_type = "smplx"
        surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_MALE.npz")
        surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_FEMALE.npz")
        dmpl_fname = None
        num_dmpls = None 
        num_expressions = None
        num_betas = 16 

        self.image_width = 1920
        self.image_height = 1080 

        self.male_bm = BodyModel(bm_fname=surface_model_male_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname)
        self.female_bm = BodyModel(bm_fname=surface_model_female_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname)

        for p in self.male_bm.parameters():
            p.requires_grad = False
        for p in self.female_bm.parameters():
            p.requires_grad = False 

        self.male_bm = self.male_bm
        self.female_bm = self.female_bm
        
        self.bm_dict = {'male' : self.male_bm, 'female' : self.female_bm}

        self.train = train
        
        self.window = window 

        self.object_name = object_name 

        self.object2kpts_idx_dict = {
            "largebox":  np.asarray([11298, 12617, 12084, 7527, 3042]), 
            "woodchair":  np.asarray([878, 4304, 4925, 11511, 6719]), 
            "monitor": np.asarray([2393, 35456, 6686, 5386, 26582]), 
            "largetable": np.asarray([10, 418, 5728, 3846, 5695]), 
            "clothesstand": np.asarray([12379, 13347, 10372, 4007, 22]),
        }

        # self.train_objects = ["largetable", "woodchair", "plasticbox", "largebox", "smallbox", "trashcan", "monitor", \
        #             "floorlamp", "clothesstand"] 
        # self.test_objects = ["smalltable", "whitechair", "suitcase", "tripod"]

        self.parents = get_smpl_parents() # 24/22 

        self.obj_geo_root_folder = os.path.join(self.data_root_folder, "captured_objects")
        self.rest_object_geo_folder = os.path.join(self.data_root_folder, "rest_object_geo")

        cam_extrinsic_list = get_multi_view_cam_extrinsic(num_views=30, farest=True)  # K X 4 X 4  
        ref_cam_extrinsic = cam_extrinsic_list[0:1] # 1 X 4 X 4 
        target_cam_extrinsic = cam_extrinsic_list[1:] # (K-1) X 4 X 4 
        ref_cam_extrinsic = ref_cam_extrinsic.repeat(target_cam_extrinsic.shape[0], 1, 1) # (K-1) X 4 X 4 
        self.epipoles = compute_epipoles(ref_cam_extrinsic, target_cam_extrinsic).cpu() # (K-1) X 2  
        # It seems all the y are 0. 

        train_subjects = []
        test_subjects = []
        num_subjects = 17 
        for s_idx in range(1, num_subjects+1):
            if s_idx >= 16:
                test_subjects.append("sub"+str(s_idx))
            else:
                train_subjects.append("sub"+str(s_idx))

        if self.train:
            seq_data_path = os.path.join(self.data_root_folder, "train_diffusion_manip_seq_joints24.p")  
            processed_data_path = os.path.join(self.data_root_folder, \
                "cvpr25_train_window_"+str(self.window)+"_"+self.object_name+".p")   
        else:    
            seq_data_path = os.path.join(self.data_root_folder, "test_diffusion_manip_seq_joints24.p")
            processed_data_path = os.path.join(self.data_root_folder, \
                "cvpr25_test_window_"+str(self.window)+"_"+self.object_name+".p")   

        if os.path.exists(processed_data_path):
            self.window_data_dict = joblib.load(processed_data_path)
        else:
            self.data_dict = joblib.load(seq_data_path)

            self.extract_rest_pose_object_geometry_and_rotation()

            self.cal_normalize_data_input()
            joblib.dump(self.window_data_dict, processed_data_path)            

        
        self.window_data_dict = self.filter_out_short_sequences() 

        # for k in self.window_data_dict:
        #     import pdb 
        #     pdb.set_trace() 
        #     if self.window_data_dict[k]["combined_jnts2d_18"].shape[1] != 23:
        #         import pdb 
        #         pdb.set_trace()
                 
        # Get train and validation statistics. 
        if self.train:
            print("Total number of windows for training:{0}".format(len(self.window_data_dict))) # all, Total number of windows for training:28859
        else:
            print("Total number of windows for validation:{0}".format(len(self.window_data_dict))) # all, 3224 

        smpl_dir = "/move/u/jiamanli/datasets/semantic_manip/processed_data/smpl_all_models/smpl"
        self.smpl = SMPL(model_path=smpl_dir, gender='MALE', batch_size=1)

    def filter_out_short_sequences(self):
        new_cnt = 0
        new_window_data_dict = {}
        for k in self.window_data_dict:
            window_data = self.window_data_dict[k]
            seq_name = window_data['seq_name']
           
            curr_seq_len = window_data['combined_jnts2d_18'].shape[0]
            n_pts = window_data['combined_jnts2d_18'].shape[1] 

            if window_data['seq_len'] < self.window:
                continue


            if curr_seq_len < self.window:
                continue 

            if n_pts != 23:
                continue 

            # if self.window_data_dict[k]['start_t_idx'] != 0:
            #     continue 

            new_window_data_dict[new_cnt] = self.window_data_dict[k]
            if "ori_w_idx" in self.window_data_dict[k]:
                new_window_data_dict[new_cnt]['ori_w_idx'] = self.window_data_dict[k]['ori_w_idx']
            else:
                new_window_data_dict[new_cnt]['ori_w_idx'] = k 
            
            new_cnt += 1

        return new_window_data_dict

    def apply_transformation_to_obj_geometry(self, obj_mesh_path, obj_scale, obj_rot, obj_trans):
        mesh = trimesh.load_mesh(obj_mesh_path)
        obj_mesh_verts = np.asarray(mesh.vertices) # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3 

        ori_obj_verts = torch.from_numpy(obj_mesh_verts).float()[None].repeat(obj_trans.shape[0], 1, 1) # T X Nv X 3 
    
        if torch.is_tensor(obj_scale):
            seq_scale = obj_scale.float() 
        else:
            seq_scale = torch.from_numpy(obj_scale).float() # T 
        
        if torch.is_tensor(obj_rot):
            seq_rot_mat = obj_rot.float()
        else:
            seq_rot_mat = torch.from_numpy(obj_rot).float() # T X 3 X 3 
        
        if obj_trans.shape[-1] != 1:
            if torch.is_tensor(obj_trans):
                seq_trans = obj_trans.float()[:, :, None]
            else:
                seq_trans = torch.from_numpy(obj_trans).float()[:, :, None] # T X 3 X 1 
        else:
            if torch.is_tensor(obj_trans):
                seq_trans = obj_trans.float()
            else:
                seq_trans = torch.from_numpy(obj_trans).float() # T X 3 X 1 

        transformed_obj_verts = seq_scale.unsqueeze(-1).unsqueeze(-1) * \
        seq_rot_mat.bmm(ori_obj_verts.transpose(1, 2).to(seq_trans.device)) + seq_trans
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 

        return transformed_obj_verts, obj_mesh_faces  

    def load_rest_pose_object_geometry(self, object_name):
        rest_obj_path = os.path.join(self.rest_object_geo_folder, object_name+".ply")
        
        mesh = trimesh.load_mesh(rest_obj_path)
        rest_verts = np.asarray(mesh.vertices) # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3

        return rest_verts, obj_mesh_faces 

    def convert_rest_pose_obj_geometry(self, object_name, obj_scale, obj_trans, obj_rot):
        # obj_scale: T, obj_trans: T X 3, obj_rot: T X 3 X 3
        # obj_mesh_verts: T X Nv X 3
        rest_obj_path = os.path.join(self.rest_object_geo_folder, object_name+".ply")
        rest_obj_json_path = os.path.join(self.rest_object_geo_folder, object_name+".json")

        if os.path.exists(rest_obj_path):
            mesh = trimesh.load_mesh(rest_obj_path)
            rest_verts = np.asarray(mesh.vertices) # Nv X 3
            obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3

            rest_verts = torch.from_numpy(rest_verts) 

            json_data = json.load(open(rest_obj_json_path, 'r'))
            rest_pose_ori_obj_rot = np.asarray(json_data['rest_pose_ori_obj_rot']) # 3 X 3 
            rest_pose_ori_obj_com_pos = np.asarray(json_data['rest_pose_ori_com_pos']) # 1 X 3 
            obj_trans_to_com_pos = np.asarray(json_data['obj_trans_to_com_pos']) # 1 X 3 

        return rest_verts, obj_mesh_faces, rest_pose_ori_obj_rot, rest_pose_ori_obj_com_pos, obj_trans_to_com_pos  

    def load_object_geometry_w_rest_geo(self, obj_rot, obj_com_pos, rest_verts):
        # obj_scale: T, obj_rot: T X 3 X 3, obj_com_pos: T X 3, rest_verts: Nv X 3 
        rest_verts = rest_verts[None].repeat(obj_rot.shape[0], 1, 1)
        transformed_obj_verts = obj_rot.bmm(rest_verts.transpose(1, 2)) + obj_com_pos[:, :, None]
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 

        return transformed_obj_verts 
    
    def load_object_geometry(self, object_name, obj_scale, obj_trans, obj_rot, \
        obj_bottom_scale=None, obj_bottom_trans=None, obj_bottom_rot=None):
        obj_mesh_path = os.path.join(self.obj_geo_root_folder, \
                    object_name+"_cleaned_simplified.obj")
       
        obj_mesh_verts, obj_mesh_faces =self.apply_transformation_to_obj_geometry(obj_mesh_path, \
        obj_scale, obj_rot, obj_trans) # T X Nv X 3 

        return obj_mesh_verts, obj_mesh_faces 

    def extract_rest_pose_object_geometry_and_rotation(self):
        self.rest_pose_object_dict = {} 

        for seq_idx in self.data_dict:
            seq_name = self.data_dict[seq_idx]['seq_name']
            object_name = seq_name.split("_")[1]
            if object_name in ["vacuum", "mop"]:
                continue 

            if object_name not in self.rest_pose_object_dict:
                obj_trans = self.data_dict[seq_idx]['obj_trans'][:, :, 0] # T X 3
                obj_rot = self.data_dict[seq_idx]['obj_rot'] # T X 3 X 3 
                obj_scale = self.data_dict[seq_idx]['obj_scale'] # T  

                rest_verts, obj_mesh_faces, rest_pose_ori_rot, rest_pose_ori_com_pos, obj_trans_to_com_pos = \
                self.convert_rest_pose_obj_geometry(object_name, obj_scale, obj_trans, obj_rot)

                self.rest_pose_object_dict[object_name] = {}
                self.rest_pose_object_dict[object_name]['ori_rotation'] = rest_pose_ori_rot # 3 X 3 
                self.rest_pose_object_dict[object_name]['ori_trans'] = rest_pose_ori_com_pos # 1 X 3 
                self.rest_pose_object_dict[object_name]['obj_trans_to_com_pos'] = obj_trans_to_com_pos # 1 X 3 

    def cal_normalize_data_input(self):
        self.window_data_dict = {}
        s_idx = 0 
        for index in self.data_dict:
            seq_name = self.data_dict[index]['seq_name']

            object_name = seq_name.split("_")[1]

            if object_name not in [self.object_name]:
                continue 

            rest_pose_obj_data = self.rest_pose_object_dict[object_name]
            rest_pose_rot_mat = rest_pose_obj_data['ori_rotation'] # 3 X 3

            rest_obj_path = os.path.join(self.rest_object_geo_folder, object_name+".ply")
            mesh = trimesh.load_mesh(rest_obj_path)
            rest_verts = np.asarray(mesh.vertices) # Nv X 3
            rest_verts = torch.from_numpy(rest_verts).float() # Nv X 3

            betas = self.data_dict[index]['betas'] # 1 X 16 
            gender = self.data_dict[index]['gender']

            seq_root_trans = self.data_dict[index]['trans'] # T X 3 
            seq_root_orient = self.data_dict[index]['root_orient'] # T X 3 
            seq_pose_body = self.data_dict[index]['pose_body'].reshape(-1, 21, 3) # T X 21 X 3

            rest_human_offsets = self.data_dict[index]['rest_offsets'] # 22 X 3/24 X 3
            trans2joint = self.data_dict[index]['trans2joint'] # 3 

            # Used in old version without defining rest object geometry. 
            seq_obj_trans = self.data_dict[index]['obj_trans'][:, :, 0] # T X 3
            seq_obj_rot = self.data_dict[index]['obj_rot'] # T X 3 X 3 
            seq_obj_scale = self.data_dict[index]['obj_scale'] # T  

            seq_obj_verts, tmp_obj_faces = self.load_object_geometry(object_name, seq_obj_scale, \
                        seq_obj_trans, seq_obj_rot) # T X Nv X 3, tensor
            seq_obj_com_pos = seq_obj_verts.mean(dim=1) # T X 3 

            obj_trans = seq_obj_com_pos.clone().detach().cpu().numpy() 

            rest_pose_rot_mat_rep = torch.from_numpy(rest_pose_rot_mat).float()[None, :, :] # 1 X 3 X 3 
            obj_rot = torch.from_numpy(self.data_dict[index]['obj_rot']) # T X 3 X 3 
            obj_rot = torch.matmul(obj_rot, rest_pose_rot_mat_rep.repeat(obj_rot.shape[0], 1, 1).transpose(1, 2)) # T X 3 X 3  
            obj_rot = obj_rot.detach().cpu().numpy() 

            num_steps = seq_root_trans.shape[0]
            # for start_t_idx in range(0, num_steps, self.window//2):
            for start_t_idx in range(0, num_steps, self.window//4):
                end_t_idx = start_t_idx + self.window - 1
                
                # Skip the segment that has a length < 30 
                if end_t_idx - start_t_idx < 60:
                    continue 

                self.window_data_dict[s_idx] = {}
                
                joint_aa_rep = torch.cat((torch.from_numpy(seq_root_orient[start_t_idx:end_t_idx+1]).float()[:, None, :], \
                    torch.from_numpy(seq_pose_body[start_t_idx:end_t_idx+1]).float()), dim=1) # T X J X 3 
                X = torch.from_numpy(rest_human_offsets).float()[:22][None].repeat(joint_aa_rep.shape[0], 1, 1).detach().cpu().numpy() # T X J X 3 
                X[:, 0, :] = seq_root_trans[start_t_idx:end_t_idx+1] 
                local_rot_mat = transforms.axis_angle_to_matrix(joint_aa_rep) # T X J X 3 X 3 
                Q = transforms.matrix_to_quaternion(local_rot_mat).detach().cpu().numpy() # T X J X 4 

                obj_x = obj_trans[start_t_idx:end_t_idx+1].copy() # T X 3 
                obj_rot_mat = torch.from_numpy(obj_rot[start_t_idx:end_t_idx+1]).float()# T X 3 X 3 
                obj_q = transforms.matrix_to_quaternion(obj_rot_mat).detach().cpu().numpy() # T X 4 

                # Canonicalize based on the first human pose's orientation. 
                X, Q, new_obj_x, new_obj_q = rotate_at_frame_w_obj(X[np.newaxis], Q[np.newaxis], \
                obj_x[np.newaxis], obj_q[np.newaxis], \
                trans2joint[np.newaxis], self.parents, n_past=1, floor_z=True)
                # 1 X T X J X 3, 1 X T X J X 4, 1 X T X 3, 1 X T X 4 

                new_seq_root_trans = X[0, :, 0, :] # T X 3 
                new_local_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(Q[0]).float()) # T X J X 3 X 3 
                new_local_aa_rep = transforms.matrix_to_axis_angle(new_local_rot_mat) # T X J X 3 
                new_seq_root_orient = new_local_aa_rep[:, 0, :] # T X 3
                new_seq_pose_body = new_local_aa_rep[:, 1:, :] # T X 21 X 3 
                
                new_obj_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(new_obj_q[0]).float()) # T X 3 X 3
                
                cano_obj_mat = torch.matmul(new_obj_rot_mat[0], obj_rot_mat[0].transpose(0, 1)) # 3 X 3 
               
                obj_verts = self.load_object_geometry_w_rest_geo(new_obj_rot_mat, \
                        torch.from_numpy(new_obj_x[0]).float().to(new_obj_rot_mat.device), rest_verts)

                center_verts = obj_verts.mean(dim=1) # T X 3 

                # Run SMPLX 
                new_seq_root_trans = torch.from_numpy(new_seq_root_trans).float() 
                smplx_jnts, smplx_verts, _ = run_smplx_model(new_seq_root_trans[None], new_local_aa_rep[None], \
                    torch.from_numpy(betas).float(), [gender], self.bm_dict)

                # (Pdb) smplx_jnts.shape
                # torch.Size([1, 120, 55, 3])
                # (Pdb) smplx_verts.shape
                # torch.Size([1, 120, 10475, 3])

                smplx_jnts = smplx_jnts[0].detach().cpu().numpy() # T X 55 X 3
                smplx_verts = smplx_verts[0].detach().cpu().numpy() # T X 55 X 3

                jpos3d_18 = convert_smpl_jnts_to_openpose18(smplx_jnts, smplx_verts) # T X 18 X 3 
                
                obj_kpts_idx_list = self.object2kpts_idx_dict[self.object_name]
                obj_kpts = obj_verts[:, obj_kpts_idx_list, :] # T X 5 X 3 
               
                self.window_data_dict[s_idx]['seq_name'] = seq_name
                self.window_data_dict[s_idx]['start_t_idx'] = start_t_idx
                self.window_data_dict[s_idx]['end_t_idx'] = end_t_idx 

                self.window_data_dict[s_idx]['betas'] = betas 
                self.window_data_dict[s_idx]['gender'] = gender

                # self.window_data_dict[s_idx]['trans2joint'] = trans2joint 
                # self.window_data_dict[s_idx]['rest_human_offsets'] = rest_human_offsets 

                # self.window_data_dict[s_idx]['obj_rot_mat'] = query['obj_rot_mat'].detach().cpu().numpy()
                # self.window_data_dict[s_idx]['window_obj_com_pos'] = query['window_obj_com_pos'].detach().cpu().numpy() 

                self.window_data_dict[s_idx]['smpl_trans'] = new_seq_root_trans.detach().cpu().numpy()  
                self.window_data_dict[s_idx]['smpl_poses'] = new_local_aa_rep.detach().cpu().numpy()  

                self.window_data_dict[s_idx]['jpos3d_18'] = jpos3d_18 
                self.window_data_dict[s_idx]['obj_kpts'] = obj_kpts.detach().cpu().numpy() 

                combined_kpts = torch.cat((torch.from_numpy(jpos3d_18).float(), obj_kpts), dim=1) # T X 23 X 3 

                combined_jnts2d_18, seq_len, cam_extrinsic, jnts3d_18 = self.gen_multi_view_2d_from_3d_seq(combined_kpts) # K X T X (18+5) X 2 
                self.window_data_dict[s_idx]['combined_jnts2d_18'] = combined_jnts2d_18.detach().cpu().numpy()[0]

                self.window_data_dict[s_idx]['combined_jnts3d_18'] = jnts3d_18.detach().cpu().numpy()

                assert combined_jnts2d_18.shape[2] == 23 

                assert jnts3d_18.shape[1] == 23 

                if combined_jnts2d_18.shape[2] != 23:
                    import pdb; pdb.set_trace() 

                self.window_data_dict[s_idx]['seq_len'] = seq_len 

                s_idx += 1 
       
    def __len__(self):
        return len(self.window_data_dict)

    def gen_multi_view_2d_from_3d_seq(self, smplx_jnts18):
        # T X 18 X 3 

        # Move the first frame's root joint to zero (x=0, y=0, z=0) 
        l_hip_idx = 8
        r_hip_idx = 11
        root_trans = (smplx_jnts18[0:1, l_hip_idx:l_hip_idx+1, :]+smplx_jnts18[0:1, r_hip_idx:r_hip_idx+1, :])/2. # 1 X 1 X 3 

        smplx_jnts18 = smplx_jnts18 - root_trans  

        # Project 3D pose to different views.  
        jnts3d_aligned_xy_cam_list, jnts3d_cam_list, \
        jnts2d_list, seq_mask_list, cam_extrinsic_list = \
                    single_seq_get_multi_view_2d_motion_from_3d_torch(smplx_jnts18, \
                    return_multiview=True, num_views=1, farest=False, unified_dist=True) 

        jnts2d_list = torch.stack(jnts2d_list) # in range width, height 
        seq_mask_list = torch.stack(seq_mask_list)
        cam_extrinsic_list = torch.stack(cam_extrinsic_list)

        seq_len = cal_min_seq_len_from_mask_torch(seq_mask_list)

        return jnts2d_list, seq_len, cam_extrinsic_list, smplx_jnts18   
        # K X T X J X 2, int value, K X 4 X 4, T X 18 X 3   

    def __getitem__(self, index):
        curr_window_data = self.window_data_dict[index]

        seq_len = curr_window_data['seq_len'] # int 

        jnts3d_18 = curr_window_data['combined_jnts3d_18'][:seq_len] # T X 23 X 3
        jnts2d_seq = curr_window_data['combined_jnts2d_18'][:seq_len]  # T X 23 X 2 
        
        obj_kpts = curr_window_data['obj_kpts'][:seq_len]  # T X 5 X 3 

        start_t_idx = 0 
        end_t_idx = start_t_idx + self.window

        jnts3d_18 = jnts3d_18[start_t_idx:end_t_idx] # sample_n_frames X 18 X 3 
        obj_kpts = obj_kpts[start_t_idx:end_t_idx] # sample_n_frames X 5 X 3 

        # if jnts3d_18.shape[0] < self.window:
        #     random_idx = random.sample(list(range(len(self.window_data_dict))), 1)[0]
        #     return self.__getitem__(random_idx)

        jnts2d_seq = torch.from_numpy(jnts2d_seq[start_t_idx:end_t_idx]).float() # sample_n_frames X 23 X 2

        motion_data = jnts2d_seq # sample_n_frames X (18+5) X 2 
        motion_data[..., 0] /= self.image_width 
        motion_data[..., 1] /= self.image_height 

        # Normalize 2d pose sequence 
        normalized_motion_data = normalize_pose2d(motion_data, input_normalized=True) # T X 18 X 2    
        # Normalize to range [-1, 1] 
        # Add padding
        actual_seq_len = motion_data.shape[0]
        if actual_seq_len < self.window:
            padding_motion_data = torch.zeros(self.window-actual_seq_len, \
                            normalized_motion_data.shape[1], normalized_motion_data.shape[2])
            normalized_motion_data = torch.cat((normalized_motion_data, padding_motion_data), dim=0) 

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
        input_dict['normalized_jnts2d'] = normalized_motion_data # T X 23 X 2 
        input_dict['seq_len'] = actual_seq_len 
        # input_dict['cam_extrinsic'] = cam_extrinsic # K X 4 X 4 

        input_dict['seq_name'] = curr_window_data['seq_name'] 

        input_dict['lines_pass_jnts2d'] = random_lines_for_joints2d 

        jnts3d_18 = torch.from_numpy(jnts3d_18).float() # T X 18 X 3 
        if jnts3d_18.shape[0] < self.window:
            padding_jnts3d_18 = torch.zeros(self.window-jnts3d_18.shape[0], \
                jnts3d_18.shape[1], jnts3d_18.shape[2])
           
            jnts3d_18 = torch.cat((jnts3d_18, padding_jnts3d_18), dim=0)

        input_dict['jnts3d_18'] = jnts3d_18 # T X 23 X 3 
       
        return input_dict 
