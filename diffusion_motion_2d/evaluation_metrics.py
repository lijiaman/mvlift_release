import os 
import numpy as np
import time 

import json 


from sklearn.cluster import DBSCAN

import trimesh 

import torch 

import torch.nn.functional as F

from eval_utils import compute_similarity_transform

def compute_accel(joints):
    """
    Computes acceleration of 3D joints.
    Args:
        joints (Nx25x3).
    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return np.mean(acceleration_normed, axis=1)

def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)

def compute_vel(joints):
    velocities = joints[1:] - joints[:-1]
    velocity_normed = np.linalg.norm(velocities, axis=2)
    return np.mean(velocity_normed, axis=1)

def compute_error_vel(joints_gt, joints_pred, vis = None):
    vel_gt = joints_gt[1:] - joints_gt[:-1] 
    vel_pred = joints_pred[1:] - joints_pred[:-1]
    normed = np.linalg.norm(vel_pred - vel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    return np.mean(normed[new_vis], axis=1)

def get_frobenious_norm_rot_only(x, y):
    # x, y: N X 3 X 3 
    error = 0.0
    for i in range(len(x)):
        x_mat = x[i][:3, :3]
        y_mat_inv = np.linalg.inv(y[i][:3, :3])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(3)
        error += np.linalg.norm(ident_mat - error_mat, 'fro')
    return error / len(x)

def get_foot_sliding(
    verts,
    up="z",
    threshold = 0.01  # 1 cm/frame
):
    # verts: T X Nv X 3
    vert_velocities = []
    up_coord = 2 if up == "z" else 1
    lowest_vert_idx = np.argmin(verts[:, :, up_coord], axis=1)
    for frame in range(1, verts.shape[0] - 1):
        vert_idx = lowest_vert_idx[frame]
        vel = np.linalg.norm(
            verts[frame + 1, vert_idx, :] - verts[frame - 1, vert_idx, :]
        ) / 2
        vert_velocities.append(vel)
    return np.sum(np.array(vert_velocities) > threshold) / verts.shape[0] * 100

def determine_floor_height_and_contacts(body_joint_seq, fps=30):
    '''
    Input: body_joint_seq N x 22 x 3 numpy array
    Contacts are N x 4 where N is number of frames and each row is left heel/toe, right heel/toe
    '''
    FLOOR_VEL_THRESH = 0.005
    FLOOR_HEIGHT_OFFSET = 0.01

    num_frames = body_joint_seq.shape[0]

    # compute toe velocities
    root_seq = body_joint_seq[:, 0, :]
    left_toe_seq = body_joint_seq[:, 10, :]
    right_toe_seq = body_joint_seq[:, 11, :]
    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    # now foot heights (z is up)
    left_toe_heights = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights = root_seq[:, 2]

    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_toe_vel < FLOOR_VEL_THRESH]
    right_static_foot_heights = right_toe_heights[right_toe_vel < FLOOR_VEL_THRESH]
    right_static_inds = all_inds[right_toe_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(left_static_foot_heights, right_static_foot_heights)
    all_static_inds = np.append(left_static_inds, right_static_inds)

    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(all_static_foot_heights.reshape(-1, 1))
        all_labels = np.unique(clustering.labels_)
        # print(all_labels)
       
        min_median = min_root_median = float('inf')
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(all_static_inds[clustering.labels_ == cur_label]) # inds in the original sequence that correspond to this cluster
           
            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)
           
            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        floor_height = min_median 
        offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET # toe joint is actually inside foot mesh a bit

    else:
        floor_height = offset_floor_height = 0.0
   
    return floor_height

def compute_foot_sliding_for_smpl(pred_global_jpos, floor_height):
    # pred_global_jpos: T X J X 3 
    seq_len = pred_global_jpos.shape[0]

    # Put human mesh to floor z = 0 and compute. 
    pred_global_jpos[:, :, 2] -= floor_height

    lankle_pos = pred_global_jpos[:, 7, :] # T X 3 
    ltoe_pos = pred_global_jpos[:, 10, :] # T X 3 

    rankle_pos = pred_global_jpos[:, 8, :] # T X 3 
    rtoe_pos = pred_global_jpos[:, 11, :] # T X 3 

    H_ankle = 0.08 # meter
    H_toe = 0.04 # meter 

    lankle_disp = np.linalg.norm(lankle_pos[1:, :2] - lankle_pos[:-1, :2], axis = 1) # T 
    ltoe_disp = np.linalg.norm(ltoe_pos[1:, :2] - ltoe_pos[:-1, :2], axis = 1) # T 
    rankle_disp = np.linalg.norm(rankle_pos[1:, :2] - rankle_pos[:-1, :2], axis = 1) # T 
    rtoe_disp = np.linalg.norm(rtoe_pos[1:, :2] - rtoe_pos[:-1, :2], axis = 1) # T 

    lankle_subset = lankle_pos[:-1, -1] < H_ankle
    ltoe_subset = ltoe_pos[:-1, -1] < H_toe
    rankle_subset = rankle_pos[:-1, -1] < H_ankle
    rtoe_subset = rtoe_pos[:-1, -1] < H_toe
   
    lankle_sliding_stats = np.abs(lankle_disp * (2 - 2 ** (lankle_pos[:-1, -1]/H_ankle)))[lankle_subset]
    lankle_sliding = np.sum(lankle_sliding_stats)/seq_len * 1000

    ltoe_sliding_stats = np.abs(ltoe_disp * (2 - 2 ** (ltoe_pos[:-1, -1]/H_toe)))[ltoe_subset]
    ltoe_sliding = np.sum(ltoe_sliding_stats)/seq_len * 1000

    rankle_sliding_stats = np.abs(rankle_disp * (2 - 2 ** (rankle_pos[:-1, -1]/H_ankle)))[rankle_subset]
    rankle_sliding = np.sum(rankle_sliding_stats)/seq_len * 1000

    rtoe_sliding_stats = np.abs(rtoe_disp * (2 - 2 ** (rtoe_pos[:-1, -1]/H_toe)))[rtoe_subset]
    rtoe_sliding = np.sum(rtoe_sliding_stats)/seq_len * 1000

    sliding = (lankle_sliding + ltoe_sliding + rankle_sliding + rtoe_sliding) / 4.

    return sliding 

def calculate_bone_lengths(joints, connections):
    # This function calculates the lengths of the bones for each frame
    # joints is expected to be of shape (T, J, 3)
    # connections is a list of tuples (start_joint, end_joint)
    bone_lengths = [torch.norm(joints[:, end] - joints[:, start], dim=1) for start, end in connections]
    return torch.stack(bone_lengths, dim=1)  # (T, num_bones)

def bone_length_consistency_metric(predictions, connections):
    # Calculate bone lengths for predictions and ground truth
    pred_lengths = calculate_bone_lengths(predictions, connections)
    # gt_lengths = calculate_bone_lengths(ground_truth, connections)

    # Convert bone lengths from tensors to numpy arrays
    pred_lengths_np = pred_lengths.detach().cpu().numpy()  # Shape (T, num_bones)
    # gt_lengths_np = gt_lengths.detach().cpu().numpy()  # Shape (T, num_bones)

    # Compute the error for bone lengths
    # This computes the L2 norm (Euclidean distance) between corresponding bone lengths across all frames,
    # and then takes the average.
    # error = np.linalg.norm(pred_lengths_np - gt_lengths_np, axis=1).mean() * 1000  # Convert from meters to millimeters

    # Calculate mean squared error of bone lengths across all frames
    # mse = torch.mean((pred_lengths - gt_lengths) ** 2).cpu().item() 

    # consistency_error = np.std(pred_lengths_np, axis=0).mean() * 1000  # Convert from meters to millimeters
    
    # return consistency_error 

    T, num_bones = pred_lengths_np.shape
    error_matrix = np.zeros((T, T))  # Matrix to hold the errors between all frames

    # Compute the absolute difference in bone lengths between each pair of frames
    for i in range(T):
        for j in range(T):
            error_matrix[i, j] = np.abs(pred_lengths_np[i] - pred_lengths_np[j]).mean()

    # Compute the mean error by averaging all the individual differences
    # Exclude the diagonal (comparison of each frame to itself) from the mean calculation
    mean_bone_length_error = np.sum(error_matrix) / (T * (T - 1)) * 1000  # Convert from meters to millimeters

    return mean_bone_length_error

def bone_scale_consistency_metric(init_pose_3d, predictions, actual_seq_len):
    # connections = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], \
    #             [9, 10], [11, 12], [12, 13], [0, 14], [14, 16], \
    #             [0, 15], [15, 17]]

    # Only compute arm and leg 
    connections = [[2, 3], [3, 4], [5, 6], [6, 7], [8, 9], \
                [9, 10], [11, 12], [12, 13]]

    # OPENPOSE_LANDMARKS = [
    #     "nose", # 0
    #     "neck", # 1
    #     "right_shoulder", # 2  
    #     "right_elbow", # 3 
    #     "right_wrist", # 4
    #     "left_shoulder", # 5
    #     "left_elbow", # 6
    #     "left_wrist", # 7
    #     "right_hip", # 8 
    #     "right_knee", # 9 
    #     "right_ankle", # 10
    #     "left_hip", # 11
    #     "left_knee", # 12
    #     "left_ankle", # 13
    #     "right_eye", # 14
    #     "left_eye", # 15
    #     "right_ear", # 16
    #     "left_ear", # 17
    # ]

    target_bone_len = calculate_bone_lengths(init_pose_3d, connections) # 1 X J 
    pred_bone_len = calculate_bone_lengths(predictions, connections) # T X J 

    target_bone_len = target_bone_len.repeat(pred_bone_len.shape[0], 1) # T X J 

    # scales = target_bone_len[:actual_seq_len, :]/pred_bone_len[:actual_seq_len, :] 
    scales = pred_bone_len[:actual_seq_len, :] / target_bone_len[:actual_seq_len, :]
    # T X J 

    return scales.mean().item()  

def compute_metrics_for_interaction_contact(gt_obj_verts, pred_obj_verts, ori_jpos_gt, ori_jpos_pred, lhand_idx, rhand_idx, use_joints24=False):
    # Compute contact score 
    num_obj_verts = gt_obj_verts.shape[1]
    if use_joints24:
        # contact_threh = 0.05
        contact_threh = 0.05
    else:
        contact_threh = 0.10 

    gt_lhand_jnt = ori_jpos_gt[:, lhand_idx, :] # T X 3 
    gt_rhand_jnt = ori_jpos_gt[:, rhand_idx, :] # T X 3 

    # What if the joint is in the object? already penetrate? 
    gt_lhand2obj_dist = torch.sqrt(((gt_lhand_jnt[:, None, :].repeat(1, num_obj_verts, 1) - gt_obj_verts.to(gt_lhand_jnt.device))**2).sum(dim=-1)) # T X N  
    gt_rhand2obj_dist = torch.sqrt(((gt_rhand_jnt[:, None, :].repeat(1, num_obj_verts, 1) - gt_obj_verts.to(gt_rhand_jnt.device))**2).sum(dim=-1)) # T X N  

    gt_lhand2obj_dist_min = gt_lhand2obj_dist.min(dim=1)[0] # T 
    gt_rhand2obj_dist_min = gt_rhand2obj_dist.min(dim=1)[0] # T 

    gt_lhand_contact = (gt_lhand2obj_dist_min < contact_threh)
    gt_rhand_contact = (gt_rhand2obj_dist_min < contact_threh)

    lhand_jnt = ori_jpos_pred[:, lhand_idx, :] # T X 3 
    rhand_jnt = ori_jpos_pred[:, rhand_idx, :] # T X 3 

    lhand2obj_dist = torch.sqrt(((lhand_jnt[:, None, :].repeat(1, num_obj_verts, 1) - pred_obj_verts.to(lhand_jnt.device))**2).sum(dim=-1)) # T X N  
    rhand2obj_dist = torch.sqrt(((rhand_jnt[:, None, :].repeat(1, num_obj_verts, 1) - pred_obj_verts.to(rhand_jnt.device))**2).sum(dim=-1)) # T X N  
   
    lhand2obj_dist_min = lhand2obj_dist.min(dim=1)[0] # T 
    rhand2obj_dist_min = rhand2obj_dist.min(dim=1)[0] # T 

    lhand_contact = (lhand2obj_dist_min < contact_threh)
    rhand_contact = (rhand2obj_dist_min < contact_threh)

    num_steps = gt_lhand_contact.shape[0]

    # Compute the distance between hand joint and object for frames that are in contact with object in GT. 
    contact_dist = 0
    gt_contact_dist = 0 

    gt_contact_cnt = 0
    for idx in range(num_steps):
        if gt_lhand_contact[idx] or gt_rhand_contact[idx]:
            gt_contact_cnt += 1 

            contact_dist += min(lhand2obj_dist_min[idx], rhand2obj_dist_min[idx])
            gt_contact_dist += min(gt_lhand2obj_dist_min[idx], gt_rhand2obj_dist_min[idx])

    if gt_contact_cnt == 0:
        contact_dist = 0 
        gt_contact_dist = 0 
    else:
        contact_dist = contact_dist.detach().cpu().numpy()/float(gt_contact_cnt)
        gt_contact_dist = gt_contact_dist.detach().cpu().numpy()/float(gt_contact_cnt)

    pred_contact_cnt = 0

    # Compute precision and recall for contact. 
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for idx in range(num_steps):
        gt_in_contact = (gt_lhand_contact[idx] or gt_rhand_contact[idx]) 
        pred_in_contact = (lhand_contact[idx] or rhand_contact[idx])
        if gt_in_contact and pred_in_contact:
            TP += 1

        if (not gt_in_contact) and pred_in_contact:
            FP += 1

        if (not gt_in_contact) and (not pred_in_contact):
            TN += 1

        if gt_in_contact and (not pred_in_contact):
            FN += 1

        if pred_in_contact:
            pred_contact_cnt += 1 

    gt_contact_percent = gt_contact_cnt /float(num_steps)
    pred_contact_percent = pred_contact_cnt / float(num_steps) 

    contact_acc = (TP+TN)/(TP+FP+TN+FN)

    if (TP+FP) == 0: # Prediction no contact!!!
        contact_precision = 0
        print("Contact precision, TP + FP == 0!!")
    else:
        contact_precision = TP/(TP+FP)
    
    if (TP+FN) == 0: # GT no contact! 
        contact_recall = 0
        print("Contact recall, TP + FN == 0!!")
    else:
        contact_recall = TP/(TP+FN)

    if contact_precision == 0 and contact_recall == 0:
        contact_f1_score = 0 
    else:
        contact_f1_score = 2 * (contact_precision * contact_recall)/(contact_precision+contact_recall) 

    return contact_dist, gt_contact_dist, gt_contact_percent, pred_contact_percent, contact_precision, contact_recall, contact_f1_score, contact_acc

def compute_metrics_for_jpos(ori_jpos_pred, ori_jpos2d_pred, \
    ori_jpos_gt, ori_jpos2d_gt, actual_len, inconsistent_jnts3d, obj_kpts_pred=None, obj_kpts_gt=None, eval_animal_data=False):
    # actual_len: scale value 
    # ori_jpos_pred: T X 18 X 3 
    # ori_jpos2d_pred: T X 18 X 2 
    # inconsistent_jnts3d: T X 18 X 3 

    # clone is important! Previous a bug here! 
    ori_jpos_gt = ori_jpos_gt[:actual_len].clone()
    ori_jpos_pred = ori_jpos_pred[:actual_len].clone()

    ori_jpos2d_gt = ori_jpos2d_gt[:actual_len].clone()
    ori_jpos2d_pred = ori_jpos2d_pred[:actual_len].clone() 

    if obj_kpts_pred is not None and obj_kpts_gt is not None:
        obj_kpts_pred = obj_kpts_pred[:actual_len].clone()
        obj_kpts_gt = obj_kpts_gt[:actual_len].clone() 

    # Move the first frame of joint 3D to have the same root 
    l_hip_index = 11
    r_hip_index = 8 
    first_hip_jpos_gt = (ori_jpos_gt[0:1, l_hip_index:l_hip_index+1, :]+ori_jpos_gt[0:1, r_hip_index:r_hip_index+1, :])/2. 
    first_hip_jpos_pred = (ori_jpos_pred[0:1, l_hip_index:l_hip_index+1, :]+ori_jpos_pred[0:1, r_hip_index:r_hip_index+1, :])/2. 

    ori_jpos_gt -= first_hip_jpos_gt 
    ori_jpos_pred -= first_hip_jpos_pred  

    # 1. Calculate jpos3d error (MPJPE w translation)
    mpjpe_w_trans = np.linalg.norm(ori_jpos_pred.detach().cpu().numpy() - \
    ori_jpos_gt.detach().cpu().numpy(), axis=2).mean() * 1000

    # 2. Calculate MPJPE 
    pred_trans = (ori_jpos_pred[:, l_hip_index:l_hip_index+1] + \
        ori_jpos_pred[:, r_hip_index:r_hip_index+1])/2. 
    gt_trans = (ori_jpos_gt[:, l_hip_index:l_hip_index+1] + \
        ori_jpos_gt[:, r_hip_index:r_hip_index+1])/2. 

    jpos_pred = ori_jpos_pred - pred_trans # zero out root
    jpos_gt = ori_jpos_gt - gt_trans  
    jpos_pred = jpos_pred.detach().cpu().numpy()
    jpos_gt = jpos_gt.detach().cpu().numpy()
    mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean() * 1000

    # 3. Calculate PA-MPJPE 
    pred3d_sym = compute_similarity_transform(jpos_pred.reshape(-1, 3), jpos_gt.reshape(-1, 3))
    pa_error = np.sqrt(np.sum((jpos_gt.reshape(-1, 3) - pred3d_sym)**2, axis=1))
    pa_mpjpe = pa_error.mean() * 1000 

    # 4. Caculate translation error 
    trans_err = np.linalg.norm(pred_trans.squeeze(0).detach().cpu().numpy() - \
            gt_trans.squeeze(0).detach().cpu().numpy(), axis=1).mean() * 1000

    # 5. Calculate 2D pixel error 
    # Move the first frame of joint 2D to be at the center (960, 540)
    if not eval_animal_data:
        first_hip_2d_gt = (ori_jpos2d_gt[0:1, l_hip_index:l_hip_index+1, :]+ori_jpos2d_gt[0:1, r_hip_index:r_hip_index+1, :])/2. 
        first_hip_2d_pred = (ori_jpos2d_pred[0:1, l_hip_index:l_hip_index+1, :]+ori_jpos2d_pred[0:1, r_hip_index:r_hip_index+1, :])/2. 

        ori_jpos2d_gt -= first_hip_2d_gt
        ori_jpos2d_pred -= first_hip_2d_pred 

    else:
        neck_idx = 5
        first_hip_2d_gt = ori_jpos2d_gt[0:1, neck_idx:neck_idx+1, :].clone()
        first_hip_2d_pred = ori_jpos2d_pred[0:1, neck_idx:neck_idx+1, :].clone() 

        ori_jpos2d_gt -= first_hip_2d_gt
        ori_jpos2d_pred -= first_hip_2d_pred 
    0
    jpos2d_err = np.linalg.norm(ori_jpos2d_pred.detach().cpu().numpy() - \
        ori_jpos2d_gt.detach().cpu().numpy(), axis=2).mean()

    seq_jpos2d_err = np.linalg.norm(ori_jpos2d_pred.detach().cpu().numpy() - \
        ori_jpos2d_gt.detach().cpu().numpy(), axis=2)

    # 6. Calculate 2D pixel error without translation influence 
    seq_hip_2d_gt = (ori_jpos2d_gt[:, l_hip_index:l_hip_index+1, :]+ori_jpos2d_gt[:, r_hip_index:r_hip_index+1, :])/2. 
    seq_hip_2d_pred = (ori_jpos2d_pred[:, l_hip_index:l_hip_index+1, :]+ori_jpos2d_pred[:, r_hip_index:r_hip_index+1, :])/2. 

    ori_jpos2d_gt -= seq_hip_2d_gt
    ori_jpos2d_pred -= seq_hip_2d_pred 

    # 2D Pose Error with Scale Adjustment per Timestep
    scaled_jpos2d_pred = []
    for t in range(actual_len):
        pred_2d = ori_jpos2d_pred[t].to(ori_jpos2d_gt.device)
        gt_2d = ori_jpos2d_gt[t]

        # Optimal scaling factor for this timestep
        scale = torch.sum(pred_2d * gt_2d) / torch.sum(pred_2d ** 2)
        scaled_pred_2d = pred_2d * scale
        scaled_jpos2d_pred.append(scaled_pred_2d)
    
    scaled_jpos2d_pred = torch.stack(scaled_jpos2d_pred) 
    ori_jpos2d_pred = scaled_jpos2d_pred.clone() 

    scaled_jpos2d_err = np.linalg.norm(ori_jpos2d_pred.detach().cpu().numpy() - \
        ori_jpos2d_gt.detach().cpu().numpy(), axis=2).mean()
    
    # Calculate foot sliding
    # floor_height = determine_floor_height_and_contacts(ori_jpos_pred.detach().cpu().numpy(), fps=30)
    # gt_floor_height = determine_floor_height_and_contacts(ori_jpos_gt.detach().cpu().numpy(), fps=30)
    # print("floor height:{0}".format(floor_height)) 
    # print("gt floor height:{0}".format(gt_floor_height)) 

    # foot_sliding_jnts = compute_foot_sliding_for_smpl(ori_jpos_pred.detach().cpu().numpy(), floor_height)
    # gt_foot_sliding_jnts = compute_foot_sliding_for_smpl(ori_jpos_gt.detach().cpu().numpy(), gt_floor_height)
   
    # eturn mpjpe, gt_foot_sliding_jnts, foot_sliding_jnts, gt_floor_height, floor_height 

    if not torch.is_tensor(inconsistent_jnts3d):
        inconsistent_jnts3d = torch.tensor(inconsistent_jnts3d, dtype=torch.float32)
    consistency = bone_scale_consistency_metric(inconsistent_jnts3d[0:1], inconsistent_jnts3d, actual_len) 

    if obj_kpts_pred is not None and obj_kpts_gt is not None:
        obj_kpts_pred -= first_hip_jpos_pred  
        obj_kpts_gt -= first_hip_jpos_gt  

        center_obj_pred = obj_kpts_pred.mean(dim=1) # T X 3
        center_obj_gt = obj_kpts_gt.mean(dim=1) # T X 3 

        center_trans_errr = np.linalg.norm(center_obj_pred.detach().cpu().numpy() - \
            center_obj_gt.detach().cpu().numpy(), axis=1).mean() * 1000 

        centered_obj_pred_kpts = obj_kpts_pred - center_obj_pred[:, None, :] 
        centered_obj_gt_kpts = obj_kpts_gt - center_obj_gt[:, None, :] 

        obj_mpjpe = np.linalg.norm(centered_obj_pred_kpts.detach().cpu().numpy() - \
            centered_obj_gt_kpts.detach().cpu().numpy(), axis=2).mean() * 1000

        return mpjpe, pa_mpjpe, trans_err, mpjpe_w_trans, jpos2d_err, scaled_jpos2d_err, consistency, obj_mpjpe, center_trans_errr   

    return mpjpe, pa_mpjpe, trans_err, mpjpe_w_trans, jpos2d_err, scaled_jpos2d_err, consistency  


def compute_metrics_for_jpos_wo_root(ori_jpos_pred, ori_jpos2d_pred, \
    ori_jpos_gt, ori_jpos2d_gt, actual_len, inconsistent_jnts3d):
    # actual_len: scale value 
    # ori_jpos_pred: T X 18 X 3 
    # ori_jpos2d_pred: T X 18 X 2 
    # inconsistent_jnts3d: T X 18 X 3 

    ori_jpos_gt = ori_jpos_gt[:actual_len]
    ori_jpos_pred = ori_jpos_pred[:actual_len]

    ori_jpos2d_gt = ori_jpos2d_gt[:actual_len]
    ori_jpos2d_pred = ori_jpos2d_pred[:actual_len]

    
    # Move the first frame of joint 3D to have the same root 
    l_hip_index = 11
    r_hip_index = 8 
    first_hip_jpos_gt = (ori_jpos_gt[:, l_hip_index:l_hip_index+1, :]+ori_jpos_gt[:, r_hip_index:r_hip_index+1, :])/2. 
    first_hip_jpos_pred = (ori_jpos_pred[:, l_hip_index:l_hip_index+1, :]+ori_jpos_pred[:, r_hip_index:r_hip_index+1, :])/2. 

    ori_jpos_gt -= first_hip_jpos_gt 
    ori_jpos_pred -= first_hip_jpos_pred  

    # Move the first frame of joint 2D to be at the center (960, 540)
    first_hip_2d_gt = (ori_jpos2d_gt[:, l_hip_index:l_hip_index+1, :]+ori_jpos2d_gt[:, r_hip_index:r_hip_index+1, :])/2. 
    first_hip_2d_pred = (ori_jpos2d_pred[:, l_hip_index:l_hip_index+1, :]+ori_jpos2d_pred[:, r_hip_index:r_hip_index+1, :])/2. 

    ori_jpos2d_gt -= first_hip_2d_gt
    ori_jpos2d_pred -= first_hip_2d_pred 

    # 2D Pose Error with Scale Adjustment per Timestep
    scaled_jpos2d_pred = []
    for t in range(actual_len):
        pred_2d = ori_jpos2d_pred[t]
        gt_2d = ori_jpos2d_gt[t]

        # Optimal scaling factor for this timestep
        scale = torch.sum(pred_2d * gt_2d) / torch.sum(pred_2d ** 2)
        scaled_pred_2d = pred_2d * scale
        scaled_jpos2d_pred.append(scaled_pred_2d)
    
    scaled_jpos2d_pred = torch.stack(scaled_jpos2d_pred) 
    ori_jpos2d_pred = scaled_jpos2d_pred.clone() 

    # Calculate 2D pixel error 
    jpos2d_err = np.linalg.norm(ori_jpos2d_pred.detach().cpu().numpy() - \
        ori_jpos2d_gt.detach().cpu().numpy(), axis=2).mean()

    # Calculate jpos3d error
    mpjpe_w_trans = np.linalg.norm(ori_jpos_pred.detach().cpu().numpy() - \
        ori_jpos_gt.detach().cpu().numpy(), axis=2).mean() * 1000

    # Calculate MPJPE 
    if ori_jpos_pred.shape[1] == 18:
        l_hip_index = 11
        r_hip_index = 8 

    pred_trans = (ori_jpos_pred[:, l_hip_index:l_hip_index+1] + \
        ori_jpos_pred[:, r_hip_index:r_hip_index+1])/2. 
    gt_trans = (ori_jpos_gt[:, l_hip_index:l_hip_index+1] + \
        ori_jpos_gt[:, r_hip_index:r_hip_index+1])/2. 

    jpos_pred = ori_jpos_pred - pred_trans # zero out root
    jpos_gt = ori_jpos_gt - gt_trans  
    jpos_pred = jpos_pred.detach().cpu().numpy()
    jpos_gt = jpos_gt.detach().cpu().numpy()
    mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean() * 1000

    pred3d_sym = compute_similarity_transform(jpos_pred.reshape(-1, 3), jpos_gt.reshape(-1, 3))
    pa_error = np.sqrt(np.sum((jpos_gt.reshape(-1, 3) - pred3d_sym)**2, axis=1))
    pa_mpjpe = pa_error.mean() * 1000 

    # pa_mpjpe = 0 # For debug 

    # mpjpe_seq = np.linalg.norm(jpos_pred - jpos_gt, axis=2)
    # import pdb 
    # pdb.set_trace() 

    # Caculate translation error 
    trans_err = np.linalg.norm(pred_trans.squeeze(0).detach().cpu().numpy() - \
            gt_trans.squeeze(0).detach().cpu().numpy(), axis=1).mean() * 1000
    
   

    if not torch.is_tensor(inconsistent_jnts3d):
        inconsistent_jnts3d = torch.tensor(inconsistent_jnts3d, dtype=torch.float32)
    consistency = bone_scale_consistency_metric(inconsistent_jnts3d[0:1], inconsistent_jnts3d, actual_len) 

    return mpjpe, pa_mpjpe, trans_err, mpjpe_w_trans, jpos2d_err, consistency  