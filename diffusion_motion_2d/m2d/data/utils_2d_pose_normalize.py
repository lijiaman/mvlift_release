import numpy as np 

import torch 

def normalize_pose2d(ori_poses, input_normalized=False, \
    image_width=1920, image_height=1080):
    # ori_obj_pts: T X K X 2/BS X T X K X 2 (in [0, width] [0, height] range)
    normalized_poses = torch.zeros_like(ori_poses)
    if input_normalized:
        image_width = 1.
        image_height = 1.

    if ori_poses.dim() == 3:
        normalized_poses[:, :, 0] = ori_poses[:, :, 0]/image_width
        normalized_poses[:, :, 1] = ori_poses[:, :, 1]/image_height 
    elif ori_poses.dim() == 4: 
        normalized_poses[:, :, :, 0] = ori_poses[:, :, :, 0]/image_width 
        normalized_poses[:, :, :, 1] = ori_poses[:, :, :, 1]/image_height 

    normalized_poses = normalized_poses * 2 - 1 # [-1, 1] range 

    return normalized_poses # BS X T X K X 2

def normalize_pose2d_w_grad(ori_poses, input_normalized=False, \
    image_width=1920, image_height=1080):
    # ori_obj_pts: T X K X 2/BS X T X K X 2 (in [0, width] [0, height] range)
    if input_normalized:
        image_width = 1.
        image_height = 1.
    
    if ori_poses.dim() == 3:
        ori_poses[:, :, 0] = ori_poses[:, :, 0]/image_width
        ori_poses[:, :, 1] = ori_poses[:, :, 1]/image_height 
    elif ori_poses.dim() == 4: 
        ori_poses[:, :, :, 0] = ori_poses[:, :, :, 0]/image_width 
        ori_poses[:, :, :, 1] = ori_poses[:, :, :, 1]/image_height 

    ori_poses = ori_poses * 2 - 1 # [-1, 1] range 

    return ori_poses # BS X T X K X 2

def de_normalize_pose2d(normalized_poses, image_width=1920, image_height=1080):
    # ori_obj_pts: T X K X 2/BS X T X K X 2 (in [-1, 1] range)
    de_poses = (normalized_poses + 1) * 0.5 # [0, 1] range
    
    de_poses[..., 0] *= image_width 
    de_poses[..., 1] *= image_height 
   
    return de_poses # BS X T X K X 2

def cal_seq_len_from_mask(seq_mask):
    # T 
    cnt = 0 
    break_seq = False 
    num_steps = seq_mask.shape[0]
    for t_idx in range(num_steps):
        if break_seq:
            break 

        if seq_mask[t_idx]:
            cnt += 1
        else:
            break_seq = True 

    return cnt 

# def move_init_pose2d_to_center(pose2d_seq):
#     # pose2d_seq: T X 18 X 2, [0, 1], move the initial pose2d to have u=0.5, v=0.5 
#     lhip_idx = 11
#     rhip_idx = 8 
#     pred_root_trans = (pose2d_seq[0, lhip_idx, :] + \
#                 pose2d_seq[0, rhip_idx, :])/2. # 2
#     init_pose_x = pred_root_trans[0]
#     init_pose_y = pred_root_trans[1] 

#     center_x = 0.5 
#     center_y = 0.5 

#     move2center_x = center_x - init_pose_x 
#     move2center_y = center_y - init_pose_y 

#     pose2d_seq[:, :, 0] += move2center_x 
#     pose2d_seq[:, :, 1] += move2center_y 

#     # Recompute visibility map 
#     x_mask = (pose2d_seq[:, :, 0] <= 0) + (pose2d_seq[:, :, 0] >= 1) # T X 18
#     y_mask = (pose2d_seq[:, :, 1] <= 0) + (pose2d_seq[:, :, 1] >= 1) # T X 18 

#     x_mask = (x_mask.sum(dim=-1) == 0) # T, 1 represents visible, 0 represents invisible  
#     y_mask = (y_mask.sum(dim=-1) == 0) # T 

#     vis_2d_mask = (x_mask * y_mask) # T 
#     if torch.is_tensor(vis_2d_mask):
#         vis_2d_mask = vis_2d_mask.detach().cpu().numpy() 

#     seq_len = cal_seq_len_from_mask(vis_2d_mask) 

#     return pose2d_seq[:seq_len] 

def move_init_pose2d_to_center(pose2d_seq, rescale=True, for_eval=False, center_all_jnts2d=False, use_bb_data=False):
    # pose2d_seq: 1 x T x 18 x 2, normalized coordinates [0, 1]
    # Indices for left and right hips are constants
    # Compute bounding box dimensions for the first pose in each sequence
    min_vals, _ = torch.min(pose2d_seq[:, 0, :, :], dim=1)
    max_vals, _ = torch.max(pose2d_seq[:, 0, :, :], dim=1)
    bbox_sizes = max_vals - min_vals  # Bounding box sizes for each sequence in the batch
    # 1 X 2 


    max_bbox_side = torch.max(bbox_sizes, dim=1).values
    if bbox_sizes[0, 0] > bbox_sizes[0, 1]: # w > h, 

        if for_eval:
            target_max = 1. / 2
            target_min = 1. / 2
        else:
            target_max = 1. / 2
            target_min = 1. / 2

        if use_bb_data:
            target_max = 1. / 3
            target_min = 1. / 3
    else: # w < h 
        if for_eval:
            target_max = 1. / 2
            target_min = 1. / 2
        else:
            target_max = 1. / 2
            target_min = 1. / 2

        if use_bb_data:
            target_max = 1. / 3
            target_min = 1. / 4

    # Calculate maximum and minimum scale factors for each sequence
    min_scale = target_min / max_bbox_side
    max_scale = target_max / max_bbox_side

    # Randomly sample a scale factor for each sequence from the calculated range
    rand_scales = torch.rand(min_scale.shape, device=pose2d_seq.device, dtype=pose2d_seq.dtype)
    scale_factors = min_scale + rand_scales * (max_scale - min_scale)

    # Scale the poses
    if rescale:
        pose2d_seq *= scale_factors[:, None, None, None]

    lhip_idx = 11
    rhip_idx = 8

    if center_all_jnts2d:
        pred_root_trans = (pose2d_seq[:, :, lhip_idx, :] + pose2d_seq[:, :, rhip_idx, :]) / 2. # 1(BS) X T X 2 
        center = torch.tensor([0.5, 0.5], device=pose2d_seq.device, dtype=pose2d_seq.dtype)

        move2center = center[None, None, :].repeat(pred_root_trans.shape[0], pred_root_trans.shape[1], 1) - pred_root_trans

        pose2d_seq += move2center[:, :, None, :].repeat(1, 1, pose2d_seq.shape[2], 1) # BS X T X J X 2 
    else:
        # Compute the initial pose translation to center
        # Average left and right hip positions to find the root (K x 2)
        pred_root_trans = (pose2d_seq[:, 0, lhip_idx, :] + pose2d_seq[:, 0, rhip_idx, :]) / 2. # 1(BS) X 2 

        # Central position in normalized coordinates
        center = torch.tensor([0.5, 0.5], device=pose2d_seq.device, dtype=pose2d_seq.dtype)

        # Compute translation needed to move initial root position to the center (K x 2)
        move2center = center[None, :] - pred_root_trans

        # Apply the translation to all positions in all sequences (broadcasting)
        pose2d_seq += move2center[:, None, None, :]

    # Recompute visibility based on bounds [0, 1] for all sequences and joints
    x_mask = (pose2d_seq[..., 0] <= 0) | (pose2d_seq[..., 0] >= 1)
    y_mask = (pose2d_seq[..., 1] <= 0) | (pose2d_seq[..., 1] >= 1)

    vis_2d_mask = ~(x_mask | y_mask)  # K x T x 18, True for visible joints

    # Determine the valid sequence length based on visibility, for each sequence
    # Correcting the reduction of dimensions:
    vis_2d_mask = vis_2d_mask.all(dim=2)  # Reduce joint dimension, results in K x T

    return pose2d_seq, vis_2d_mask, scale_factors # Return truncated sequences and the common minimum length

def move_init_pose2d_to_center_for_animal_data(pose2d_seq, rescale=True, for_eval=False, center_all_jnts2d=False):
    # pose2d_seq: 1 x T x 20 x 2, normalized coordinates [0, 1]
    # Indices for left and right hips are constants
    # Compute bounding box dimensions for the first pose in each sequence
    min_vals, _ = torch.min(pose2d_seq[:, 0, :, :], dim=1)
    max_vals, _ = torch.max(pose2d_seq[:, 0, :, :], dim=1)
    bbox_sizes = max_vals - min_vals  # Bounding box sizes for each sequence in the batch
    # 1 X 2 

    max_bbox_side = torch.max(bbox_sizes, dim=1).values
    if bbox_sizes[0, 0] > bbox_sizes[0, 1]: # w > h, 
        target_max = 1. / 6
        target_min = 1. / 6
    else: # w < h 
        target_max = 1. / 4
        target_min = 1. / 4

    # Calculate maximum and minimum scale factors for each sequence
    min_scale = target_min / max_bbox_side
    max_scale = target_max / max_bbox_side

    # Randomly sample a scale factor for each sequence from the calculated range
    rand_scales = torch.rand(min_scale.shape, device=pose2d_seq.device, dtype=pose2d_seq.dtype)
    scale_factors = min_scale + rand_scales * (max_scale - min_scale)

    # Scale the poses
    if rescale:
        pose2d_seq *= scale_factors[:, None, None, None]

    neck_idx = 5

    # Compute the initial pose translation to center
    pred_root_trans = pose2d_seq[:, 0, neck_idx, :] # BS X 2 

    # Central position in normalized coordinates
    center = torch.tensor([0.5, 0.5], device=pose2d_seq.device, dtype=pose2d_seq.dtype)

    # Compute translation needed to move initial root position to the center (K x 2)
    move2center = center[None, :] - pred_root_trans

    # Apply the translation to all positions in all sequences (broadcasting)
    pose2d_seq += move2center[:, None, None, :]

    # Recompute visibility based on bounds [0, 1] for all sequences and joints
    x_mask = (pose2d_seq[..., 0] <= 0) | (pose2d_seq[..., 0] >= 1)
    y_mask = (pose2d_seq[..., 1] <= 0) | (pose2d_seq[..., 1] >= 1)

    vis_2d_mask = ~(x_mask | y_mask)  # K x T x 18, True for visible joints

    # Determine the valid sequence length based on visibility, for each sequence
    # Correcting the reduction of dimensions:
    vis_2d_mask = vis_2d_mask.all(dim=2)  # Reduce joint dimension, results in K x T

    return pose2d_seq, vis_2d_mask, scale_factors # Return truncated sequences and the common minimum length
