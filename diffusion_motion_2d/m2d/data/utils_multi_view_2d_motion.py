import sys 
sys.path.append("../../")

import numpy as np
import os 
import math 
import cv2 
import random 
import pickle as pkl 

import torch 
        
# from m2d.vis.vis_jnts import plot_3d_motion 
from m2d.vis.vis_jnts import plot_3d_motion, gen_2d_motion_vis

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def degrees_to_radians(degrees):
    return degrees * (torch.pi / 180)

def perspective_projection(points, rotation, translation,
                           focal_length=1000, image_width=1920, image_height=1080):
    batch_size = points.shape[0]
    camera_center = torch.tensor([image_width / 2, image_height / 2], device=points.device)

    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    aligned_3d = torch.cat([projected_points[:, :, :-1], points[:, :, 2:3]], dim=-1)

    return aligned_3d, points, projected_points[:, :, :-1]
    # aligned_3d has the same xy values as projected 2D. points is 3D joint position in camera's coordinate frame.

def visualize_cameras(rotations, translations, dest_fig_path):
    """
    Visualize multiple camera poses where rotations is K x 3 x 3 and translations is K x 3.
    
    Args:
    rotations (np.array): Array of shape K x 3 x 3, where each element is a 3x3 rotation matrix.
    translations (np.array): Array of shape K x 3, where each element is a 3x1 translation vector.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the scene size based on translations extents
    max_extent = np.max(np.abs(translations)) * 1.2
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])

    # Define axis colors
    axis_colors = ['red', 'green', 'blue']  # RGB for x, y, z axes

    for R, t in zip(rotations, translations):
        # Camera position is the translation vector
        camera_pos = t

        ax.scatter(*t, color='blue')

        # Draw a vertical line from the x-y plane to the camera position
        ax.plot([t[0], t[0]], [t[1], t[1]], [0, t[2]], 'gray', linestyle='--', linewidth=1)

        ax.text(t[0], t[1], t[2], f"{t[2]:.2f}", color='red') 
        
        # Plot each axis as an arrow
        for i in range(3):
            axis_vector = R[:, i]
            axis_end = camera_pos + 0.5 * axis_vector  # Scale the direction for visualization
            
            # Plot camera axis
            ax.quiver(*camera_pos, *(axis_end - camera_pos), color=axis_colors[i], arrow_length_ratio=0.05)

    plt.title('Visualization of Multiple Cameras with Axes')
    # plt.show()
    plt.savefig(dest_fig_path) 

def reverse_perspective_projection(aligned_3d, rotation, translation, focal_length=1000, image_width=1920, image_height=1080):
    batch_size = aligned_3d.shape[0]
    # Assuming camera_center is needed for reconstruction as well, similar to its use in perspective_projection
    camera_center = torch.tensor([image_width / 2, image_height / 2], device=aligned_3d.device)

    # Prepare the inverse of camera matrix K
    K_inv = torch.zeros([batch_size, 3, 3], device=aligned_3d.device)
    K_inv[:, 0, 0] = 1 / focal_length
    K_inv[:, 1, 1] = 1 / focal_length
    K_inv[:, 2, 2] = 1.
    K_inv[:, :-1, -1] = -camera_center / focal_length

    # Expand aligned_3d to homogeneous coordinates by adding ones
    ones = torch.ones(batch_size, aligned_3d.shape[1], 1, device=aligned_3d.device)
    aligned_3d_homogeneous = torch.cat([aligned_3d[:, :, :-1], ones], dim=-1)

    # Apply inverse camera intrinsics
    points_camera_space = torch.einsum('bij,bkj->bki', K_inv, aligned_3d_homogeneous)

    # Multiply with the z-values to undo the perspective divide
    points_camera_space[:, :, 0] *= aligned_3d[:, :, 2]
    points_camera_space[:, :, 1] *= aligned_3d[:, :, 2]
    points_camera_space[:, :, 2] = aligned_3d[:, :, 2]

    # Apply inverse transformation (rotation and translation)
    rotation_inv = rotation.transpose(1, 2)
    translation_inv = -torch.einsum('bij,bj->bi', rotation_inv, translation)

    # Move points back to global space
    points_global_space = torch.einsum('bij,bkj->bki', rotation_inv, points_camera_space - translation_inv.unsqueeze(1))

    return points_camera_space, points_global_space

def convert_jnts_cam_to_world(points_camera_space, rotation, translation):
    # Apply inverse transformation (rotation and translation)
    rotation_inv = rotation.transpose(1, 2)
    translation_inv = -torch.einsum('bij,bj->bi', rotation_inv, translation)

    # Move points back to global space
    points_global_space = torch.einsum('bij,bkj->bki', rotation_inv, points_camera_space - translation_inv.unsqueeze(1))

    return points_global_space

def camera_extrinsic_from_azimuth_elevation_torch(distance, azimuth, elevation):
    # Convert azimuth and elevation angles to radians
    azimuth_rad = degrees_to_radians(torch.tensor(azimuth))
    elevation_rad = degrees_to_radians(torch.tensor(elevation))

    # Calculate camera position in Cartesian coordinates
    x = distance * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    y = distance * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    z = distance * torch.sin(elevation_rad)

    # cam_height_above_floor = 0.8
    cam_height_above_floor = 0
    camera_position = torch.tensor([x, y, z + cam_height_above_floor])

    # Define the target (assuming we're looking at the origin) and calculate the direction vector
    look_at_target = torch.tensor([0, 0, 0 + cam_height_above_floor]).float()
    direction = look_at_target - camera_position
    forward = -direction / torch.norm(direction)  # Camera faces toward the negative of direction

    # World's up vector for Z-up system
    world_up = torch.tensor([0, 0, 1]).float()

    # Calculate right vector as cross product of forward vector and world's up vector
    right = torch.cross(world_up, forward)
    right /= torch.norm(right)

    # Recalculate the up vector as cross product of right and forward vector
    up = torch.cross(forward, right)
    up /= torch.norm(up)

    # Construct rotation matrix
    rotation_matrix = torch.stack([right, up, forward])

    # Create extrinsic matrix
    extrinsic_matrix = torch.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = -torch.matmul(rotation_matrix, camera_position)

    # Flip Z-axis for camera coordinate system
    extrinsic_matrix = torch.diag(torch.tensor([1, -1, -1, 1])).float() @ extrinsic_matrix

    return extrinsic_matrix

def draw_bodypose(canvas, candidate):
    H, W, C = canvas.shape
    candidate = np.array(candidate) # T(1) X 18 X 2 
    num_steps = candidate.shape[0] 

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(num_steps):
            limb_idx = limbSeq[i] 
            # Y = candidate[n, limb_idx, 0] * float(W)
            # X = candidate[n, limb_idx, 1] * float(H)
            skeleton_idx = np.asarray(limb_idx)-1
            skeleton_idx = skeleton_idx.tolist() 
            Y = candidate[n, skeleton_idx, 0]
            X = candidate[n, skeleton_idx, 1] 
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    # canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(num_steps):
            x, y = candidate[n, i][0:2]
            # x = int(x * W)
            # y = int(y * H)
            x = int(x)
            y = int(y)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas

def extract_smplx2openpose_jnts19(joint_seq, vtx_seq):
    # joint_seq: T X J X 3, numpy array 
    # vtx_seq: T X Nv X 3, numpy array 

    # Extract selected verex index 
    # 'nose':		    9120,
    # 'reye':		    9929,
    # 'leye':		    9448,
    # 'rear':		    616,
    # 'lear':		    6,
    selected_vertex_idx = [9120, 9929, 9448, 616, 6] 
    selected_verts = vtx_seq[:, selected_vertex_idx, :] # T X 5 X 3 

    smpl2openpose_idx = [22, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, \
                23, 24, 25, 26] 

    smplx_jnts27 = np.concatenate((joint_seq[:, :22, :], selected_verts), \
                        axis=1) # T X (22+5) X 3 
    smplx_for_openpose_jnts19 = smplx_jnts27[:, smpl2openpose_idx, :] # T X 19 X 3  

    return smplx_for_openpose_jnts19 

def extract_smplx2openpose_jnts19_torch(joint_seq, vtx_seq):
    # joint_seq: T X J X 3, numpy array 
    # vtx_seq: T X Nv X 3, numpy array 

    # Extract selected verex index 
    # 'nose':		    9120,
    # 'reye':		    9929,
    # 'leye':		    9448,
    # 'rear':		    616,
    # 'lear':		    6,
    selected_vertex_idx = [9120, 9929, 9448, 616, 6] 
    selected_verts = vtx_seq[:, selected_vertex_idx, :] # T X 5 X 3 

    smpl2openpose_idx = [22, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, \
                23, 24, 25, 26] 

    smplx_jnts27 = torch.cat((joint_seq[:, :22, :], selected_verts), \
                        dim=1) # T X (22+5) X 3 
    smplx_for_openpose_jnts19 = smplx_jnts27[:, smpl2openpose_idx, :] # T X 19 X 3  

    return smplx_for_openpose_jnts19 

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

def single_seq_get_multi_view_2d_motion_from_3d(smplx_jnts18, return_multiview=False, num_views=6):
    # smplx_jnts18: T X 18 X 3, numpy array 

    jnts3d_aligned_xy_cam_list = []
    jnts3d_cam_list = []
    jnts2d_list = [] 

    cam_extrinsic_list = []

    seq_mask_list = [] 

    image_width = 1920
    image_height = 1080

    focal_length = 1000 
    # focal_length = 1300 

    # num_views = 6 # Currently not using multi-view data, just use a random view 
    elevation = 0 
    # Sample a distance in a small range 
    # distance = random.uniform(3.5, 4) 
    distance = random.uniform(2, 3) 
    # Random sample an angle 
    init_azimuth = random.sample(list(range(360)), 1)[0]   
    offset_angle = 360/num_views 
    for view_idx in range(num_views):

        # Camera extrinsic parameters
        azimuth = init_azimuth + view_idx * offset_angle
        extrinsic_matrix = camera_extrinsic_from_azimuth_elevation(distance, azimuth, elevation)
        # 4 X 4 

        # Convert to image coordinates
        num_steps = smplx_jnts18.shape[0]
        
        cam_rot_mat = torch.from_numpy(extrinsic_matrix[:3, :3]).float()[None].repeat(num_steps, 1, 1) # T X 3 X 3 
        cam_trans = torch.from_numpy(extrinsic_matrix[:3, -1]).float()[None].repeat(num_steps, 1) # T X 3 

        if not torch.is_tensor(smplx_jnts18):
            smplx_jnts_3d = torch.from_numpy(smplx_jnts18).float() # T X 18 X 3 
        else:
            smplx_jnts_3d = smplx_jnts18 

        jnts_3d_aligned_xy_cam, jnts3d_cam, jnts2d = perspective_projection(smplx_jnts_3d, \
                cam_rot_mat, cam_trans, focal_length, \
                image_width, image_height) 

        # Get mask for invisible positions 
        x_mask = (jnts2d[:, :, 0] < 0) + (jnts2d[:, :, 0] > image_width) # T X 18
        y_mask = (jnts2d[:, :, 1] < 0) + (jnts2d[:, :, 1] > image_height) # T X 18 

        x_mask = (x_mask.sum(dim=-1) == 0) # T 
        y_mask = (y_mask.sum(dim=-1) == 0) # T

        vis_2d_mask = (x_mask * y_mask)

        seq_mask_list.append(vis_2d_mask) 

        jnts3d_aligned_xy_cam_list.append(jnts_3d_aligned_xy_cam)
        jnts3d_cam_list.append(jnts3d_cam)
        jnts2d_list.append(jnts2d)

        if return_multiview:
            cam_extrinsic_list.append(torch.from_numpy(extrinsic_matrix).float())
        else:
            cam_extrinsic_list.append(extrinsic_matrix)

    if return_multiview:
        return jnts3d_aligned_xy_cam_list, jnts3d_cam_list, jnts2d_list, \
        seq_mask_list, cam_extrinsic_list 
    else:
        # Select a view that has more visible frames 
        max_len = 0 
        max_len_view_idx = 0 
        for view_idx in range(num_views):
            curr_view_visible_len = cal_seq_len_from_mask(seq_mask_list[view_idx]) 

            if curr_view_visible_len > max_len:
                max_len = curr_view_visible_len 
                max_len_view_idx = view_idx 

        new_jnts3d_aligned_xy_cam_list = [jnts3d_aligned_xy_cam_list[max_len_view_idx]]
        new_jnts3d_cam_list = [jnts3d_cam_list[max_len_view_idx]]
        new_jnts2d_list = [jnts2d_list[max_len_view_idx]]
        new_cam_extrinsic_list = [cam_extrinsic_list[max_len_view_idx]]
        seq_mask = seq_mask_list[max_len_view_idx] 

        return new_jnts3d_aligned_xy_cam_list, new_jnts3d_cam_list, \
            new_jnts2d_list, seq_mask, new_cam_extrinsic_list  

def single_seq_get_multi_view_2d_motion_from_3d_torch(smplx_jnts18, \
    return_multiview=False, num_views=6, farest=False, unified_dist=False, use_animal_data=False, use_baby_data=False):
    if not torch.is_tensor(smplx_jnts18):
        smplx_jnts18 = torch.from_numpy(smplx_jnts18).float()

    # num_views = 6
    elevation = 0
    if farest: # FineGym 
        distance = 3 
    else:
        distance = random.uniform(2, 3)

    if unified_dist: 
        distance = 2.8 
    elif use_animal_data:
        distance = 2.5 
    elif use_baby_data:
        distance = 1

    init_azimuth = random.choice(range(360))
    offset_angle = 360/num_views

    image_width, image_height = 1920, 1080
    focal_length = 1000
    # focal_length = 1300

    results = []

    for view_idx in range(num_views):
        azimuth = init_azimuth + view_idx * offset_angle
        extrinsic_matrix = camera_extrinsic_from_azimuth_elevation_torch(distance, \
            azimuth, elevation)

        extrinsic_matrix = extrinsic_matrix.to(smplx_jnts18.device)
        
        cam_rot_mat = extrinsic_matrix[:3, :3].unsqueeze(0).repeat(len(smplx_jnts18), 1, 1)
        cam_trans = extrinsic_matrix[:3, -1].unsqueeze(0).repeat(len(smplx_jnts18), 1)

        jnts_3d_aligned_xy_cam, jnts3d_cam, jnts2d = perspective_projection(
            smplx_jnts18, cam_rot_mat, cam_trans, focal_length, image_width, image_height
        )

        x_mask = (jnts2d[..., 0] < 0) | (jnts2d[..., 0] > image_width) # 1 represents invisible 
        y_mask = (jnts2d[..., 1] < 0) | (jnts2d[..., 1] > image_height)
        vis_2d_mask = (~(x_mask | y_mask)).all(dim=-1) # T 

        results.append({
            'jnts3d_aligned_xy_cam': jnts_3d_aligned_xy_cam,
            'jnts3d_cam': jnts3d_cam,
            'jnts2d': jnts2d,
            'vis_2d_mask': vis_2d_mask,
            'extrinsic_matrix': extrinsic_matrix if return_multiview else extrinsic_matrix.numpy()
        })

    if return_multiview:
        return [r['jnts3d_aligned_xy_cam'] for r in results], \
               [r['jnts3d_cam'] for r in results], \
               [r['jnts2d'] for r in results], \
               [r['vis_2d_mask'] for r in results], \
               [r['extrinsic_matrix'] for r in results]
    else:
        max_len = max((torch.sum(r['vis_2d_mask']).item(), idx) for idx, r in enumerate(results))[1]
        selected = results[max_len]
        return [selected['jnts3d_aligned_xy_cam']], [selected['jnts3d_cam']], \
               [selected['jnts2d']], selected['vis_2d_mask'], [selected['extrinsic_matrix']]

def get_multi_view_cam_extrinsic(num_views=6, farest=False, add_elevation=False):
    # num_views = 6
    if add_elevation:
        elevation = 10
    else:
        elevation = 0
    
    if farest:
        distance = 3 
    else:
        distance = random.uniform(2, 3)
    init_azimuth = random.choice(range(360))
    offset_angle = 360/num_views

    results = []
    for view_idx in range(num_views):
        azimuth = init_azimuth + view_idx * offset_angle
        extrinsic_matrix = camera_extrinsic_from_azimuth_elevation_torch(distance, \
            azimuth, elevation)

        extrinsic_matrix = extrinsic_matrix.cuda() 
        
        # cam_rot_mat = extrinsic_matrix[:3, :3].unsqueeze(0).repeat(len(smplx_jnts18), 1, 1)
        # cam_trans = extrinsic_matrix[:3, -1].unsqueeze(0).repeat(len(smplx_jnts18), 1)

        results.append(extrinsic_matrix)

    results = torch.stack(results) # K X 4 X 4 

    return results  
   
def cal_min_seq_len_from_mask_torch(seq_mask):
    # seq_mask: K X T 
    # Assuming seq_mask is a PyTorch tensor of dtype torch.bool with shape K x T
    # Convert the Boolean condition (where seq_mask is False) to an integer tensor
    # False becomes 1 (first False) and True becomes 0
    seq_mask_int = (seq_mask == False).int()

    # Use argmax to find the first occurrence of False (1) in each row (dim=1)
    first_false_indices = torch.argmax(seq_mask_int, dim=1)

    # Check if there is any False in each sequence
    has_false = seq_mask_int.any(dim=1)

    # Compute valid lengths: if a row has False, use the index of the first False, otherwise use the row's length
    valid_lengths = torch.where(has_false, first_false_indices, torch.tensor(seq_mask.size(1), device=seq_mask.device))

    # Return the minimum of these valid lengths
    return torch.min(valid_lengths).item()  # Convert tensor to Python int

# def cal_seq_len_from_mask_torch(seq_mask):
#     # seq_mask: T 
#     # Assuming seq_mask is a PyTorch tensor of dtype torch.bool
#     # Find the first occurrence of False. torch.argmax will be used to find the index.
#     if torch.any(seq_mask == False):
#         return torch.argmax(seq_mask == False).item()  # Convert tensor to Python int
#     return len(seq_mask)  # If no False is found, the sequence length is the full length of the tensor.

def project_3d_random_view(smplx_jnts18):
    # smplx_jnts18: BS X T X 18 X 3, tensor 
    bs, seq_len, num_joints, _ = smplx_jnts18.shape 
    smplx_jnts18 = smplx_jnts18.reshape(bs*seq_len, -1, 3) # (BS*T) X 18 X 3 

    jnts2d_list = [] 

    cam_extrinsic_list = []

    seq_mask_list = [] 

    image_width = 1920
    image_height = 1080

    focal_length = 1000 

    num_views = 1 # Currently not using multi-view data, just use a random view 
    elevation = 0 
    # Sample a distance in a small range 
    # distance = random.uniform(3.5, 4) 
    distance = random.uniform(2, 3) 
    # Random sample an angle 
    init_azimuth = random.sample(list(range(360)), 1)[0]   
    offset_angle = 60 
    for view_idx in range(num_views):

        # Camera extrinsic parameters
        azimuth = init_azimuth + view_idx * offset_angle
        extrinsic_matrix = camera_extrinsic_from_azimuth_elevation(distance, azimuth, elevation)
        # 4 X 4 

        # Convert to image coordinates
        num_steps = smplx_jnts18.shape[0]
        
        cam_rot_mat = torch.from_numpy(extrinsic_matrix[:3, :3]).float()[None].repeat(num_steps, 1, 1).to(smplx_jnts18.device) # T X 3 X 3 
        cam_trans = torch.from_numpy(extrinsic_matrix[:3, -1]).float()[None].repeat(num_steps, 1).to(smplx_jnts18.device) # T X 3 

        jnts_3d_aligned_xy_cam, jnts3d_cam, jnts2d = perspective_projection(smplx_jnts18, \
                cam_rot_mat, cam_trans, focal_length, \
                image_width, image_height) 
        
        jnts2d = jnts2d.reshape(bs, seq_len, -1, 2) # BS X T X J X 2  

        # Get mask for invisible positions 
        x_mask = (jnts2d[:, :, :, 0] < 0) + (jnts2d[:, :, :, 0] > image_width) # BS X T X 18
        y_mask = (jnts2d[:, :, :, 1] < 0) + (jnts2d[:, :, :, 1] > image_height) # BS X T X 18 

        x_mask = (x_mask.sum(dim=-1) == 0) # BS X T 
        y_mask = (y_mask.sum(dim=-1) == 0) # BS X T 

        vis_2d_mask = (x_mask * y_mask)
        if torch.is_tensor(vis_2d_mask):
            vis_2d_mask = vis_2d_mask.detach().cpu().numpy() 

        seq_mask_list.append(vis_2d_mask) # BS X T 

        jnts2d_list.append(jnts2d) # BS X T X J X 2  

        extrinsic_matrix = torch.from_numpy(extrinsic_matrix).float()[None, \
                        :, :].repeat(num_steps, 1, 1) # (BS*T) X 4 X 4 
        extrinsic_matrix = extrinsic_matrix.reshape(bs, seq_len, 4, 4) 
        cam_extrinsic_list.append(extrinsic_matrix) # BS X T X 4 X 4 

    # Select a view that has more visible frames 
    max_len = 0 
    max_len_view_idx = 0 
    # for view_idx in range(num_views):
    #     import pdb 
    #     pdb.set_trace() 
    #     curr_view_visible_len = cal_seq_len_from_mask(seq_mask_list[view_idx]) 

    #     if curr_view_visible_len > max_len:
    #         max_len = curr_view_visible_len 
    #         max_len_view_idx = view_idx 

    new_jnts2d_list = [jnts2d_list[max_len_view_idx]] # [BS X T X J X 2]
    new_cam_extrinsic_list = [cam_extrinsic_list[max_len_view_idx]] # [BS X T X 4 X 4]
    seq_mask = seq_mask_list[max_len_view_idx] # BS X T

    return new_jnts2d_list[0].reshape(bs, seq_len, -1), seq_mask, new_cam_extrinsic_list[0]
    # BS X T X (J*2), BS X T, BS X T X 4 X 4 

def get_smpl_parents_w_addtional_jnts5():
    # SMPL_JOINTS = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 2, 'spine' : 3, 'leftLeg' : 4, 'rightLeg' : 5,
    #         'spine1' : 6, 'leftFoot' : 7, 'rightFoot' : 8, 'spine2' : 9, 
    #         'leftToeBase' : 10, 'rightToeBase' : 11, 'neck' : 12, 'leftShoulder' : 13, 'rightShoulder' : 14, 
    #         'head' : 15, 'leftArm' : 16, 'rightArm' : 17,
    #         'leftForeArm' : 18, 'rightForeArm' : 19, 'leftHand' : 20, 'rightHand' : 21}
    
    # Add 5 joints for projecting to joints 18. 
    # 'nose': 22, 'lefteye': 23, 'righteye': 24, 
    # 'leftear': 25, 'rightear': 26 

    parents = [-1,          0,          0,          0,          1,
                2,          3,          4,          5,          6,
                7,          8,          9,          9,          9,
               12,         13,         14,         16,         17,
               18,         19,          15,        22,         22,
               23,         24] 

    return parents

def quat_fk_torch_w_addtional_jnts5(lrot_mat, lpos):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    
    parents = get_smpl_parents_w_addtional_jnts5() 

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

def get_other_camera_extrinsics(cam_rot_mat, cam_trans, num_positions=5):
    """
    Generates an array of camera extrinsics where cameras are distributed uniformly around the 
    original point in the x-y plane, at the same height (z-coordinate), rotating around the z-axis.
    Each extrinsic is a 4x4 matrix combining rotation and translation.
    
    Args:
    cam_rot_mat (np.array): The original 3x3 camera rotation matrix.
    cam_trans (np.array): The original 3x1 camera translation vector.
    num_positions (int): Number of camera positions to generate.
    
    Returns:
    np.array: A numpy array of shape (num_positions, 4, 4) containing the extrinsic matrices.
    """
    cam_trans = cam_trans[:, 0]

    d = np.linalg.norm(cam_trans[:2])  # Distance from the origin in the x-y plane
    original_z = cam_trans[2]  # Height of the camera
    extrinsics = np.zeros((num_positions+1, 4, 4))  # Preallocate array of extrinsics

    extrinsics[0, :3, :3] = cam_rot_mat 
    extrinsics[0, :3, 3] = cam_trans 
    extrinsics[0, 3, 3] = 1. 

    for i in range(num_positions):
        theta = 2 * np.pi * (i+1) / (num_positions+1)
        x = d * np.cos(theta)
        y = d * np.sin(theta)
        new_t = np.array([x, y, original_z])
        new_t = torch.from_numpy(new_t).float() 
        
        # R_theta = np.array([
        #     [np.cos(theta), -np.sin(theta), 0],
        #     [np.sin(theta), np.cos(theta), 0],
        #     [0, 0, 1]
        # ])
        # new_R = R_theta @ cam_rot_mat  # New rotation matrix

        # # Combine into a 4x4 matrix
        # extrinsics[i+1, :3, :3] = new_R
        # extrinsics[i+1, :3, 3] = new_t
        # extrinsics[i+1, 3, 3] = 1

        # Define the target (assuming we're looking at the origin) and calculate the direction vector
        # look_at_target = torch.tensor([0, 0, 0 + original_z]).float()
        look_at_target = torch.tensor([0, 0, 0]).float()
        direction = look_at_target - new_t 
        forward = -direction / torch.norm(direction)  # Camera faces toward the negative of direction

        # World's up vector for Z-up system
        world_up = torch.tensor([0, 0, 1]).float()

        # Calculate right vector as cross product of forward vector and world's up vector
        right = torch.cross(forward, world_up)
        right /= torch.norm(right)

        # Recalculate the up vector as cross product of right and forward vector
        up = torch.cross(right, forward)
        up /= torch.norm(up)

        # Construct rotation matrix
        rotation_matrix = torch.stack([right, up, forward])

        # Create extrinsic matrix
        extrinsics[i+1, :3, :3] = rotation_matrix.transpose(0, 1)
        extrinsics[i+1, :3, 3] = new_t 
    
    return extrinsics
