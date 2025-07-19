import numpy as np
import os 
import math 
import cv2 
import random 
import pickle as pkl 

import torch 

def perspective_projection(points, rotation, translation,
                           focal_length=1000, image_width=1920, image_height=1080):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    cx, cy = image_width / 2, image_height / 2
    K = torch.zeros([batch_size, 3, 4], device=points.device)
    K[:, 0, 0] = K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    
    # Transform to camera coordinates
    points_cam = torch.einsum('bij,bkj->bki', rotation, points) + translation.unsqueeze(1)
    
    # Convert to homogeneous coordinates for projection
    ones = torch.ones((points_cam.shape[0], points_cam.shape[1], 1), device=points.device)
    points_cam_homog = torch.cat([points_cam, ones], dim=-1)
    
    # Apply camera intrinsics
    points_img_homog = torch.einsum('bij,bkj->bki', K, points_cam_homog)
    
    # Perspective division
    epsilon = 1e-6 
    safe_z = points_img_homog[:, :, 2].unsqueeze(-1).float() + epsilon 
    points_img = points_img_homog[:, :, :2].float() / safe_z 

    return points_img

def perspective_projection_from_cam(points_cam, focal_length=1000, \
        image_width=1920, image_height=1080):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points_cam.shape[0]
    cx, cy = image_width / 2, image_height / 2
    K = torch.zeros([batch_size, 3, 4], device=points_cam.device)
    K[:, 0, 0] = K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    
    # Transform to camera coordinates
    # points_cam = torch.einsum('bij,bkj->bki', rotation, points) + translation.unsqueeze(1)

    # Convert to homogeneous coordinates for projection
    ones = torch.ones((points_cam.shape[0], points_cam.shape[1], 1), device=points_cam.device)
    points_cam_homog = torch.cat([points_cam, ones], dim=-1)
    
    # Apply camera intrinsics
    points_img_homog = torch.einsum('bij,bkj->bki', K, points_cam_homog)
    
    # Perspective division
    epsilon = 1e-6 
    safe_z = points_img_homog[:, :, 2].unsqueeze(-1).float() + epsilon 
    points_img = points_img_homog[:, :, :2].float() / safe_z 

    return points_img

def camera_to_world_jnts3d(points_camera_space, rotation, translation):

    # Apply inverse transformation (rotation and translation)
    rotation_inv = rotation.transpose(1, 2)
    translation_inv = -torch.einsum('bij,bj->bi', rotation_inv, translation)

    # Move points back to global space
    points_global_space = torch.einsum('bij,bkj->bki', rotation_inv, points_camera_space - translation_inv.unsqueeze(1))

    return points_global_space
    