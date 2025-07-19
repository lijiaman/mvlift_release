# import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
import cv2 
import math 

from matplotlib import cm

from mpl_toolkits.mplot3d import axes3d, Axes3D


import torch 

def get_skeleton():
    OPENPOSE_LANDMARKS = [
        "nose", # 0
        "neck", # 1
        "right_shoulder", # 2  
        "right_elbow", # 3 
        "right_wrist", # 4
        "left_shoulder", # 5
        "left_elbow", # 6
        "left_wrist", # 7
        "right_hip", # 8 
        "right_knee", # 9 
        "right_ankle", # 10
        "left_hip", # 11
        "left_knee", # 12
        "left_ankle", # 13
        "right_eye", # 14
        "left_eye", # 15
        "right_ear", # 16
        "left_ear", # 17
    ]

    SKELETON_NAMES = [
        ["neck", "left_ear", "left_eye", "nose", "right_eye", "right_ear", "neck"],
        ["neck", "left_shoulder", "left_elbow", "left_wrist"],
        ["neck", "right_shoulder", "right_elbow", "right_wrist"],
        ["left_shoulder", "right_shoulder", "right_hip", "left_hip", "left_shoulder"],
        ["left_hip", "left_knee", "left_ankle"],
        ["right_hip", "right_knee", "right_ankle"]
    ]

    SKELETON = [[OPENPOSE_LANDMARKS.index(name) for name in names] for names in SKELETON_NAMES]

    return SKELETON 


def get_skeleton_new():
    OPENPOSE_LANDMARKS = [
        "nose", # 0
        "neck", # 1
        "right_shoulder", # 2  
        "right_elbow", # 3 
        "right_wrist", # 4
        "left_shoulder", # 5
        "left_elbow", # 6
        "left_wrist", # 7
        "right_hip", # 8 
        "right_knee", # 9 
        "right_ankle", # 10
        "left_hip", # 11
        "left_knee", # 12
        "left_ankle", # 13
        "right_eye", # 14
        "left_eye", # 15
        "right_ear", # 16
        "left_ear", # 17
    ]

    SKELETON_NAMES = [
        ["neck", "left_ear", "left_eye", "nose", "right_eye", "right_ear", "neck"],
        ["neck", "left_shoulder", "left_elbow", "left_wrist"],
        ["neck", "right_shoulder", "right_elbow", "right_wrist"],
        ["left_shoulder", "right_shoulder", "right_hip", "left_hip", "left_shoulder"],
        ["left_hip", "left_knee", "left_ankle"],
        ["right_hip", "right_knee", "right_ankle"]
    ]

    SKELETON = [[OPENPOSE_LANDMARKS.index(name) for name in names] for names in SKELETON_NAMES]

    return SKELETON 

def get_skeleton_jnts13():
    OPENPOSE_LANDMARKS = [
        "neck", # 1
        "right_shoulder", # 2  
        "right_elbow", # 3 
        "right_wrist", # 4
        "left_shoulder", # 5
        "left_elbow", # 6
        "left_wrist", # 7
        "right_hip", # 8 
        "right_knee", # 9 
        "right_ankle", # 10
        "left_hip", # 11
        "left_knee", # 12
        "left_ankle", # 13
    ]

    SKELETON_NAMES = [
        ["neck", "left_shoulder", "left_elbow", "left_wrist"],
        ["neck", "right_shoulder", "right_elbow", "right_wrist"],
        ["left_shoulder", "right_shoulder", "right_hip", "left_hip", "left_shoulder"],
        ["left_hip", "left_knee", "left_ankle"],
        ["right_hip", "right_knee", "right_ankle"]
    ]

    SKELETON = [[OPENPOSE_LANDMARKS.index(name) for name in names] for names in SKELETON_NAMES]

    return SKELETON 

def save_animation(anim: FuncAnimation, save_path, fps):
    save_path = f"{save_path}.mp4"
    print(f"saving to [{os.path.abspath(save_path)}]...", end="", flush=True)
    anim.save(save_path, writer="imagemagick", fps=fps)
    print("\033[0K\r\033[0K\r", end="")
    print(f"saved to [{os.path.abspath(save_path)}]")

def plot_3d_motion_new(save_path, channels, use_omomo_jnts23=False):
    # channels: K X T X n_joints X 3
    # parents: a list containing the parent joint index 
    fig = plt.figure(figsize=(9, 7))
    ax = Axes3D(fig) 

    color_list = ['#27AE60', '#6931A4', '#E74C3C'] # green, purple, red   

    vals = channels # K X T X 24 X 3, K represents how many skeleton showing in same figure(K=2: show gt and generation)
    num_cmp = vals.shape[0]

    # import pdb 
    # pdb.set_trace() 
   
    # # Generate connnections list based on parents list 
    # connections = []
    # num_joints = len(parents)
    # for j_idx in range(num_joints):
    #     if j_idx > 0:
    #         connections.append([parents[j_idx], j_idx])

    #  "nose", # 0
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

    

    if use_omomo_jnts23:
        connections = [
            [0, 1],  # nose to neck
            [1, 2],  # neck to right shoulder
            [2, 3],  # right shoulder to right elbow
            [3, 4],  # right elbow to right wrist
            [1, 5],  # neck to left shoulder
            [5, 6],  # left shoulder to left elbow
            [6, 7],  # left elbow to left wrist
            [1, 8],  # neck to right hip
            [8, 9],  # right hip to right knee
            [9, 10], # right knee to right ankle
            [1, 11], # neck to left hip
            [11, 12],# left hip to left knee
            [12, 13],# left knee to left ankle
            [14, 0], # right eye to nose
            [15, 0], # left eye to nose
            [16, 14],# right ear to right eye
            [17, 15], # left ear to left eye
            [18, 19], 
            [19, 20],
            [20, 21], 
            [21, 22],
        ]

    else:
        connections = [
            [0, 1],  # nose to neck
            [1, 2],  # neck to right shoulder
            [2, 3],  # right shoulder to right elbow
            [3, 4],  # right elbow to right wrist
            [1, 5],  # neck to left shoulder
            [5, 6],  # left shoulder to left elbow
            [6, 7],  # left elbow to left wrist
            [1, 8],  # neck to right hip
            [8, 9],  # right hip to right knee
            [9, 10], # right knee to right ankle
            [1, 11], # neck to left hip
            [11, 12],# left hip to left knee
            [12, 13],# left knee to left ankle
            [14, 0], # right eye to nose
            [15, 0], # left eye to nose
            [16, 14],# right ear to right eye
            [17, 15] # left ear to left eye
        ]


    lines = []
    for cmp_idx in range(num_cmp):
        cur_line = []
        for ind, (i,j) in enumerate(connections):
            cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=color_list[cmp_idx])[0])
        lines.append(cur_line)

    # ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], marker='o')
  
    def animate(i):
        changed = []
        for ai in range(len(vals)):
            for ind, (p_idx, j_idx) in enumerate(connections):
                lines[ai][ind].set_data([vals[ai][i, j_idx, 0], vals[ai][i, p_idx, 0]], \
                    [vals[ai][i, j_idx, 1], vals[ai][i, p_idx, 1]])
                lines[ai][ind].set_3d_properties(
                    [vals[ai][i, j_idx, 2], vals[ai][i, p_idx, 2]])
            changed += lines

        return changed

    RADIUS = 2  # space around the subject
    xroot, yroot, zroot = vals[0, 0, 11, 0], vals[0, 0, 11, 1], vals[0, 0, 11, 2]
    # xroot, yroot, zroot = 0, 0, 0 # For debug

    # ax.view_init(-90, 90) # Used in LAFAN data
    
    ax.view_init(0, 0)
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    # ax.set_axis_off()
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    ani = FuncAnimation(fig,                                            
                        animate,                                        
                        np.arange(len(vals[0])),                  
                        interval=33.33)  

    ani.save(save_path+".mp4",                       
            writer="imagemagick",          
            # dpi=120,                                      
            fps=30) 

    # plt.draw()
    # plt.savefig(dest_img_path)
    plt.cla()
    plt.close()

def get_skeleton_for_animal():
    # Define the 17 animal joints
    ANIMAL_LANDMARKS = [
        "left eye",      # 0
        "right eye",     # 1
        "nose",          # 2
        "neck",          # 3
        "root of tail",  # 4
        "left shoulder", # 5
        "left elbow",    # 6
        "left front paw",# 7
        "right shoulder",# 8
        "right elbow",   # 9
        "right front paw",# 10
        "left hip",      # 11
        "left knee",     # 12
        "left back paw", # 13
        "right hip",     # 14
        "right knee",    # 15
        "right back paw" # 16
    ]

    # Define the skeleton connections (joint pairs)
    SKELETON_NAMES = [
        ["left eye", "right eye"],
        ["left eye", "nose"],
        ["right eye", "nose"],
        ["nose", "neck"],
        ["neck", "root of tail"],
        ["neck", "left shoulder"],
        ["left shoulder", "left elbow"],
        ["left elbow", "left front paw"],
        ["neck", "right shoulder"],
        ["right shoulder", "right elbow"],
        ["right elbow", "right front paw"],
        ["root of tail", "left hip"],
        ["left hip", "left knee"],
        ["left knee", "left back paw"],
        ["root of tail", "right hip"],
        ["right hip", "right knee"],
        ["right knee", "right back paw"]
    ]

    # Create the skeleton index pairs based on the landmarks
    SKELETON = [[ANIMAL_LANDMARKS.index(name) for name in names] for names in SKELETON_NAMES]

    return SKELETON

def get_skeleton_for_omomo():
    OPENPOSE_LANDMARKS = [
        "nose", # 0
        "neck", # 1
        "right_shoulder", # 2  
        "right_elbow", # 3 
        "right_wrist", # 4
        "left_shoulder", # 5
        "left_elbow", # 6
        "left_wrist", # 7
        "right_hip", # 8 
        "right_knee", # 9 
        "right_ankle", # 10
        "left_hip", # 11
        "left_knee", # 12
        "left_ankle", # 13
        "right_eye", # 14
        "left_eye", # 15
        "right_ear", # 16
        "left_ear", # 17
        "obj_kpts1", # 18
        "obj_kpts2", # 19
        "obj_kpts3", # 20
        "obj_kpts4", # 21
        "obj_kpts5", # 22
    ]

    SKELETON_NAMES = [
        ["neck", "left_ear", "left_eye", "nose", "right_eye", "right_ear", "neck"],
        ["neck", "left_shoulder", "left_elbow", "left_wrist"],
        ["neck", "right_shoulder", "right_elbow", "right_wrist"],
        ["left_shoulder", "right_shoulder", "right_hip", "left_hip", "left_shoulder"],
        ["left_hip", "left_knee", "left_ankle"],
        ["right_hip", "right_knee", "right_ankle"], 
        ["obj_kpts1", "obj_kpts2", "obj_kpts3", "obj_kpts4", "obj_kpts5"], 
    ]

    SKELETON = [[OPENPOSE_LANDMARKS.index(name) for name in names] for names in SKELETON_NAMES]

    return SKELETON 

def plot_3d_motion(save_path, joints, title="", figsize=(5, 5), \
                fps=30, radius=3, elev=10, azim=None, \
                rotate=True, repeats=1, linewidth=1.0, use_smpl_jnts13=False, use_animal_pose17=False, use_omomo_jnts23=False):
    matplotlib.use("Agg")
    title = "\n".join(wrap(title, 20))
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)


    def init():
        ax.clear()
        ratio = figsize[0] / figsize[1]
        ax.set_xlim3d([-radius * ratio / 2, radius * ratio / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius * ratio / 3.0, radius * ratio / 3.0])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        verts = np.array(verts)
        verts[:, [0, 1, 2]] = verts[:, [0, 2, 1]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # data: [seq_len, joints_num, 3]
    data = joints.copy().reshape(len(joints), -1, 3)
    data = np.concatenate([data] * repeats, axis=0)

    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"] * 10

    frame_number = data.shape[0]

    height_offset = MINS[2]
    data[:, :, 2] -= height_offset

    trajec = np.zeros((frame_number, 2)) # # T X 2
    trajec[:, 0] = data[0, 0, 0] 
    trajec[:, 1] = data[0, 0, 1]

    data[..., [0, 1]] -= trajec.reshape(-1, 1, 2)  # place subject in the center

    if use_smpl_jnts13:
        skeleton = get_skeleton_jnts13() 
    elif use_animal_pose17:
        skeleton = get_skeleton_for_animal()
    elif use_omomo_jnts23:
        skeleton = get_skeleton_for_omomo() 
    else:
        skeleton = get_skeleton()

    def update(index):
        init()
        ax.view_init(elev=elev, azim=azim if azim is not None else (index / 300 * 360 if rotate else 0))
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], \
            0, MINS[1] - trajec[index, 1], MAXS[1] - trajec[index, 1])
        ax.scatter(data[index, :, 0], data[index, :, 1], \
            data[index, :, 2], s=3, c=colors_blue[: data.shape[1]])

        for i, (chain, color) in enumerate(zip(skeleton, colors_blue)):
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], \
                data[index, chain, 2], linewidth=linewidth, color=color)

        if not use_animal_pose17:
            # Calculate and draw the facing direction
            left_hip_idx = 11
            right_hip_idx = 8 
            right_hip = data[index, right_hip_idx] 
            left_hip = data[index, left_hip_idx]

            hip_pos = (data[index, left_hip_idx] + data[index, right_hip_idx])/2. 

            left_shoulder_idx = 5
            right_shoulder_idx = 2 
            left_shoulder = data[index, left_shoulder_idx] 
            right_shoulder = data[index, right_shoulder_idx] 

            # Calculate midpoints
            M_shoulder = (left_shoulder + right_shoulder) / 2.0
            M_hip = (left_hip + right_hip) / 2.0
            
            # Calculate vectors
            body_axis = M_shoulder - M_hip
            hip_span = right_hip - left_hip 

            facing_direction = np.cross(body_axis, hip_span)
            
            # Normalize the facing direction and body axis
            direction_vector = facing_direction / np.linalg.norm(facing_direction)
        
            ax.quiver(hip_pos[0], hip_pos[1], hip_pos[2], direction_vector[0], direction_vector[1], \
                direction_vector[2], length=1.0, color='red')

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    save_animation(ani, save_path, fps)

    plt.close()

def plot_pose_2d_for_paper(joints, dest_pose2d_fig_path):
    """
    Plot a 2D pose given joint coordinates and limb connections.

    Parameters:
    - joints: A NumPy array of shape (num_joints, 2), where each row is [x, y].
    - limbSeq: A list of lists, where each sublist represents a limb connection as [joint_index1, joint_index2].
    """
    joints[:, 0] *= 1.77 

    point_color = np.array([224, 187, 228, 255]) / 255
    line_color = np.array([149, 125, 173, 255]) / 255

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18]]

    fig, ax = plt.subplots()
    # Plot joints
    ax.scatter(joints[:, 0], joints[:, 1], c=point_color)
    
    # Draw limbs
    for limb in limbSeq:
        joint_start, joint_end = limb
        joint_start -= 1  # Adjust for zero-based indexing
        joint_end -= 1
        x_coords = [joints[joint_start][0], joints[joint_end][0]]
        y_coords = [joints[joint_start][1], joints[joint_end][1]]
        ax.plot(x_coords, y_coords, c=line_color, linewidth=5)
    
    ax.axis('off')  # Turn off the axis
    ax.set_aspect('equal', adjustable='datalim')
    ax.autoscale_view()

    ax.invert_yaxis()  # Invert the y-axis to match the common image coordinate system
    ax.set_aspect('equal', adjustable='box')  # Maintain the aspect ratio of the pose
    # plt.title('2D Pose Visualization')
    plt.savefig(dest_pose2d_fig_path, transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.savefig(dest_pose2d_fig_path)

def plot_pose_2d_omomo_for_paper(joints, dest_pose2d_fig_path):
    """
    Plot a 2D pose given joint coordinates and limb connections.

    Parameters:
    - joints: A NumPy array of shape (num_joints, 2), where each row is [x, y].
    - limbSeq: A list of lists, where each sublist represents a limb connection as [joint_index1, joint_index2].
    """
    joints[:, 0] *= 1.77 

    point_color = np.array([224, 187, 228, 255]) / 255
    line_color = np.array([149, 125, 173, 255]) / 255

    line_obj_color = np.array([222, 202, 135, 255]) / 255
    point_obj_color = np.array([245, 228, 144, 255]) / 255

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18]] 

    limbSeq_obj = [[19, 20], [19, 21], [19, 22], [19, 23]]

    # Get point color list 
    point_color_list = [] 
    for j_idx in range(18):
        point_color_list.append(point_color)
    
    for j_idx in range(5):
        point_color_list.append(point_obj_color) 

    # point_color_list = np.array(point_color_list) 

    # Get line color list 
    line_color_list = []
    for j_idx in range(17):
        line_color_list.append(line_color)
    
    for j_idx in range(4):
        line_color_list.append(line_obj_color)
    
    # line_color_list = np.array(line_color_list) 

    fig, ax = plt.subplots()
    # Plot joints
    ax.scatter(joints[:18, 0], joints[:18, 1], c=point_color)
    ax.scatter(joints[18:, 0], joints[18:, 1], c=point_obj_color)
    
    # Draw limbs
    for limb in limbSeq:
        joint_start, joint_end = limb
        joint_start -= 1  # Adjust for zero-based indexing
        joint_end -= 1
        x_coords = [joints[joint_start][0], joints[joint_end][0]]
        y_coords = [joints[joint_start][1], joints[joint_end][1]]
        ax.plot(x_coords, y_coords, c=line_color, linewidth=5)

    for limb in limbSeq_obj:
        joint_start, joint_end = limb
        joint_start -= 1  # Adjust for zero-based indexing
        joint_end -= 1
        x_coords = [joints[joint_start][0], joints[joint_end][0]]
        y_coords = [joints[joint_start][1], joints[joint_end][1]]
        ax.plot(x_coords, y_coords, c=line_obj_color, linewidth=5)
    
    ax.axis('off')  # Turn off the axis
    ax.set_aspect('equal', adjustable='datalim')
    ax.autoscale_view()

    ax.invert_yaxis()  # Invert the y-axis to match the common image coordinate system
    ax.set_aspect('equal', adjustable='box')  # Maintain the aspect ratio of the pose
    # plt.title('2D Pose Visualization')
    plt.savefig(dest_pose2d_fig_path, transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.savefig(dest_pose2d_fig_path)

def plot_pose_2d_cat_for_paper(joints, dest_pose2d_fig_path):
    """
    Plot a 2D pose given joint coordinates and limb connections.

    Parameters:
    - joints: A NumPy array of shape (num_joints, 2), where each row is [x, y].
    - limbSeq: A list of lists, where each sublist represents a limb connection as [joint_index1, joint_index2].
    """
    joints[:, 0] *= 1.77 

    point_color = np.array([223, 166, 25, 255]) / 255
    line_color = np.array([207, 145, 18, 255]) / 255

    limbSeq = [[1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [6, 7], 
            [7, 8], [4, 9], [9, 10], [10, 11], [5, 12], [12, 13], [13, 14], [5, 15], [15, 16], [16, 17]]

    fig, ax = plt.subplots()
    # Plot joints
    ax.scatter(joints[:, 0], joints[:, 1], c=point_color)
    
    # Draw limbs
    for limb in limbSeq:
        joint_start, joint_end = limb
        joint_start -= 1  # Adjust for zero-based indexing
        joint_end -= 1
        x_coords = [joints[joint_start][0], joints[joint_end][0]]
        y_coords = [joints[joint_start][1], joints[joint_end][1]]
        ax.plot(x_coords, y_coords, c=line_color, linewidth=5)
    
    ax.axis('off')  # Turn off the axis
    ax.set_aspect('equal', adjustable='datalim')
    ax.autoscale_view()

    ax.invert_yaxis()  # Invert the y-axis to match the common image coordinate system
    ax.set_aspect('equal', adjustable='box')  # Maintain the aspect ratio of the pose
    # plt.title('2D Pose Visualization')
    plt.savefig(dest_pose2d_fig_path, transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.savefig(dest_pose2d_fig_path)

def visualize_pose_and_lines_for_paper(self, pose_sequence, line_coeffs, output_folder, \
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

    # colors = [
    #     [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], 
    #     [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], 
    #     [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], 
    #     [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], 
    #     [255, 0, 170], [255, 0, 85], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], 
    #     [170, 255, 0],
    # ]
    colors = [[245, 228, 144]*J]
    colors = np.array(colors) / 255  # Scale RGB values to [0, 1]

    for t in range(T):
        plt.figure(figsize=(8, 6))
        plt.title(f"Frame {t+1}")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.grid(True)

        if epipoles is not None:
            plt.plot(epipoles[0], epipoles[1], 'o', color=colors[0]) 

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

                if j % 3 == 0:
                    plt.axvline(-c/a, color=colors[j])
                # plt.axvline(-c/a, color=colors[j])

        # Save the plot to an image file
        if vis_gt:
            plt.savefig(f"{output_folder}/frame_{t+1}_gt.png")
        else:
            plt.savefig(f"{output_folder}/frame_{t+1}.png")
        plt.close()

def draw_bodypose(canvas, candidate, use_smpl_jnts13=False, use_animal_pose17=False, use_interaction_pose23=False):
    H, W, C = canvas.shape
    candidate = np.array(candidate) # T(1) X 18 X 2 
    num_steps = candidate.shape[0] 

    stickwidth = 4

    if use_smpl_jnts13:
        num_joints = 13 
        limbSeq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], \
                [9, 10], [1, 11], [11, 12], [12, 13]]

        colors = [[255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                ]
    elif use_animal_pose17:
        num_joints = 17
        # For 17 joints, start from inex 1 
        limbSeq = [[1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [6, 7], 
            [7, 8], [4, 9], [9, 10], [10, 11], [5, 12], [12, 13], [13, 14], [5, 15], [15, 16], [16, 17]]

        # Updated colors for 20 joints
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], 
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], 
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 85, 85], [170, 85, 85]]
    elif use_interaction_pose23:
        num_joints = 23 
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18],  \
                [19, 20], [19, 21], [19, 22], [19, 23]] 

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], \
                [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0]]

    else: # 18 joints 
        num_joints = 18 
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    if use_animal_pose17:
        num_libms = num_joints
    elif use_interaction_pose23:
        num_libms = num_joints - 2 
    else:
        num_libms = num_joints - 1

    for i in range(num_libms): 
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

    for i in range(num_joints):
        for n in range(num_steps):
            x, y = candidate[n, i][0:2]
            # x = int(x * W)
            # y = int(y * H)
            x = int(x)
            y = int(y)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas

def gen_2d_motion_vis(jnts_2d, output_video_path, use_smpl_jnts13=False, use_animal_pose17=False, use_interaction_pose23=False):
    # jnts_2d: T X 18 X 2 
    image_height = 1080 
    image_width = 1920 
    fps = 30 
    size = (image_width, image_height)
    num_steps = jnts_2d.shape[0]

    video_writer = cv2.VideoWriter(output_video_path, \
        cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for t_idx in range(num_steps):
        canvas = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)
        # canvas = np.ones(shape=(image_height, image_width, 3), dtype=np.uint8)
        pose_map = draw_bodypose(canvas, jnts_2d[t_idx:t_idx+1], use_smpl_jnts13=use_smpl_jnts13, \
            use_animal_pose17=use_animal_pose17, use_interaction_pose23=use_interaction_pose23)  
        video_writer.write(pose_map)

    video_writer.release()

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

    return aligned_3d, projected_points[:, :, :-1]

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

    return points_global_space

def visualize_3d_pose(joint_positions, parents, dest_fig_path):
    """
    Visualize a 3D pose given joint positions and the hierarchy information.

    Parameters:
    joint_positions (numpy array): An array of shape (1, J, 3) containing the 3D positions of the joints.
    parents (list): A list where each index corresponds to a joint and the value at that index is the parent joint index.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Squeeze the joint_positions in case there's an unnecessary first dimension
    joint_positions = np.squeeze(joint_positions)

    # Check if joint_positions is now (J, 3)
    if joint_positions.ndim != 2 or joint_positions.shape[1] != 3:
        raise ValueError("joint_positions must be of shape (J, 3)")

    # Extract x, y, z coordinates
    x, y, z = joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2]

    # Plot the joints
    ax.scatter(x, y, z)

    # Draw lines between each joint and its parent
    for i, parent_idx in enumerate(parents):
        if parent_idx == -1:  # No parent for this joint
            continue
        # Line from parent to current joint
        ax.plot([x[parent_idx], x[i]], [y[parent_idx], y[i]], [z[parent_idx], z[i]], 'blue')

    # Setting labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Setting equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=0, azim=120) # Adjust viewing angle for better visualization

    # plt.show()
    plt.savefig(dest_fig_path)

def visualize_3d_pose_with_indices_and_skeleton(joint_positions, parents, dest_fig_path):
    """
    Visualize a 3D pose with each joint annotated by its index and connected by skeletal lines.

    Parameters:
    joint_positions (numpy array): An array of shape (J, 3) containing the 3D positions of the joints.
    parents (list or numpy array): An array of the same length as the number of joints (J) where each
                                  entry is the index of the parent joint or -1 if it has no parent.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Check if joint_positions is in the correct shape
    if joint_positions.ndim != 2 or joint_positions.shape[1] != 3:
        raise ValueError("joint_positions must be of shape (J, 3)")

    # Extract x, y, z coordinates
    x, y, z = joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2]

    # Plot the joints
    ax.scatter(x, y, z)

    # Draw lines between joints and their parents
    for i, parent_idx in enumerate(parents):
        if parent_idx != -1:  # Make sure the joint has a parent
            ax.plot([x[i], x[parent_idx]], [y[i], y[parent_idx]], [z[i], z[parent_idx]], 'blue')

    # Annotate each joint with its index
    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], f'{i}', color='red', size=5)

    # Setting labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Setting equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.view_init(elev=0, azim=120) # Adjust viewing angle for better visualization
    
    # plt.show()
    plt.savefig(dest_fig_path, dpi=300)

def plot_3d_joint_positions(joint_positions, dest_fig_path):
    """
    Plot 3D joint positions without bones.

    Parameters:
    joint_positions (numpy array): An array of shape (J, 3) containing the 3D positions of the joints.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Check if joint_positions is in the correct shape
    if joint_positions.ndim != 2 or joint_positions.shape[1] != 3:
        raise ValueError("joint_positions must be of shape (J, 3)")

    # Extract x, y, z coordinates
    x, y, z = joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2]

    # Plot the joints as scatter points
    ax.scatter(x, y, z, c='red', marker='o', s=5)  # Customize color and marker style as needed

    # Annotate each joint with its index
    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], f'{i}', color='blue', fontsize=4, ha='right')

    # Setting labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Setting equal aspect ratio for better visualization
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.title('3D Joint Positions Visualization')

    # ax.view_init(elev=0, azim=120) # Adjust viewing angle for better visualization

    ax.view_init(elev=120, azim=-90) # Adjust viewing angle for better visualization
    # plt.show()
    plt.savefig(dest_fig_path, dpi=300)

def visualize_root_trajectory_for_figure(trajectory, dest_fig_path):
    """Plots a 3D trajectory with a color gradient that reflects the time steps."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    trajectory = trajectory[::10] # Downsample for better visualization 
    
    xlim = (-0.5, 0.5)
    ylim = (-0.5, 0.5) 
    zlim = (0, 1) 

    # Set plot limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title('3D Trajectory Visualization with Time-Based Color Gradient')

    # Generate consistent colors based on time step
    num_points = trajectory.shape[0]
    cmap = cm.get_cmap("plasma", num_points)  # Consistent colormap with `num_points` colors

    # Plot each point with a unique color and add arrows
    for i in range(num_points - 1):
        # Plot the segment between consecutive points
        color = cmap(i)
        ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2], color=color, marker='o')
        
        # Calculate direction vector without scaling
        direction = trajectory[i+1] - trajectory[i]

        # Add an arrow pointing from point i to point i+1 with full segment length
        ax.quiver(
            trajectory[i, 0], trajectory[i, 1], trajectory[i, 2],
            direction[0], direction[1], direction[2],
            color=color, arrow_length_ratio=0.2  # Increased arrow length ratio for visibility
        )
    
    plt.savefig(dest_fig_path, dpi=300) 

def plot_trajectory_components_for_figure(trajectory, save_dir=""):
    """
    Saves the x, y, and z components of a 3D trajectory over time as separate images.

    Parameters:
    - trajectory (numpy.ndarray): A T x 3 array where each row represents a time step,
                                  and the columns represent the x, y, and z values respectively.
    - save_dir (str): Directory to save the images. Default is the current directory.
    """
    if ".png" in save_dir:
        save_dir = save_dir.replace(".png", "/") 

    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 

    # Ensure the trajectory has the correct shape
    if trajectory.shape[1] != 3:
        raise ValueError("Input trajectory should be a T x 3 array.")
    
    # Create time steps
    time_steps = np.arange(trajectory.shape[0])
    
    # Plot x component over time and save
    plt.figure(figsize=(40, 16))
    plt.plot(time_steps, trajectory[:, 0], label='x', color='blue')
    plt.xlabel("Time Step")
    plt.ylabel("X Value")
    plt.title("X Component of Trajectory Over Time")
    plt.grid(True)
    plt.savefig(f"{save_dir}trajectory_x.png", dpi=600)
    plt.close()
    
    # Plot y component over time and save
    plt.figure(figsize=(40, 16))
    plt.plot(time_steps, trajectory[:, 1], label='y', color='green')
    plt.xlabel("Time Step")
    plt.ylabel("Y Value")
    plt.title("Y Component of Trajectory Over Time")
    plt.grid(True)
    plt.savefig(f"{save_dir}trajectory_y.png", dpi=600)
    plt.close()
    
    # Plot z component over time and save
    plt.figure(figsize=(40, 16))
    plt.plot(time_steps, trajectory[:, 2], label='z', color='red')
    plt.xlabel("Time Step")
    plt.ylabel("Z Value")
    plt.title("Z Component of Trajectory Over Time")
    plt.grid(True)
    plt.savefig(f"{save_dir}trajectory_z.png", dpi=600)
    plt.close()

def plot_multiple_trajectories(trajectories, labels=None, save_dir=""):
    """
    Plots the x, y, and z components of multiple 3D trajectories over time in separate figures.
    Each figure will contain the corresponding component (x, y, or z) for all trajectories.

    Parameters:
    - trajectories (list of numpy.ndarray): A list of T x 3 arrays where each array represents
                                            a trajectory with time steps in rows and x, y, z values in columns.
    - labels (list of str): Optional. List of labels for each trajectory to use in the legend.
    - save_dir (str): Directory to save the images. Default is the current directory.
    """
    if ".png" in save_dir:
        save_dir = save_dir.replace(".png", "/")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
   
    # Check that all trajectories have the correct shape
    for traj in trajectories:
        if traj.shape[1] != 3:
            raise ValueError("Each trajectory should be a T x 3 array.")
    
    # Set up labels if not provided
    if labels is None:
        labels = [f"Trajectory {i+1}" for i in range(len(trajectories))]
    
    # Generate time steps (assumes all trajectories have the same length)
    time_steps = np.arange(trajectories[0].shape[0])
    
    # Define common style options for plots
    plt.style.use('seaborn-whitegrid')

    # Plot x component for all trajectories
    plt.figure(figsize=(10, 4))
    for i, traj in enumerate(trajectories):
        plt.plot(time_steps, traj[:, 0], label=labels[i], linewidth=1.5)
    plt.xlabel("Time Step")
    plt.ylabel("X Value")
    plt.title("X Component of Trajectories Over Time")
    plt.legend()
    plt.savefig(f"{save_dir}multiple_trajectory_x.png", bbox_inches='tight', transparent=True)
    plt.close()
    
    # Plot y component for all trajectories
    plt.figure(figsize=(10, 4))
    for i, traj in enumerate(trajectories):
        plt.plot(time_steps, traj[:, 1], label=labels[i], linewidth=1.5)
    plt.xlabel("Time Step")
    plt.ylabel("Y Value")
    plt.title("Y Component of Trajectories Over Time")
    plt.legend()
    plt.savefig(f"{save_dir}multiple_trajectory_y.png", bbox_inches='tight', transparent=True)
    plt.close()
    
    # Plot z component for all trajectories
    plt.figure(figsize=(10, 4))
    for i, traj in enumerate(trajectories):
        plt.plot(time_steps, traj[:, 2], label=labels[i], linewidth=1.5)
    plt.xlabel("Time Step")
    plt.ylabel("Z Value")
    plt.title("Z Component of Trajectories Over Time")
    plt.legend()
    plt.savefig(f"{save_dir}multiple_trajectory_z.png", bbox_inches='tight', transparent=True)
    plt.close()

def plot_pose_2d_for_demo(joints, limbSeq, ax, point_color, line_color):
    """
    Helper function to plot a single 2D pose on a given Matplotlib axis.
    """
    ax.scatter(joints[:, 0], joints[:, 1], c=[point_color], s=50, zorder=2)
    for limb in limbSeq:
        joint_start, joint_end = limb
        joint_start -= 1  # Adjust for zero-based indexing
        joint_end -= 1
        x_coords = [joints[joint_start][0], joints[joint_end][0]]
        y_coords = [joints[joint_start][1], joints[joint_end][1]]
        ax.plot(x_coords, y_coords, c=line_color, linewidth=5, zorder=1)

def visualize_pose_sequence_to_video_for_demo(
        pose_sequence, output_video_path, video_size=(1920, 1080), fps=30, for_animal=False):
    """
    Visualize a sequence of 2D poses and save as a video with fixed axes.

    Parameters:
    - pose_sequence: A NumPy array of shape (T, J, 2), where T is the number of frames,
      J is the number of joints, and 2 is for (x, y) coordinates.
    - output_video_path: Path to save the output video.
    - video_size: Tuple specifying the video size (width, height).
    - fps: Frames per second for the output video.
    """
    if for_animal:
        point_color = np.array([223, 166, 25, 255]) / 255
        line_color = np.array([207, 145, 18, 255]) / 255

        limbSeq = [[1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [6, 7], 
                [7, 8], [4, 9], [9, 10], [10, 11], [5, 12], [12, 13], [13, 14], [5, 15], [15, 16], [16, 17]]
    else:
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                [1, 16], [16, 18]]

        point_color = np.array([224, 187, 228, 255]) / 255
        line_color = np.array([149, 125, 173, 255]) / 255

    # Determine the global axis limits
    x_min, x_max = 0, 1920  # Fixed to the 2D image range
    y_min, y_max = 0, 1080  # Fixed to the 2D image range

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, video_size)

    fig, ax = plt.subplots(figsize=(video_size[0] / 100, video_size[1] / 100), dpi=500)
    for frame_idx in range(len(pose_sequence)):
        ax.clear()
        ax.axis('off')
        joints = pose_sequence[frame_idx]
        plot_pose_2d_for_demo(joints, limbSeq, ax, point_color, line_color)

        # Set fixed axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Invert y-axis for image-like coordinates

        # Save the frame to an image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Convert RGB to BGR (OpenCV format) and resize to video size
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, video_size)
        out.write(img)

    out.release()
    plt.close(fig)

def visualize_pose_sequence_to_video_for_demo_interaction(
        pose_sequence, output_video_path, video_size=(1920, 1080), fps=30):
    """
    Visualize a sequence of 2D poses and save as a video with fixed axes.

    Parameters:
    - pose_sequence: A NumPy array of shape (T, J, 2), where T is the number of frames,
      J is the number of joints, and 2 is for (x, y) coordinates.
    - output_video_path: Path to save the output video.
    - video_size: Tuple specifying the video size (width, height).
    - fps: Frames per second for the output video.
    """
    point_color = np.array([224, 187, 228, 255]) / 255
    line_color = np.array([149, 125, 173, 255]) / 255

    line_obj_color = np.array([222, 202, 135, 255]) / 255
    point_obj_color = np.array([245, 228, 144, 255]) / 255

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18]] 

    # limbSeq_obj = [[19, 20], [19, 21], [19, 22], [19, 23]]

    limbSeq_obj = [[0, 1], [0, 2], [0, 3], [0, 4]]

    # Get point color list 
    point_color_list = [] 
    for j_idx in range(18):
        point_color_list.append(point_color)
    
    for j_idx in range(5):
        point_color_list.append(point_obj_color) 

    # point_color_list = np.array(point_color_list) 

    # Get line color list 
    line_color_list = []
    for j_idx in range(17):
        line_color_list.append(line_color)
    
    for j_idx in range(4):
        line_color_list.append(line_obj_color)

    # Determine the global axis limits
    x_min, x_max = 0, 1920  # Fixed to the 2D image range
    y_min, y_max = 0, 1080  # Fixed to the 2D image range

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, video_size)

    fig, ax = plt.subplots(figsize=(video_size[0] / 100, video_size[1] / 100), dpi=500)
    for frame_idx in range(len(pose_sequence)):
        ax.clear()
        ax.axis('off')
        joints = pose_sequence[frame_idx]
        plot_pose_2d_for_demo(joints[:18], limbSeq, ax, point_color, line_color)

        plot_pose_2d_for_demo(joints[18:], limbSeq_obj, ax, point_obj_color, line_obj_color)

        # Set fixed axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Invert y-axis for image-like coordinates

        # Save the frame to an image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Convert RGB to BGR (OpenCV format) and resize to video size
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, video_size)
        out.write(img)

    out.release()
    plt.close(fig)
