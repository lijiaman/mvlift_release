import os 
import math
import numpy as np
import matplotlib
import cv2
import tqdm 

def vis_pose2d_seq(motion_2d_data, dest_vid_path):
    # motion_2d_data: T X 18 X 2 ([0, 1] range)
    num_steps = motion_2d_data.shape[0]
    pose2d_map_list = []
    for t_idx in range(num_steps):
        pose2d_map = draw_bodypose(motion_2d_data[t_idx])

        pose2d_map_list.append(pose2d_map)

    write_pose2d_to_video(pose2d_map_list, dest_vid_path)

def draw_bodypose(candidate):
    # candidate: 18 X 2 
    canvas = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
    H, W, C = canvas.shape
    candidate = np.array(candidate)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        index = np.array(limbSeq[i]) - 1
        if -1 in index:
            continue
        Y = candidate[index.astype(int), 0] * float(W)
        X = candidate[index.astype(int), 1] * float(H)
       
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        x, y = candidate[i][0:2]
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas

def write_pose2d_to_video(pose2d_map, output_video_path, fps=25, size=[512, 512]):
    # pose2d_map: T X 512 X 512 
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    num_steps = len(pose2d_map)
    for idx in range(num_steps):
        video_writer.write(pose2d_map[idx])

    video_writer.release()
