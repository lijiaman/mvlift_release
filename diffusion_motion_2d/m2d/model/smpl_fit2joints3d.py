import os
import pickle

import numpy as np
from smplx import SMPL
import torch

np.random.seed(0)
torch.manual_seed(0)

def compute_time_loss(poses):
    
    pose_delta = poses[1:] - poses[:-1]
    time_loss = torch.linalg.norm(pose_delta, ord=2)
    return time_loss

class SMPLRegressor:
  """SMPL fitting based on 3D keypoints."""

  def __init__(self, smpl_model_path, smpl_model_gener='MALE'):
    # Fitting hyper-parameters
    self.base_lr = 100.0
    self.niter = 10000
    self.metric = torch.nn.MSELoss()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.smpl_model_path = smpl_model_path
    self.smpl_model_gender = smpl_model_gener

    # Mapping to unify joint definations
    # self.joints_mapping_smpl = unify_joint_mappings(dataset='smpl')

  def get_optimizer(self, smpl, step, base_lr):
    """Setup opimizer with a warm up learning rate."""
    if step < 100:
      optimizer = torch.optim.SGD([
          {'params': [smpl.transl], 'lr': base_lr},
          {'params': [smpl.scaling], 'lr': base_lr * 0.01},
          {'params': [smpl.global_orient], 'lr': 0.0},
          {'params': [smpl.body_pose], 'lr': 0.0},
          {'params': [smpl.betas], 'lr': 0.0},
      ])

    elif step < 400:
      optimizer = torch.optim.SGD([
          {'params': [smpl.transl], 'lr': base_lr},
          {'params': [smpl.scaling], 'lr': base_lr * 0.01},
          {'params': [smpl.global_orient], 'lr': base_lr * 0.001},
          {'params': [smpl.body_pose], 'lr': 0.0},
          {'params': [smpl.betas], 'lr': 0.0},
      ])

    else:
      optimizer = torch.optim.SGD([
          {'params': [smpl.transl], 'lr': base_lr},
          {'params': [smpl.scaling], 'lr': base_lr * 0.01},
          # {'params': [smpl.global_orient], 'lr': base_lr * 0.001},
          # {'params': [smpl.body_pose], 'lr': base_lr * 0.001},
          {'params': [smpl.global_orient], 'lr': base_lr * 1},
          {'params': [smpl.body_pose], 'lr': base_lr * 1},
          # {'params': [smpl.body_pose], 'lr': base_lr * 0.01},
          {'params': [smpl.betas], 'lr': 0.0},
      ])
    return optimizer

  def fit(self, keypoints3d, verbose=True):
    """Run fitting to optimize the SMPL parameters."""
    assert len(keypoints3d.shape) == 3, 'input shape should be [N, njoints, 3]'
    # mapping_target = unify_joint_mappings(dataset=dtype)
    # keypoints3d = keypoints3d[:, mapping_target, :]

    # Apply rotation back to put human on floor z = 0 
    # align_rot_mat = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    align_rot_mat = np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    keypoints3d = np.dot(keypoints3d.reshape(-1, 3), \
      align_rot_mat.T).reshape(-1, 18, 3)

    keypoints3d = torch.from_numpy(keypoints3d).float().to(self.device) # T X 18 X 3 

    # Remove neck joints since the neck joint is actually computed by taking mean for lshoulder and rshoulder
    # keypoints3d = torch.cat((keypoints3d[:, 0:1, :], keypoints3d[:, 2:, :]), dim=1) 

    batch_size, njoints = keypoints3d.shape[0:2]

    debug = True 
    if debug:
      from m2d.vis.vis_jnts import plot_3d_joint_positions 
      dest_target3d_vis_folder = "./opt_target3d_jnts_vis"
      if not os.path.exists(dest_target3d_vis_folder):
        os.makedirs(dest_target3d_vis_folder)
      for t_idx in range(batch_size):
        dest_vid_3d_path = os.path.join(dest_target3d_vis_folder, "%05d"%(t_idx)+".png")
        plot_3d_joint_positions(keypoints3d[t_idx].detach().cpu().numpy(), dest_vid_3d_path) 

    # Init learnable smpl model
    smpl = SMPL(
        model_path=self.smpl_model_path,
        gender=self.smpl_model_gender,
        batch_size=batch_size).to(self.device)

    # Start fitting
    for step in range(self.niter):
      optimizer = self.get_optimizer(smpl, step, self.base_lr)

      output = smpl.forward()
      # smpl2jnts17_idx = [24, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, \
      #           25, 26, 27, 28] 
      # note SMPL needs to be "left-right flipped" to be consistent
      # with others

      smpl2jnts18_idx = [24, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28]

      # smpl2jnts17_idx = [24, 16, 18, 20, 17, 19, 21, 1, 4, 7, 2, 5, 8, 26, 25, 28, 27]

      # smpl2jnts17_idx = [24, 16, 18, 20, 17, 19, 21, 2, 5, 8, 1, 4, 7, 26, 25, 28, 27]

      # Remove neck joints for SMPL joints fitting 
      # joints18 names:  "nose", # 0
      # "neck", # 1
      # "right_shoulder", # 2  
      # "right_elbow", # 3 
      # "right_wrist", # 4
      # "left_shoulder", # 5
      # "left_elbow", # 6
      # "left_wrist", # 7
      # "right_hip", # 8 
      # "right_knee", # 9 
      # "right_ankle", # 10
      # "left_hip", # 11
      # "left_knee", # 12
      # "left_ankle", # 13
      # "right_eye", # 14
      # "left_eye", # 15
      # "right_ear", # 16
      # "left_ear", # 17
      # joints = output.joints[:, self.joints_mapping_smpl[:njoints], :]

      joints = output.joints[:, smpl2jnts18_idx, :] # T X 17 X 3 

      loss_jnts = self.metric(joints, keypoints3d)

      if step > 500:
        loss_time = compute_time_loss(smpl.body_pose) * 1e-3
      else:
        loss_time = torch.zeros(1).cuda()  

      loss = loss_jnts + loss_time 

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if verbose and step % 10 == 0:
        print("Step:{0}, SMPL Fitting Loss:{1}, Time Loss:{2}, Total Loss:{3}".format(step, \
          loss_jnts.item(), loss_time.item(), loss.item()))

      # if FLAGS.visualize:
      #   vertices = output.vertices[0].detach().cpu().numpy()  # first frame
      #   mesh = trimesh.Trimesh(vertices, smpl.faces)
      #   mesh.visual.face_colors = [200, 200, 250, 100]
      #   pts = vedo.Points(keypoints3d[0].detach().cpu().numpy(), r=20)  # first frame
      #   vedo.show(mesh, pts, interactive=False)

    if debug:
      dest_target3d_vis_folder = "./opt_smpl_opt_jnts_vis"
      if not os.path.exists(dest_target3d_vis_folder):
        os.makedirs(dest_target3d_vis_folder)
      for t_idx in range(batch_size):
        dest_vid_3d_path = os.path.join(dest_target3d_vis_folder, "%05d"%(t_idx)+".png")
        plot_3d_joint_positions(joints[t_idx].detach().cpu().numpy(), dest_vid_3d_path) 

        dest_vid_3d_path = os.path.join(dest_target3d_vis_folder, "%05d"%(t_idx)+"_ori.png")
        plot_3d_joint_positions(output.joints[t_idx].detach().cpu().numpy(), dest_vid_3d_path) 

    # Return results
    return smpl, loss.item()


def main(_):
  if FLAGS.visualize:
    assert SUPPORT_VIS, "--visualize is not support! Fail to import vedo or trimesh."

  aist_dataset = AISTDataset(FLAGS.anno_dir)
  smpl_regressor = SMPLRegressor(FLAGS.smpl_dir, 'MALE')

  if FLAGS.sequence_names:
    seq_names = FLAGS.sequence_names
  else:
    seq_names = aist_dataset.mapping_seq2env.keys()

  for seq_name in seq_names:
    logging.info('processing %s', seq_name)

    # load 3D keypoints
    keypoints3d = AISTDataset.load_keypoint3d(
        aist_dataset.keypoint3d_dir, seq_name, use_optim=True)

    # SMPL fitting
    if FLAGS.data_type == "internal":
      smpl, loss = smpl_regressor.fit(keypoints3d, dtype='coco', verbose=True)
    elif FLAGS.data_type == "openpose":
      smpl, loss = smpl_regressor.fit(keypoints3d, dtype='openpose25', verbose=True)
    else:
      raise ValueError(FLAGS.data_type)

    # One last time forward
    with torch.no_grad():
      _ = smpl.forward()
    body_pose = smpl.body_pose.detach().cpu().numpy()
    global_orient = smpl.global_orient.detach().cpu().numpy()
    smpl_poses = np.concatenate([global_orient, body_pose], axis=1)
    smpl_scaling = smpl.scaling.detach().cpu().numpy()
    smpl_trans = smpl.transl.detach().cpu().numpy()

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    motion_file = os.path.join(FLAGS.save_dir, f'{seq_name}.pkl')
    with open(motion_file, 'wb') as f:
      pickle.dump({
          'smpl_poses': smpl_poses,
          'smpl_scaling': smpl_scaling,
          'smpl_trans': smpl_trans,
          'smpl_loss': loss,
      }, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  app.run(main)