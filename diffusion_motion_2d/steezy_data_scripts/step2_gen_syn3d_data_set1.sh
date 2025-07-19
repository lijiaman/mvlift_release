#!/bin/bash
#SBATCH --partition=move --account=move --qos=normal
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=move2

#SBATCH --gres=gpu:1

#SBATCH --job-name="gen_synthetic_3d_steezy"
#SBATCH --output=/move/u/jiamanli/github/mofitness/diffusion_motion_2d/final_cvpr25_slurm_logs/slurm_output_gen_synthetic_3d_steezy_train.log

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

source ~/.bashrc
export PATH="/viscam/u/jiamanli/my_gcc/gcc-5.4.0/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-11.3"
cd /move/u/jiamanli/github/mofitness/diffusion_motion_2d 
conda activate goal

python trainer_youtube_motion_2d.py \
--project="/move/u/jiamanli/mofitness_out_cvpr25" \
--exp_name="final_cvpr25_steezy_2d_w_line_cond_set1" \
--wandb_pj_name="cvpr25_line_cond_motion_2d_diffusion" \
--entity="jiamanli" \
--window=120 \
--batch_size=32 \
--youtube_train_val_json_path="/viscam/projects/mofitness/datasets/steezy_new/processed_data/train_val_split.json" \
--youtube_data_npz_folder="/viscam/projects/mofitness/datasets/steezy_new/clip_vit_pose2d_res" \
--train_2d_diffusion_w_line_cond \
--test_sample_res \
--test_2d_diffusion_w_line_cond \
--gen_more_consistent_2d_seq \
--gen_synthetic_3d_data_w_our_model \
--eval_w_best_mpjpe
