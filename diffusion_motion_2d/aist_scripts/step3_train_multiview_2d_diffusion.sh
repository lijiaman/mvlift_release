python trainer_multiview_motion2d.py \
--project="/viscam/projects/mofitness/for_mvlift_release/mvlift_out" \
--exp_name="aist_2d_mv_diffusion_set1" \
--wandb_pj_name="mvlift_release_step3_mv_2d_diffusion" \
--entity="jiamanli" \
--window=120 \
--batch_size=32 \
--youtube_data_npz_folder="/viscam/projects/mofitness/for_mvlift_release/datasets/AIST/synthetic_gen_3d_data" \
--train_num_views=4 