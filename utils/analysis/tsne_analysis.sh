#!/bin/bash
BASE_PATH="/user/frosa/robotic/Multi-Task-LFD-Framework"
MODEL="${BASE_PATH}/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Stable-Policy-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat-simclr128x512"
EMBEDDING="${BASE_PATH}/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Stable-Policy-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat-simclr128x512/tsne_step_72900_average_False_full_traj_True_val_frames_False"
STEP=72900

# --average
# --full_traj
# --validation_frames
python3 tsne_analysis.py --model ${MODEL} --step ${STEP} --full_traj --test_folder
# python tsne_analysis.py --model ${MODEL} --step ${STEP} --full_traj