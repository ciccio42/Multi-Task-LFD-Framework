#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
BASE_PATH="/media/ciccio/Sandisk/mosaic-baseline-sav-folder/mosaic-baseline-sav-folder/baseline-1"
MODEL="${BASE_PATH}/1Task-Pick-Place-Target-Obj-Batch64-1gpu-Attn2ly128-Act2ly256mix4-actCat-simclr128x512"
# 1: Run inference from a given sample list
# 2: Run inference for a specific sub-task
# 3: Given a sample indices, return the corresponding agent and demo files
# 4:
# 5: Run classic inference with random environment
# 6: Load trajectories from rollout
EXP_NUMBER=5
STEP=22275 #35280 #37800

RESULTS_DIR="${BASE_PATH}/policy_with_obj_detector_frame0"
NUM_WORKERS=1
PROJECT_NAME=None

python test_target_obj_detector.py --step ${STEP} --model ${MODEL} --results_dir ${RESULTS_DIR} --num_workers ${NUM_WORKERS} --experiment_number ${EXP_NUMBER}
