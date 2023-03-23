#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
BASE_PATH="/home/Multi-Task-LFD-Framework"
MODEL="${BASE_PATH}/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Balanced-Dataset-Adam-No-Strong-Aug-No-Crop-Twice-RGB-lr-schedule-Batch64-1gpu-Attn2ly128-Act2ly256mix4-actCat-simclr128x512"
EXP_NUMBER=1
STEP=232875 #35280 #37800
TASK_ID=12 #2

RESULTS_DIR="${BASE_PATH}/utils/analysis/dataset_analysis_results/tr_overfitting/pick_place_adam_lr"
NUM_WORKERS=5
PROJECT_NAME="pick_place_adam_rgb"

python3 ${BASE_PATH}/utils/analysis/dataset_analysis.py --step ${STEP} --model ${MODEL} --task_indx ${TASK_ID} --results_dir ${RESULTS_DIR} --num_workers ${NUM_WORKERS} --experiment_number ${EXP_NUMBER}  --training_trj --project_name ${PROJECT_NAME}
