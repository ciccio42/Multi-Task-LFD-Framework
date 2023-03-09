#!/bin/bash
BASE_PATH="/home/Multi-Task-LFD-Framework"
MODEL="${BASE_PATH}/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Stable-Policy-Fixed-Conditioning-Balanced-Training-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat-simclr128x512"
#"${BASE_PATH}/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Stable-Policy-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat-simclr128x512"
EXP_NUMBER=4
STEP=35280 #37800
TASK_ID=12 #2

RESULTS_DIR="${BASE_PATH}/utils/analysis/dataset_analysis_results/demonstrator_variation/pick_place_fixed_demo_target_left"
#NUM_WORKERS=4
#PROJECT_NAME="pick_place_fixed_demo_target_left"

python3 ${BASE_PATH}/utils/analysis/dataset_analysis.py --step ${STEP} --model ${MODEL} --task_indx ${TASK_ID} --results_dir ${RESULTS_DIR} --num_workers ${NUM_WORKERS} --experiment_number ${EXP_NUMBER} --debug #--project_name ${PROJECT_NAME} --training_trj --run_inference 
