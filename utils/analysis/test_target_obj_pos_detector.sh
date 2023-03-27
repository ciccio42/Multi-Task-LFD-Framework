#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
BASE_PATH="/home/ciccio/Desktop/multi_task_lfd/Multi-Task-LFD-Framework"
MODEL="${BASE_PATH}/mosaic-baseline-sav-folder/1Task-Pick-Place-Target-Obj-Batch64-1gpu-Attn2ly128-Act2ly256mix4-actCat-simclr128x512"
EXP_NUMBER=5
STEP=34425 #35280 #37800

RESULTS_DIR="${BASE_PATH}/utils/analysis/dataset_analysis_results/target_object_detector"
NUM_WORKERS=1
PROJECT_NAME="target_object_detector"

python3 ${BASE_PATH}/utils/analysis/test_target_obj_detector.py --step ${STEP} --model ${MODEL} --results_dir ${RESULTS_DIR} --num_workers ${NUM_WORKERS} --experiment_number ${EXP_NUMBER} --debug