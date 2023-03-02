#!/bin/bash
BASE_PATH="/home/Multi-Task-LFD-Framework"
MODEL="${BASE_PATH}/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Stable-Policy-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat-simclr128x512"
RESULTS_DIR="${BASE_PATH}/utils/analysis/dataset_analysis_results"
TASK_ID=12
NUM_WORKERS=6
STEP=72900
python3 ${BASE_PATH}/utils/analysis/dataset_analysis.py --step ${STEP} --model ${MODEL} --task_indx ${TASK_ID} --results_dir ${RESULTS_DIR} --num_workers ${NUM_WORKERS} --training_trj --debug