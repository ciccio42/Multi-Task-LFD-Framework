#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0
MODEL=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/1Task-Pick-Place-Mosaic-200-360-Target-Obj-Detector-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat
# 1: Run inference from a given sample list
# 2: Run inference for a specific sub-task
# 3: Given a sample indices, return the corresponding agent and demo files
# 4:
# 5: Run classic inference with random environment
# 6: Load trajectories from rollout
EXP_NUMBER=5
STEP=13000

RESULTS_DIR="/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/1Task-Pick-Place-Mosaic-200-360-Target-Obj-Detector-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat"
NUM_WORKERS=1
PROJECT_NAME=None
BASE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
TASK=pick_place

python test_target_obj_detector.py --step ${STEP} --model ${MODEL} --results_dir ${RESULTS_DIR} --env $TASK --num_workers ${NUM_WORKERS} --controller_path ${CONTROLLER_PATH} --experiment_number ${EXP_NUMBER}
