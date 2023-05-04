#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_loc/.mujoco/mujoco210"
export CUDA_VISIBLE_DEVICES=2
export HYDRA_FULL_ERROR=1

# 1 - BASELINE - mosaic experiment
#EXP_NAME=Task-NutAssembly
#TASK_str=nut_assembly
#EPOCH=20
#BSIZE=27
#python repo/mosaic/train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=${EXP_NAME} bsize=${BSIZE} vsize=${BSIZE} epochs=${EPOCH}

# 2 - BASELINE - mosaic experiment
EXPERT_DATA=/raid/home/miviacluster/data/frosa/multitask_dataset/multitask_dataset_baseline
SAVE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/baseline-1
POLICY='${mosaic}'

EXP_NAME=1Task-Pick-Place-Mosaic-Baseline
TASK_str=pick_place
EPOCH=250
BSIZE=128 #64 #32
CONFIG_PATH=experiments/
PROJECT_NAME="mosaic-baseline-box"
CONFIG_NAME=config.yaml
LOADER_WORKERS=20

LOAD_TARGET_OBJ_DETECTOR=false
TARGET_OBJ_DETECTOR_STEP=17204
TARGET_OBJ_DETECTOR_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Target-Obj-Random-Frames-Batch128-1gpu-Attn2ly128-Act2ly256mix4-actCat
FREEZE_TARGET_OBJ_DETECTOR=false

RESUME_PATH=None
RESUME_STEP=-1

python repo/mosaic/train_any.py --config-path ${CONFIG_PATH} --config-name ${CONFIG_NAME} policy=${POLICY} task_names=${TASK_str} exp_name=${EXP_NAME} bsize=${BSIZE} vsize=${BSIZE} epochs=${EPOCH} mosaic.load_target_obj_detector=${LOAD_TARGET_OBJ_DETECTOR} mosaic.target_obj_detector_step=${TARGET_OBJ_DETECTOR_STEP} mosaic.target_obj_detector_path=${TARGET_OBJ_DETECTOR_PATH} mosaic.freeze_target_obj_detector=${FREEZE_TARGET_OBJ_DETECTOR} project_name=${PROJECT_NAME} EXPERT_DATA=${EXPERT_DATA} save_path=${SAVE_PATH} resume_path=${RESUME_PATH} resume_step=${RESUME_STEP} debug=false wandb_log=true resume=false loader_workers=${LOADER_WORKERS}

# 3 - BASELINE - mosaic experiment
# EXP_NAME=Task-Stack-Block
# TASK_str=stack_block
# EPOCH=20
# BSIZE=30 #32
# python repo/mosaic/train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=${EXP_NAME} bsize=${BSIZE} vsize=${BSIZE} epochs=${EPOCH}
