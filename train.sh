#!/bin/sh
#export MUJOCO_PY_MUJOCO_PATH="/user/frosa/robotic/.mujoco/mujoco210"
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
# 1 - BASELINE - mosaic experiment
#EXP_NAME=Task-NutAssembly
#TASK_str=nut_assembly
#EPOCH=20
#BSIZE=27
#python repo/mosaic/train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=${EXP_NAME} bsize=${BSIZE} vsize=${BSIZE} epochs=${EPOCH}

# 2 - BASELINE - mosaic experiment
EXP_NAME=1Task-Pick-Place-Pre-Obj-Embedding
TASK_str=pick_place
EPOCH=250
BSIZE=16 #64 #32
CONFIG_PATH=experiments/
PROJECT_NAME="adam_lr_obj_pretrain"
CONFIG_NAME=config.yaml

LOAD_TARGET_OBJ_DETECTOR=true
TARGET_OBJ_DETECTOR_STEP=22275
TARGET_OBJ_DETECTOR_PATH=/home/ciccio/Desktop/multi_task_lfd/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Target-Obj-Batch64-1gpu-Attn2ly128-Act2ly256mix4-actCat-simclr128x512
FREEZE_TARGET_OBJ_DETECTOR=true

python repo/mosaic/train_any.py --config-path ${CONFIG_PATH} --config-name ${CONFIG_NAME} policy='${mosaic}' task_names=${TASK_str} exp_name=${EXP_NAME} bsize=${BSIZE} vsize=${BSIZE} epochs=${EPOCH} mosaic.load_target_obj_detector=${LOAD_TARGET_OBJ_DETECTOR} mosaic.target_obj_detector_step=${TARGET_OBJ_DETECTOR_STEP} mosaic.target_obj_detector_path=${TARGET_OBJ_DETECTOR_PATH} mosaic.freeze_target_obj_detector=${FREEZE_TARGET_OBJ_DETECTOR} project_name=${PROJECT_NAME} debug=false wandb_log=false resume=false

# 3 - BASELINE - mosaic experiment
# EXP_NAME=Task-Stack-Block
# TASK_str=stack_block
# EPOCH=20
# BSIZE=30 #32
# python repo/mosaic/train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=${EXP_NAME} bsize=${BSIZE} vsize=${BSIZE} epochs=${EPOCH}
