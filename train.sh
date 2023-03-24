#!/bin/sh
#export MUJOCO_PY_MUJOCO_PATH="/user/frosa/robotic/.mujoco/mujoco210"
export CUDA_VISIBLE_DEVICES=0
# 1 - BASELINE - mosaic experiment
#EXP_NAME=Task-NutAssembly
#TASK_str=nut_assembly
#EPOCH=20
#BSIZE=27
#python repo/mosaic/train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=${EXP_NAME} bsize=${BSIZE} vsize=${BSIZE} epochs=${EPOCH}

# 2 - BASELINE - mosaic experiment
EXP_NAME=1Task-Pick-Place-Balanced-Dataset-Adam-No-Strong-Aug-No-Crop-Twice-RGB-lr-schedule
TASK_str=pick_place
EPOCH=250
BSIZE=64 #32
CONFIG_PATH=experiments/ #/user/frosa/robotic/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Balanced-Dataset-Adam-No-Strong-Aug-No-Crop-Twice-RGB-lr-schedule-Batch64-1gpu-Attn2ly128-Act2ly256mix4-actCat-simclr128x512/
CONFIG_NAME=config.yaml
python repo/mosaic/train_any.py --config-path ${CONFIG_PATH} --config-name ${CONFIG_NAME} policy='${mosaic}' task_names=${TASK_str} exp_name=${EXP_NAME} bsize=${BSIZE} vsize=${BSIZE} epochs=${EPOCH} debug=true wandb_log=false resume=false

# 3 - BASELINE - mosaic experiment
# EXP_NAME=Task-Stack-Block
# TASK_str=stack_block
# EPOCH=20
# BSIZE=30 #32
# python repo/mosaic/train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=${EXP_NAME} bsize=${BSIZE} vsize=${BSIZE} epochs=${EPOCH}