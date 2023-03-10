#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH="/user/frosa/robotic/.mujoco/mujoco210"
export CUDA_VISIBLE_DEVICES=2
# 1 - BASELINE - mosaic experiment
#EXP_NAME=Task-NutAssembly
#TASK_str=nut_assembly
#EPOCH=20
#BSIZE=27
#python repo/mosaic/train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=${EXP_NAME} bsize=${BSIZE} vsize=${BSIZE} epochs=${EPOCH}

# 2 - BASELINE - mosaic experiment
EXP_NAME=1Task-Pick-Place-Balanced-Dataset-RMSProp
TASK_str=pick_place
EPOCH=250
BSIZE=64 #32
python repo/mosaic/train_any.py policy='${mosaic}' task_names=${TASK_str} exp_name=${EXP_NAME} bsize=${BSIZE} vsize=${BSIZE} epochs=${EPOCH} debug=false wandb_log=true

# 3 - BASELINE - mosaic experiment
# EXP_NAME=Task-Stack-Block
# TASK_str=stack_block
# EPOCH=20
# BSIZE=30 #32
# python repo/mosaic/train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=${EXP_NAME} bsize=${BSIZE} vsize=${BSIZE} epochs=${EPOCH}