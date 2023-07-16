#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH=/user/frosa/robotic/.mujoco/mujoco210
export CUDA_VISIBLE_DEVICES=2

EXP_NAME=3Task-Pick-Nut-Stack
CONFIG_PATH="/user/frosa/robotic/Multi-Task-LFD-Framework/experiments"
CONFIG_NAME="3_tasks_config.yaml"
EPOCH=50
SAME_N=3
python repo/mosaic/train_any.py --config-path ${CONFIG_PATH} --config-name ${CONFIG_NAME} policy='${mosaic}' exp_name=${EXP_NAME} set_same_n=${SAME_N} epochs=${EPOCH}