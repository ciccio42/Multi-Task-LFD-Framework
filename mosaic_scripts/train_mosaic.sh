#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_loc/.mujoco/mujoco210"
export CUDA_VISIBLE_DEVICES=4
export HYDRA_FULL_ERROR=1

EXPERT_DATA=/home/frosa_loc/Multi-Task-LFD-Framework/multitask_dataset/multitask_dataset_repo_box
SAVE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/mosaic-parameters
POLICY='${mosaic}'

EXP_NAME=1Task-Pick-Place-Mosaic-Prova
TASK_str=pick_place
EPOCH=40
BSIZE=32 #128 #64 #32
SET_SAME_N=2
COMPUTE_OBJ_DISTRIBUTION=false
CONFIG_PATH=experiments/
PROJECT_NAME="mosaic-parameters-paper-object-cropped-no-norm-box"
CONFIG_NAME=config.yaml
LOADER_WORKERS=8

LOAD_TARGET_OBJ_DETECTOR=false
TARGET_OBJ_DETECTOR_STEP=17204
TARGET_OBJ_DETECTOR_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Target-Obj-Random-Frames-Batch128-1gpu-Attn2ly128-Act2ly256mix4-actCat
FREEZE_TARGET_OBJ_DETECTOR=false
CONCAT_STATE=false

RESUME=false
RESUME_STEP=84100
RESUME_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/1Task-Pick-Place-Mosaic-cropped-no-normalized-Batch32

ACTION_DIM=8
N_MIXTURES=6
OUT_DIM=128
ATTN_FF=256
COMPRESSOR_DIM=256
HIDDEN_DIM=512
CONCAT_DEMO_HEAD=true
CONCAT_DEMO_ACT=false
PRETRAINED=false

EARLY_STOPPING_PATIECE=-1
OPTIMIZER='Adam'
LR=0.0005
WEIGHT_DECAY=0.00
SCHEDULER=None

SAVE_FREQ=1000
LOG_FREQ=1000
VAL_FREQ=1000

# Policy 1: At each slot is assigned a RandomSampler
BALANCING_POLICY=0

python /home/frosa_loc/Multi-Task-LFD-Framework/repo/mosaic/train_any.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    task_names=${TASK_str} \
    exp_name=${EXP_NAME} \
    save_freq=${SAVE_FREQ} \
    log_freq=${LOG_FREQ} \
    val_freq=${VAL_FREQ} \
    set_same_n=${SET_SAME_N} \
    bsize=${BSIZE} \
    vsize=${BSIZE} \
    epochs=${EPOCH} \
    dataset_cfg.compute_obj_distribution=${COMPUTE_OBJ_DISTRIBUTION} \
    samplers.balancing_policy=${BALANCING_POLICY} \
    mosaic.load_target_obj_detector=${LOAD_TARGET_OBJ_DETECTOR} \
    mosaic.target_obj_detector_step=${TARGET_OBJ_DETECTOR_STEP} \
    mosaic.target_obj_detector_path=${TARGET_OBJ_DETECTOR_PATH} \
    mosaic.freeze_target_obj_detector=${FREEZE_TARGET_OBJ_DETECTOR} \
    actions.adim=${ACTION_DIM} \
    actions.n_mixtures=${N_MIXTURES} \
    actions.out_dim=${OUT_DIM} \
    attn.attn_ff=${ATTN_FF} \
    simclr.compressor_dim=${COMPRESSOR_DIM} \
    simclr.hidden_dim=${HIDDEN_DIM} \
    mosaic.concat_state=${CONCAT_STATE} \
    mosaic.concat_demo_head=${CONCAT_DEMO_HEAD} \
    mosaic.concat_demo_act=${CONCAT_DEMO_ACT} \
    early_stopping_cfg.patience=${EARLY_STOPPING_PATIECE} \
    attn.img_cfg.pretrained=${PRETRAINED} \
    project_name=${PROJECT_NAME} \
    EXPERT_DATA=${EXPERT_DATA} \
    save_path=${SAVE_PATH} \
    resume_path=${RESUME_PATH} \
    resume_step=${RESUME_STEP} \
    optimizer=${OPTIMIZER} \
    train_cfg.lr=${LR} \
    train_cfg.weight_decay=${WEIGHT_DECAY} \
    train_cfg.lr_schedule=${SCHEDULER} \
    debug=false \
    wandb_log=true \
    resume=${RESUME} \
    loader_workers=${LOADER_WORKERS}
