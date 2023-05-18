#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

SAVE_FOLDER=/home/frosa_loc/object_detector
TRJ_PATH=/home/frosa_loc/multitask_dataset_ur/multitask_dataset_language_command/pick_place/
TASK_NAME=pick_place
ROBOT_NAME=ur5e
CAMERA_NAME=camera_front
NUM_WORKERS=30
SPLIT_TRAINIG_VALIDATION=true
MODEL=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/1Task-Pick-Place-Tosil-No-Obj-Detector-Batch128

python create_dataset.py --save_folder ${SAVE_FOLDER} --trj_path ${TRJ_PATH} --task_name ${TASK_NAME} --robot_name ${ROBOT_NAME} --camera_name ${CAMERA_NAME} --num_workers ${NUM_WORKERS} --split_trainig_validation --model ${MODEL}
