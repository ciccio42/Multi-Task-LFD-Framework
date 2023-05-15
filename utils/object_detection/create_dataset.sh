#!/bin/bash
SAVE_FOLDER=/media/ciccio/Sandisk/object_detector
TRJ_PATH=/media/ciccio/Sandisk/multitask_dataset_ur/multitask_dataset_language_command/pick_place/
TASK_NAME=pick_place
ROBOT_NAME=ur5e
CAMERA_NAME=camera_front
NUM_WORKERS=1

python create_dataset.py --save_folder ${SAVE_FOLDER} --trj_path ${TRJ_PATH} --task_name ${TASK_NAME} --robot_name ${ROBOT_NAME} --camera_name ${CAMERA_NAME} --num_workers ${NUM_WORKERS}
