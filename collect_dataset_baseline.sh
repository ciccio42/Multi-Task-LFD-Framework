#!/bin/bash
# task name, possible values [door, drawer, basketball, nut_assembly,
#                              stack_block, pick_place, button
#                              stack_new_color, stack_new_shape]

# path to folder where save trajectories
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

BASEPATH=/home/frosa_loc/Multi-Task-LFD-Framework
PATH_TO_DATA=$BASEPATH/multitask_dataset/multitask_dataset_repo
SUITE=${PATH_TO_DATA}
echo ${SUITE}
WORKERS=4 # number of workers
GPU_ID_INDX=0
SCRIPT=$BASEPATH/repo/mosaic/tasks/collect_data/collect_any_task.py
echo "---- Start to collect dataset ----"

# ---- Pick-place ----
TASK_name=pick_place ## NOTE different size
N_tasks=16
NUM=1600
N_env=800
per_task=100
HEIGHT=200
WIDTH=360
for ROBOT in sawyer panda; do
        python ${SCRIPT} ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
                -tsk ${TASK_name} -ro ${ROBOT} --n_tasks ${N_tasks} --n_env ${N_env} \
                --N ${NUM} --per_task_group ${per_task} \
                --num_workers ${WORKERS} --collect_cam \
                --heights ${HEIGHT} --widths ${WIDTH} \
                --overwrite
done
