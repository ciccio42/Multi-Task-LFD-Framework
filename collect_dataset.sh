#!/bin/bash
# task name, possible values [door, drawer, basketball, nut_assembly, 
#                              stack_block, pick_place, button
#                              stack_new_color, stack_new_shape]
export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# path to folder where save trajectories
BASEPATH=~/Desktop/multi_task_lfd/Multi-Task-LFD-Framework
PATH_TO_DATA=~/Desktop/multi_task_lfd/
SUITE=${PATH_TO_DATA}/multitask_dataset_language_command
echo ${SUITE}
WORKERS=1 # number of workers
GPU_ID_INDX=0
SCRIPT=$BASEPATH/repo/Multi-Task-LFD-Training-Framework/tasks/collect_data/collect_task.py
PATH_TO_CONTROL_CONFIG=$BASEPATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json 
echo ${SUITE}
echo "---- Start to collect dataset ----"


TASK_name=pick_place  ## NOTE different size
N_tasks=16
NUM=1600
N_env=800
per_task=100
for ROBOT in panda ur5e
do
 python3 ${SCRIPT} ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
         -tsk ${TASK_name} -ro ${ROBOT} --n_tasks ${N_tasks}  --n_env ${N_env} \
         --N ${NUM} --per_task_group ${per_task} \
         --num_workers ${WORKERS} \
         --overwrite \
         --ctrl_config ${PATH_TO_CONTROL_CONFIG} \
         --renderer \
         #--debug 
         #--collect_cam \
         #--debug \
         
done 
