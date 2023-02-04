#!/bin/bash
# task name, possible values [door, drawer, basketball, nut_assembly, 
#                              stack_block, pick_place, button
#                              stack_new_color, stack_new_shape]

cd repo/Multi-Task-LFD-Training-Framework
# path to folder where save trajectories
PATH_TO_DATA="/home/multitask_dataset" #"/home/ciccio/Desktop/multitask_dataset"
# path to the controller configuration file
PATH_TO_CONTROL_CONFIG="/home/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/tasks/robosuite_env/controllers/config/osc_pose.json" #"/home/ciccio/Desktop/multi_task_lfd/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/tasks/robosuite_env/controllers/config/osc_pose.json"
SUITE=${PATH_TO_DATA}/multitask_dataset
echo ${SUITE}
NUM_WORKERS=1 # number of workers
GPU_ID_INDX=0
echo "---- Start to collect dataset ----"


TASK_name=nut_assembly 
for ROBOT in ur5e
do
echo "Robot ${ROBOT} - Task ${TASK_name}" 
N_VARS=9 # number of variations
NUM=450 # number of trajectories to collect
PER_TASK_GROUP=50 # Number of trajectories of same task in row 
python3 tasks/collect_data/collect_task.py ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
    -tsk ${TASK_name} -ro ${ROBOT} \
    --n_tasks ${N_VARS} \
    --N ${NUM} --per_task_group ${PER_TASK_GROUP} \
    --num_workers ${NUM_WORKERS} \
    --ctrl_config ${PATH_TO_CONTROL_CONFIG} \
    --overwrite \
    --gpu_id_indx ${GPU_ID_INDX} \
    --collect_cam
    #--debugger
    #--renderer 
    
done 

TASK_name=pick_place 
for ROBOT in ur5e
do
echo "Robot ${ROBOT} - Task ${TASK_name}" 
N_VARS=16 # number of variations
NUM=800 # number of trajectories to collect
PER_TASK_GROUP=50 # Number of trajectories of same task in row 
python3 tasks/collect_data/collect_task.py ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
    -tsk ${TASK_name} -ro ${ROBOT} \
    --n_tasks ${N_VARS} \
    --N ${NUM} --per_task_group ${PER_TASK_GROUP} \
    --num_workers ${NUM_WORKERS} \
    --ctrl_config ${PATH_TO_CONTROL_CONFIG} \
    --overwrite \
    --gpu_id_indx ${GPU_ID_INDX} \
    --collect_cam
    #--collect_cam
    #--debugger
done 


cd ../../