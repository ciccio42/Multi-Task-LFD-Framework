#!/bin/bash
# task name, possible values [door, drawer, basketball, nut_assembly, 
#                              stack_block, pick_place, button
#                              stack_new_color, stack_new_shape]
cd repo/mosaic
PATH_TO_DATA="/home/multitask_dataset"
PATH_TO_CONTROL_CONFIG="/home/multitask_lfd/repo/mosaic/tasks/robosuite_env/controllers/config/osc_pose.json"
SUITE=${PATH_TO_DATA}/mosaic_multitask_dataset
echo ${SUITE}
NUM_WORKERS=3 # number of workers
GPU_ID_INDX=0
echo "---- Start to collect dataset ----"


#TASK_name=nut_assembly 
#for ROBOT in ur5e
#do
#echo "Robot ${ROBOT} - Task ${TASK_name}" 
#N_VARS=3 # number of variations
#NUM=30 # number of trajectories to collect
#PER_TASK_GROUP=10 # Number of trajectories of same task in row 
#python3 tasks/collect_data/collect_task.py ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
#    -tsk ${TASK_name} -ro ${ROBOT} \
#    --n_tasks ${N_VARS} \
#    --N ${NUM} --per_task_group ${PER_TASK_GROUP} \
#    --num_workers ${NUM_WORKERS} \
#    --ctrl_config ${PATH_TO_CONTROL_CONFIG} \
#    --overwrite \
#    --gpu_id_indx ${GPU_ID_INDX} \
#    --collect_cam
#    #--renderer
#    #--collect_cam
#    #
#    # 
#    #--debugger
#done 

TASK_name=pick_place 
for ROBOT in ur5e
do
echo "Robot ${ROBOT} - Task ${TASK_name}" 
N_VARS=6 # number of variations
NUM=60 # number of trajectories to collect
PER_TASK_GROUP=10 # Number of trajectories of same task in row 
python3 tasks/collect_data/collect_task.py ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
    -tsk ${TASK_name} -ro ${ROBOT} \
    --n_tasks ${N_VARS} \
    --N ${NUM} --per_task_group ${PER_TASK_GROUP} \
    --num_workers ${NUM_WORKERS} \
    --ctrl_config ${PATH_TO_CONTROL_CONFIG} \
    --overwrite \
    --gpu_id_indx ${GPU_ID_INDX} \
    --collect_cam
    #--renderer
    #--collect_cam
    #
    # 
    #--debugger
done 


cd ~/Desktop/multi_task_lfd/Multi-Task-LFD-Framework