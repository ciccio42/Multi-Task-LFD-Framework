#!/bin/bash

cd mosaic
PATH_TO_DATA="/home/ciccio/Desktop/mosaic_dataset"
PATH_TO_CONTROL_CONFIG="/home/ciccio/Desktop/multi_task_lfd/mosaic/tasks/robosuite_env/controllers/config/osc_pose.json"
TASK_name=pick_place # task name, possible values [door, drawer, basketball, nut_assembly, 
                     #                              stack_block, pick_place, button
                     #                              stack_new_color, stack_new_shape]
N_VARS=16 # number of variations for this task
NUM=20 # number of trajectory to collect
N_ENV=1 # number of environment
SUITE=${PATH_TO_DATA}/mosaic_multitask_dataset
echo ${SUITE}
PER_TASK_GROUP=1
NUM_WORKERS=1 # number of workers
echo "---- Start to collect dataset ----"


for ROBOT in ur5e
do
echo "Robot ${ROBOT} - Task ${TASK_name}" 
python3 tasks/collect_data/collect_task.py ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
    -tsk ${TASK_name} -ro ${ROBOT} \
    --n_tasks ${N_VARS}  --n_env ${N_ENV} \
    --N ${NUM} --per_task_group ${PER_TASK_GROUP} \
    --num_workers ${NUM_WORKERS} \
    --ctrl_config ${PATH_TO_CONTROL_CONFIG} \
    --overwrite \
    --collect_cam \
    --debugger
    #--renderer 
    #--collect_cam
    #
    # 
    #--debugger
    
done 

cd /home/ciccio/Desktop/multi_task_lfd
