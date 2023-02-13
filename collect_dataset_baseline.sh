#!/bin/bash
# task name, possible values [door, drawer, basketball, nut_assembly, 
#                              stack_block, pick_place, button
#                              stack_new_color, stack_new_shape]

# path to folder where save trajectories
PATH_TO_DATA="/home/multitask_dataset" #"/home/ciccio/Desktop/multitask_dataset"
SUITE=${PATH_TO_DATA}/multitask_dataset_baseline
echo ${SUITE}
WORKERS=8 # number of workers
GPU_ID_INDX=0
CPUS=8
SCRIPT=/home/Multi-Task-LFD-Framework/repo/mosaic/tasks/collect_data/collect_any_task.py
echo "---- Start to collect dataset ----"

cd /home/Multi-Task-LFD-Framework/repo/mosaic
TASK_name=nut_assembly
N_tasks=9
NUM=100
N_env=800
per_task=100
HEIGHT=100
WIDTH=180
for ROBOT in panda
do
 python3 ${SCRIPT} ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
         -tsk ${TASK_name} -ro ${ROBOT} --n_tasks ${N_tasks}  --n_env ${N_env} \
         --N ${NUM} --per_task_group ${per_task} \
         --num_workers ${WORKERS} --collect_cam \
         --heights ${HEIGHT} --widths ${WIDTH} \
         --overwrite
done 

cd ../../