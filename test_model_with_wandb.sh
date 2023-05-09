export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# ---- Pick-Place ---- #

BASE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/
PROJECT_NAME="mosaic_baseline_box_no_obj_box"
for MODEL in mosaic-baseline-sav-folder/baseline-no-obj-detector/1Task-Pick-Place-Mosaic-Box-Batch128; do
    for S in 51612 62744; do
        for TASK in pick_place; do
            python $BASE_PATH/repo/mosaic/tasks/test_models/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers 1 --project_name ${PROJECT_NAME} --wandb_log
        done
    done
done

BASE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/
PROJECT_NAME="tosil_baseline_box_no_obj_box"
for MODEL in mosaic-baseline-sav-folder/baseline-no-obj-detector/1Task-Pick-Place-TOSIL-Box-Batch128; do
    for S in 51612 62744; do
        for TASK in pick_place; do
            python $BASE_PATH/repo/mosaic/tasks/test_models/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers 1 --project_name ${PROJECT_NAME} --wandb_log
        done
    done
done
