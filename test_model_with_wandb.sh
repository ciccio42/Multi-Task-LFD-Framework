export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# ---- Pick-Place ---- #

BASE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework
PROJECT_NAME="mosaic-parameters-paper-object-cropped-no-norm"
for MODEL in mosaic-baseline-sav-folder/mosaic-parameters/1Task-Pick-Place-Mosaic-Parameters-Paper-Object-Cropped-No-Norm-Batch32; do
    for S in 81000; do
        for TASK in pick_place; do
            python $BASE_PATH/repo/mosaic/tasks/test_models/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers 4 --wandb_log --project_name ${PROJECT_NAME}
        done
    done
done

# BASE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/
# PROJECT_NAME="tosil_baseline_box_no_obj_box"
# for MODEL in mosaic-baseline-sav-folder/baseline-no-obj-detector/1Task-Pick-Place-TOSIL-Box-Batch128; do
#     for S in 72864 81972 112332 132572; do
#         for TASK in pick_place; do
#             python $BASE_PATH/repo/mosaic/tasks/test_models/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers 8 --project_name ${PROJECT_NAME} --wandb_log
#         done
#     done
# done
