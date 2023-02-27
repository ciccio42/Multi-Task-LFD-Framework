BASE_PATH=/home/Multi-Task-LFD-Framework

# ---- Nut Assembly ---- #
# for MODEL in $BASE_PATH/mosaic-baseline-sav-folder/baseline-1/Task-NutAssembly-Batch27-1gpu-Attn2ly128-Act2ly256mix4-headCat-simclr128x512

# do
# for S in 29700 35100 37800 54000
# do
# for TASK in nut_assembly
# do

# python3 $BASE_PATH/repo/mosaic/tasks/test_models/test_any_task.py $MODEL --wandb_log --env $TASK --saved_step $S --eval_each_task 10 --num_workers 5 --project_name 'mosaic_baseline_1'
# done

# done
# done


# ---- Pick-Place ---- #
for MODEL in $BASE_PATH/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Stable-Policy-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat-simclr128x512
do
for S in 56700 72900 81000
do
for TASK in pick_place
do

python3 $BASE_PATH/repo/mosaic/tasks/test_models/test_any_task.py $MODEL --wandb_log --env $TASK --saved_step $S --eval_each_task 10 --num_workers 5 --project_name 'mosaic_baseline_1_stable_policy'
done
done
done

# ---- Stack Block ----#
# for MODEL in $BASE_PATH/mosaic-baseline-sav-folder/baseline-1/Task-Pick-Place-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat-simclr128x512

# do
# for S in 24300 36450 52650 81000
# do
# for TASK in stack_block
# do

# python3 $BASE_PATH/repo/mosaic/tasks/test_models/test_any_task.py $MODEL --wandb_log --env $TASK --saved_step $S --eval_each_task 10 --num_workers 6 --project_name 'mosaic_baseline_1'
# done

# done
# done