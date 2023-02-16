BASE_PATH=/home/Multi-Task-LFD-Framework

for MODEL in $BASE_PATH/mosaic-baseline-sav-folder/Task-Pick-and-Place-Batch32-1gpu-Attn2ly128-Act2ly64mix2-headCat-simclr128x256

do
for S in 0 4050 8100 12150 16200 20250 32400
do
for TASK in pick_place
do

python3 $BASE_PATH/repo/mosaic/tasks/test_models/test_any_task.py $MODEL --wandb_log --env $TASK --saved_step $S --eval_each_task 10
done

done
done
