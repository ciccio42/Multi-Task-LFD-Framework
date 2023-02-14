BASE_PATH=/home/Multi-Task-LFD-Framework

for MODEL in $BASE_PATH/mosaic-baseline-sav-folder/2Task-NutAssembly-Batch45-1gpu-Attn2ly128-Act2ly64mix2-headCat-simclr128x256

do
for S in 0 1620 3240 4860 6480 9720 11340 12960 14580 17820 19440 21060 24300 25920 32400
do

for TASK in nut_assembly
do

python3 $BASE_PATH/repo/mosaic/tasks/test_models/test_any_task.py $MODEL --wandb_log --env $TASK --saved_step $S --eval_each_task 10
done

done
done
