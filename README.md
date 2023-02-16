``` bash
mkdir repo
cd repo
git clone https://github.com/ciccio42/mosaic.git
cd mosaic
git checkout ur5e_task
cd ../
git clone https://github.com/ciccio42/robosuite.git
cd robosuite
git checkout ur5e_ik
cd ../
git clone git@github.com:ciccio42/Multi-Task-LFD-Training-Framework.git
```

``` bash
# Remove previously docker with the same name
docker image rm multitask_lfd
docker rm -f multitask_lfd
# Build Docker
export GID=$(id -g frosa)
docker build -t multitask_lfd --build-arg USER_UID=$UID --build-arg USERNAME=$USER --build-arg USER_GID=$GID .

# Run docker
docker run  --name multitask_lfd -it --user frosa --rm --gpus all multitask_lfd
docker exec -it multitask_lfd /bin/bash
#---- Command Tamplate ----#
# docker run --name multitask_lfd -v [PATH-TO-CLONED-REPOSITORY]:/home/multitask_lfd -v [PATH-TO-DATASET-FOLDER]:/home/multitask_dataset -p 5678:5678 -it --gpus '"device=[GPU TO MAKE AVAILABLE]"' --shm-size 8G --rm multitask_lfd

#---- Command Example ----#
docker run --name multitask_lfd -v /user/frosa/robotic/Multi-Task-LFD-Framework:/home/Multi-Task-LFD-Framework -v /mnt/sdc1/frosa/multitask_dataset:/home/multitask_dataset -v /usr/lib/nvidia:/usr/lib/nvidia-000 -p 5678:5678 -it --gpus all --rm --pid=host --detach-keys 'ctrl-p' multitask_lfd
```
# Note
If you use **online rendering**
``` bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

If you use **offline rendering**
``` bash
unset LD_PRELOAD
```

# Training
```bash
conda create -n multi_task_lfd python=3.7
pip install -r mosaic_requirements.txt
pip install -r robosuite_requirements.txt
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

# Dldev1
```bash
conda create -n multi_task_lfd python=3.7 pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=10.1 -c pytorch 
```

export MUJOCO_PY_MUJOCO_PATH=/user/frosa/robotic/.mujoco/mujoco210