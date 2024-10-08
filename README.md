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
docker run  --name multitask_lfd -it --user frosa --rm --gpus all  multitask_lfd
docker exec -it --detach-keys 'ctrl-p'  multitask_lfd /bin/bash
#---- Command Tamplate ----#
# docker run --name multitask_lfd -v [PATH-TO-CLONED-REPOSITORY]:/home/multitask_lfd -v [PATH-TO-DATASET-FOLDER]:/home/multitask_dataset -p 5678:5678 -it --gpus '"device=[GPU TO MAKE AVAILABLE]"' --shm-size 8G --rm multitask_lfd

#---- Command Example ----#
docker run --privileged --name multitask_lfd -v /user/frosa/robotic/Multi-Task-LFD-Framework:/home/Multi-Task-LFD-Framework -v /mnt/sdc1/frosa/multitask_dataset:/home/multitask_dataset -v /usr/lib/nvidia:/usr/lib/nvidia-000 --shm-size 8G -p 5678:5678 -it --gpus all --rm --pid=host --detach-keys 'ctrl-p' multitask_lfd 
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
pip install -r requirements.txt
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

# Dldev1
```bash
conda create -n multi_task_lfd python=3.7 pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=10.1 -c pytorch 
export MUJOCO_PY_MUJOCO_PATH=/user/frosa/robotic/.mujoco/mujoco210
```

## Dataset Collector
This folder contains the code used to collect the dataset in real-world application.
It depends dependens on the


# To use with conda
```bash
conda install -c conda-forge glew && conda install -c conda-forge patchelf && conda install -c conda-forge mesalib && conda install -c menpo glfw3
echo $CONDA_PREFIX # chech the conda prefix
export CPATH=$CONDA_PREFIX/include
```

# Vima and Vima-bench
```bash
git clone git@github.com:ciccio42/VIMA.git
git clone git@github.com:ciccio42/VIMABench.git
```

# Installation procedure
```bash
conda create -n multi_task_lfd python=3.9
conda activate multi_task_lfd
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# 1 Install qpth-0.0.15 manually
# 2 Install mujoco follow https://pytorch.org/rl/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html
cd ~ && mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
pip install setuptools==65.5.0
tar -xvf mujoco210-linux-x86_64.tar.gz
conda install -c conda-forge glew -y
conda install -c conda-forge mesalib -y
conda install -c anaconda mesa-libgl-cos6-x86_64 -y
conda install -c menpo glfw3 -y
conda install libgcc -y
conda install patchelf -y
conda install -c anaconda mesa-libegl-cos6-x86_64 -y
conda install -c conda-forge gcc==12.1.0
conda install -c conda-forge gxx_linux-64


# 3 Install requirements
pip install -r requirements.txt
pip install -r requirements_multi_task_lfd.txt 

# 4 Install robosuite
pip install -e /user/frosa/multi_task_lfd/Multi-Task-LFD-Framework/repo/robosuite/.

# 5 Install Multi-Task IL Framework
pip install -e /user/frosa/multi_task_lfd/Multi-Task-LFD-Framework/repo/


```