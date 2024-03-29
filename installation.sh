#!/bin/bash
echo ----- User -----
whoami
sudo apt-get install htop
pip install --user seaborn
pip install --user torch==1.8.1+cu101 torchvision==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
BASEPATH=/home/Multi-Task-LFD-Framework
echo ----- Installing mujoco-py -----

# install mujoco-py
unset LD_LIBRARY_PATH
echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco210/bin:/usr/local/cuda-10.1/lib64/:/usr/lib/nvidia-000 >>/home/$USER/.bashrc
source /home/$USER/.bashrc
pip install --user mujoco-py

# install robosuite
echo ----- Installing robosuite -----
cd $BASEPATH/repo/robosuite
pip install --user .

echo ----- Installing mosaic -----
cp $BASEPATH/repo/mosaic/tasks/robosuite_env/sawyer/sawyer_arm.urdf $BASEPATH/repo/robosuite/models/assets/bullet_data/sawyer_description/urdf/sawyer_arm.urdf
cd $BASEPATH/repo/mosaic
pip install --user .
cd $BASEPATH/repo/mosaic/tasks
pip install --user .

# install code for running scripted policies
#echo ----- Installing Training Framework -----
# 1. Install training
cd $BASEPATH/repo/Multi-Task-LFD-Training-Framework/training
pip install --user .
## 2. Install tasks
cd $BASEPATH/repo/Multi-Task-LFD-Training-Framework/tasks
pip install --user .

export CUDA_VISIBLE_DEVICES=0

cd /
./bin/bash
