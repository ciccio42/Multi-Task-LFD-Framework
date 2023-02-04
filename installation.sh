#!/bin/bash
echo ----- User -----
whoami
echo ----- Installing mujoco-py -----
# install mujoco-py
sudo mkdir /usr/lib/nvidia-000
echo export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/.mujoco/mujoco210/bin:/usr/lib/nvidia-000:/root/.mujoco/bin:/home/$USER/.mujoco/mujoco210/bin>> /home/$USER/.bashrc
source /home/$USER/.bashrc
pip3 install --user mujoco-py

# install robosuite
echo ----- Installing robosuite -----
cd /home/Multi-Task-LFD-Framework/repo/robosuite
pip3 install --user .

# install code for running scripted policies
echo ----- Installing Training Framework -----
# 1. Install training 
cd /home/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training
pip3 install --user .
# 2. Install tasks
cd /home/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/tasks
pip3 install --user .

export CUDA_VISIBLE_DEVICES=0

cd /
./bin/bash