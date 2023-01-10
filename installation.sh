#!/bin/bash
echo ----- User -----
whoami
echo ----- Installing mujoco-py -----
# install mujoco-py
sudo mkdir /usr/lib/nvidia-000
echo export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia-000:/root/.mujoco/bin:/home/$USER/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH} >> /home/$USER/.bashrc
source /home/$USER/.bashrc
pip3 install --user mujoco-py

# install robosuite
echo ----- Installing robosuite -----
cd /home/multitask_lfd/repo/robosuite
pip3 install --user .

# install mosaic
echo ----- Installing mosaic -----
cd /home/multitask_lfd/repo/mosaic
pip3 install --user . 
echo export PYTHONPATH=${PYTHONPATH}:/home/multitask_lfd/mosaic >> /home/$USER/.bashrc
source /home/$USER/.bashrc
cd tasks
pip3 install --user .

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /
./bin/bash