FROM nvidia/cuda:10.1-devel-ubuntu18.04
#FROM ubuntu:bionic
# Linux updating key (https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771)
RUN apt-key del 7fa2af80
COPY cuda/cuda-keyring_1.0-1_all.deb cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/*
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/*

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get --assume-yes install cmake libopenmpi-dev zlib1g-dev xvfb git g++ libfontconfig1 libxrender1 libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev xpra patchelf libglfw3-dev libglfw3 libglew2.0 virtualenv xserver-xorg-dev
RUN apt-get update && apt-get install -y curl wget gcc build-essential vim nano unzip sudo

# Install python
RUN apt -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt  -y install python3.7 
RUN apt-get -y install python3.7-dev python3.7-venv 
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN apt-get -y install python3-setuptools

# Install pip
RUN apt-get install curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
RUN python3 /tmp/get-pip.py
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install debugpy

RUN chmod 777 /root /etc

# Add user
ARG USERNAME
ARG USER_UID
ARG USER_GID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
ENV PATH=/home/$USERNAME/.local/bin:$PATH

# Mujoco set
ENV LANG C.UTF-8

RUN sudo mkdir -p /home/$USERNAME/.mujoco/ \
    && sudo wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && sudo  tar -xf mujoco.tar.gz -C /home/$USERNAME/.mujoco/ \
    && sudo rm mujoco.tar.gz

# ---- install robosuite requirement ----
COPY robosuite_requirements.txt /home/robosuite_requirements.txt
RUN pip3 install -r  /home/robosuite_requirements.txt

# ---- install mosaic requirement ----
COPY mosaic_requirements.txt /home/mosaic_requirements.txt
RUN pip3 install -r  /home/mosaic_requirements.txt 


# Install
WORKDIR / 
ENV USER=$USERNAME
CMD ["/bin/bash", "/home/multitask_lfd/installation.sh"]