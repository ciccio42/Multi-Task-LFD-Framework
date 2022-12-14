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
```

``` bash
# Remove previously docker with the same name
docker image rm multitask_lfd
docker rm -f multitask_lfd
# Build Docker
export GID=$(id -g frosa)
docker build -t multitask_lfd --build-arg USER_UID=$UID --build-arg USERNAME=$USER --build-arg USER_GID=$GID .

# Rrun docker
docker run  --name multitask_lfd -it --user frosa --rm --gpus all multitask_lfd
docker exec -it multitask_lfd /bin/bash
docker run --name multitask_lfd -v /user/frosa/robotic/multitask_lfd:/home/multitask_lfd -v /mnt/sdc1/frosa/multitask_dataset:/home/multitask_dataset -p 5678:5678 -it --gpus all --rm multitask_lfd
```
