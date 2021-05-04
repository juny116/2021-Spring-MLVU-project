# mlvu-dalle
2021 MLVU project repo (DALL-E reproduce)
Trainging DALL-E for fashion datasets

Codes are from [lucidrains/DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch), [VQGAN (Taming transformer)](https://github.com/CompVis/taming-transformers) 


### How to train with horovod
* use the docker file within this repo or use horovod docker image from docker hub juny116/horovod
* WORKDIR for juny116/horovod is workspace
* additional python packages are not included in docker images, should be installed within the container
* 
** e.g. nvidia-docker run --shm-size=8g -it --volume="$PWD:/workspace" --volume="/home/juny116/data/fashion-dataset/image_text:/data" --volume="/home/juny116/.cache:/root/.cache" --name  horovod juny116/horovod
