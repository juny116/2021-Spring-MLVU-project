# mlvu-dalle
2021 MLVU project repo (DALL-E reproduce)
Trainging DALL-E for fashion datasets

Codes are from [lucidrains/DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch), [VQGAN (Taming transformer)](https://github.com/CompVis/taming-transformers) 


### How to train with horovod
* use the docker file within this repo or use horovod docker image from docker hub juny116/horovod
* mount the codebase to the workspace dir within container, WORKDIR for juny116/horovod is workspace
* additional python packages are not included in docker images, should be installed within the container
* mounting .cache dir is preferred

### Command line example
* nvidia-docker run --shm-size=8g -it --volume="$PWD:/workspace" --volume="path_to_data:/data" --volume="path_to_cache:/root/.cache" --name  horovod juny116/horovod
within container
* pip install -r requirements.txt
* wandb login
* horovodrun -np 4 -H localhost:4 python train_dalle.py --taming --image_text_folder /data --distr_backend horovod
