import argparse
from random import choice
from pathlib import Path

# torch

import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

# vision imports

from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

# dalle related classes and utils

from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE1024, DiscreteVAE, DALLE, VQGanVAE16384
from dalle_pytorch.simple_tokenizer import tokenize, tokenizer, VOCAB_SIZE

# argument parsing
import io
import os, sys
import requests
import PIL

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from IPython.display import display, display_markdown

target_image_size = 128

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))

def preprocess(img):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)

parser = argparse.ArgumentParser()

# group = parser.add_mutually_exclusive_group(required = False)
parser.add_argument('--vae_class', type = str, default='discrete',
                    help='VAE class name')
parser.add_argument('--vae_path', type = str,
                    help='path to your trained discrete VAE')
parser.add_argument('--target', type = str,
                    help='path to your target sample')
parser.add_argument('--image_size', type = int,
                    help='path to your trained discrete VAE')
args = parser.parse_args()
# helpers

def exists(val):
    return val is not None

IMAGE_SIZE = args.image_size
if args.vae_class == 'VQGAN1024':
    vae = VQGanVAE1024()
elif args.vae_class == 'VQGAN16384':
    vae = VQGanVAE16384()
elif args.vae_class == 'DALLE':
    vae = OpenAIDiscreteVAE()
else:
    VAE_PATH = args.vae_path
    vae_path = Path(VAE_PATH)
    loaded_obj = torch.load(str(vae_path))
    vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']
    vae = DiscreteVAE(**vae_params)
    vae.load_state_dict(weights)

target = args.target
img = PIL.Image.open(f'/home/juny116/Workspace/DALLE-pytorch/samples/{target}/org.jpg')

composed = T.Compose([
    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    T.Resize(IMAGE_SIZE),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor()
])

img = composed(img)
org_img = T.ToPILImage(mode='RGB')(img)
org_img.save(f'samples/{target}/{IMAGE_SIZE}.jpg')
img = torch.unsqueeze(img, 0)
lat = vae.get_codebook_indices(img)
# or_lat = lat[:,:128]

# lat = torch.randint(low=0, high=1024, size=(1,128))
# lat = torch.cat([or_lat,lat],dim=1)

imgs = vae.decode(lat)
img = imgs[0]
if args.vae_class == 'discrete':
    img = make_grid(img.float(), normalize = True, range = (-1, 1))
img = T.ToPILImage(mode='RGB')(img)
img.save(f'samples/{target}/{args.vae_class}_{lat.size(1)}_{IMAGE_SIZE}.jpg')

