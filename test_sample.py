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

from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE1024, DiscreteVAE, DALLE, VQGanVAE16384, VQGanVAECustom
# from dalle_pytorch.simple_tokenizer import tokenize, tokenizer, VOCAB_SIZE

# argument parsing
import io
import os, sys
import requests
import PIL

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from IPython.display import display, display_markdown
from tqdm import tqdm

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
VAE_CLASS = args.vae_class
if VAE_CLASS == 'VQGAN1024':
    vae = VQGanVAE1024()
elif VAE_CLASS == 'VQGAN16384':
    vae = VQGanVAE16384()
elif VAE_CLASS == 'VQGAN_CUSTOM':
    vae = VQGanVAECustom()
elif VAE_CLASS == 'DALLE':
    vae = OpenAIDiscreteVAE()
elif VAE_CLASS == 'DALLE_TRAIN':
    VAE_PATH = args.vae_path
    vae_path = Path(VAE_PATH)
    loaded_obj = torch.load(str(vae_path), map_location='cuda')
    vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']
    vae = DiscreteVAE(**vae_params)
    vae.load_state_dict(weights)
    vae.to('cuda')


filenames = os.listdir(args.target)

for filename in tqdm(filenames):
    TARGET_IMG_PATH = args.target + '/' + filename
    TARGET_SAVE_PATH  = args.target + '/output/'
    filename = filename.split('.')[0]
    
    img = PIL.Image.open(TARGET_IMG_PATH)

    composed = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor()
    ])
    
    img = composed(img)
    org_img = T.ToPILImage(mode='RGB')(img)
#     org_img.save(TARGET_SAVE_PATH+'/original/'+filename+'_original.jpg')
    img = torch.unsqueeze(img, 0).to('cuda')
    lat = vae.get_codebook_indices(img)

    imgs = vae.decode(lat)
    img = imgs[0]
    if VAE_CLASS == 'DALLE_TRAIN':
        img = make_grid(img.float(), normalize = True, range = (-1, 1))
    img = T.ToPILImage(mode='RGB')(img)
    img.save(f'{TARGET_SAVE_PATH}/{VAE_CLASS}/{filename}_{lat.size(1)}_{IMAGE_SIZE}.jpg')

    