import torch
import clip
from PIL import Image
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('--text', type = str, required = True, help='your text prompt')

parser.add_argument('--outputs_dir', type = str, default = './outputs', required = False,
                    help='output directory')

parser.add_argument('--num_images', type = int, default = 512, required = False,
                    help='number of images')

parser.add_argument('--top_k', type = int, default = 10, required = False,
                    help='top k images by clip score')

args = parser.parse_args()

base_path = Path(args.outputs_dir) / args.text.replace(' ', '_')

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = []
for i in range(args.num_images):
    image.append(preprocess(Image.open(f'{base_path}/{i}.jpg')).unsqueeze(0).to(device))

image = torch.cat(image)
text = clip.tokenize([args.text]).to(device)


with torch.no_grad():
    # image_features = model.encode_image(image)
    # text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    probs = logits_per_text.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)
print(torch.mean(logits_per_text))
print(torch.topk(logits_per_text, 10))