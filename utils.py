import torch
import json
from PIL import Image


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def denormalize(images, device="cuda:0"):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    new_images = (images - mean[None, :, None, None])/ std[None, :, None, None]
    return new_images

def normalize(images, device="cuda:0"):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    new_images = (images * std[None, :, None, None])+ mean[None, :, None, None]
    return new_images


def data_read(text_file,mode,K=1000):
    dataset = []
    for obj in json.load(open(text_file, 'r')):
        if obj['difficult_direct_answer']==False:
            dataset.append([obj['image_id'],obj['question'],obj['choices'],obj['correct_choice_idx'],obj['direct_answers']])

    print(dataset[:2])
    return dataset[:K]