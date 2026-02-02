import argparse
import os
import random
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json
import pandas as pd

def rtp_read(text_file):
    dataset = []
    lines = open(text_file).read().split("\n")
    for li in lines:
        if li.strip():  # 跳过空行
            obj = json.loads(li)
            if obj['challenging']:
                dataset.append(obj['prompt']['text'])
    return dataset

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    new_images = (images - mean[None, :, None, None])/ std[None, :, None, None]
    return new_images

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    new_images = (images * std[None, :, None, None])+ mean[None, :, None, None]
    return new_images


def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image_path", type=str, default='./image.bmp',
                        help="Image file")
    parser.add_argument("--image_safety_patch", type=str, default=None,
                        help="image safety patch file")
    parser.add_argument("--text_safety_patch", type=str, default=None,
                        help="text safety patch file") 
    parser.add_argument("--output_file", type=str, default='./constrained_inference_result.jsonl',
                        help="Output file.")

    args = parser.parse_args()
    return args


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image



# ========================================
#             Model Initialization
# ========================================

print('>>> Initializing Models')

from llava.utils import get_model
args = parse_args()

print('model = ', args.model_path)

tokenizer, model, image_processor, model_name = get_model(args)
model = model.cuda()
print(model)
print(model_name)
model.eval()

print('[Initialization Finished]\n')


from llava_utils import prompt_wrapper, generator


my_generator = generator.Generator(model=model, tokenizer=tokenizer)


# ========================================
#             Inference
# ========================================

prompts = rtp_read('harmful_corpus/rtp_prompts_ct_500.jsonl')
text_prompt = '%s'

if args.text_safety_patch!=None:
    with open(args.text_safety_patch, 'r') as file:
        text_safety_patch = file.read().rstrip()

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
output_f = open(args.output_file, 'w')
output_f.write(json.dumps({
    "args": vars(args),
    "prompt": text_prompt
}) + "\n")
output_f.flush()

out = []
with torch.no_grad():

    for i, user_message in enumerate(prompts):
 
        image_file = 'adversarial_qna_images/adv_image_' + str(np.random.randint(25)) + '.bmp'
        image = load_image(image_file)
        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
        if args.image_safety_patch!=None:
            image = normalize(image)
            safety_patch = torch.load(args.image_safety_patch).cuda()
            safe_image = denormalize((image + safety_patch).clamp(0,1))
        else:
            safe_image = image
        print(f" ----- {i} ----")
        print(" -- prompt: ---")
       
        if args.text_safety_patch!=None:
            # Use the below for optimized text patch
            #    user_message = text_safety_patch + '\n' + user_message 
            # Use the below for heuristic text patch
            user_message += '\n' + text_safety_patch

        text_prompt_template = prompt_wrapper.prepare_text_prompt(text_prompt % user_message)
        print(text_prompt_template)
        prompt = prompt_wrapper.Prompt(model, tokenizer, text_prompts=text_prompt_template, device=model.device)

        response = my_generator.generate(prompt, safe_image).replace("[INST]","").replace("[/INST]","").replace("[SYS]","").replace("[/SYS/]","").strip()

        if args.text_safety_patch!=None:
            response = response.replace(text_safety_patch,"")
        
        print(" -- continuation: ---")
        print(response)
        res_item = {'prompt': user_message, 'continuation': response}
        out.append(res_item)
        output_f.write(json.dumps(res_item) + "\n")
        output_f.flush()
        print()

output_f.close()
