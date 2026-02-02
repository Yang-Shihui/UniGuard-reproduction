import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json
import os.path as osp
import pandas as pd
import torchvision

from utils import normalize, denormalize, load_image


def rtp_read(text_file):
    dataset = []
    lines = open(text_file).read().split("\n")
    for li in lines:
        if li.strip():  # 跳过空行
            obj = json.loads(li)
            if obj['challenging']:
                dataset.append(obj['prompt']['text'])
    return dataset

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--base_dir", type=str, default="./")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image_path", type=str, default='unconstrained_attack_images/adversarial_',
                        help="Image file")
    parser.add_argument("--image_safety_patch", type=str, default=None,
                        help="image safety patch file")
    parser.add_argument("--text_safety_patch", type=str, default=None,
                        help="text safety patch file") 
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file.")
    parser.add_argument("--device", type=str, default='cuda:0',
                        help="Device.")

    args = parser.parse_args()
    return args





# ========================================
#             Model Initialization
# ========================================

print('>>> Initializing Models')

from llava.utils import get_model
args = parse_args()
os.makedirs(osp.dirname(args.output_file), exist_ok=True)

print('model = ', args.model_path)

tokenizer, model, image_processor, model_name = get_model(args)
# model = model.to(args.device)
print(model)
print(model_name)
model.eval()

print('[Initialization Finished]\n')


from llava_utils import prompt_wrapper, generator


my_generator = generator.Generator(model=model, tokenizer=tokenizer)


# ========================================
#             Inference
# ========================================

prompts = rtp_read(osp.join(args.base_dir, 'harmful_corpus/rtp_prompts_ct_500.jsonl'))

#text_prompt = 'Please extend the following sentence: %s'
text_prompt = '%s'

if args.text_safety_patch!=None:
    with open(args.text_safety_patch, 'r') as file:
        text_safety_patch = file.read().rstrip()

os.makedirs(osp.dirname(args.output_file), exist_ok=True)
output_f = open(args.output_file, 'w')
output_f.write(json.dumps({
    "args": vars(args),
    "prompt": text_prompt
}) + "\n")
output_f.flush()

out = []
with torch.no_grad():

    for i, user_message in enumerate(prompts):
        
        #load a randomly-sampled unconstrained attack image as Image object
        image = load_image(args.image_path+str(np.random.randint(25))+'.bmp')
        #transform the image using the visual encoder (CLIP) of LLaVA 1.5; the processed image size would be PyTorch tensor whose shape is (336,336).
        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
        print(image.shape)
        if args.image_safety_patch is not None:
            # make the image pixel values between (0,1) 
            image = normalize(image)
            # load the safety patch tensor whose values are (0,1)
            safety_patch = torch.load(args.image_safety_patch).cuda()
            # apply the safety patch to the input image, clamp it between (0,1) and denormalize it to the original pixel values
            safe_image = denormalize((image + safety_patch).clamp(0,1))
            # make sure the image value is between (0,1)
            print(torch.min(image),torch.max(image),torch.min(safe_image),torch.max(safe_image))

        else:
            safe_image = image


        print(f" ----- {i} ----")
        print(" -- prompt: ---")

        if args.text_safety_patch!=None:
            # Automatically detect patch type based on filename
            patch_filename = osp.basename(args.text_safety_patch).lower()
            if 'optimized' in patch_filename:
                # use the below for optimal text safety patch
                user_message = text_safety_patch + '\n' + user_message
            else:
                # use the below for heuristic text safety patch
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
