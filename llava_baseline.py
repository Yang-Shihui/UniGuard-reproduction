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
from smooth import smooth

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
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--mode", type=int, default=1, help="testing against benign prompts (mode=0) and adversarial prompts (mode=1).")
    parser.add_argument("--image_path", type=str, default='unconstrained_attack_images/adversarial_',
                        help="Image file")
    parser.add_argument("--output_file", type=str, default='./baseline_result.jsonl',
                        help="Output file.")
    parser.add_argument("--baseline_method", type=str, default="compress", 
                        choices=["blur", "compress", "smoothllm"],
                        help="Baseline defense method: 'blur' for blur kernel or 'compress' for JPEG compression")

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
print("Model loaded, first param device:", next(model.parameters()).device)
model = model.cuda()
print("After .cuda(), first param device:", next(model.parameters()).device)
print(model)
print(model_name)
model.eval()

print('[Initialization Finished]\n')


from llava_utils import prompt_wrapper, generator


my_generator = generator.Generator(model=model, tokenizer=tokenizer)


# ========================================
#             Inference
# ========================================

##  TODO: expose interface.
if args.mode==1:
    prompts = rtp_read('harmful_corpus/rtp_prompts_ct_500.jsonl')
else:
    lines = open('prompts/benign').read().split("\n")
    prompts = [li for li in lines]

#text_prompt = 'Please extend the following sentence: %s'
text_prompt = '%s'

out = []
with torch.no_grad():

    for i, user_message in enumerate(prompts):

        print(f" ----- {i} ----")
        print(" -- prompt: ---")


        # Apply baseline defense method
        image_file = args.image_path + str(np.random.randint(25)) + '.bmp'
        
        if args.baseline_method == "blur":
            # blur kernel baseline
            image_cv = cv2.imread(image_file)
            image_blurred = cv2.blur(image_cv, (3, 3))
            # Convert from BGR (OpenCV) to RGB (PIL)
            image_blurred_rgb = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_blurred_rgb)
            
        elif args.baseline_method == "compress":
            # compression decompression baseline
            os.makedirs('baseline', exist_ok=True)
            image = load_image(image_file)
            image.save('baseline/compressed_q10.jpg', quality=10)
            image = load_image('baseline/compressed_q10.jpg')
            
        elif args.baseline_method == "smoothllm":
            user_message = smooth(user_message, perturb_pct=0.1)
            image = load_image(image_file)
        else:
            raise ValueError(f"Unknown baseline method: {args.baseline_method}")

        print(text_prompt % user_message)

        safe_image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
        
        text_prompt_template = prompt_wrapper.prepare_text_prompt(text_prompt % user_message)
        print(text_prompt_template)
        prompt = prompt_wrapper.Prompt(model, tokenizer, text_prompts=text_prompt_template, device=model.device)

        response = my_generator.generate(prompt, safe_image)

        print(" -- continuation: ---")
        print(response)
        out.append({'prompt': user_message, 'continuation': response})
        print()

with open(args.output_file, 'w') as f:
    f.write(json.dumps({
        "args": vars(args),
        "prompt": text_prompt
    }))
    f.write("\n")

    for li in out:
        f.write(json.dumps(li))
        f.write("\n")
