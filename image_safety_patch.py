"""

Requirements:

pip install transformers==4.34 sentencepiece protobuf accelerate

"""

import argparse
import torch
import os
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from datetime import datetime


def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=5000, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=16, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--save_dir", type=str, default='output/new',
                        help="save directory")

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

model.resize_token_embeddings(len(tokenizer))
model.eval()
print('[Initialization Finished]\n')


if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

#read the small corpus including harmful content, which is needed for safety patch generation
lines = open(os.path.join(args.data_dir, 'harmful_corpus', 'harmful_strings.csv')).read().split("\n")
neg_targets = [li for li in lines if len(li)>0]

from llava_utils import visual_defender

print('device = ', model.device)
# pos_targets is not used in the optimization process, so pass an empty list
my_defender = visual_defender.Defender(args, model, tokenizer, [], neg_targets, device=model.device, image_processor=image_processor)

template_img = 'unconstrained_attack.bmp'
image = load_image(template_img)
image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()

from llava_utils import prompt_wrapper
text_prompt_template = prompt_wrapper.prepare_text_prompt('')
print(text_prompt_template)
    
safety_patch = my_defender.defense_constrained(text_prompt_template,
                                                            img=image, batch_size=1,
                                                            num_iter=args.n_iters, alpha=args.alpha / 255,
                                                            epsilon=args.eps / 255)

# Generate filename with timestamp and parameters to avoid overwriting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
patch_filename = f'safety_patch_eps{args.eps}_alpha{args.alpha}_iters{args.n_iters}_{timestamp}.pt'
patch_path = os.path.join(args.save_dir, patch_filename)

torch.save(safety_patch, patch_path)
print(f'Safety patch saved to: {patch_path}')

# Optionally save as image
# save_image(safety_patch, os.path.join(args.save_dir, f'safety_patch_{timestamp}.bmp'))

print('[Done]')
