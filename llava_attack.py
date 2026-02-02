import argparse
import torch
import os
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import cv2
import torchvision

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-path", type=str, default="ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=5000, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=64, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", default=False, action='store_true')

    parser.add_argument("--save_dir", type=str, default='output',
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
model.eval()
print('[Initialization Finished]\n')


if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

lines = open('harmful_corpus/harmful_strings.csv').read().split("\n")
targets = [li for li in lines if len(li)>0]
print(targets)

files = []
for file_name in os.listdir('datasets/coco/val2017/'):
    file_path = os.path.join('datasets/coco/val2017/', file_name)
    if os.path.isfile(file_path):
        files.append(file_path)
print(files[:2])

for i in range(25):
    image_path = files[np.random.randint(len(files))]
    original_image = load_image(image_path)
    original_image.save(args.save_dir+'original_'+str(i)+'.bmp')
    image = image_processor.preprocess(original_image, return_tensors='pt')['pixel_values'].cuda()
    print(image.shape)

    from llava_utils import visual_attacker

    print('device = ', model.device)
    my_attacker = visual_attacker.Attacker(args, model, tokenizer, targets, device=model.device, image_processor=image_processor)

    from llava_utils import prompt_wrapper
    text_prompt_template = prompt_wrapper.prepare_text_prompt('')
    print(text_prompt_template)

    if not args.constrained:
        print('[unconstrained]')
        adv_img_prompt = my_attacker.attack_unconstrained(text_prompt_template,
                                                                img=image, batch_size = 1,
                                                                num_iter=args.n_iters, alpha=args.alpha/255)

    else:
        adv_img_prompt = my_attacker.attack_constrained(text_prompt_template,
                                                                img=image, batch_size= 1,
                                                                num_iter=args.n_iters, alpha=args.alpha / 255,
                                                                epsilon=args.eps / 255)

    save_image(adv_img_prompt, args.save_dir+'adversarial_'+str(i)+'.bmp')
print('[Done]')
