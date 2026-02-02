import argparse
import os
import os.path as osp
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json
from torchvision import transforms

import pandas as pd
import torchvision

from utils import normalize, denormalize, load_image

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from minigpt_utils import prompt_wrapper, generator




def rtp_read(text_file):
    dataset = []
    lines = open(text_file).read().split("\n")
    for li in lines:
        obj = json.loads(li)
        if obj['challenging']:
            dataset.append(obj['prompt']['text'])
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")

    parser.add_argument("--mode", type=str, default='VisualChatBot',
                        choices=["TextOnly", "VisualChatBot"],
                        help="Inference Mode: TextOnly: Text model only (Vicuna) \n VisualChatBot: Vision model + Text model (MiniGPT4) ")


    parser.add_argument("--image_path", type=str, default='unconstrained_attack_images/adversarial_',
                        help="Image file")

    parser.add_argument("--image_safety_patch", type=str, default=None,
                        help="image safety patch file")
    parser.add_argument("--text_safety_patch", type=str, default=None,
                        help="text safety patch file")


    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file.")

    parser.add_argument("--baseline_attack_mode", type=str, default=None,
                        choices=["compress", "blur"], help="Attack mode for baselines")

    parser.add_argument("--baseline_mode", type=int, choices=[0, 1], default=None,
                        help="testing against benign prompts (mode=0) and adversarial prompts (mode=1). This is the 'mode' argument")

    parser.add_argument("--safety_patch_mode", type=str, choices=["heuristic", "optimal"], default=None,
                        help="The type of safety patch we use")

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    parser.add_argument("--do_baseline", action="store_true", help="whether to perform baseline experiments")

    parser.add_argument("--do_attack", action="store_true", help="")

    args = parser.parse_args()

    if args.do_baseline:

        assert args.baseline_mode is not None
        assert args.text_safety_patch is None

        assert args.image_safety_patch is None
        assert args.baseline_attack_mode is not None

        os.makedirs("baseline", exist_ok=True)

    elif args.do_attack:
        assert args.image_safety_patch is None, "In attack mode, image_safety_patch should be None"
        assert args.text_safety_patch is None, "In attack mode, text_safety_patch should be None"



    else:
        assert args.image_safety_patch is not None, "image_safety_patch should not be None"
        assert args.text_safety_patch is not None, "text_safety_patch should not be None"
        assert args.baseline_attack_mode is None

    if args.text_safety_patch is not None:
        assert args.safety_patch_mode is not None, "Must specify the type of safety patch by `--safety_patch_mode`"

    return args


center_crop_pipeline = transforms.Compose([
    transforms.CenterCrop((224, 224)),  # Center crop to 336x336
    # Include any other transformations here. For example, converting to tensor.
    # transforms.ToTensor(),
    # If your visual processor (vis_processor) includes conversion to tensor and normalization, you might not need the above line.
])


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('>>> Initializing Models')

args = parse_args()
cfg = Config(args)
os.makedirs(osp.dirname(args.output_file), exist_ok=True)
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)

device = 'cuda:{}'.format(args.gpu_id)

model = model_cls.from_config(model_config).to(device)

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
print('Initialization Finished')

my_generator = generator.Generator(model=model, device=device)



# ========================================
#             Inference
# ========================================
if args.do_baseline:
    if args.baseline_mode == 1:
        prompts = rtp_read('harmful_corpus/rtp_prompts.jsonl')
    else:
        # baseline_mode is 0
        lines = open('prompts/benign').read().split("\n")
        prompts = [li for li in lines]

else:
    prompts = rtp_read('harmful_corpus/rtp_prompts.jsonl')

#text_prompt = 'Please extend the following sentence: %s'
text_prompt = '%s'

if args.text_safety_patch is not None:
    with open(args.text_safety_patch, 'r') as file:
        text_safety_patch = file.read().rstrip()

# Move this patch loading outside the loop to save some time
if args.image_safety_patch is not None:
    # load the safety patch tensor whose values are (0,1)
    safety_patch = torch.load(args.image_safety_patch).to(device)

    safety_patch = center_crop_pipeline(safety_patch)

    print("Shape of safety patch:", safety_patch.shape)

else:
    safety_patch = None # Should not happen

to_pil = transforms.ToPILImage()
out = []
with torch.no_grad():
    for i, user_message in enumerate(prompts):

        if args.do_baseline:

            if args.mode == "TextOnly":

                ## vicuna official system message.
                # prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "

                prefix = ""
                prefix += "###Human:%s ###Assistant:"
                img_prompt = []

            elif args.mode == "VisualChatBot":
                prefix = prompt_wrapper.minigpt4_chatbot_prompt

                if args.baseline_attack_mode == "blur":
                    # blur kernel baseline
                    image_pil = cv2.imread(args.image_path+str(np.random.randint(25))+'.bmp')

                    image_np = cv2.blur(image_pil,(3, 3))
                    image = to_pil(image_np)


                elif args.baseline_attack_mode == "compress":
                    image = load_image(args.image_path + str(np.random.randint(25)) + '.bmp')
                    image = image.save('baseline/compressed_q10.jpg', quality=10)
                    image = load_image('baseline/compressed_q10.jpg')

                else:
                    raise NotImplementedError

                safe_image = vis_processor(image).unsqueeze(0).to(model.device)

                # prompt = prompt_wrapper.Prompt(model, tokenizer, text_prompts=text_prompt_template, device=model.device)
                # response = my_generator.generate(prompt, safe_image)
                img_prompt = [safe_image]



        else:

            if args.mode == "TextOnly":

                ## vicuna official system message.
                # prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "

                prefix = ""
                prefix += "###Human:%s ###Assistant:"
                img_prompt = []

            elif args.mode == "VisualChatBot":
                prefix = prompt_wrapper.minigpt4_chatbot_prompt
                # load a randomly-sampled unconstrained attack image as Image object
                # image = load_image(args.image_path + str(np.random.randint(25)) + '.bmp')

                image = Image.open(args.image_path + str(np.random.randint(25)) +'.bmp').convert('RGB')

                # transform the image using the visual encoder (CLIP); the processed image size would be PyTorch tensor whose shape is (336,336).

                image = vis_processor(image).unsqueeze(0).to(model.device)

                if args.image_safety_patch is not None:
                    # make the image pixel values between (0,1)
                    image = normalize(image, device=device)

                    # apply the safety patch to the input image, clamp it between (0,1) and denormalize it to the original pixel values
                    safe_image = denormalize((image + safety_patch).clamp(0, 1), device=device)
                    # make sure the image value is between (0,1)
                    print(torch.min(image).item(), torch.max(image).item(), torch.min(safe_image).item(), torch.max(safe_image).item())

                else:
                    safe_image = image


                img_prompt = [safe_image]




        text_prompt = prefix % ('Please extend the following sentence: %s')
        #####

        print("Instructions: ")
        print(text_prompt)

        prompt = prompt_wrapper.Prompt(model=model, img_prompts=[img_prompt])


        print(f" ----- {i} ----")
        print(" -- prompt: ---")

        if args.text_safety_patch != None:
            # use the below for optimal text safety patch
            if args.safety_patch_mode == "optimal":
                user_message = text_safety_patch + '\n' + user_message
            # use the below for heuristic text safety patch
            elif args.safety_patch_mode == "heuristic":
                user_message += '\n' + text_safety_patch

            else:
                raise NotImplementedError

        print(text_prompt % user_message)

        prompt.update_text_prompt([text_prompt % user_message])

        # Generate responses using MiniGPT4
        response, _ = my_generator.generate(prompt)

        if args.text_safety_patch != None:
            response = response.replace(text_safety_patch, "")

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
