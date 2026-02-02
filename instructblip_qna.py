import argparse
import json
import os
import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from utils import normalize, denormalize, load_image, data_read


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
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    # "/workingdir/models_hf/lmsys/vicuna-13b-v1.1"
    parser.add_argument("--model_path", type=str, default='unconstrained_attack_images/adversarial_',
                        help="Model file")
    parser.add_argument("--image_path", type=str, default='unconstrained_attack_images/adversarial_',
                        help="Image file")
    parser.add_argument("--image_safety_patch", type=str, default=None,
                        help="image safety patch file")
    parser.add_argument("--text_safety_patch", type=str, default=None,
                        help="text safety patch file")

    parser.add_argument("--output_file", type=str, default='./result.jsonl',
                        help="Output file.")

    parser.add_argument("--baseline_attack_mode", type=str, default=None,
                        choices=["compress", "blur"], help="Attack mode for baselines")

    parser.add_argument("--baseline_mode", type=int, choices=[0, 1], default=1,
                        help="testing against benign prompts (mode=0) and adversarial prompts (mode=1). This is the 'mode' argument")

    parser.add_argument("--safety_patch_mode", type=str, choices=["heuristic", "optimized"], default=None,
                        help="The type of safety patch we use")

    parser.add_argument("--do_baseline", action="store_true", help="whether to perform baseline experiments")
    parser.add_argument("--do_attack", action="store_true", help="whether to perform attack")

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

# ========================================
#             Model Initialization
# ========================================

print('>>> Initializing Models')

from lavis.models import load_model_and_preprocess



args = parse_args()
device = f"cuda:{args.gpu_id}"

# remember to modify the parameter llm_model in ./lavis/configs/models/blip2/blip2_instruct_vicuna13b.yaml to the path that store the vicuna weights
model, vis_processor, _ = load_model_and_preprocess(
        name='blip2_vicuna_instruct',
        model_type="vicuna13b", # args.model_path,
        is_eval=True,
        device=device,
    )
model.eval()
"""
Source code of the model in:
    ./lavis/models/blip2_models/blip2_vicuna_instruct.py
"""

# img = Image.open(args.image_file).convert('RGB')
# img = vis_processor["eval"](img).unsqueeze(0).to(device)

print('[Initialization Finished]\n')


# ========================================
#             Inference
# ========================================

##  TODO: expose interface.

qna = data_read('datasets/aokvqa/aokvqa_v1p0_val.json','question',K=1000)

if args.do_baseline:
    if args.baseline_mode == 1:
        prompts = rtp_read('harmful_corpus/rtp_prompts.jsonl')
    else:
        lines = open('prompts/benign').read().split("\n")
        prompts = [li for li in lines]

else:
    prompts = rtp_read('harmful_corpus/rtp_prompts.jsonl')

text_prompt = 'Please answer this single-choice question: %s'


if args.text_safety_patch is not None:
    with open(args.text_safety_patch, 'r') as file:
        text_safety_patch = file.read().rstrip()

if args.image_safety_patch is not None:
    # load the safety patch tensor whose values are (0,1)
    safety_patch = torch.load(args.image_safety_patch).to(device)
    safety_patch = center_crop_pipeline(safety_patch)

else:
    safety_patch = None

to_pil = transforms.ToPILImage()

os.makedirs(osp.dirname(args.output_file), exist_ok=True)
output_f = open(args.output_file, 'w')

out = []
acc = []
responses = []
num_valid_answers = 0
with torch.no_grad():
    for i in range(len(qna)):
        # load a randomly-sampled unconstrained attack image as Image object
        # image = load_image(args.image_path + str(np.random.randint(25)) + '.bmp')
        # transform the image using the visual encoder (CLIP) of LLaVA 1.5; the processed image size would be PyTorch tensor whose shape is (336,336).
        # image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()

        image_id, question, choices, answer_id, direct_answers = qna[i]
        image_id = (12 - len(str(image_id))) * '0' + str(image_id)

        image_file = 'datasets/coco/val2017/' + image_id + '.jpg'




        if args.do_baseline:
            if args.baseline_attack_mode == "blur":
                # blur kernel baseline
                image_pil = cv2.imread(image_file)

                image_np = cv2.blur(image_pil, (3, 3))
                image = to_pil(image_np)

            elif args.baseline_attack_mode == "compress":
                image = load_image(image_file)
                image = image.save('baseline/compressed_q10.jpg', quality=10)
                image = load_image('baseline/compressed_q10.jpg')

            else:
                raise NotImplementedError

            image = vis_processor["eval"](image).unsqueeze(0).to(device)

            print(image.shape)
            safe_image = image

        else:
            image = Image.open(image_file).convert('RGB')
            image = vis_processor["eval"](image).unsqueeze(0).to(device)

            print(image.shape)

            if args.image_safety_patch != None:
                # make the image pixel values between (0,1)
                image = normalize(image, device)

                # apply the safety patch to the input image, clamp it between (0,1) and denormalize it to the original pixel values
                safe_image = denormalize((image + safety_patch).clamp(0, 1), device=device)


                # make sure the image value is between (0,1)
                print(torch.min(image), torch.max(image), torch.min(safe_image), torch.max(safe_image))

            else:
                safe_image = image


        print(f" ----- {i} ----")
        print(" -- prompt: ---")

        if args.text_safety_patch != None:
            # use the below for optimal text safety patch
            if args.safety_patch_mode == "optimized":
                question = text_safety_patch + '\n' + question + '\nA. ' + choices[0] + '\nB. ' + choices[1] + '\nC. ' + \
                           choices[2] + '\nD. ' + choices[
                               3] + "\nAnswer this single-choice question with the option's letter (A or or C or D) directly."
            # use the below for heuristic text safety patch
            elif args.safety_patch_mode == "heuristic":
                question = question + '\nA. ' + choices[0] + '\nB. ' + choices[1] + '\nC. ' + choices[2] + '\nD. ' + \
                           choices[
                               3] + "\nAnswer this single-choice question with the option's letter (A or B or C or D) directly." + '\n' + text_safety_patch
            else:
                raise NotImplementedError
        else:
            question = question + '\nA. ' + choices[0] + '\nB. ' + choices[1] + '\nC. ' + choices[2] + '\nD. ' + \
                       choices[3] + "\nAnswer this single-choice question with the option's letter (A or B or C or D) directly."

        print(text_prompt % question)


        # I set `remove_invalid_values=True` in the model
        # More in the [documentation](https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation#transformers.GenerationConfig.remove_invalid_values)

        # Need to increase temperature above 0.5 to avoid NaN
        # https://github.com/facebookresearch/llama/issues/380
        response = model.generate({"image": safe_image, "prompt": text_prompt % question},
                       use_nucleus_sampling=True, top_p=0.9, temperature=1,
                                  min_length=128, max_length=1024
                                  )[0]

        response = response.replace("[INST]", "").replace("[/INST]", "").replace("<<SYS>>", "").replace("<</SYS>>",
                                                                                                        "").replace(
            "[SYS]", "").replace("[/SYS]", "").strip()

        if args.text_safety_patch != None:
            response = response.replace(text_safety_patch, "")

        print(" -- response: ---")

        maxv = 99999
        response_id = -1
        for idx in range(4):
            loc = response.find(chr(ord('A') + idx) + '.')
            if loc != -1 and maxv > loc:
                maxv = loc
                response_id = chr(ord('A') + idx)

            loc = response.find(chr(ord('A') + idx) + '\n')
            if loc != -1 and maxv > loc:
                maxv = loc
                response_id = chr(ord('A') + idx)

            loc = response.find(chr(ord('A') + idx) + ' ')
            if loc == 0 and maxv > loc:
                maxv = loc
                response_id = chr(ord('A') + idx)

            loc = response.find(chr(ord('A') + idx) + '\t')
            if loc == 0 and maxv > loc:
                maxv = loc
                response_id = chr(ord('A') + idx)

        print(" -- continuation: ---")
        print(response)
        answer_id = chr(ord('A') + answer_id)
        print(f"Response ID: {response_id}")
        print(f"Answer ID: {answer_id}")
        print(f"Response: {response}")

        acc.append(response_id == answer_id if len(response) != 0 else 0)
        num_valid_answers += 0 if response_id == -1 else 1

        res_item = {'prompt': question, 'continuation': response, "response_id": response_id, "answer_id": answer_id}
        responses.append(res_item)
        output_f.write(json.dumps(res_item) + "\n")
        output_f.flush()
        print()


overall_accuracy = np.average(acc)
accuracy_valid_answers_only = np.sum(acc) / num_valid_answers


print(f'overall_accuracy = {overall_accuracy}')
print(f'accuracy (valid answers only) = {accuracy_valid_answers_only}')
print(f'num_valid_answers = {num_valid_answers}')


output_f.close()

# Rewrite file with header at the end or just append? 
# The original script writes accuracy at the top. 
# For streaming, we might need to write a separate summary file or accept that accuracy is at the end.
# Actually, the original script REWRITES the whole file at the end.
# Let's write the summary to a separate file or just accept it's at the end.
# Wait, let's keep the original logic for summary BUT stream the responses to a TEMP file? No, that's complex.
# Let's just write the accuracy to the console and a separate summary file maybe? 
# Or just write it at the end of the jsonl.

with open(args.output_file + '.summary', 'w') as f:
    f.write(f'overall_accuracy = {overall_accuracy}' + '\n')
    f.write(f'accuracy (valid answers only) = {accuracy_valid_answers_only}\n')
    f.write(f'num_valid_answers = {num_valid_answers}\n')