import json
import os.path as osp
import threading

import gradio as gr
import numpy as np
import torch

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava_utils import prompt_wrapper, generator
from utils import normalize, denormalize, load_image

BASE_DIR = "/workingdir/soh337/llavaguard"
# BASE_DIR = "/Users/ahren/Workspace/Multimodal/llavaguard"
UNCONSTRAINED_ATTACK_IMAGE_PATH = 'unconstrained_attack_images/adversarial_'
CONSTRAINED_ATTACK_IMAGE_PATH = 'adversarial_qna_images/adv_image_'
MODEL_PATH = "/workingdir/models_hf/liuhaotian/llava-v1.5-13b"

TEXT_SAFETY_PATCHES = {
    "optimized": "text_patch_optimized",
    "heuristic": "text_patch_heuristic"
}

IMAGE_SAFETY_PATCHES = {
    "default": "safety_patch.pt"
}

DEVICE = "cuda"


def rtp_read(text_file):
    dataset = []
    lines = open(text_file).read().split("\n")
    for li in lines:
        obj = json.loads(li)
        if obj['challenging']:
            dataset.append(obj['prompt']['text'])
    return dataset


model = loaded_model_name = tokenizer = image_processor = context_len = my_generator = None

def load_model_async(model_path, model_name):
    global tokenizer, model, image_processor, context_len, loaded_model_name, my_generator
    print(f"Loading {model_name} model ... ")
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_4bit=True)
    if "llava" in model_name.lower():
        loaded_model_name = "LLaVA"
    else:
        raise NotImplementedError

    my_generator = generator.Generator(model=model, tokenizer=tokenizer)




print('>>> Initializing Models')
threading.Thread(target=load_model_async, args=(MODEL_PATH, get_model_name_from_path(MODEL_PATH))).start()
# load_model_async(MODEL_PATH, get_model_name_from_path(MODEL_PATH))


prompts = rtp_read(osp.join(BASE_DIR, 'harmful_corpus/rtp_prompts.jsonl'))

# out_unprotected: responses without the safety patch
out, out_unprotected = [], []
text_prompt = 'Please extend the following sentence: %s'


def generate_answer(image, user_message: str, requested_model_name: str,
                    image_safety_patch_type: str, text_safety_patch_type: str,
                    ):
    global tokenizer, model, image_processor, context_len, loaded_model_name, my_generator

    with open(TEXT_SAFETY_PATCHES[text_safety_patch_type], 'r') as file:
        text_safety_patch = file.read().rstrip()

    image_safety_patch = IMAGE_SAFETY_PATCHES[image_safety_patch_type]
    if requested_model_name == "LLaVA":

        if requested_model_name == loaded_model_name:

            print(f"{requested_model_name} model already loaded.")

        else:
            print(f"Loading {requested_model_name} model ... ")

            threading.Thread(target=load_model_async, args=(MODEL_PATH, get_model_name_from_path(MODEL_PATH))).start()
            my_generator = generator.Generator(model=model, tokenizer=tokenizer)

        # load a randomly-sampled unconstrained attack image as Image object
        if isinstance(image, str):
            image = load_image(image)

        # transform the image using the visual encoder (CLIP) of LLaVA 1.5; the processed image size would be PyTorch tensor whose shape is (336,336).
        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(DEVICE)

        if image_safety_patch != None:
            # make the image pixel values between (0,1)
            image = normalize(image, device=DEVICE)
            # load the safety patch tensor whose values are (0,1)
            safety_patch = torch.load(image_safety_patch).to(DEVICE)
            # apply the safety patch to the input image, clamp it between (0,1) and denormalize it to the original pixel values
            safe_image = denormalize((image + safety_patch).clamp(0, 1), device=DEVICE)
            # make sure the image value is between (0,1)
            print(torch.min(image), torch.max(image), torch.min(safe_image), torch.max(safe_image))

        else:
            safe_image = image

        model.eval()

        user_message_unprotected = user_message
        if text_safety_patch != None:
            if text_safety_patch_type == "optimal":
                # use the below for optimal text safety patch
                user_message = text_safety_patch + '\n' + user_message

            elif text_safety_patch_type == "heuristic":
                # use the below for heuristic text safety patch
                user_message += '\n' + text_safety_patch
            else:
                raise ValueError(f"Invalid safety patch type: {user_message}")

        text_prompt_template_unprotected = prompt_wrapper.prepare_text_prompt(text_prompt % user_message_unprotected)
        prompt_unprotected = prompt_wrapper.Prompt(model, tokenizer, text_prompts=text_prompt_template_unprotected,
                                                   device=model.device)

        text_prompt_template = prompt_wrapper.prepare_text_prompt(text_prompt % user_message)
        prompt = prompt_wrapper.Prompt(model, tokenizer, text_prompts=text_prompt_template, device=model.device)

        response_unprotected = my_generator.generate(prompt_unprotected, image).replace("[INST]", "").replace("[/INST]",
                                                                                                              "").replace(
            "[SYS]", "").replace("[/SYS/]", "").strip()

        response = my_generator.generate(prompt, safe_image).replace("[INST]", "").replace("[/INST]", "").replace(
            "[SYS]", "").replace("[/SYS/]", "").strip()

        if text_safety_patch != None:
            response = response.replace(text_safety_patch, "")

        response_unprotected = response_unprotected.replace(text_safety_patch, "")

        print(" -- [Unprotected] continuation: ---")
        print(response_unprotected)
        print(" -- [Protected] continuation: ---")
        print(response)

        out.append({'prompt': user_message, 'continuation': response})
        out_unprotected.append({'prompt': user_message, 'continuation': response_unprotected})

        return response, response_unprotected


def get_list_of_examples():
    global rtp
    examples = []

    # Use the first 3 prompts for constrained attack
    for i, prompt in enumerate(prompts[:3]):
        image_num = np.random.randint(25)  # Randomly select an image number
        image_path = f'{CONSTRAINED_ATTACK_IMAGE_PATH}{image_num}.bmp'

        examples.append(
            [image_path, prompt]
        )

    # Use the 3-6th prompts for unconstrained attack
    for i, prompt in enumerate(prompts[3:6]):
        image_num = np.random.randint(25)  # Randomly select an image number
        image_path = f'{UNCONSTRAINED_ATTACK_IMAGE_PATH}{image_num}.bmp'

        examples.append(
            [image_path, prompt]
        )

    return examples


css = """#col-container {max-width: 90%; margin-left: auto; margin-right: auto; display: flex; flex-direction: column;}
#header {text-align: center;}
#col-chatbox {flex: 1; max-height: min(750px, 100%);}
#label {font-size: 2em; padding: 0.5em; margin: 0;}
.message {font-size: 1.2em;}
.message-wrap {max-height: min(700px, 100vh);}
"""


def get_empty_state():
    # TODO: Not sure what this means
    return gr.State({"arena": None})


examples = get_list_of_examples()


# Define a function to update inputs based on selected example
def update_inputs(example_id):
    selected_example = examples[int(example_id)]
    return selected_example['image_path'], selected_example['text']


model_selector, image_patch_selector, text_patch_selector = None, None, None


def process_text_and_image(image_path: str, user_message: str):
    global model_selector, image_patch_selector, text_patch_selector
    print(f"User Message: {user_message}")
    # print(f"Text Safety Patch: {safety_patch}")
    print(f"Image Path: {image_path}")
    print(model_selector.value)

    # generate_answer(user_message, image_path, "LLaVA", "heuristic", "default")
    response, response_unprotected = generate_answer(image_path, user_message, model_selector.value, image_patch_selector.value,
                    text_patch_selector.value)

    return response, response_unprotected


with gr.Blocks(css=css) as demo:
    state = get_empty_state()
    all_components = []

    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """# 🦙LLaVAGuard🔥<br>
    Safeguarding your Multimodal LLM
    **[Project Homepage](#)**""",
            elem_id="header",
        )

        # example_selector = gr.Dropdown(choices=[f"Example {i}" for i, e in enumerate(examples)],
        #                                label="Select an Example")

        with gr.Row():
            model_selector = gr.Dropdown(choices=["LLaVA"], label="Model", info="Select Model", value="LLaVA")
            image_patch_selector = gr.Dropdown(choices=["default"], label="Image Patch", info="Select Image Safety "
                                                                                              "Patch", value="default")
            text_patch_selector = gr.Dropdown(choices=["heuristic", "optimized"], label="Text Patch", info="Select "
                                                                                                           "Text "
                                                                                                           "Safety "
                                                                                                           "Patch",
                                              value="heuristic")

        image_and_text_uploader = gr.Interface(
            fn=process_text_and_image,
            inputs=[gr.Image(type="pil", label="Upload your image", interactive=True),

                    gr.Textbox(placeholder="Input a question", label="Your Question"),
                    ],
            examples=examples,
            outputs=[
                gr.Textbox(label="With Safety Patches"),
                gr.Textbox(label="NO Safety Patches")
            ])

# Launch the demo
demo.launch(share=False)
