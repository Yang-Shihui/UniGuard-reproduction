import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image
from llava_utils import prompt_wrapper, text_defender


def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None) 
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=50, help="specify the number of iterations for attack.")

    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory for loss curves and metrics")
    parser.add_argument("--n_candidates", type=int, default=100,
                        help="n_candidates")
    parser.add_argument("--chunk_size", type=int, default=4,
                        help="Chunk size for processing batches to avoid OOM (default: 2, reduce if still OOM)")
    parser.add_argument("--image_safety_patch", type=str, default=None,
                        help="Path to the image safety patch file (.pt) for joint optimization")
    parser.add_argument("--template_image", type=str, default='unconstrained_attack.bmp',
                        help="Template image for text patch generation")
    parser.add_argument("--output_patch_file", type=str, default=None,
                        help="Output file path for optimized text patch (default: auto-generate with timestamp)")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite existing patch file if it exists")
    parser.add_argument("--patch_length", type=int, default=16,
                        choices=[8, 16, 32], help="Length of the text safety patch in tokens (default: 16)")
    parser.add_argument("--generate_natural_lang_patch", action='store_true', default=True,
                        help="Generate natural language text patch file (default: True)")
    parser.add_argument("--no_natural_lang_patch", action='store_false', dest='generate_natural_lang_patch',
                        help="Skip generating natural language text patch file")

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


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

from llava.utils import get_model
args = parse_args()

print('model = ', args.model_path)

tokenizer, model, image_processor, model_name = get_model(args)
print(model.base_model)
model.eval()
print('[Initialization Finished]\n')


if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

lines = open(os.path.join(args.data_dir, 'harmful_corpus', 'harmful_strings.csv')).read().split("\n")
targets = [li for li in lines if len(li)>0]
print(targets[0])

# Get the actual device where model parameters are located
model_device = next(model.parameters()).device

# Load template image (always needed since prompt contains <image> token)
safe_image = None
if os.path.exists(args.template_image):
    template_image = Image.open(args.template_image).convert('RGB')
    template_image = image_processor.preprocess(template_image, return_tensors='pt')['pixel_values'].cuda()
    
    # Apply image safety patch if provided (for joint optimization)
    if args.image_safety_patch is not None:
        from llava_utils.visual_defender import normalize, denormalize
        
        print(f'\n>>> Loading image safety patch from: {args.image_safety_patch}')
        image_safety_patch = torch.load(args.image_safety_patch).cuda()
        
        # Apply image safety patch: normalize -> add patch -> clamp -> denormalize
        normalized_image = normalize(template_image)
        safe_image = denormalize((normalized_image + image_safety_patch).clamp(0, 1))
        print(f'>>> Image safety patch applied. Safe image shape: {safe_image.shape}')
        print(f'>>> Joint optimization enabled: P(harmful | image+image_patch, prompt+text_patch)')
    else:
        # Use template image without patch (baseline)
        safe_image = template_image
        print(f'\n>>> Using template image without patch. Image shape: {safe_image.shape}')
        print(f'>>> Optimization: P(harmful | image, prompt+text_patch)')
else:
    print(f'>>> Warning: Template image {args.template_image} not found, using text-only optimization')
    print(f'>>> Note: Prompt contains <image> token, but no image will be used')

# Generate unique identifier for this run to avoid overwriting loss files
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
patch_type = 'joint' if args.image_safety_patch is not None else 'textonly'
args.run_id = f'text_patch_{patch_type}_{run_timestamp}'

print(f'\n>>> Patch length set to: {args.patch_length} tokens')

my_attacker = text_defender.Attacker(args, model, tokenizer, targets, device=model_device, safe_image=safe_image)

from llava_utils import prompt_wrapper
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token

text_prompt_template = prompt_wrapper.prepare_text_prompt('')
print(text_prompt_template)

prompt_segs = text_prompt_template.split('<image>')  # each <ImageHere> corresponds to one image
print(prompt_segs)
seg_tokens = [
     tokenizer(
         seg if seg.strip() != "" else " ",  # Add fallback or default text
         return_tensors="pt", add_special_tokens=i == 0).to(model.device).input_ids
     # only add bos to the first seg
     for i, seg in enumerate(prompt_segs)
 ]
embs = [model.model.embed_tokens(seg_t) for seg_t in seg_tokens] # text to embeddings
mixed_embs = torch.cat(embs, dim=1)
offset = mixed_embs.shape[1]
print(offset)

# Evaluate the safety patches based on natural language and save the one with minimum loss
safety_patch = my_attacker.evaluate_safety_patch(text_prompt_template=text_prompt_template, offset=offset,
                                    num_iter=args.n_iters, batch_size=8)

# Save natural language patch to output directory (if enabled)
if args.generate_natural_lang_patch:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    natural_lang_patch_file = os.path.join(args.save_dir, f'text_patch_optimized_natural_language_len{args.patch_length}_{timestamp}.txt')
    with open(natural_lang_patch_file, 'w') as f:
        f.write(safety_patch)
    print(f'\n>>> Natural language text patch saved to: {natural_lang_patch_file}')
else:
    print('\n>>> Natural language text patch generation skipped (--no_natural_lang_patch)')

# Generate optimized text patch using gradient-based attack method
print('\n>>> Generating optimized text patch using attack method...')
adv_prompt = my_attacker.attack(text_prompt_template=text_prompt_template, offset=offset,
                                    num_iter=args.n_iters, batch_size=8)

# Determine output file path - save to output directory with text patch prefix
if args.output_patch_file is None:
    # Auto-generate filename with timestamp in output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patch_suffix = '_joint' if args.image_safety_patch is not None else ''
    args.output_patch_file = os.path.join(args.save_dir, f'text_patch_optimized{patch_suffix}_len{args.patch_length}_iters{args.n_iters}_{timestamp}.txt')
else:
    # If user specified a path, ensure it's in output directory
    if not os.path.isabs(args.output_patch_file):
        # Relative path - put in output directory
        args.output_patch_file = os.path.join(args.save_dir, args.output_patch_file)
    # Ensure it has .txt extension
    if not args.output_patch_file.endswith('.txt'):
        args.output_patch_file = args.output_patch_file + '.txt'

# Check if file exists and handle overwrite
if os.path.exists(args.output_patch_file) and not args.overwrite:
    # Generate a new filename with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name, ext = os.path.splitext(args.output_patch_file)
    args.output_patch_file = f'{base_name}_{timestamp}{ext}'
    print(f'>>> Warning: File exists, saving to: {args.output_patch_file}')

# Ensure output directory exists
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)

# Save the optimized text patch
with open(args.output_patch_file, 'w') as f:
    f.write(adv_prompt)

print(f'\n>>> Optimized text patch saved to: {args.output_patch_file}')
print(f'>>> Patch content: {adv_prompt[:100]}...' if len(adv_prompt) > 100 else f'>>> Patch content: {adv_prompt}')
