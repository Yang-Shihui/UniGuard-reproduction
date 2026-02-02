import argparse
import os
import os.path as osp
import random
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json
from torchvision.utils import save_image


from utils import data_read

# 导入smooth模块用于smoothllm方法
from smooth import smooth

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
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--image_safety_patch", type=str, default=None,
                        help="image safety patch file")
    parser.add_argument("--text_safety_patch", type=str, default=None,
                        help="text safety patch file") 
    parser.add_argument("--output_file", type=str, default='./qna_result.jsonl',
                        help="Output file.")
    # 新增: baseline方法参数
    parser.add_argument("--baseline_method", type=str, default=None,
                        choices=["blur", "compress", "smoothllm"],
                        help="Baseline defense method: 'blur' for blur kernel, "
                             "'compress' for JPEG compression, or 'smoothllm' for text smoothing")

    args = parser.parse_args()
    return args


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def load_processed_results(output_file):
    """加载已处理的结果，返回已处理的prompt集合和准确率列表"""
    processed_prompts = set()
    acc = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()
                # 跳过第一行（元数据）和最后一行（准确率）
                for line in lines[1:]:
                    line = line.strip()
                    if not line or line.startswith('overall_accuracy'):
                        continue
                    try:
                        data = json.loads(line)
                        if 'prompt' in data:
                            processed_prompts.add(data['prompt'])
                            # 尝试从已保存的结果中恢复准确率（如果有的话）
                            # 这里我们只记录prompt，准确率会在重新处理时计算
                    except json.JSONDecodeError:
                        continue
            print(f'[断点续传] 发现已处理 {len(processed_prompts)} 个样本')
        except Exception as e:
            print(f'[警告] 读取已处理结果时出错: {e}，将从头开始')
    return processed_prompts, acc


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

from llava_utils import prompt_wrapper, generator_qna

my_generator = generator_qna.Generator(model=model, tokenizer=tokenizer)

# ========================================
#             Inference
# ========================================

##  TODO: expose interface.
qna = data_read('datasets/aokvqa/aokvqa_v1p0_val.json','question',K=200)

if args.text_safety_patch!=None:
    with open(args.text_safety_patch, 'r') as file:
        text_safety_patch = file.read().rstrip()

# 创建baseline目录（用于compress方法）
if args.baseline_method == "compress":
    os.makedirs('baseline', exist_ok=True)

responses = []
acc = []
text_prompt = '%s'

# 检查是否存在已处理的结果（断点续传）
processed_prompts, saved_acc = load_processed_results(args.output_file)
start_idx = 0

# 初始化输出文件
if not os.path.exists(args.output_file):
    # 如果文件不存在，创建新文件并写入元数据
    with open(args.output_file, 'w') as f:
        f.write(json.dumps({
            "args": vars(args),
            "prompt": text_prompt
        }))
        f.write("\n")
else:
    # 如果文件存在，读取已保存的结果
    print(f'[断点续传] 检测到输出文件已存在: {args.output_file}')
    with open(args.output_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('overall_accuracy'):
                continue
            try:
                data = json.loads(line)
                if 'prompt' in data and 'continuation' in data:
                    responses.append(data)
            except json.JSONDecodeError:
                continue
    print(f'[断点续传] 已加载 {len(responses)} 个已处理的结果')

with torch.no_grad():

    for i in range(len(qna)):
        
        image_id,question,choices,answer_id,direct_answers = qna[i]
        image_id = (12-len(str(image_id)))*'0'+str(image_id)
        
        # 构建问题（用于检查是否已处理）
        if args.text_safety_patch!=None:
            # Automatically detect patch type based on filename
            patch_filename = osp.basename(args.text_safety_patch).lower()
            if 'optimized' in patch_filename:
                # use the below for optimal text safety patch
                question_full = text_safety_patch + '\n' + question + '\nA. '+choices[0]+'\nB. '+choices[1]+'\nC. '+choices[2]+'\nD. '+choices[3] + "\nAnswer with the option's letter from the given choices directly."
            else:
                # use the below for heuristic text safety patch
                question_full = question + '\nA. '+choices[0]+'\nB. '+choices[1]+'\nC. '+choices[2]+'\nD. '+choices[3] + "\nAnswer with the option's letter from the given choices directly." + '\n' + text_safety_patch    
        else:
            question_full = question + '\nA. '+choices[0]+'\nB. '+choices[1]+'\nC. '+choices[2]+'\nD. '+choices[3] + "\nAnswer with the option's letter from the given choices directly."
        
        # 检查是否已处理过（断点续传）
        if question_full in processed_prompts:
            print(f" ----- {i} ---- [已处理，跳过]")
            # 从已保存的结果中找到对应的响应来计算准确率
            for saved_resp in responses:
                if saved_resp.get('prompt') == question_full:
                    response = saved_resp.get('continuation', '')
                    break
            else:
                continue  # 如果找不到对应的响应，跳过
            
            # 计算准确率
            maxv = 99999
            response_id = -1
            for idx in range(4):
                loc = response.find(chr(ord('A')+idx)+'.')
                if loc!=-1 and maxv>loc:
                    maxv = loc
                    response_id = chr(ord('A')+idx)
     
                loc = response.find(chr(ord('A')+idx)+'\n')
                if loc!=-1 and maxv>loc:
                    maxv = loc
                    response_id = chr(ord('A')+idx)
        
                loc = response.find(chr(ord('A')+idx)+' ')
                if loc==0 and maxv>loc:
                    maxv = loc
                    response_id = chr(ord('A')+idx)
     
                loc = response.find(chr(ord('A')+idx)+'\t')
                if loc==0 and maxv>loc:
                    maxv = loc
                    response_id = chr(ord('A')+idx)
                               
            answer_id_char = chr(ord('A')+answer_id)
            acc.append(response_id==answer_id_char if len(response)!=0 else 0)
            continue
        
        # 处理新样本 - 应用baseline方法
        image_path = 'datasets/coco/val2017/'+image_id+'.jpg'
        image = load_image(image_path)
        image.save("llava_original.png")

        # ===== 新增: baseline方法处理逻辑 =====
        if args.baseline_method == "blur":
            # blur kernel baseline
            image_cv = cv2.imread(image_path)
            image_blurred = cv2.blur(image_cv, (3, 3))
            # Convert from BGR (OpenCV) to RGB (PIL)
            image_blurred_rgb = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_blurred_rgb)
            
        elif args.baseline_method == "compress":
            # compression decompression baseline
            image.save('baseline/compressed_q10.jpg', quality=10)
            image = load_image('baseline/compressed_q10.jpg')
            
        elif args.baseline_method == "smoothllm":
            # 对question进行平滑处理
            question = smooth(question, perturb_pct=0.1)
        # ===== baseline方法处理逻辑结束 =====

        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
        
        if args.image_safety_patch!=None:
            image = normalize(image)
            safety_patch = torch.load(args.image_safety_patch).cuda()
            safe_image = denormalize((image + safety_patch).clamp(0,1))
        else:
            safe_image = image

        print(f" ----- {i} ----")
        print(" -- prompt: ---")

        question = question_full
        text_prompt_template = prompt_wrapper.prepare_text_prompt(question)
        
        print(text_prompt_template)
        prompt = prompt_wrapper.Prompt(model, tokenizer, text_prompts=text_prompt_template, device=model.device)

        response = my_generator.generate(prompt, safe_image).replace("[INST]","").replace("[/INST]","").replace("<<SYS>>","").replace("<</SYS>>","").replace("[SYS]","").replace("[/SYS]","").strip()
        if args.text_safety_patch!=None:
            response = response.replace(text_safety_patch,"")
        print(" -- response: ---")
        response_data = {'prompt': question, 'continuation': response}
        responses.append(response_data)
        
        # 边跑边保存：每处理完一个样本立即写入文件
        with open(args.output_file, 'a') as f:
            f.write(json.dumps(response_data))
            f.write("\n")
        
        maxv = 99999
        response_id = -1
        for idx in range(4):
            loc = response.find(chr(ord('A')+idx)+'.')
            if loc!=-1 and maxv>loc:
                maxv = loc
                response_id = chr(ord('A')+idx)
 
            loc = response.find(chr(ord('A')+idx)+'\n')
            if loc!=-1 and maxv>loc:
                maxv = loc
                response_id = chr(ord('A')+idx)
    
            loc = response.find(chr(ord('A')+idx)+' ')
            if loc==0 and maxv>loc:
                maxv = loc
                response_id = chr(ord('A')+idx)
 
            loc = response.find(chr(ord('A')+idx)+'\t')
            if loc==0 and maxv>loc:
                maxv = loc
                response_id = chr(ord('A')+idx)
                           
        answer_id_char = chr(ord('A')+answer_id)
        print(response,response_id,answer_id_char)
        acc.append(response_id==answer_id_char if len(response)!=0 else 0)

print('overall_accuracy = {}'.format(np.average(acc)))

# 最后追加写入总体准确率
with open(args.output_file, 'a') as f:
    print('overall_accuracy = {}'.format(np.average(acc)),file=f)
