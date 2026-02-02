import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Perplexity efficiently")
    parser.add_argument("--inputs", type=str, nargs='+', required=True, help="Input JSONL files")
    parser.add_argument("--outputs", type=str, nargs='+', required=True, help="Output JSON files")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default="lmsys/vicuna-13b-v1.5")
    return parser.parse_args()

def calculate_ppl(texts, model, tokenizer, device, batch_size=1):
    model.eval()
    nlls = []
    
    # 逐条计算以确保稳定性
    for text in tqdm(texts, desc="Calculating", leave=False):
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)

    if not nlls: return 0.0
    return torch.exp(torch.stack(nlls).mean()).item()

def main():
    args = parse_args()
    if len(args.inputs) != len(args.outputs):
        print("Error: Inputs and outputs count mismatch.")
        return

    print(f"Loading model and tokenizer: {args.model_path}")
    # 关键点：使用 half() 或 float16 减少一半显存需求
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16, 
        device_map=args.device
    ).eval()

    for input_file, output_file in zip(args.inputs, args.outputs):
        if not os.path.exists(input_file):
            print(f"Skipping {input_file} (not found)")
            continue
            
        print(f"\nProcessing: {input_file}")
        all_text = []
        with open(input_file, 'r') as f:
            for i, line in enumerate(f):
                if i == 0 or (i <= 3 and "_qna" in input_file): continue
                try:
                    obj = json.loads(line)
                    text = obj.get('continuation', "")
                    if isinstance(text, list): text = text[0]
                    if text and text.strip():
                        all_text.append(text)
                except: continue

        if not all_text:
            print(f"No valid text in {input_file}")
            continue

        mean_ppl = calculate_ppl(all_text, model, tokenizer, args.device, args.batch_size)
        print(f"Mean Perplexity: {mean_ppl:.4f}")

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({"input": input_file, "model": args.model_path, "mean_perplexity": mean_ppl}, f, indent=4)

if __name__ == "__main__":
    main()