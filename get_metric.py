# ========== 第一步：先配置代理（必须放在最开头） ==========
import os
import socket
import socks
import httplib2

# 替换为你本地的代理信息（Clash/V2Ray 等）
PROXY_HOST = "127.0.0.1"  # 本地代理地址，一般是 127.0.0.1
PROXY_PORT = 7891         # Socks5 代理端口（Clash 默认 7890，V2Ray 常为 10808）

# 1. 配置 Socket 层代理（覆盖所有 Python 网络请求）
socks.set_default_proxy(socks.SOCKS5, PROXY_HOST, PROXY_PORT)
socket.socket = socks.socksocket

# 2. 配置 httplib2 代理（适配 googleapiclient 底层依赖）
def get_proxied_http():
    http = httplib2.Http(proxy_info=httplib2.ProxyInfo(
        socks.PROXY_TYPE_SOCKS5,
        PROXY_HOST,
        PROXY_PORT
    ))
    return http

# ========== 第二步：导入其他模块 ==========
import json
from metric.perspective_api import PerspectiveApiScorer
import argparse
from tqdm import tqdm

api_key = "AIzaSyCRQyj4x_oxe7mcqoBv4-YJ9YqKdmDiSDQ"
perspective_scorer = PerspectiveApiScorer(api_key, http=get_proxied_http())

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--load_existing_generation", action="store_true")

    args = parser.parse_args()
    return args

args = parse_args()
input_file = args.input
output_file = args.output
if output_file == "":
    output_file = input_file.split('.jsonl')[0] + '_eval.jsonl'

def float_it(score):
    for key in score.keys():
        score[key] = float(score[key])
    return score

print(f"input file: {input_file}")
print(f"Output file {output_file}")
inputs = open(input_file).read().strip().split('\n')
all_text = []

if args.load_existing_generation:
    # 检查输出文件是否存在
    if os.path.exists(args.output):
        print(f"从已有文件读取: {args.output}")
        with open(args.output, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0 or (i <= 3 and "_qna" in args.input):
                    print(f"Skipping {i}-th line")
                    continue
                obj = json.loads(line)
                if 'continuation' in obj:
                    all_text += [obj['continuation']]
    else:
        print(f"⚠️  输出文件不存在: {args.output}")
        print("将使用正常模式：从输入文件读取并计算指标")
        args.load_existing_generation = False  # 回退到正常模式

if not args.load_existing_generation:
    print("正在计算 Perspective API 指标...")
    with open(output_file, 'w') as f:
        for i, li in enumerate(tqdm(inputs)):

            if i == 0 or (i <= 3 and "_qna" in args.input):
                print(f"Skipping {i}-th line")
                continue
            obj = json.loads(li)
            if 'continuation' in obj:

                text = obj['continuation']
                if type(text) == list:
                    text = text[0]

                if text != "":
                    all_text.append(text)
                    res = {
                        'perspective_api': float_it(perspective_scorer.get_scores(text))
                    }
                    obj['continuation_metric'] = res

            f.write(json.dumps(obj))
            f.write('\n')

print("Output file", output_file)
print("✓ 完成！已计算 Perspective API 指标")
