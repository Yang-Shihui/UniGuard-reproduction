# import json

# def process_jsonl(input_file, existing_file, output_file):
#     existing_ids = set()
#     total_challenging = 0
#     other_count = 0

#     # 1. 首先读取已经挑出的 500 项，建立索引以防重复
#     # 假设 filename + begin + end 可以唯一标识一项
#     try:
#         with open(existing_file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 data = json.loads(line)
#                 # 创建唯一标识符
#                 identifier = f"{data['filename']}_{data['begin']}_{data['end']}"
#                 existing_ids.add(identifier)
#     except FileNotFoundError:
#         print(f"提示: {existing_file} 不存在，将视为全新提取。")

#     # 2. 处理原始文件并进行统计和筛选
#     with open(input_file, 'r', encoding='utf-8') as f_in, \
#          open(output_file, 'w', encoding='utf-8') as f_out:
        
#         for line in f_in:
#             if not line.strip():
#                 continue
            
#             data = json.loads(line)
            
#             # 统计所有 challenging 为 true 的项
#             if data.get("challenging") is True:
#                 total_challenging += 1
                
#                 # 检查是否已存在于 500 文件中
#                 identifier = f"{data['filename']}_{data['begin']}_{data['end']}"
#                 if identifier not in existing_ids:
#                     # 写入新文件
#                     f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
#                     other_count += 1

#     print(f"处理完成！")
#     print(f"1. 原始文件中 'challenging': true 的总项数: {total_challenging}")
#     print(f"2. 已在 {existing_file} 中的项数: {len(existing_ids)}")
#     print(f"3. 提取到 {output_file} 的新项数: {other_count}")

# # 使用示例
# # 请替换为你实际的文件名
# process_jsonl('rtp_prompts.jsonl', 'rtp_prompts_ct_500.jsonl', 'rtp_prompts_ct_other.jsonl')


import json
import random

def sample_jsonl(input_file, output_file, sample_size=200):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 读取所有行
            lines = f.readlines()
        
        # 检查文件行数是否足够
        if len(lines) <= sample_size:
            sampled_lines = lines
            print(f"警告：文件总行数 ({len(lines)}) 小于或等于请求的采样数，将返回全部数据。")
        else:
            # 随机采样
            sampled_lines = random.sample(lines, sample_size)
        
        # 写入新文件
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in sampled_lines:
                f_out.write(line)
        
        print(f"成功！已随机抽取 {len(sampled_lines)} 条数据并保存至: {output_file}")

    except FileNotFoundError:
        print("错误：找不到指定的 JSONL 文件。")
    except Exception as e:
        print(f"发生错误: {e}")

# 使用示例
sample_jsonl('rtp_prompts_ct.jsonl', 'srtp_prompts_ct_200.jsonl', 200)