import os
import csv
import glob

def extract_metrics_from_csv(csv_path):
    """从CSV文件中提取指定指标的值"""
    metrics_dict = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                metric_name = row[0].strip()
                metric_value = row[1].strip()
                metrics_dict[metric_name] = metric_value
    
    # 提取需要的字段，并格式化为两位小数
    required_fields = ['overall', 'identity_attack', 'profanity', 
                      'sexually_explicit', 'threat', 'toxicity']
    
    result = {}
    for field in required_fields:
        value = metrics_dict.get(field, '')
        if value != '':
            try:
                # 转换为浮点数并保留两位小数
                result[field] = f"{float(value):.2%}"
            except (ValueError, TypeError):
                result[field] = ''
        else:
            result[field] = ''
    
    return result

def main():
    metric_dir = '/home/s202510007/workspace/UniGuard/outputs/llava_new/eval/metric'
    
    # 获取所有CSV文件（排除输出文件）
    csv_files = glob.glob(os.path.join(metric_dir, '*.csv'))
    csv_files = [f for f in csv_files if not f.endswith('_summary.csv')]
    
    constrained_data = []
    unconstrained_data = []
    
    for csv_file in csv_files:
        # 获取文件名（不含路径和扩展名）
        filename = os.path.basename(csv_file)
        filename_without_ext = os.path.splitext(filename)[0]
        
        # 提取指标
        metrics = extract_metrics_from_csv(csv_file)
        
        # 构建数据行：文件名作为第一列
        row = {'filename': filename_without_ext}
        row.update(metrics)
        
        # 根据文件名分类（先检查更具体的匹配）
        if 'unconstrained' in filename:
            unconstrained_data.append(row)
        elif 'constrained' in filename:
            constrained_data.append(row)
    
    # 定义列名
    fieldnames = ['filename', 'overall', 'identity_attack', 'profanity', 
                  'sexually_explicit', 'threat', 'toxicity']
    
    # 保存constrained数据
    if constrained_data:
        output_path_constrained = os.path.join(metric_dir, 'constrained_summary.csv')
        with open(output_path_constrained, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(constrained_data)
        print(f"✓ 已保存 constrained 汇总文件: {output_path_constrained}")
        print(f"  包含 {len(constrained_data)} 个文件")
    
    # 保存unconstrained数据
    if unconstrained_data:
        output_path_unconstrained = os.path.join(metric_dir, 'unconstrained_summary.csv')
        with open(output_path_unconstrained, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(unconstrained_data)
        print(f"✓ 已保存 unconstrained 汇总文件: {output_path_unconstrained}")
        print(f"  包含 {len(unconstrained_data)} 个文件")
    
    print("\n完成！")

if __name__ == '__main__':
    main()
