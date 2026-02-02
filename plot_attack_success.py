import csv
import matplotlib.pyplot as plt
import numpy as np
import os

def read_csv_data(csv_path):
    """读取CSV文件并提取attack_success数据"""
    methods = []
    attack_success = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            methods.append(row['methods'])
            # 将百分比字符串转换为浮点数
            value = row['attack_success']
            if '%' in value:
                attack_success.append(float(value.replace('%', '')) / 100)
            else:
                attack_success.append(float(value))
    
    return methods, attack_success

def plot_single_chart(methods, values, title, output_filename, base_dir):
    """生成单个attack_success柱状图"""
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 设置颜色 - 根据图片描述，使用渐变色
    colors = []
    for i, method in enumerate(methods):
        if method == 'no_defense' or method == 'defense':
            colors.append('#FFD700')  # 黄色
        elif method in ['O', 'P']:
            colors.append('#FFA500')  # 橙色/浅黄色
        else:
            colors.append('#FF6B6B')  # 红色/粉色
    
    # 创建柱状图
    bars = plt.bar(range(len(methods)), values, color=colors, edgecolor='black', linewidth=1.2)
    
    # 设置标题
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 设置Y轴
    plt.ylabel('Attack Success Ratio', fontsize=12, fontweight='bold')
    plt.ylim(0.40, 0.70)
    plt.yticks(np.arange(0.40, 0.71, 0.05), fontsize=10)
    
    # 设置X轴
    plt.xlabel('Methods', fontsize=12, fontweight='bold')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right', fontsize=10)
    
    # 在柱子上添加数值标签
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # 添加网格线
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(base_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存柱状图: {output_path}")
    
    plt.close()

def plot_attack_success_bar_chart():
    """生成attack_success柱状图"""
    
    # 读取两个CSV文件
    base_dir = '/home/s202510007/workspace/UniGuard/outputs/instructblip_new/eval/metric'
    constrained_csv = os.path.join(base_dir, 'constrained_summary.csv')
    unconstrained_csv = os.path.join(base_dir, 'unconstrained_summary.csv')
    
    # 读取数据
    constrained_methods, constrained_values = read_csv_data(constrained_csv)
    unconstrained_methods, unconstrained_values = read_csv_data(unconstrained_csv)
    
    # 绘制constrained图表
    plot_single_chart(
        constrained_methods, 
        constrained_values,
        'Constrained Attack Success Ratio on InstructBLIP ↓',
        'constrained_attack_success_bar_chart.png',
        base_dir
    )
    
    # 绘制unconstrained图表
    plot_single_chart(
        unconstrained_methods, 
        unconstrained_values,
        'Unconstrained Attack Success Ratio on InstructBLIP ↓',
        'unconstrained_attack_success_bar_chart.png',
        base_dir
    )

if __name__ == '__main__':
    plot_attack_success_bar_chart()
    print("\n完成！")
