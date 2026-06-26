"""
单列图版本：分面点图 (Faceted Dot Plot)
适合与表格配合使用，强调B vs E的对比
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Table 1 → evaluation → RQ1_2 → Code
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config

# 配置
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = config.RQ12_DIR / "data" / "evaluation_SciBERT"

# 指标映射
METRIC_MAPPING = {
    'Pla-S': 'feasibility',
    'SemDist': 'novelty',
    'KWCR': 'coverage',
    'AvgDis': 'pairwise_distance'
}

# 领域映射
DOMAIN_MAPPING = {
    'Com': ('com', 'single'),
    'Ele': ('ele', 'single'),
    'AI': ('ai', 'single'),
    'Com+Ele': ('com', 'crossele'),
    'Com+AI': ('com', 'crossai'),
    'Ele+AI': ('ele', 'crossai')
}

# 模型映射
MODEL_FOLDERS = ['4o', '4omini', 'qwen-max']
MODEL_FILE_PREFIXES = {
    '4o': '4o',
    '4omini': '4omini',
    'qwen-max': 'qw'
}


def load_metric_value(model_folder: str, domain_prefix: str, strategy: str, 
                      metric_type: str, method: str) -> List[float]:
    """加载指定配置的指标值"""
    file_prefix = MODEL_FILE_PREFIXES[model_folder]
    filename = f"{domain_prefix}_150_{file_prefix}_{strategy}.json"
    
    if metric_type == 'feasibility':
        json_file = f"{filename}_feasibility.json"
    else:
        json_file = f"{filename}_{metric_type}.json"
    
    json_path = BASE_DIR / model_folder / filename / json_file
    
    if not json_path.exists():
        raise FileNotFoundError(f"找不到文件: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    values = [item[method] for item in data]
    return values


def calculate_domain_statistics(domain_key: str, metric_type: str, method: str) -> Tuple[float, float]:
    """计算指定领域、指标、方法的统计信息"""
    domain_prefix, strategy = DOMAIN_MAPPING[domain_key]
    
    model_means = []
    for model_folder in MODEL_FOLDERS:
        values = load_metric_value(model_folder, domain_prefix, strategy, metric_type, method)
        model_mean = np.mean(values)
        model_means.append(model_mean)
    
    overall_mean = np.mean(model_means)
    overall_std = np.std(model_means, ddof=1)
    
    return overall_mean, overall_std


def plot_single_column_figure():
    """绘制2x2布局点图"""
    print("=" * 80)
    print("绘制2x2分面点图")
    print(f"数据目录: {BASE_DIR}")
    print("=" * 80)
    
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 20
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.95, 
                       hspace=0.15, wspace=0.2)
    
    color_b = '#4A90E2'
    color_e = '#E94B3C'
    
    domains = list(DOMAIN_MAPPING.keys())
    domain_labels = domains
    
    axes_flat = axes.flatten()
    
    for idx, (metric_name, metric_type) in enumerate(METRIC_MAPPING.items()):
        ax = axes_flat[idx]
        
        if metric_type == 'feasibility':
            method_b = 'base_F_avg'
            method_e = 'der_F_avg'
        else:
            method_b = 'baseline_avg'
            method_e = 'derived_avg'
        
        means_b, stds_b = [], []
        means_e, stds_e = [], []
        
        print(f"\n{metric_name}:")
        for domain in domains:
            mean_b, std_b = calculate_domain_statistics(domain, metric_type, method_b)
            mean_e, std_e = calculate_domain_statistics(domain, metric_type, method_e)
            
            means_b.append(mean_b)
            stds_b.append(std_b)
            means_e.append(mean_e)
            stds_e.append(std_e)
            
            print(f"  {domain:12s} - B: {mean_b:.3f}±{std_b:.3f}  E: {mean_e:.3f}±{std_e:.3f}")
        
        x = np.arange(len(domains))
        offset = 0.15
        
        ax.errorbar(x - offset, means_b, yerr=stds_b, fmt='o', color=color_b,
                   markersize=10, capsize=5, capthick=2.5, linewidth=2.5,
                   label='Baseline', alpha=0.85)
        
        ax.errorbar(x + offset, means_e, yerr=stds_e, fmt='s', color=color_e,
                   markersize=10, capsize=5, capthick=2.5, linewidth=2.5,
                   label='Element', alpha=0.85)
        
        for i in range(len(x)):
            ax.plot([x[i] - offset, x[i] + offset], 
                   [means_b[i], means_e[i]], 
                   'k-', alpha=0.2, linewidth=1.2, zorder=0)
        
        ax.set_ylabel(metric_name, fontsize=24, weight='bold')
        ax.set_xticks(x)
        
        row = idx // 2
        if row == 1:
            ax.set_xticklabels(domain_labels, fontsize=20, rotation=45, ha='right')
        else:
            ax.set_xticklabels([])
        
        ax.tick_params(axis='y', labelsize=20)
        
        if idx == 0:
            ax.legend(fontsize=20, loc='upper center', frameon=True, 
                     framealpha=0.95, edgecolor='gray',
                     bbox_to_anchor=(1.1, 1.25), ncol=2)
        
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1, axis='y')
        ax.set_axisbelow(True)
        
        ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
        
        label = chr(97 + idx)
        ax.text(-0.12, 1.08, f'({label})', transform=ax.transAxes,
               fontsize=26, weight='bold', va='top')
    
    output_file = SCRIPT_DIR / "metrics_single_column_SciBERT.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\nPDF已保存: {output_file}")
    
    plt.close()
    
    print("\n" + "=" * 80)
    print("绘图完成")
    print("=" * 80)


if __name__ == '__main__':
    plot_single_column_figure()
