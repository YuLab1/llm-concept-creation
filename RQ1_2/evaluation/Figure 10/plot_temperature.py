"""
温度实验可视化 — 读取评估数据，生成柱状图+折线图。

指标: SemDist (novelty), KWCR (coverage), AvgDis (pairwise_distance)
布局: (a) Baseline  (b) Element-Based
默认使用 all-MiniLM-L6-v2，支持 SciBERT 可选。
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RQ12_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = RQ12_DIR / "data"

TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9]

EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "temp_folder": "all-MiniLM-L6-v2",
        "eval_data":   "evaluation_all-MiniLM-L6-v2",
    },
    "SciBERT": {
        "temp_folder": "SciBERT",
        "eval_data":   "evaluation_SciBERT",
    },
}

METRICS = {
    "SemDist": ("novelty",           "baseline_avg", "derived_avg"),
    "KWCR":    ("coverage",          "baseline_avg", "derived_avg"),
    "AvgDis":  ("pairwise_distance", "baseline_avg", "derived_avg"),
}

colors = {
    'avg_dis':  '#2E86B5',
    'semdist':  '#2E9999',
    'kwcr':     '#F8C9C9',
}


def load_metric(json_path: Path, field: str) -> float:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.mean([item[field] for item in data])


def get_metric_path(emb_cfg: dict, temp: float, metric_file: str) -> Path:
    if temp == 0.7:
        return (DATA_DIR / emb_cfg["eval_data"] / "4o"
                / "com_150_4o_single.json"
                / f"com_150_4o_single.json_{metric_file}.json")
    else:
        stem = f"com_150_4o_single_t{temp}"
        return (SCRIPT_DIR / emb_cfg["temp_folder"]
                / stem / f"{stem}.json_{metric_file}.json")


def load_all_data(emb_cfg: dict):
    result = {}
    for metric_label, (metric_file, base_field, der_field) in METRICS.items():
        base_vals, der_vals = [], []
        for t in TEMPERATURES:
            path = get_metric_path(emb_cfg, t, metric_file)
            base_vals.append(load_metric(path, base_field))
            der_vals.append(load_metric(path, der_field))
        result[metric_label] = {
            "base": [round(v, 3) for v in base_vals],
            "der":  [round(v, 3) for v in der_vals],
        }
    return result


def plot_subplot(ax_bar, ax_line, x, kwcr, avg_dis, semdist, title_label):
    bar_width = 0.5
    ax_bar.bar(x, kwcr, width=bar_width, label='KWCR',
               color=colors['kwcr'], edgecolor='#994444', linewidth=0.8, zorder=3)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(t) for t in TEMPERATURES], fontsize=18)
    ax_bar.set_xlabel("Temperature", fontsize=18, color='black', fontweight='bold')
    ax_bar.tick_params(axis='y', labelcolor='black', labelsize=18)

    ax_line.plot(x, avg_dis, label='AvgDis', marker='^',
                 color=colors['avg_dis'], markersize=9, linewidth=2.5, zorder=2)
    ax_line.plot(x, semdist, label='SemDist', marker='s',
                 color=colors['semdist'], markersize=9, linewidth=2.5, zorder=2)
    ax_line.tick_params(axis='y', labelcolor='black', labelsize=18)

    for spine in ax_bar.spines.values():
        spine.set_linewidth(1.1)
    for spine in ax_line.spines.values():
        spine.set_linewidth(1.1)

    ax_bar.text(0.02, 0.98, title_label, transform=ax_bar.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left')


def plot_for_model(emb_name: str, emb_cfg: dict):
    data = load_all_data(emb_cfg)
    x = np.arange(len(TEMPERATURES))

    fig, (ax1_base, ax1_der) = plt.subplots(1, 2, figsize=(16, 6))

    ax2_base = ax1_base.twinx()
    plot_subplot(ax1_base, ax2_base, x,
                 data["KWCR"]["base"], data["AvgDis"]["base"],
                 data["SemDist"]["base"], '(a)')

    ax2_der = ax1_der.twinx()
    plot_subplot(ax1_der, ax2_der, x,
                 data["KWCR"]["der"], data["AvgDis"]["der"],
                 data["SemDist"]["der"], '(b)')

    lines_1, labels_1 = ax1_base.get_legend_handles_labels()
    lines_2, labels_2 = ax2_base.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), frameon=True, framealpha=0.9,
               fontsize=18, ncol=3)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_pdf = SCRIPT_DIR / f"temperature_{emb_cfg['temp_folder']}.pdf"
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"  保存: {out_pdf.name}")


def check_data_complete(emb_cfg: dict) -> list:
    missing = []
    for t in TEMPERATURES:
        path = get_metric_path(emb_cfg, t, "novelty")
        if not path.exists():
            missing.append(t)
    return missing


def main():
    for emb_name, emb_cfg in EMBEDDING_MODELS.items():
        missing = check_data_complete(emb_cfg)
        if missing:
            print(f"[SKIP] {emb_name}: 缺少温度 {missing}")
            continue
        print(f"绘图: {emb_name}")
        plot_for_model(emb_name, emb_cfg)

    print("完成")


if __name__ == "__main__":
    main()
