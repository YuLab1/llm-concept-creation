"""
语义熵分析与可视化 — 从评估数据计算并绘图。

读取 data/evaluation_{embedding}/ 下所有数据集的 entropy.json，
为每个嵌入模型生成柱状图，保存为 PDF。
默认使用 all-MiniLM-L6-v2 和 SciBERT。
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = config.RQ12_DIR / "data"

DATASETS = ["Comm", "Ele", "AI", "Comm+Ele", "Ele+AI", "Comm+AI"]
MODELS = ["GPT-4o", "GPT-4o-mini", "Qwen"]

FILE_CONFIG = [
    ("com_150_4o_single.json",       "4o",       "Comm"),
    ("com_150_4omini_single.json",   "4omini",   "Comm"),
    ("com_150_qw_single.json",       "qwen-max", "Comm"),
    ("ele_150_4o_single.json",       "4o",       "Ele"),
    ("ele_150_4omini_single.json",   "4omini",   "Ele"),
    ("ele_150_qw_single.json",       "qwen-max", "Ele"),
    ("ai_150_4o_single.json",        "4o",       "AI"),
    ("ai_150_4omini_single.json",    "4omini",   "AI"),
    ("ai_150_qw_single.json",        "qwen-max", "AI"),
    ("com_150_4o_crossele.json",     "4o",       "Comm+Ele"),
    ("com_150_4omini_crossele.json", "4omini",   "Comm+Ele"),
    ("com_150_qw_crossele.json",     "qwen-max", "Comm+Ele"),
    ("ele_150_4o_crossai.json",      "4o",       "Ele+AI"),
    ("ele_150_4omini_crossai.json",  "4omini",   "Ele+AI"),
    ("ele_150_qw_crossai.json",      "qwen-max", "Ele+AI"),
    ("com_150_4o_crossai.json",      "4o",       "Comm+AI"),
    ("com_150_4omini_crossai.json",  "4omini",   "Comm+AI"),
    ("com_150_qw_crossai.json",      "qwen-max", "Comm+AI"),
]

MODEL_IDX = {"4o": 0, "4omini": 1, "qwen-max": 2}

EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "evaluation_all-MiniLM-L6-v2",
    "SciBERT":          "evaluation_SciBERT",
}

colors = ['#D3D0EE', '#C4EEF5', '#EDF7D0']


def process_entropy(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ele_key = "element_entropy" if "element_entropy" in data[0] else "elment_entropy"
    element_vals = [item[ele_key] for item in data]
    domain_vals = [item["domain_entropy"] for item in data]
    return round(np.mean(element_vals), 3), round(np.mean(domain_vals), 3)


def build_entropy_table(eval_dir: Path, entropy_type: str = "domain"):
    table = np.zeros((len(DATASETS), len(MODELS)))
    for filename, model_folder, dataset in FILE_CONFIG:
        entropy_path = eval_dir / model_folder / filename / f"{filename}_entropy.json"
        if not entropy_path.exists():
            continue
        element_avg, domain_avg = process_entropy(entropy_path)
        ds_idx = DATASETS.index(dataset)
        m_idx = MODEL_IDX[model_folder]
        table[ds_idx, m_idx] = element_avg if entropy_type == "element" else domain_avg
    return table


def plot_entropy(table, emb_name: str, entropy_type: str = "domain"):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(14, 7))

    bar_width = 0.22
    x_single = np.arange(3) * 0.7
    x_cross = np.arange(3) * 0.7 + (x_single[-1] + 0.7) + 0.7
    x = np.concatenate((x_single, x_cross))

    for i in range(len(MODELS)):
        vals = table[:, i]
        bars = ax.bar(x + i * bar_width, vals,
                      width=bar_width, color=colors[i], label=MODELS[i], zorder=2)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom',
                    fontsize=14, color='black', weight='bold')

    x_ticks = x + (bar_width * (len(MODELS) - 1)) / 2
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(DATASETS, fontsize=18, weight='bold')

    legend = ax.legend(loc='upper left', bbox_to_anchor=(0.01, 1.05), frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor('none')
    frame.set_edgecolor('black')
    frame.set_alpha(0.0)
    for text in legend.get_texts():
        text.set_fontsize(16)

    last_left = x_single[-1] + (bar_width * len(MODELS) - bar_width) / 2
    first_right = x_cross[0] + (bar_width * len(MODELS) - bar_width) / 2
    ext = 0.4
    arrow_y = 0.4
    ax.annotate('', xy=(first_right - ext, arrow_y),
                xytext=(last_left + ext, arrow_y),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='gray'), fontsize=13)
    ax.text((last_left + ext + first_right - ext) / 2, arrow_y + 0.03,
            'Cross-domain', fontsize=18, ha='center', va='bottom',
            color='gray', weight='bold')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
        spine.set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.yaxis.set_tick_params(labelleft=False, left=False)
    ax.set_ylabel("")
    ax.set_ylim(bottom=0, top=table.max() * 1.15)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    plt.tight_layout()

    safe_name = emb_name.replace(" ", "_")
    out_pdf = SCRIPT_DIR / f"entropy_{entropy_type}_{safe_name}.pdf"
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    return out_pdf.name


def main():
    for emb_name, eval_folder in EMBEDDING_MODELS.items():
        eval_dir = DATA_DIR / eval_folder
        if not eval_dir.exists():
            print(f"[SKIP] {emb_name}: 目录不存在")
            continue

        print(f"处理: {emb_name}")
        for etype in ["domain", "element"]:
            table = build_entropy_table(eval_dir, etype)
            name = plot_entropy(table, emb_name, etype)
            print(f"  {name}")

    print("完成")


if __name__ == "__main__":
    main()
