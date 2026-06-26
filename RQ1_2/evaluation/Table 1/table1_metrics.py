"""
Table 1 — 评估指标 LaTeX 表格生成

从 data/evaluation_all-MiniLM-L6-v2 文件夹读取 novelty / coverage / pairwise_distance / feasibility
四类评估结果，计算各指标平均值，生成 LaTeX 表格。
支持通过 --embedding 参数切换嵌入模型（all-MiniLM-L6-v2 / SciBERT）。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config

# ============================================================
#  参数
# ============================================================
EMBEDDING_DIRS = {
    "all-MiniLM-L6-v2": config.RQ12_DIR / "data" / "evaluation_all-MiniLM-L6-v2",
    "SciBERT": config.RQ12_DIR / "data" / "evaluation_SciBERT",
}
DEFAULT_EMBEDDING = "all-MiniLM-L6-v2"
EVAL_DATA_DIR = EMBEDDING_DIRS[DEFAULT_EMBEDDING]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

DATASETS = ["Comm", "Ele", "AI", "Comm+Ele", "Comm+AI", "Ele+AI"]
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
    ("com_150_4o_crossai.json",      "4o",       "Comm+AI"),
    ("com_150_4omini_crossai.json",  "4omini",   "Comm+AI"),
    ("com_150_qw_crossai.json",      "qwen-max", "Comm+AI"),
    ("ele_150_4o_crossai.json",      "4o",       "Ele+AI"),
    ("ele_150_4omini_crossai.json",  "4omini",   "Ele+AI"),
    ("ele_150_qw_crossai.json",      "qwen-max", "Ele+AI"),
]

MODEL_FOLDER_TO_IDX = {"4o": 0, "4omini": 1, "qwen-max": 2}

METRIC_ROWS = [
    ("SemDist", "B", "novelty.json",            "baseline_avg"),
    ("SemDist", "E", "novelty.json",            "derived_avg"),
    ("KWCR",    "B", "coverage.json",           "baseline_avg"),
    ("KWCR",    "E", "coverage.json",           "derived_avg"),
    ("AvgDis",  "B", "pairwise_distance.json",  "baseline_avg"),
    ("AvgDis",  "E", "pairwise_distance.json",  "derived_avg"),
    ("Pla-S",   "B", "feasibility.json",        "base_F_avg"),
    ("Pla-S",   "E", "feasibility.json",        "der_F_avg"),
]


# ============================================================
#  数据处理
# ============================================================

def load_metric_avg(data_dir: Path, model_folder: str,
                    filename: str, metric_file: str,
                    field_name: str) -> float:
    full_path = (data_dir / model_folder / filename
                 / f"{filename}_{metric_file}")
    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    values = [item[field_name] for item in data]
    return round(np.mean(values), 3)


def build_all_values(data_dir: Path = None) -> dict:
    if data_dir is None:
        data_dir = EVAL_DATA_DIR

    result = {}
    for metric_name, method, metric_file, field_name in METRIC_ROWS:
        key = f"{metric_name}_{method}"
        values = []
        for filename, model_folder, dataset in FILE_CONFIG:
            v = load_metric_avg(data_dir, model_folder, filename,
                                metric_file, field_name)
            values.append(v)
        result[key] = values

    return result


# ============================================================
#  LaTeX 表格生成
# ============================================================

def generate_latex_table(all_values: dict) -> list:
    lines = []
    lines.append("\\begin{table*}[hbtp]")
    lines.append("\\centering")
    lines.append("\\adjustbox{width=0.98\\textwidth,center}")
    lines.append("{")
    lines.append("\\scriptsize")
    lines.append("\\renewcommand{\\arraystretch}{1.3}")
    lines.append(
        "\\begin{tabular}{@{}ll@{\\hspace{0.8pt}}|@{\\hspace{0.8pt}}"
        "ccc@{\\hspace{0.8pt}}|@{\\hspace{0.8pt}}"
        "ccc@{\\hspace{0.8pt}}|@{\\hspace{0.8pt}}"
        "ccc@{\\hspace{1pt}}|@{\\hspace{1pt}}"
        "ccc@{\\hspace{0.8pt}}|@{\\hspace{0.8pt}}"
        "ccc@{\\hspace{0.8pt}}|@{\\hspace{0.8pt}}"
        "ccc@{}}"
    )
    lines.append("\\toprule")

    lines.append(
        "& & \\multicolumn{9}{c|}{\\textbf{Intra-domain}}"
        " & \\multicolumn{9}{c}{\\textbf{Cross-domain}} \\\\"
    )
    lines.append("\\cmidrule(lr){3-11} \\cmidrule(l){12-20}")
    lines.append("\\textbf{Metric} & \\textbf{Method}")
    lines.append("& \\multicolumn{3}{c@{\\hspace{0.8pt}}|}{\\textbf{Comm}}")
    lines.append("& \\multicolumn{3}{c@{\\hspace{0.8pt}}|}{\\textbf{Ele}}")
    lines.append("& \\multicolumn{3}{c@{\\hspace{1pt}}|}{\\textbf{AI}}")
    lines.append("& \\multicolumn{3}{c@{\\hspace{0.8pt}}|}{\\textbf{Comm+Ele}}")
    lines.append("& \\multicolumn{3}{c@{\\hspace{0.8pt}}|}{\\textbf{Comm+AI}}")
    lines.append("& \\multicolumn{3}{c}{\\textbf{Ele+AI}} \\\\")
    lines.append(
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}"
        " \\cmidrule(lr){12-14} \\cmidrule(lr){15-17} \\cmidrule(l){18-20}"
    )
    lines.append(
        "& & \\textbf{4o} & \\textbf{4o-mini} & \\textbf{Qwen}\n"
        "& \\textbf{4o} & \\textbf{4o-mini} & \\textbf{Qwen}\n"
        "& \\textbf{4o} & \\textbf{4o-mini} & \\textbf{Qwen}\n"
        "& \\textbf{4o} & \\textbf{4o-mini} & \\textbf{Qwen}\n"
        "& \\textbf{4o} & \\textbf{4o-mini} & \\textbf{Qwen}\n"
        "& \\textbf{4o} & \\textbf{4o-mini} & \\textbf{Qwen} \\\\"
    )
    lines.append("\\midrule")

    metric_groups = [
        ("SemDist", "SemDist_B", "SemDist_E"),
        ("KWCR",    "KWCR_B",    "KWCR_E"),
        ("AvgDis",  "AvgDis_B",  "AvgDis_E"),
        ("Pla-S",   "Pla-S_B",   "Pla-S_E"),
    ]

    for gi, (label, b_key, e_key) in enumerate(metric_groups):
        lines.append(f"\\multirow{{2}}{{*}}{{\\textbf{{{label}}}}}")

        b_vals = all_values[b_key]
        b_line = "& B & " + " & ".join(f"{v:.3f}" for v in b_vals) + " \\\\"
        lines.append(b_line)

        e_vals = all_values[e_key]
        e_line = ("& E & "
                  + " & ".join(f"\\cellcolor{{gray!15}}{v:.3f}" for v in e_vals)
                  + " \\\\")
        lines.append(e_line)

        if gi < len(metric_groups) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append(
        "\\caption{Transposed quantitative evaluation across domains and "
        "models. B: Prompt-Only Baseline, E: Structured Element-Based "
        "Innovation (highlighted with gray background). "
        "Models: 4o (GPT-4o), 4o-mini (GPT-4o-mini), Qwen (Qwen-max).}"
    )
    lines.append("\\label{tab:eval_metrics_template}")
    lines.append("\\end{table*}")

    return lines


# ============================================================
#  主流程
# ============================================================

def main(output_dir=None, embedding=None):
    global EVAL_DATA_DIR
    if embedding and embedding in EMBEDDING_DIRS:
        EVAL_DATA_DIR = EMBEDDING_DIRS[embedding]

    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_label = embedding or DEFAULT_EMBEDDING
    print(f"Table 1 生成 [{emb_label}]")

    all_values = build_all_values()

    suffix = "" if emb_label == DEFAULT_EMBEDDING else f"_{emb_label}"
    out_file = output_dir / f"Table 1{suffix}.txt"
    latex_lines = generate_latex_table(all_values)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"已保存: {out_file}")

    return all_values


def parse_args():
    p = argparse.ArgumentParser(description="Table 1 — 评估指标 LaTeX 表格生成")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--embedding", default=DEFAULT_EMBEDDING,
                   choices=list(EMBEDDING_DIRS.keys()))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(output_dir=args.output_dir, embedding=args.embedding)
