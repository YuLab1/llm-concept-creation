"""
Table 5 — Feasibility 评估结果 LaTeX 表格生成

从 data/evaluation_all-MiniLM-L6-v2 文件夹下的 18 个 feasibility JSON 文件中读取数据，
计算各指标平均值，生成包含 S_fwd / S_bwd / Pla-S 的 LaTeX 表格。
支持通过 --embedding 参数切换嵌入模型。
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
METRICS = [
    'base_F1_avg', 'base_F2_avg', 'base_F_avg',
    'der_F1_avg', 'der_F2_avg', 'der_F_avg',
]

FILE_CONFIG = [
    ("com_150_4o_single.json", "4o", "Comm"),
    ("com_150_4omini_single.json", "4omini", "Comm"),
    ("com_150_qw_single.json", "qwen-max", "Comm"),
    ("ele_150_4o_single.json", "4o", "Ele"),
    ("ele_150_4omini_single.json", "4omini", "Ele"),
    ("ele_150_qw_single.json", "qwen-max", "Ele"),
    ("ai_150_4o_single.json", "4o", "AI"),
    ("ai_150_4omini_single.json", "4omini", "AI"),
    ("ai_150_qw_single.json", "qwen-max", "AI"),
    ("com_150_4o_crossele.json", "4o", "Comm+Ele"),
    ("com_150_4omini_crossele.json", "4omini", "Comm+Ele"),
    ("com_150_qw_crossele.json", "qwen-max", "Comm+Ele"),
    ("com_150_4o_crossai.json", "4o", "Comm+AI"),
    ("com_150_4omini_crossai.json", "4omini", "Comm+AI"),
    ("com_150_qw_crossai.json", "qwen-max", "Comm+AI"),
    ("ele_150_4o_crossai.json", "4o", "Ele+AI"),
    ("ele_150_4omini_crossai.json", "4omini", "Ele+AI"),
    ("ele_150_qw_crossai.json", "qwen-max", "Ele+AI"),
]

MODEL_FOLDER_TO_IDX = {"4o": 0, "4omini": 1, "qwen-max": 2}


# ============================================================
#  数据处理
# ============================================================

def process_feasibility(file_path: Path) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"数据格式错误: {file_path}")

    collectors = {m: [] for m in METRICS}
    for item in data:
        for m in METRICS:
            if m in item:
                collectors[m].append(item[m])

    return {
        m: round(np.mean(v), 3) if v else 0.0
        for m, v in collectors.items()
    }


def build_tables(data_dir: Path = None):
    if data_dir is None:
        data_dir = EVAL_DATA_DIR

    tables = {m: np.zeros((6, 3)) for m in METRICS}

    for filename, model_folder, dataset in FILE_CONFIG:
        feasibility_path = (
            data_dir / model_folder / filename
            / f"{filename}_feasibility.json"
        )

        if not feasibility_path.exists():
            print(f"  [WARN] 文件不存在: {feasibility_path.name}")
            continue

        result = process_feasibility(feasibility_path)
        ds_idx = DATASETS.index(dataset)
        m_idx = MODEL_FOLDER_TO_IDX[model_folder]

        for metric in METRICS:
            tables[metric][ds_idx, m_idx] = result[metric]

    return tables


# ============================================================
#  LaTeX 表格生成
# ============================================================

def generate_latex_table(tables: dict) -> list:
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
        ('S_{fwd}', 'base_F1_avg', 'der_F1_avg'),
        ('S_{bwd}', 'base_F2_avg', 'der_F2_avg'),
        ('Pla-S',   'base_F_avg',  'der_F_avg'),
    ]

    for gi, (label, base_key, der_key) in enumerate(metric_groups):
        lines.append(f"\\multirow{{2}}{{*}}{{\\textbf{{{label}}}}}")

        b_parts = ["& B"]
        for ds_idx in range(6):
            for m_idx in range(3):
                b_parts.append(f" & {tables[base_key][ds_idx, m_idx]:.3f}")
        b_parts.append(" \\\\")
        lines.append("".join(b_parts))

        e_parts = ["& E"]
        for ds_idx in range(6):
            for m_idx in range(3):
                v = tables[der_key][ds_idx, m_idx]
                e_parts.append(f" & \\cellcolor{{gray!15}}{v:.3f}")
        e_parts.append(" \\\\")
        lines.append("".join(e_parts))

        if gi < len(metric_groups) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append(
        "\\caption{Feasibility evaluation across domains and models. "
        "B: Prompt-Only Baseline, E: Structured Element-Based Innovation "
        "(highlighted with gray background). "
        "Models: 4o (GPT-4o), 4o-mini (GPT-4o-mini), Qwen (Qwen-max).}"
    )
    lines.append("\\label{tab:feasibility_eval}")
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
    print(f"Table 5 Feasibility 生成 [{emb_label}]")

    tables = build_tables()

    latex_lines = generate_latex_table(tables)
    latex_text = "\n".join(latex_lines)

    out_file = output_dir / "feasibility_table_with_pla.txt"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(latex_text)

    summary = {
        'datasets': DATASETS,
        'models': MODELS,
        'metrics': {m: tables[m].tolist() for m in METRICS},
    }
    summary_file = output_dir / "feasibility_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"完成: {out_file.name}, {summary_file.name}")
    return tables


def parse_args():
    p = argparse.ArgumentParser(description="Table 5 — Feasibility LaTeX 表格生成")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--embedding", default=DEFAULT_EMBEDDING,
                   choices=list(EMBEDDING_DIRS.keys()))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(output_dir=args.output_dir, embedding=args.embedding)
