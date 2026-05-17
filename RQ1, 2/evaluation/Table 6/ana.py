"""
语义偏移分析 — 从嵌入 pkl 计算欧氏距离 / 角度偏移，生成 LaTeX 表格。

默认使用 all-MiniLM-L6-v2 嵌入，可通过 --embedding 切换为 SciBERT。
"""

import argparse
import pickle
import sys
import math
import numpy as np
from scipy.spatial.distance import euclidean
from numpy.linalg import norm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config

SCRIPT_DIR = Path(__file__).parent

EMBEDDING_DIRS = {
    "all-MiniLM-L6-v2": config.RQ12_DIR / "data" / "evaluation_all-MiniLM-L6-v2" / "4o",
    "SciBERT": config.RQ12_DIR / "data" / "evaluation_SciBERT" / "4o",
}
DEFAULT_EMBEDDING = "all-MiniLM-L6-v2"

FILE_CONFIG = [
    ("com_150_4o_single.json",   "Comm"),
    ("com_150_4o_crossele.json", "Comm+Ele"),
    ("com_150_4o_crossai.json",  "Comm+AI"),
]


def load_emb_data(base_dir: Path, filename: str):
    pkl_path = base_dir / filename / f"{filename}_emb_data.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_distances_and_angles(origin_embed, generated_embeds):
    """计算生成向量与原始向量的欧氏距离与方向角度。"""
    if len(generated_embeds) == 0:
        return {
            "euclidean_mean": 0.0, "euclidean_variance": 0.0,
            "angle_mean_deg": 0.0, "angle_variance_deg": 0.0,
        }

    origin_embed = np.array(origin_embed)
    generated_embeds_array = np.array(generated_embeds)

    ref_direction = np.mean(generated_embeds_array, axis=0) - origin_embed
    ref_norm = norm(ref_direction)
    ref_unit = ref_direction / ref_norm if ref_norm > 0 else np.zeros_like(ref_direction)

    results = []
    for gen_embed in generated_embeds_array:
        vec = gen_embed - origin_embed
        euc = euclidean(origin_embed, gen_embed)
        vec_norm = norm(vec)
        unit_vec = vec / vec_norm if vec_norm > 0 else np.zeros_like(vec)
        dot_product = np.clip(np.dot(unit_vec, ref_unit), -1.0, 1.0)
        angle_deg = math.degrees(np.arccos(dot_product))
        results.append({"euclidean": round(euc, 4), "angle_deg": round(angle_deg, 2)})

    euclidean_distances = np.array([r["euclidean"] for r in results])
    angles_deg = np.array([r["angle_deg"] for r in results])

    return {
        "euclidean_mean": round(float(np.mean(euclidean_distances)), 4),
        "euclidean_variance": round(float(np.var(euclidean_distances)), 4),
        "angle_mean_deg": round(float(np.mean(angles_deg)), 2),
        "angle_variance_deg": round(float(np.var(angles_deg)), 2),
    }


def compute_deviation(emb_data):
    concept_stats = []
    for rec in emb_data:
        ori_emb = rec["desc_ori_embs"]
        der_embs = []
        for group in rec["derived"]:
            der_embs.extend(group["desc_embs"])
        if len(der_embs) > 0:
            stats = compute_distances_and_angles(ori_emb, der_embs)
            concept_stats.append(stats)
    return concept_stats


def summarize(concept_stats):
    eu_means = [s["euclidean_mean"] for s in concept_stats]
    eu_vars = [s["euclidean_variance"] for s in concept_stats]
    ang_means = [s["angle_mean_deg"] for s in concept_stats]
    ang_vars = [s["angle_variance_deg"] for s in concept_stats]
    return (
        round(np.mean(eu_means), 3),
        round(np.mean(eu_vars), 3),
        round(np.mean(ang_means), 2),
        round(np.mean(ang_vars), 2),
    )


def generate_latex(rows):
    lines = [
        r"\begin{table}[hbtp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Euc.Mean} & \textbf{Euc.Var} & \textbf{Ang.Mean} & \textbf{Ang.Var} \\",
        r"\midrule",
    ]
    for domain, (em, ev, am, av) in rows:
        lines.append(f"{domain:<14s} &{em:.3f} &{ev} & {am} & {av} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Semantic deviation analysis.}",
        r"\label{tab:distance_angle_stats}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main(embedding=None):
    emb_label = embedding or DEFAULT_EMBEDDING
    base_dir = EMBEDDING_DIRS[emb_label]

    print(f"语义偏移分析 [{emb_label}]")

    rows = []
    for filename, domain in FILE_CONFIG:
        emb_data = load_emb_data(base_dir, filename)
        stats = compute_deviation(emb_data)
        result = summarize(stats)
        print(f"  {domain}: Euc={result[0]:.3f}, Ang={result[2]:.2f}")
        rows.append((domain, result))

    latex = generate_latex(rows)
    suffix = "" if emb_label == DEFAULT_EMBEDDING else f"_{emb_label}"
    out_file = SCRIPT_DIR / f"deviation_table{suffix}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(latex + "\n")
    print(f"完成: {out_file.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="语义偏移分析")
    p.add_argument("--embedding", default=DEFAULT_EMBEDDING,
                   choices=list(EMBEDDING_DIRS.keys()))
    args = p.parse_args()
    main(embedding=args.embedding)
