"""
步骤四：评估结果可视化

生成：
  1. 正文用精简图表（饼图 + 直方图）
  2. 附录用完整热力图（20×20 分数矩阵）
  3. 统计信息 JSON 摘要
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ------------------------------------------------------------------
#  数据加载
# ------------------------------------------------------------------

def load_data() -> Tuple[List[Dict], List[Dict], List[Dict],
                          List[Dict], List[Dict]]:
    with open(config.MATCH_RESULTS_FILE, 'r', encoding='utf-8') as f:
        baseline_matches = json.load(f)

    with open(config.GENERATED_FILE, 'r', encoding='utf-8') as f:
        generated = json.load(f)

    evaluation_scores = []
    if config.EVAL_SCORES_FILE.exists():
        with open(config.EVAL_SCORES_FILE, 'r', encoding='utf-8') as f:
            evaluation_scores = json.load(f)

    with open(config.COM_CONCEPTS_FILE, 'r', encoding='utf-8') as f:
        com_concepts = json.load(f)

    with open(config.AI_CONCEPTS_FILE, 'r', encoding='utf-8') as f:
        ai_concepts = json.load(f)

    return baseline_matches, generated, evaluation_scores, com_concepts, ai_concepts


# ------------------------------------------------------------------
#  分数合并与校验
# ------------------------------------------------------------------

def merge_and_complete_scores(baseline_matches: List[Dict],
                              generated: List[Dict],
                              evaluation_scores: List[Dict]) -> List[Dict]:
    exact_matched_ids = set()
    for m in baseline_matches:
        if m.get("definition_matched", False) and m.get("found_generated", False):
            gid = m.get("generated_id")
            if gid is not None:
                exact_matched_ids.add(gid)

    eval_baseline_ids = {s["id"] for s in evaluation_scores
                         if s.get("is_baseline_match", False)}

    if exact_matched_ids != eval_baseline_ids:
        print("[warn] Baseline-match ID mismatch")

    merged = sorted(evaluation_scores.copy(), key=lambda x: x["id"])
    expected_total = len(generated)
    if len(merged) != expected_total:
        print(f"[warn] Count mismatch: {len(merged)} vs {expected_total}")

    return merged


# ------------------------------------------------------------------
#  统计
# ------------------------------------------------------------------

def calculate_statistics(scores: List[Dict]) -> Dict:
    vals = [s["score"] for s in scores]
    return {
        "total": len(scores),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "baseline_matches": sum(1 for s in scores
                                if s.get("is_baseline_match", False)),
        "score_distribution": {
            "5.0 (Baseline)": sum(1 for s in scores if s["score"] == 5.0),
            "4.0": sum(1 for s in scores
                       if s["score"] == 4.0
                       and not s.get("is_baseline_match", False)),
            "3.0-3.99": sum(1 for s in scores if 3.0 <= s["score"] < 4.0),
            "2.0-2.99": sum(1 for s in scores if 2.0 <= s["score"] < 3.0),
            "1.0-1.99": sum(1 for s in scores if 1.0 <= s["score"] < 2.0),
        },
    }


# ------------------------------------------------------------------
#  正文图表
# ------------------------------------------------------------------

def create_main_figure(scores: List[Dict], stats: Dict, output_dir: Path):
    baseline_scores = [s["score"] for s in scores
                       if s.get("is_baseline_match", False)]
    other_scores = [s["score"] for s in scores
                    if not s.get("is_baseline_match", False)]

    bl_match = len(baseline_scores)
    bl_total = config.BASELINE_TOTAL

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    labels = [f'Matched\n({bl_match}/{bl_total})',
              f'Not Matched\n({bl_total - bl_match}/{bl_total})']
    sizes = [bl_match, bl_total - bl_match]
    colors = [config.SCIENTIFIC_COLORS['baseline'], '#E0E0E0']

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=(0.05, 0), labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=False, startangle=90,
        textprops={'fontsize': 22, 'weight': 'bold'})
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(24)
        at.set_weight('bold')

    ax2 = axes[1]
    bins = np.arange(1, 4.25, 0.25)
    n, bins_out, patches = ax2.hist(
        other_scores, bins=bins, alpha=0.7,
        color=config.SCIENTIFIC_COLORS['medium'],
        edgecolor='black', linewidth=0.8)

    for i, patch in enumerate(patches):
        center = (bins_out[i] + bins_out[i + 1]) / 2
        if center >= 3.5:
            patch.set_facecolor(config.SCIENTIFIC_COLORS['high'])
        elif center >= 2.5:
            patch.set_facecolor(config.SCIENTIFIC_COLORS['medium'])
        else:
            patch.set_facecolor(config.SCIENTIFIC_COLORS['low'])

    if other_scores:
        omean = np.mean(other_scores)
        omed = np.median(other_scores)
        ax2.axvline(omean, color='blue', linestyle='--', linewidth=2,
                    label=f'Mean: {omean:.2f}')
        ax2.axvline(omed, color='orange', linestyle='--', linewidth=2,
                    label=f'Median: {omed:.2f}')

    ax2.set_xlabel('Quality Score (1-4)', fontsize=26, weight='bold')
    ax2.set_ylabel('Frequency', fontsize=26, weight='bold')
    ax2.set_xlim(0.8, 4.2)
    if len(n) > 0:
        ax2.set_ylim(0, max(n) * 1.1)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.legend(fontsize=22, loc='upper left')

    plt.tight_layout()

    pdf_path = output_dir / "evaluation_main_figure.pdf"
    png_path = output_dir / "evaluation_main_figure.png"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------
#  附录热力图
# ------------------------------------------------------------------

def create_appendix_heatmap(scores: List[Dict],
                            com_concepts: List[Dict],
                            ai_concepts: List[Dict],
                            output_dir: Path):
    com_list = [c["concept"] for c in com_concepts]
    ai_list = [c["concept"] for c in sorted(ai_concepts, key=lambda x: x["id"])]
    n = len(com_list)

    score_matrix = np.zeros((n, n))
    baseline_matrix = np.zeros((n, n))

    for s in scores:
        if s["communication_concept"] in com_list:
            row = com_list.index(s["communication_concept"])
            col = (s["id"] - 1) % n
            score_matrix[row, col] = s["score"]
            if s.get("is_baseline_match", False):
                baseline_matrix[row, col] = 1

    def _wrap(text, maxlen=25):
        if len(text) <= maxlen:
            return text
        if ' / ' in text:
            parts = text.split(' / ')
            res = parts[0]
            for p in parts[1:]:
                if len(res) + len(p) + 3 <= maxlen:
                    res += ' / ' + p
                else:
                    res += '\n' + p
            return res
        if ' (' in text:
            idx = text.find(' (')
            if 0 < idx < maxlen:
                return text[:idx] + '\n' + text[idx + 1:]
        words = text.split()
        res = words[0]
        for w in words[1:]:
            if len(res) + len(w) + 1 <= maxlen:
                res += ' ' + w
            else:
                res += '\n' + w
        return res

    com_labels = [_wrap(l) for l in com_list]
    ai_labels = [_wrap(l) for l in ai_list]

    fig, ax = plt.subplots(figsize=(36, 32))
    sns.heatmap(
        score_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
        vmin=1, vmax=5, center=3, cbar=True,
        cbar_kws={'shrink': 0.8, 'label': 'Quality Score'},
        xticklabels=ai_labels, yticklabels=com_labels,
        linewidths=0.5, linecolor='white', square=True, ax=ax,
        annot_kws={'size': 20, 'weight': 'bold'})

    for i in range(n):
        for j in range(n):
            if baseline_matrix[i, j] == 1:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=False,
                    edgecolor='blue', linewidth=2.5))

    ax.set_xlabel('AI Concept', fontsize=28, weight='bold')
    ax.set_ylabel('Communication Concept', fontsize=28, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    for lb in ax.get_xticklabels():
        lb.set_fontsize(20)
        lb.set_weight('bold')
    for lb in ax.get_yticklabels():
        lb.set_fontsize(20)
        lb.set_weight('bold')

    if len(fig.axes) > 1:
        cbar_ax = fig.axes[-1]
        cbar_ax.set_ylabel('Quality Score', fontsize=20,
                           weight='bold', labelpad=20)
        cbar_ax.tick_params(labelsize=16)

    plt.tight_layout()

    pdf_path = output_dir / "evaluation_appendix_heatmap.pdf"
    png_path = output_dir / "evaluation_appendix_heatmap.png"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------
#  主函数
# ------------------------------------------------------------------

def main(output_dir: Path = None):
    plt.rcParams.update(config.PLOT_PARAMS)

    output_dir = Path(output_dir) if output_dir else config.RQ3_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("RQ3 可视化")

    baseline_matches, generated, eval_scores, com_concepts, ai_concepts = load_data()
    merged = merge_and_complete_scores(baseline_matches, generated, eval_scores)
    stats = calculate_statistics(merged)

    stats_path = output_dir / "evaluation_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    create_main_figure(merged, stats, output_dir)
    create_appendix_heatmap(merged, com_concepts, ai_concepts, output_dir)

    print("完成")


if __name__ == "__main__":
    main()
