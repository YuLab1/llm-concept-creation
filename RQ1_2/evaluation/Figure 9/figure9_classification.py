"""
Figure 9 — 概念分类分析与双层饼图生成

从 generation concepts 文件夹下的 18 个文件中提取 baseline 数据，
使用 gpt-4o-mini 进行分类分析（Rephrasing / Incremental / Radical），
生成双层饼图。
"""

import argparse
import json
import random
import sys
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm

# Figure 9 → evaluation → RQ1_2 → Code
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config
from llm_utils import create_client, extract_json_from_text

# ============================================================
#  参数
# ============================================================
SAMPLES_PER_FILE = 100
CLASSIFICATION_MODEL = "gpt-4o-mini"
CLASSIFICATION_TEMPERATURE = 0.3
CLASSIFICATION_MAX_TOKENS = 200

DATA_DIR = config.RQ12_DIR / "data" / "generation concepts"
MODEL_FOLDERS = ["GPT-4o", "GPT-4omini", "Qwen-max"]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

random.seed(42)


# ============================================================
#  缓存管理
# ============================================================

def load_cache(cache_file: Path) -> Dict[str, Dict]:
    """从磁盘加载分类缓存。"""
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cache(cache: Dict[str, Dict], cache_file: Path):
    """保存分类缓存到磁盘。"""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def make_cache_key(source_name: str, source_def: str,
                   gen_name: str, gen_def: str) -> str:
    content = f"{source_name}|{source_def}|{gen_name}|{gen_def}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()


# ============================================================
#  LLM 分类
# ============================================================

def classify_concept(client, source_name: str, source_def: str,
                     gen_name: str, gen_def: str,
                     cache: Dict, max_retries: int = config.MAX_RETRIES
                     ) -> Dict[str, Any]:
    """使用 gpt-4o-mini 对概念对进行三分类。"""
    cache_key = make_cache_key(source_name, source_def, gen_name, gen_def)
    if cache_key in cache:
        return cache[cache_key]

    prompt = f"""You are an expert taxonomy analyst for scientific concepts.
Your task is to classify the relationship between a Source Concept and a Generated Concept into one of the following three categories.

**Input Data:**
- Source Concept: {source_name} - {source_def}
- Generated Concept: {gen_name} - {gen_def}

**Categories:**
1. [Rephrasing]: The generated concept is merely a paraphrase or a trivial retrieval of the source. No new semantic information is added.
2. [Incremental]: The generated concept extends the source with logical but predictable modifications (e.g., adding a common attribute, combining with a closely related concept). It represents a "safe" step.
3. [Radical]: The generated concept deviates significantly from the source. It introduces entirely new paradigms, OR it is logically disconnected/nonsensical.

Output JSON format only, no additional content:
{{
    "category": "Rephrasing" or "Incremental" or "Radical",
    "explanation": "1-2 sentence explanation"
}}"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=CLASSIFICATION_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are an expert taxonomy analyst. "
                                "Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=CLASSIFICATION_TEMPERATURE,
                max_tokens=CLASSIFICATION_MAX_TOKENS,
            )
            text = response.choices[0].message.content
            result = extract_json_from_text(text)

            if result is None:
                raise ValueError("JSON 提取失败")
            if "category" not in result or "explanation" not in result:
                raise ValueError("响应缺少必要字段")
            if result["category"] not in ("Rephrasing", "Incremental", "Radical"):
                raise ValueError(f"无效分类: {result['category']}")

            cache[cache_key] = result
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)
            else:
                error_result = {
                    "category": "Radical",
                    "explanation": f"分类失败: {str(e)}"
                }
                cache[cache_key] = error_result
                return error_result

    return {"category": "Radical", "explanation": "分类失败"}


# ============================================================
#  数据提取与抽样
# ============================================================

def load_and_extract_data(file_path: Path) -> List[Dict[str, Any]]:
    """从单个 JSON 文件提取 (source, baseline) 概念对。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    extracted = []
    filename = file_path.name
    folder_name = file_path.parent.name

    for item in data:
        original_concept = item.get('original_concept', '')
        original_description = item.get('original_description', '')
        baseline = item.get('baseline', [])

        if not original_concept or not original_description or not baseline:
            continue

        for bl_item in baseline:
            if isinstance(bl_item, dict):
                term = bl_item.get('redefined_term', '')
                concept = bl_item.get('redefined_concept', '')
                if term and concept:
                    extracted.append({
                        'source_name': original_concept,
                        'source_definition': original_description,
                        'generated_name': term,
                        'generated_definition': concept,
                        'source_file': filename,
                        'source_folder': folder_name,
                    })

    return extracted


def process_all_files(data_dir: Path = None) -> List[Dict[str, Any]]:
    """遍历所有模型文件夹，提取全部 baseline 概念对。"""
    if data_dir is None:
        data_dir = DATA_DIR
    all_data = []

    for folder_name in MODEL_FOLDERS:
        folder_path = data_dir / folder_name
        if not folder_path.exists():
            print(f"  警告: 文件夹 {folder_path} 不存在")
            continue

        json_files = sorted(folder_path.glob("*.json"))
        print(f"\n  处理文件夹: {folder_name} ({len(json_files)} 个文件)")

        for jf in json_files:
            try:
                extracted = load_and_extract_data(jf)
                all_data.extend(extracted)
                print(f"    {jf.name}: {len(extracted)} 组")
            except Exception as e:
                print(f"    {jf.name}: 错误 - {e}")

    print(f"\n  总计提取 {len(all_data)} 组数据")
    return all_data


def stratified_sampling(all_data: List[Dict],
                        samples_per_file: int = SAMPLES_PER_FILE
                        ) -> List[Dict]:
    """按文件分层随机抽样。"""
    file_groups = defaultdict(list)
    for item in all_data:
        file_groups[item['source_file']].append(item)

    sampled = []
    for fname, items in sorted(file_groups.items()):
        n = min(samples_per_file, len(items))
        sampled.extend(random.sample(items, n))
        print(f"    {fname}: 抽取 {n} 组")

    return sampled


# ============================================================
#  可视化 — 双层饼图 (Figure 10)
# ============================================================

def detect_domain_type(filename: str) -> str:
    """从文件名判断单领域 / 跨领域。"""
    fn = filename.lower()
    if "single" in fn:
        return "single"
    elif "cross" in fn:
        return "cross"
    return "unknown"


def hex_to_rgb(hex_color: str):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def create_nested_pie_chart(df: pd.DataFrame, output_dir: Path):
    """生成双层饼图（外层：主分类，内层：single/cross，图例在右侧）。"""
    sns.set_style("white")
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # --- 统计 ---
    stats = defaultdict(lambda: {"single": 0, "cross": 0})
    for _, row in df.iterrows():
        cat = row['category']
        dt = detect_domain_type(row['source_file'])
        if dt in ("single", "cross"):
            stats[cat][dt] += 1

    category_stats = {}
    for cat, dc in stats.items():
        total = dc["single"] + dc["cross"]
        if total > 0:
            category_stats[cat] = {
                "single": dc["single"], "cross": dc["cross"], "total": total
            }

    if not category_stats:
        print("  警告: 无有效分类数据，跳过饼图生成")
        return

    sorted_categories = sorted(category_stats.items(),
                               key=lambda x: x[1]["total"], reverse=True)

    # --- 配色 ---
    outer_colors = [hex_to_rgb("6488B3"), hex_to_rgb("C1BD3D")]
    inner_colors = [hex_to_rgb("55AC88"), hex_to_rgb("FFAF34"),
                    hex_to_rgb("BB7692"), hex_to_rgb("F06F69")]

    category_colors = {
        cat: outer_colors[i % len(outer_colors)]
        for i, (cat, _) in enumerate(sorted_categories)
    }

    inner_color_dict = {
        ("Radical", "Single"): inner_colors[0],
        ("Incremental", "Single"): inner_colors[1],
        ("Incremental", "Cross"): inner_colors[2],
        ("Radical", "Cross"): inner_colors[3],
    }

    # --- 构造绘图数据 ---
    total_all = sum(s["total"] for _, s in sorted_categories)
    outer_sizes, outer_labels = [], []
    inner_sizes, inner_labels_full, inner_color_map = [], [], []
    inner_entries_all = []

    for category, s in sorted_categories:
        outer_labels.append(category)
        outer_sizes.append(s["total"])

        entries = []
        if s["single"] > 0:
            entries.append(("Single", s["single"], category))
        if s["cross"] > 0:
            entries.append(("Cross", s["cross"], category))
        entries.sort(key=lambda x: x[1], reverse=True)

        for domain_type, count, cat in entries:
            inner_labels_full.append(f"{cat} {domain_type}")
            inner_sizes.append(count)
            inner_color_map.append(
                inner_color_dict.get((cat, domain_type), (0.8, 0.8, 0.8)))
            inner_entries_all.append(
                (cat, domain_type, count, count / total_all * 100))

    outer_percentages = [sz / total_all * 100 for sz in outer_sizes]
    inner_pct_map = {
        f"{cat} {dt}": pct
        for cat, dt, _, pct in inner_entries_all
    }

    # --- 画布与布局 ---
    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=300)
    fig.subplots_adjust(left=0.03, right=0.83, top=0.97, bottom=0.06)
    ax.set_position([0.03, 0.10, 0.58, 0.80])

    # --- 外环：主类别 ---
    outer_wedges, _ = ax.pie(
        outer_sizes, labels=None, startangle=90, counterclock=False,
        colors=[category_colors[c] for c in outer_labels],
        wedgeprops=dict(width=0.34, edgecolor='white', linewidth=1.6),
        radius=1.0,
    )

    # --- 内环：细分类 ---
    ax.pie(
        inner_sizes, labels=None, startangle=90, counterclock=False,
        colors=inner_color_map,
        wedgeprops=dict(width=0.28, edgecolor='white', linewidth=1.2),
        radius=0.66,
    )

    # --- 外层标签 ---
    for wedge, pct, label in zip(outer_wedges, outer_percentages, outer_labels):
        angle = np.deg2rad((wedge.theta2 + wedge.theta1) / 2)
        x = 0.76 * np.cos(angle)
        y = 0.76 * np.sin(angle)
        ax.text(x, y, f"{label}\n{pct:.1f}%",
                ha='center', va='center', fontsize=15.5,
                fontweight='bold', color='black')

    # --- 图例：右侧，按同类相邻排列 ---
    inner_legend_order = [
        ("Incremental Single", inner_colors[1]),
        ("Incremental Cross", inner_colors[2]),
        ("Radical Single", inner_colors[0]),
        ("Radical Cross", inner_colors[3]),
    ]

    seen_labels = set(inner_labels_full)
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='none', label='  Subcategory')
    ]

    for label, color in inner_legend_order:
        if label in seen_labels:
            pct = inner_pct_map[label]
            legend_elements.append(
                mpatches.Patch(
                    facecolor=color, edgecolor='white', linewidth=1.0,
                    label=f"   {label} ({pct:.1f}%)",
                ))

    leg = ax.legend(
        handles=legend_elements, loc='center left',
        bbox_to_anchor=(0.95, 0.5), fontsize=12.5,
        frameon=True, fancybox=False, edgecolor='#cccccc',
        framealpha=1.0, handlelength=1.05, handleheight=0.95,
        handletextpad=0.28, labelspacing=0.36, borderpad=0.78,
    )
    leg.get_frame().set_linewidth(0.6)

    for i, text in enumerate(leg.get_texts()):
        if text.get_text().strip() == "Subcategory":
            leg.legend_handles[i].set_visible(False)
            text.set_fontsize(13.5)
            text.set_fontweight('heavy')

    # --- 收紧边界 ---
    ax.set_xlim(-1.05, 1.08)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- 导出 ---
    for ext in ('pdf', 'png'):
        out_path = output_dir / f"classification_pie_chart.{ext}"
        plt.savefig(str(out_path), bbox_inches='tight',
                    facecolor='white', pad_inches=0.01, dpi=300)
    plt.close()

    print(f"  饼图已保存: {output_dir / 'classification_pie_chart.pdf'}")

    # --- 统计信息 ---
    print("\n  双层饼图统计:")
    for cat, s in sorted_categories:
        pct = s["total"] / total_all * 100
        s_pct = s["single"] / s["total"] * 100 if s["total"] > 0 else 0
        c_pct = s["cross"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"    {cat}: {s['total']} ({pct:.1f}%)"
              f"  Single={s['single']}({s_pct:.1f}%)"
              f"  Cross={s['cross']}({c_pct:.1f}%)")


# ============================================================
#  主流程
# ============================================================

def main(output_dir=None):
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / "classification_cache.json"

    print("Figure 9 — 概念分类分析")

    all_data = process_all_files()
    sampled = stratified_sampling(all_data)

    cache = load_cache(cache_file)
    openai_client = create_client()
    classified = []

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                classify_concept, openai_client,
                it['source_name'], it['source_definition'],
                it['generated_name'], it['generated_definition'],
                cache,
            ): it
            for it in sampled
        }
        with tqdm(total=len(sampled), desc="分类") as pbar:
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    result = fut.result()
                    row = item.copy()
                    row['category'] = result['category']
                    row['explanation'] = result['explanation']
                    classified.append(row)
                except Exception as e:
                    print(f"  错误: {e}")
                pbar.update(1)

    save_cache(cache, cache_file)

    df = pd.DataFrame(classified)
    csv_cols = [
        'source_file', 'source_folder', 'source_name', 'source_definition',
        'generated_name', 'generated_definition', 'category', 'explanation',
    ]

    df[csv_cols].to_csv(
        output_dir / 'classification_results.csv',
        index=False, encoding='utf-8-sig')

    radical_df = df[df['category'] == 'Radical']
    if not radical_df.empty:
        radical_df[csv_cols].to_csv(
            output_dir / 'radical_classification_results.csv',
            index=False, encoding='utf-8-sig')

    file_stats = (df.groupby(['source_folder', 'source_file', 'category'])
                    .size().unstack(fill_value=0))
    file_stats['Total'] = file_stats.sum(axis=1)
    file_stats.to_csv(output_dir / 'classification_by_file.csv',
                      encoding='utf-8-sig')

    create_nested_pie_chart(df, output_dir)
    print("完成")


def parse_args():
    p = argparse.ArgumentParser(description="Figure 9 — 概念分类分析")
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(output_dir=args.output_dir)
