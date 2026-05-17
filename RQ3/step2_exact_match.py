"""
步骤二：精确匹配评估

针对 20 个基准概念，通过「通信概念 + AI 概念」精确定位对应的生成概念，
再调用 GPT-4o 判断定义描述是否语义等价（一对一）。
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from llm_utils import create_client, extract_json_response


# ------------------------------------------------------------------
#  辅助函数
# ------------------------------------------------------------------

def normalize_concept(concept: str) -> str:
    return concept.strip().lower()


# ------------------------------------------------------------------
#  缓存
# ------------------------------------------------------------------

def _load_cache(cache_dir: Path, baseline_no: int) -> Optional[bool]:
    cache_file = cache_dir / f"baseline_{baseline_no}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f).get("is_match")
        except Exception:
            return None
    return None


def _save_cache(cache_dir: Path, baseline_no: int, is_match: bool):
    cache_file = cache_dir / f"baseline_{baseline_no}.json"
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({"is_match": is_match}, f)
    except Exception:
        pass


# ------------------------------------------------------------------
#  提示词与 LLM 调用
# ------------------------------------------------------------------

def _create_match_prompt(baseline_def: str, generated_def: str) -> str:
    return f"""You are an expert in wireless communication and AI. Your task is to determine if two concept definitions describe essentially the same composite concept.

**Baseline Concept Definition**:
{baseline_def}

**Generated Concept Definition**:
{generated_def}

Please determine if these two definitions describe the same or highly similar composite concept. Consider:
1. Core functionality and purpose
2. Technical approach and methods
3. Application domain and use cases

Answer with ONLY a JSON object containing a single boolean field:
{{
  "is_match": true or false
}}

Output only the JSON, no additional text."""


def check_definition_match(cache_dir: Path, baseline_no: int,
                           baseline_def: str, generated_def: str) -> bool:
    """调用 GPT-4o 检查两个定义是否语义匹配（带缓存）。"""
    cached = _load_cache(cache_dir, baseline_no)
    if cached is not None:
        return cached

    client = create_client()
    prompt = _create_match_prompt(baseline_def, generated_def)

    for attempt in range(config.EVALUATION_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=config.EVALUATION_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are an expert evaluator of technical concepts."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=100,
            )
            result_text = response.choices[0].message.content.strip()
            result = extract_json_response(result_text)

            if result and "is_match" in result:
                is_match = bool(result["is_match"])
                _save_cache(cache_dir, baseline_no, is_match)
                return is_match

        except Exception as e:
            print(f"    [warn] attempt {attempt + 1}/{config.EVALUATION_MAX_RETRY}: "
                  f"{str(e)[:100]}")
            time.sleep(0.5 * (attempt + 1))

    return False


# ------------------------------------------------------------------
#  匹配逻辑
# ------------------------------------------------------------------

def find_exact_match(baseline_item: Dict, generated: List[Dict]) -> Optional[Dict]:
    """根据「通信概念 + AI 概念」在生成概念中查找精确匹配项。"""
    base_com = normalize_concept(baseline_item["Communication Concept"])
    base_ai = normalize_concept(baseline_item["AI Concept"])

    matches = []
    for gen_item in generated:
        if (normalize_concept(gen_item["communication_concept"]) == base_com and
                normalize_concept(gen_item["ai_concept"]) == base_ai):
            matches.append(gen_item)

    if len(matches) == 0:
        return None
    if len(matches) == 1:
        return matches[0]

    print(f"  [warn] {len(matches)} duplicates found for "
          f"{baseline_item['Communication Concept']} + {baseline_item['AI Concept']}")
    return matches[0]


# ------------------------------------------------------------------
#  数据加载
# ------------------------------------------------------------------

def load_data() -> Tuple[List[Dict], List[Dict]]:
    with open(config.BASELINE_FILE, 'r', encoding='utf-8') as f:
        baseline = json.load(f)
    with open(config.GENERATED_FILE, 'r', encoding='utf-8') as f:
        generated = json.load(f)
    return baseline, generated


def check_duplicates_in_generated(generated: List[Dict]):
    """检查生成概念中是否存在重复的（通信, AI）组合。"""
    seen = {}
    duplicates = []
    for item in generated:
        key = f"{normalize_concept(item['communication_concept'])}||" \
              f"{normalize_concept(item['ai_concept'])}"
        if key in seen:
            duplicates.append((seen[key], item["id"]))
        else:
            seen[key] = item["id"]

    if duplicates:
        print(f"[warn] {len(duplicates)} duplicate pair(s) found")


# ------------------------------------------------------------------
#  评估
# ------------------------------------------------------------------

def evaluate_exact_matches(baseline: List[Dict], generated: List[Dict],
                           cache_dir: Path) -> List[Dict]:
    """执行一对一精确匹配评估。"""
    results = []
    found_count = 0
    def_matched_count = 0
    start_time = time.time()

    for idx, b_item in enumerate(baseline, 1):
        b_no = b_item["No"]
        b_name = b_item["Composite Concept Name"]
        b_com = b_item["Communication Concept"]
        b_ai = b_item["AI Concept"]
        b_def = b_item["Definition"]

        matched = find_exact_match(b_item, generated)

        if matched:
            found_count += 1
            is_match = check_definition_match(
                cache_dir, b_no, b_def, matched["definition"])
            if is_match:
                def_matched_count += 1

            result = {
                "baseline_no": b_no,
                "baseline_name": b_name,
                "baseline_communication": b_com,
                "baseline_ai": b_ai,
                "baseline_definition": b_def,
                "found_generated": True,
                "definition_matched": is_match,
                "generated_id": matched["id"],
                "generated_name": matched["composite_concept_name"],
                "generated_communication": matched["communication_concept"],
                "generated_ai": matched["ai_concept"],
                "generated_definition": matched["definition"],
            }
        else:
            result = {
                "baseline_no": b_no,
                "baseline_name": b_name,
                "baseline_communication": b_com,
                "baseline_ai": b_ai,
                "baseline_definition": b_def,
                "found_generated": False,
                "definition_matched": False,
                "generated_id": None,
                "generated_name": None,
                "generated_communication": None,
                "generated_ai": None,
                "generated_definition": None,
            }

        results.append(result)

    elapsed = time.time() - start_time
    print(f"  Found: {found_count}/{len(baseline)}, "
          f"Def matched: {def_matched_count}, Time: {elapsed:.1f}s")

    return results


# ------------------------------------------------------------------
#  报告
# ------------------------------------------------------------------

def generate_report(results: List[Dict], report_path: Path):
    """生成可读的评估报告。"""
    lines = []
    lines.append("=" * 80)
    lines.append("Exact-Match Evaluation Report")
    lines.append("=" * 80)
    lines.append("")

    found = [r for r in results if r["found_generated"]]
    not_found = [r for r in results if not r["found_generated"]]
    def_matched = [r for r in results if r["definition_matched"]]
    def_not_matched = [r for r in found if not r["definition_matched"]]

    lines.append(f"Total baseline concepts: {len(results)}")
    lines.append("")
    lines.append(f"[Step 1] Exact locating (Com + AI concept):")
    lines.append(f"  Found    : {len(found)} ({len(found)/len(results)*100:.1f}%)")
    lines.append(f"  Not found: {len(not_found)} ({len(not_found)/len(results)*100:.1f}%)")
    lines.append("")
    lines.append(f"[Step 2] Definition match (GPT-4o):")
    lines.append(f"  Matched    : {len(def_matched)} ({len(def_matched)/len(results)*100:.1f}%)")
    lines.append(f"  Not matched: {len(def_not_matched)} ({len(def_not_matched)/len(results)*100:.1f}%)")
    lines.append("")

    lines.append("=" * 80)
    lines.append("Details")
    lines.append("=" * 80)
    lines.append("")

    for r in results:
        lines.append(f"[{r['baseline_no']}] {r['baseline_name']}")
        lines.append(f"  Com: {r['baseline_communication']}")
        lines.append(f"  AI : {r['baseline_ai']}")
        if r["found_generated"]:
            lines.append(f"  -> Generated (ID {r['generated_id']}): {r['generated_name']}")
            lines.append(f"     Definition match: {'YES' if r['definition_matched'] else 'NO'}")
            lines.append(f"  Baseline def : {r['baseline_definition']}")
            lines.append(f"  Generated def: {r['generated_definition']}")
        else:
            lines.append(f"  -> NOT found")
        lines.append("")
        lines.append("-" * 80)
        lines.append("")

    if def_matched:
        lines.append("=" * 80)
        lines.append("Matched concepts summary")
        lines.append("=" * 80)
        lines.append("")
        for r in def_matched:
            lines.append(f"  [{r['baseline_no']}] {r['baseline_name']} -> {r['generated_name']}")

    if not_found:
        lines.append("")
        lines.append("=" * 80)
        lines.append("Not-found concepts summary")
        lines.append("=" * 80)
        lines.append("")
        for r in not_found:
            lines.append(f"  [{r['baseline_no']}] {r['baseline_name']}")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Report saved: {report_path}")


# ------------------------------------------------------------------
#  主函数
# ------------------------------------------------------------------

def main(output_dir: Path = None):
    output_dir = Path(output_dir) if output_dir else config.RQ3_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "match_cache"
    cache_dir.mkdir(exist_ok=True)

    print("Step 2: 精确匹配评估")
    baseline, generated = load_data()
    check_duplicates_in_generated(generated)
    results = evaluate_exact_matches(baseline, generated, cache_dir)

    result_path = output_dir / "exact_match_results.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    report_path = output_dir / "exact_match_report.txt"
    generate_report(results, report_path)
    print("完成")


if __name__ == '__main__':
    main()
