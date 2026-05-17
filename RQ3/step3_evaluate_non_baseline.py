"""
步骤三：非基准评估（1-4 分制）

对每个非基准匹配的生成概念，调用 GPT-4o 评估合理性（1-4 分），
每个概念评估 EVAL_TIMES 次取平均。基准匹配概念固定赋 5 分。
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from llm_utils import create_client, extract_json_response

print_lock = Lock()
save_lock = Lock()


# ------------------------------------------------------------------
#  缓存
# ------------------------------------------------------------------

def _load_quality_cache(cache_dir: Path, concept_id: int) -> Optional[float]:
    cache_file = cache_dir / f"{concept_id}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                score = json.load(f).get("score")
                if score is not None and 1.0 <= score <= 4.0:
                    return float(score)
        except Exception:
            pass
    return None


def _save_quality_cache(cache_dir: Path, concept_id: int, score: float):
    cache_file = cache_dir / f"{concept_id}.json"
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                "concept_id": concept_id,
                "score": score,
                "timestamp": time.time(),
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ------------------------------------------------------------------
#  提示词
# ------------------------------------------------------------------

def _create_quality_prompt(com_concept: str, ai_concept: str,
                           concept_name: str, definition: str) -> str:
    return f"""You are an expert in wireless communication and AI. Your task is to evaluate the quality and plausibility of a composite concept.

**Communication Concept**: {com_concept}
**AI Concept**: {ai_concept}
**Composite Concept Name**: {concept_name}
**Definition**: {definition}

Please evaluate the **plausibility and logical coherence** of this composite concept on a scale of 1-4:
- **4**: Good plausibility, reasonable integration, some technical value
- **3**: Moderately plausible, acceptable integration, limited novelty
- **2**: Weak plausibility, forced integration, unclear benefit
- **1**: Not plausible, illogical integration, no clear value

**Important**: The maximum score is 4. Do not use 5.

Answer with a JSON object:
{{
  "score": 1 to 4 (integer),
  "reasoning": "Brief explanation"
}}

Output only the JSON, no additional text."""


# ------------------------------------------------------------------
#  单概念评估
# ------------------------------------------------------------------

def _evaluate_single(concept: Dict, cache_dir: Path) -> float:
    """评估单个概念（EVAL_TIMES 轮取平均，分数限制在 1-4）。"""
    concept_id = concept["id"]

    cached = _load_quality_cache(cache_dir, concept_id)
    if cached is not None:
        with print_lock:
            print(f"  [cache] ID {concept_id}: "
                  f"{concept['composite_concept_name']} = {cached:.2f}")
        return cached

    client = create_client()
    prompt = _create_quality_prompt(
        concept["communication_concept"],
        concept["ai_concept"],
        concept["composite_concept_name"],
        concept["definition"],
    )

    scores = []
    for eval_round in range(config.EVAL_TIMES):
        success = False
        for retry in range(config.EVALUATION_MAX_RETRY):
            try:
                response = client.chat.completions.create(
                    model=config.EVALUATION_MODEL,
                    messages=[
                        {"role": "system",
                         "content": "You are an expert evaluator of technical concepts. "
                                    "You must only use scores 1-4, never 5."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )
                result_text = response.choices[0].message.content.strip()
                result = extract_json_response(result_text)

                if result and "score" in result:
                    score = int(result["score"])
                    if 1 <= score <= 4:
                        scores.append(score)
                        success = True
                        break
                    elif score == 5:
                        with print_lock:
                            print(f"    [warn] round {eval_round+1} returned 5, clamped to 4")
                        scores.append(4)
                        success = True
                        break
                    else:
                        if retry == config.EVALUATION_MAX_RETRY - 1:
                            with print_lock:
                                print(f"    [warn] round {eval_round+1} invalid score {score}, "
                                      f"using 3")
                            scores.append(3)
                            success = True
                            break

            except Exception as e:
                if retry == config.EVALUATION_MAX_RETRY - 1:
                    with print_lock:
                        print(f"    [warn] round {eval_round+1} failed: {str(e)[:100]}")
                time.sleep(0.5 * (retry + 1))

        if not success:
            with print_lock:
                print(f"    [warn] round {eval_round+1} fully failed, using default 3")
            scores.append(3)

    if scores:
        avg = min(sum(scores) / len(scores), 4.0)
        final = round(avg, 2)
    else:
        final = 3.0

    _save_quality_cache(cache_dir, concept_id, final)
    return final


def _evaluate_wrapper(concept: Dict, cache_dir: Path) -> Dict:
    """线程安全的包装函数，返回结果字典。"""
    cid = concept["id"]
    cname = concept["composite_concept_name"]
    try:
        score = _evaluate_single(concept, cache_dir)
        with print_lock:
            print(f"  ID {cid}: {cname} = {score:.2f}")
        return {
            "id": cid,
            "communication_concept": concept["communication_concept"],
            "ai_concept": concept["ai_concept"],
            "concept_name": cname,
            "score": score,
            "is_baseline_match": False,
            "evaluation_method": "quality_evaluation_1to4",
        }
    except Exception as e:
        with print_lock:
            print(f"  ID {cid}: {cname} FAILED: {str(e)[:200]}")
        return {
            "id": cid,
            "communication_concept": concept["communication_concept"],
            "ai_concept": concept["ai_concept"],
            "concept_name": cname,
            "score": 3.0,
            "is_baseline_match": False,
            "evaluation_method": "quality_evaluation_1to4_failed",
        }


# ------------------------------------------------------------------
#  数据加载
# ------------------------------------------------------------------

def load_data() -> tuple:
    baseline_matched_ids = set()
    if config.MATCH_RESULTS_FILE.exists():
        with open(config.MATCH_RESULTS_FILE, 'r', encoding='utf-8') as f:
            for m in json.load(f):
                if m.get("definition_matched", False) or m.get("is_match", False):
                    gid = m.get("generated_id")
                    if gid is not None:
                        baseline_matched_ids.add(gid)

    with open(config.GENERATED_FILE, 'r', encoding='utf-8') as f:
        generated = json.load(f)

    to_eval = [c for c in generated if c["id"] not in baseline_matched_ids]
    return to_eval, baseline_matched_ids


# ------------------------------------------------------------------
#  主函数
# ------------------------------------------------------------------

def main(output_dir: Path = None):
    output_dir = Path(output_dir) if output_dir else config.RQ3_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "reeval_cache" / "quality"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Step 3: 非基准评估 (1-4 分)")

    concepts_to_eval, baseline_matched_ids = load_data()
    if not concepts_to_eval:
        print("  无需评估")
        return

    cached_count = sum(1 for c in concepts_to_eval
                       if _load_quality_cache(cache_dir, c["id"]) is not None)
    need_eval = [c for c in concepts_to_eval
                 if _load_quality_cache(cache_dir, c["id"]) is None]

    if not need_eval:
        results = []
        for c in concepts_to_eval:
            s = _load_quality_cache(cache_dir, c["id"])
            results.append({
                "id": c["id"],
                "communication_concept": c["communication_concept"],
                "ai_concept": c["ai_concept"],
                "concept_name": c["composite_concept_name"],
                "score": s,
                "is_baseline_match": False,
                "evaluation_method": "quality_evaluation_1to4_cached",
            })
    else:
        results = []
        with ThreadPoolExecutor(max_workers=config.RQ3_MAX_WORKERS) as executor:
            futures = {
                executor.submit(_evaluate_wrapper, c, cache_dir): c
                for c in need_eval
            }
            with tqdm(total=len(need_eval), desc="Evaluating") as pbar:
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        c = futures[future]
                        with print_lock:
                            print(f"  [error] ID {c['id']}: {str(e)[:200]}")
                    pbar.update(1)

        for c in concepts_to_eval:
            if c["id"] not in {r["id"] for r in results}:
                s = _load_quality_cache(cache_dir, c["id"])
                if s is not None:
                    results.append({
                        "id": c["id"],
                        "communication_concept": c["communication_concept"],
                        "ai_concept": c["ai_concept"],
                        "concept_name": c["composite_concept_name"],
                        "score": s,
                        "is_baseline_match": False,
                        "evaluation_method": "quality_evaluation_1to4_cached",
                    })

    # Baseline-matched concepts get score 5
    baseline_results = []
    if config.MATCH_RESULTS_FILE.exists():
        with open(config.MATCH_RESULTS_FILE, 'r', encoding='utf-8') as f:
            for m in json.load(f):
                if m.get("definition_matched", False) or m.get("is_match", False):
                    gid = m.get("generated_id")
                    if gid is not None:
                        baseline_results.append({
                            "id": gid,
                            "communication_concept": (m.get("generated_communication")
                                                      or m.get("communication_concept")),
                            "ai_concept": (m.get("generated_ai")
                                           or m.get("ai_concept")),
                            "concept_name": m.get("generated_name"),
                            "score": 5.0,
                            "is_baseline_match": True,
                            "evaluation_method": "baseline_match",
                        })

    all_results = sorted(baseline_results + results, key=lambda x: x["id"])

    output_path = output_dir / "evaluation_scores_1to4.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"  Total: {len(all_results)}, Baseline: {len(baseline_results)}, "
          f"Non-BL: {len(results)}")
    print("完成")


if __name__ == "__main__":
    main()
