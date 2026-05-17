"""
步骤一：穷举复合概念生成

遍历所有「通信概念 × AI 概念」组合，调用 LLM 生成新的复合概念。
支持多线程、磁盘缓存和断点续传。
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from llm_utils import create_client, extract_json_from_text, get_cache_key

print_lock = Lock()
save_lock = Lock()


# ------------------------------------------------------------------
#  缓存工具
# ------------------------------------------------------------------

def _cache_path(cache_dir: Path, com_concept: str, ai_concept: str) -> Path:
    key = get_cache_key(f"{com_concept}|||{ai_concept}")
    return cache_dir / f"{key}.json"


def _load_cache(cache_dir: Path, com_concept: str, ai_concept: str) -> Optional[Dict]:
    path = _cache_path(cache_dir, com_concept, ai_concept)
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache(cache_dir: Path, com_concept: str, ai_concept: str, result: Dict):
    path = _cache_path(cache_dir, com_concept, ai_concept)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ------------------------------------------------------------------
#  提示词
# ------------------------------------------------------------------

def create_prompt(com_concept: str, ai_concept: str) -> str:
    return f"""You are an expert in wireless communication and artificial intelligence. Your task is to naturally combine two concepts into an innovative composite concept.

**Communication Concept**: {com_concept}
**AI Concept**: {ai_concept}

Please create a novel composite concept that naturally integrates these two concepts. The combination should be technically meaningful and innovative.

Provide your answer in the following JSON format:
{{
  "concept_name": "A concise and catchy name for the composite concept",
  "definition": "A clear 1-2 sentence definition explaining what this composite concept is and how it works"
}}

Requirements:
1. The concept name should be creative and reflect both domains
2. Keep the definition concise (1-2 sentences)

Output only the JSON, no additional explanation."""


# ------------------------------------------------------------------
#  核心生成逻辑
# ------------------------------------------------------------------

def generate_composite_concept(com_concept: str, ai_concept: str,
                               cache_dir: Path) -> Optional[Dict]:
    """调用 LLM 生成复合概念（带缓存）。"""
    cached = _load_cache(cache_dir, com_concept, ai_concept)
    if cached:
        with print_lock:
            print(f"  [cache] {com_concept} + {ai_concept}")
        return cached

    client = create_client()
    prompt = create_prompt(com_concept, ai_concept)

    for attempt in range(config.GENERATION_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=config.GENERATION_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are an expert in wireless communication and AI innovation."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            result_text = response.choices[0].message.content.strip()
            result = extract_json_from_text(result_text)

            if result and "concept_name" in result and "definition" in result:
                final = {
                    "concept_name": result["concept_name"],
                    "definition": result["definition"],
                }
                _save_cache(cache_dir, com_concept, ai_concept, final)
                return final
            else:
                with print_lock:
                    print(f"  [warn] missing fields (attempt {attempt + 1}/{config.GENERATION_MAX_RETRY})")

        except Exception as e:
            with print_lock:
                print(f"  [error] attempt {attempt + 1}/{config.GENERATION_MAX_RETRY}: {e}")
            if attempt == config.GENERATION_MAX_RETRY - 1:
                return None

        time.sleep(0.5 * (attempt + 1))

    return None


def _process_single(com_item: Dict, ai_item: Dict, combo_id: int,
                     total: int, cache_dir: Path) -> Optional[Dict]:
    """处理单个组合（线程工作函数）。"""
    com_concept = com_item["concept"]
    ai_concept = ai_item["concept"]

    with print_lock:
        print(f"[{combo_id}/{total}] ({combo_id / total * 100:.1f}%) "
              f"Com: {com_concept[:30]}... | AI: {ai_concept[:30]}...")

    generated = generate_composite_concept(com_concept, ai_concept, cache_dir)
    if generated is None:
        with print_lock:
            print(f"  [fail] skipped")
        return None

    result = {
        "id": combo_id,
        "communication_concept": com_concept,
        "ai_concept": ai_concept,
        "composite_concept_name": generated["concept_name"],
        "definition": generated["definition"],
    }
    with print_lock:
        print(f"  -> {generated['concept_name']}")
    return result


# ------------------------------------------------------------------
#  主函数
# ------------------------------------------------------------------

def main(output_dir: Path = None):
    """运行穷举复合概念生成流程。"""
    output_dir = Path(output_dir) if output_dir else config.RQ3_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "generation_cache"
    cache_dir.mkdir(exist_ok=True)
    output_file = output_dir / "combined_concepts_generated.json"

    print("Step 1: 穷举复合概念生成")

    with open(config.COM_CONCEPTS_FILE, 'r', encoding='utf-8') as f:
        com_pool = json.load(f)
    with open(config.AI_CONCEPTS_FILE, 'r', encoding='utf-8') as f:
        ai_pool = json.load(f)

    existing_results: List[Dict] = []
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except Exception:
            pass

    tasks = []
    combo_id = 0
    existing_ids = {r["id"] for r in existing_results}
    for com_item in com_pool:
        for ai_item in ai_pool:
            combo_id += 1
            if combo_id not in existing_ids:
                tasks.append((com_item, ai_item, combo_id))

    total = len(com_pool) * len(ai_pool)
    print(f"  组合: {total}, 剩余: {len(tasks)}")

    results = existing_results.copy()
    completed = len(existing_results)
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=config.RQ3_MAX_WORKERS) as executor:
        future_map = {
            executor.submit(_process_single, ci, ai, cid, total, cache_dir): cid
            for ci, ai, cid in tasks
        }
        for future in as_completed(future_map):
            try:
                result = future.result()
                if result:
                    with save_lock:
                        results.append(result)
                        completed += 1
                        if completed % 10 == 0:
                            sorted_results = sorted(results, key=lambda x: x["id"])
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(sorted_results, f, indent=2, ensure_ascii=False)
                            with print_lock:
                                print(f"  [saved] {completed} results")
                else:
                    failed += 1
            except Exception as e:
                with print_lock:
                    print(f"  [error] task exception: {e}")
                failed += 1

    sorted_results = sorted(results, key=lambda x: x["id"])
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"  完成: {completed}, 失败: {failed}, 耗时: {elapsed / 60:.1f} min")


if __name__ == '__main__':
    main()
