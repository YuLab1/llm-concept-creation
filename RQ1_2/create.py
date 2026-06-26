"""
统一概念创造流程 — 合并 create_4o / create_4omini / create_qianwen 三个脚本。

通过命令行参数选择 LLM 后端，支持单领域和跨领域两种创造模式。
baseline 和 derived 生成阶段使用线程池并发调用以加速。

用法示例:
    python create.py --model gpt-4o    --input concepts.json --mode single --domain Communication --output out.json
    python create.py --model qwen-max  --input concepts.json --mode cross  --domain1 Communication --domain2 Electromagnetism --output out.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TOP_K_ELEMENTS, N_REPEAT, MAX_WORKERS
from llm_utils import build_llm_client, generate_with_retry, load_json, save_json


# ====================================================================
#  阶段 1 — 概念解构（单领域与跨领域共用）
# ====================================================================

def concept_decomposition(llm_call, concept_name, concept_description, domain):
    """提取核心要素并按创新潜力排序，保留 top-k 个。"""
    prompt = f"""
    You are an expert in {domain}. Analyze the given concept description and extract its core elements.
    Then, evaluate the **innovation potential** of each element on a scale of 1 to 5 (higher means more potential for modification and innovation) without any explanation.

    **Concept Name:** {concept_name}
    **Concept Description:** {concept_description}

    **Output Format (JSON)**:
    {{
        "original_concept": "{concept_name}",
        "core_elements": [
            {{
                "element_name": "element1",
                "innovativeness_score": 2
            }},
            {{
                "element_name": "element2",
                "innovativeness_score": 5
            }}
        ]
    }}
    """
    result = generate_with_retry(llm_call, prompt, {"original_concept", "core_elements"})
    elements = result["core_elements"]
    sorted_elements = sorted(
        elements, key=lambda x: (-x["innovativeness_score"], x["element_name"])
    )
    return sorted_elements[:TOP_K_ELEMENTS]


# ====================================================================
#  策略 1 — 单领域概念创新
# ====================================================================

def element_modification(llm_call, concept_name, concept_description, element_name, domain):
    """阶段 2: 要素改造"""
    prompt = f"""
    You are an expert in innovative {domain}. Your task is to creatively transform a specific technical element related to a {domain} concept.

    Your task is to:
    1. Analyze the element's role within its technical and functional context.
    2. Creatively redesign or recombine it to produce a novel variant with distinct utility.
    3. Ensure the innovation reflects a substantive transformation, not just superficial modification.

    **Input Context:**
    - Concept Name: {concept_name}
    - Concept Description: {concept_description}
    - Element Name: {element_name}

    **Output Format (JSON):**
    {{
      "new_element": "Your creative new element name",
      "element_description": "A brief explanation of the new element and how you creatively transformed the original element"
    }}
    """
    return generate_with_retry(llm_call, prompt, {"new_element", "element_description"})


def concept_modification(llm_call, concept_description, raw_element, new_element, domain):
    """阶段 3: 融入新要素实现概念创新"""
    prompt = f"""
    You are an expert in the field of {domain}. Given an original concept and a distinctly innovated core element, your task is to revise the original concept by substituting the old element with this new, innovative one.

    **Original Concept Description:**
    {concept_description}
    **Original Element to be replaced:**
    {raw_element}

    **New Element Information:**
    - New Element: {json.dumps(new_element, indent=2)}

    Make sure that:
    - The newly revised concept is logically consistent and technically robust.
    - The innovative element is seamlessly and meaningfully integrated, bringing novel and substantial value to the concept.

    **Output Format (JSON):**
    {{
      "redefined_term": "Updated concept name after integration."
      "redefined_concept": "A concise and rigorous new definition for the new term."
    }}
    """
    redefined = generate_with_retry(llm_call, prompt, {"redefined_term", "redefined_concept"})
    new_element.update(redefined)
    return new_element


def generate_single_domain_baseline(llm_call, concept_name, concept_description, domain):
    """对照组: 单领域直接创新（无解构步骤）"""
    prompt = f"""
    You are an expert in {domain}. Your task is to creatively evolve the following concept within the {domain} domain.
    Rather than merely modifying or extending existing knowledge, your goal is to generate a novel concept by rethinking, restructuring, or recombining core elements in an original way.

    The newly derived concept should:
    - Represent a meaningful reconfiguration of existing principles within {domain}.
    - Offer a clear advancement, reinterpretation, or novel direction compared to the original.
    - Be technically rigorous, logically coherent, and precisely articulated.

    **Original Concept Name:** {concept_name}
    **Original Concept Description:** {concept_description}

    **Output Format (JSON):**
    {{
        "redefined_term": "The new term created from the innovation",
        "redefined_concept": "A concise and rigorous new definition for the new term"
    }}
    """
    return generate_with_retry(llm_call, prompt, {"redefined_term", "redefined_concept"})


# ====================================================================
#  策略 2 — 跨领域交叉创新
# ====================================================================

def cross_domain_mapping(llm_call, concept_name, concept_description, element_name,
                         domain1, domain2):
    """阶段 2: 跨领域映射"""
    prompt = f"""
    You are an expert in both {domain1} and {domain2}. Your task is to analyze the core elements of a given concept from {domain1}, select the most appropriate analogous or complementary concept from {domain2}, and establish a formal mapping relationship between these two concepts.
    **Original Concept:** {concept_name}
    **Concept Description:** {concept_description}
    **Elements Name:** {element_name}

    For each core element, please:
    1. Identify a corresponding element or principle from {domain2} that exhibits functional, structural, or theoretical alignment.
    2. Justify this mapping by referencing foundational principles, technical mechanisms, or emergent synergies between the domains.

    **Output Format (JSON):**
    {{
        "mapped": "The concept name mapped in {domain2}",
        "justification": "A brief explanation to why such a mapping is established"
    }}
    """
    return generate_with_retry(llm_call, prompt, {"mapped", "justification"})


def cross_domain_construction(llm_call, concept_description, raw_element, mapped_data,
                              domain1, domain2):
    """阶段 3: 跨领域概念构造"""
    prompt = f"""
    You are an expert in both the fields of {domain1} and {domain2}. Your task is to incorporate this transformation into the original {domain1} concept based on the mapping results, so as to reconstruct a brand-new concept.
    This new concept should possess the following characteristics:
    - Fuse the principles from both the {domain1} field and the {domain2} field in a meaningful way.
    - Present a concept that is technically sound and innovative.

    - **Concept Description:** {concept_description}
    - **Raw element:** {raw_element}
    - **Mapped Element:** {json.dumps(mapped_data, indent=2)}

    **Output Format (JSON):**
    {{
        "redefined_term": "Updated concept name."
        "redefined_concept": "A concise and rigorous new definition for the new term."
    }}
    """
    redefined = generate_with_retry(llm_call, prompt, {"redefined_term", "redefined_concept"})
    mapped_data.update(redefined)
    return mapped_data


def generate_cross_domain_baseline(llm_call, concept_name, concept_description,
                                   domain1, domain2):
    """对照组: 跨领域直接创新（无解构步骤）"""
    prompt = f"""
    You are an expert in both {domain1} and {domain2}. Your task is to creatively generate a novel cross-domain concept by meaningfully integrating the following concept from {domain1} with relevant principles, methodologies, or paradigms from {domain2}.
    The fusion should go beyond superficial combination — it must reflect a deep, innovative integration that leverages the strengths and insights of both fields.
    The new concept should:
        - Combine core ideas, models, or approaches from both {domain1} and {domain2} in a technically coherent and creative way.
        - Offer improvements or advancements in the context of both fields.
        - Be technically rigorous and innovative.

    **Original Concept Name:** {concept_name}
    **Original Concept Description:** {concept_description}

    **Output Format (JSON):**
    {{
        "redefined_term": "The new term created from the integration of both domains",
        "redefined_concept": "A concise and rigorous new definition for the new term"
    }}
    """
    return generate_with_retry(llm_call, prompt, {"redefined_term", "redefined_concept"})


# ====================================================================
#  完整流程（使用线程池并发生成）
# ====================================================================

def _run_single_derived_once(llm_call, concept_name, concept_description, e_name, domain):
    """单领域: 单次 element_modification → concept_modification 链。"""
    modified = element_modification(llm_call, concept_name, concept_description, e_name, domain)
    merged = concept_modification(llm_call, concept_description, e_name, modified, domain)
    return merged


def _run_cross_derived_once(llm_call, concept_name, concept_description, e_name,
                            domain1, domain2):
    """跨领域: 单次 mapping → construction 链。"""
    mapped = cross_domain_mapping(llm_call, concept_name, concept_description,
                                  e_name, domain1, domain2)
    merged = cross_domain_construction(llm_call, concept_description, e_name,
                                       mapped, domain1, domain2)
    return merged


def execute_pipeline_single(llm_call, concept_name, concept_description, domain):
    """单领域概念创造完整流程（线程池并发）"""
    concept_dict = {
        "original_concept": concept_name,
        "original_description": concept_description,
    }
    print(f"\n===== 正在处理概念: {concept_name} =====\n")

    # --- baseline: 并发 N_REPEAT 次 ---
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(generate_single_domain_baseline,
                        llm_call, concept_name, concept_description, domain)
            for _ in range(N_REPEAT)
        ]
        baseline = [f.result() for f in futures]
    concept_dict["baseline"] = baseline

    # --- 概念解构 ---
    top_elements = concept_decomposition(llm_call, concept_name, concept_description, domain)
    print("\n[阶段 1: 概念解构]\n", json.dumps(top_elements, indent=2))
    concept_dict["top_elements"] = top_elements

    # --- derived: 所有 element × N_REPEAT 并发提交 ---
    derived = [[] for _ in top_elements]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {}
        for idx, elem in enumerate(top_elements):
            e_name = elem["element_name"]
            for _ in range(N_REPEAT):
                fut = pool.submit(_run_single_derived_once,
                                  llm_call, concept_name, concept_description, e_name, domain)
                future_map[fut] = idx

        for fut in as_completed(future_map):
            idx = future_map[fut]
            derived[idx].append(fut.result())

    concept_dict["derived"] = derived
    return concept_dict


def execute_pipeline_cross(llm_call, concept_name, concept_description, domain1, domain2):
    """跨领域概念创造完整流程（线程池并发）"""
    concept_dict = {
        "original_concept": concept_name,
        "original_description": concept_description,
    }
    print(f"\n===== 正在处理概念: {concept_name} =====\n")

    # --- baseline: 并发 N_REPEAT 次 ---
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(generate_cross_domain_baseline,
                        llm_call, concept_name, concept_description, domain1, domain2)
            for _ in range(N_REPEAT)
        ]
        baseline = [f.result() for f in futures]
    concept_dict["baseline"] = baseline

    # --- 概念解构 ---
    top_elements = concept_decomposition(llm_call, concept_name, concept_description, domain1)
    print("\n[阶段 1: 概念解构]\n", json.dumps(top_elements, indent=2))
    concept_dict["top_elements"] = top_elements

    # --- derived: 所有 element × N_REPEAT 并发提交 ---
    derived = [[] for _ in top_elements]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {}
        for idx, elem in enumerate(top_elements):
            e_name = elem["element_name"]
            for _ in range(N_REPEAT):
                fut = pool.submit(_run_cross_derived_once,
                                  llm_call, concept_name, concept_description, e_name,
                                  domain1, domain2)
                future_map[fut] = idx

        for fut in as_completed(future_map):
            idx = future_map[fut]
            derived[idx].append(fut.result())

    concept_dict["derived"] = derived
    return concept_dict


# ====================================================================
#  命令行入口
# ====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="使用指定 LLM 运行概念创造流程"
    )
    parser.add_argument("--model", required=True,
                        choices=list((__import__("config")).MODEL_PRESETS.keys()),
                        help="config.py 中定义的 LLM 预设名称")
    parser.add_argument("--input", required=True,
                        help="输入概念列表 JSON 文件路径")
    parser.add_argument("--output", required=True,
                        help="输出结果 JSON 文件路径")
    parser.add_argument("--mode", required=True, choices=["single", "cross"],
                        help="'single' 单领域创新 / 'cross' 跨领域创新")
    parser.add_argument("--domain", default="Communication",
                        help="主领域（单领域模式使用）")
    parser.add_argument("--domain1", default=None,
                        help="源领域（跨领域模式使用）")
    parser.add_argument("--domain2", default=None,
                        help="目标领域（跨领域模式使用）")
    return parser.parse_args()


def main():
    args = parse_args()

    llm_call = build_llm_client(args.model)
    data = load_json(args.input)

    results = []
    for concept in tqdm(data, desc="处理中", unit="概念"):
        name = concept.get("EnglishTerm", "")
        desc = concept.get("Description", "")

        if args.mode == "single":
            result = execute_pipeline_single(llm_call, name, desc, args.domain)
        else:
            d1 = args.domain1 or args.domain
            d2 = args.domain2 or "Electromagnetism"
            result = execute_pipeline_cross(llm_call, name, desc, d1, d2)

        results.append(result)

    save_json(results, args.output)
    print(f"\n结果已保存至 {args.output}")


if __name__ == "__main__":
    main()
