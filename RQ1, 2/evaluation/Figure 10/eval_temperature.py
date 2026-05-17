"""
温度实验评估 — 使用 SciBERT / all-MiniLM-L6-v2 计算嵌入指标。

对 evaluation/Figure 10 下的温度 JSON 文件，
计算 novelty, coverage, pairwise_distance, entropy，
保存到对应嵌入模型子目录。
"""

import sys
import traceback
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
RQ12_DIR = SCRIPT_DIR.parent.parent
CODE_DIR = RQ12_DIR.parent

sys.path.insert(0, str(RQ12_DIR))
sys.path.insert(0, str(CODE_DIR))

from llm_utils import load_json
from evaluation import (
    get_embedding_model,
    embed_terms,
    compute_novelty,
    compute_coverage,
    compute_pairwise_distance,
    compute_semantic_entropy,
    save_outputs,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_OUTPUT_MAP = {
    "minilm":   "all-MiniLM-L6-v2",
    "scibert":  "SciBERT",
}

EMBEDDING_METRICS = ["novelty", "coverage", "pairwise_distance", "entropy"]


def discover_temperature_files():
    return sorted(SCRIPT_DIR.glob("com_150_4o_single_t*.json"))


def metrics_already_exist(output_dir: Path, prefix: str) -> bool:
    return all(
        (output_dir / f"{prefix}_{m}.json").exists()
        for m in EMBEDDING_METRICS
    )


def process_file(json_path: Path, emb_model, emb_model_name: str):
    filename = json_path.name
    stem = json_path.stem

    output_dir = SCRIPT_DIR / MODEL_OUTPUT_MAP[emb_model_name] / stem
    prefix = filename

    if metrics_already_exist(output_dir, prefix):
        return

    data = load_json(str(json_path))
    emb_data = embed_terms(data, emb_model)
    novelty = compute_novelty(emb_data)
    coverage = compute_coverage(data, emb_model)
    pairwise = compute_pairwise_distance(emb_data)
    entropy = compute_semantic_entropy(emb_data)

    metrics = {
        "novelty": novelty,
        "coverage": coverage,
        "pairwise_distance": pairwise,
        "entropy": entropy,
    }

    save_outputs(str(output_dir), prefix, emb_data, metrics)
    print(f"  saved: {output_dir.relative_to(SCRIPT_DIR)}")


def run(model_names: list):
    files = discover_temperature_files()
    if not files:
        raise FileNotFoundError(f"No temperature JSON files in {SCRIPT_DIR}")

    print(f"Device: {DEVICE}, Files: {len(files)}, Models: {model_names}")

    for model_name in model_names:
        print(f"Embedding: {model_name}")
        emb_model = get_embedding_model(model_name)

        for fpath in files:
            try:
                process_file(fpath, emb_model, model_name)
            except Exception as exc:
                print(f"  [ERROR] {fpath.name}: {exc}")
                traceback.print_exc()

    print("完成")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Temperature evaluation")
    p.add_argument("--models", nargs="+", default=["minilm", "scibert"],
                   choices=list(MODEL_OUTPUT_MAP.keys()))
    args = p.parse_args()
    run(args.models)
