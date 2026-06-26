"""
批量评估脚本 — 使用 SciBERT 计算四个嵌入指标。

对 data/generation concepts 中所有 JSON 文件，
计算 novelty, coverage, pairwise_distance, entropy，
保存到 data/evaluation_SciBERT/ 对应位置。
Feasibility（LLM 打分）已提前放置，此脚本不重复计算。
"""

import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "data" / "generation concepts"
DATA_DIR = SCRIPT_DIR / "data"

MODEL_OUTPUT_MAP = {
    "scibert": "evaluation_SciBERT",
}

FOLDER_NAME_MAP = {
    "GPT-4o":     "4o",
    "GPT-4omini": "4omini",
    "Qwen-max":   "qwen-max",
}

EMBEDDING_METRICS = ["novelty", "coverage", "pairwise_distance", "entropy"]


def discover_files(input_dir: Path) -> list:
    files = []
    for model_folder in sorted(input_dir.iterdir()):
        if not model_folder.is_dir():
            continue
        for json_file in sorted(model_folder.glob("*.json")):
            output_folder = FOLDER_NAME_MAP.get(model_folder.name, model_folder.name)
            files.append({
                "path": json_file,
                "filename": json_file.name,
                "model_folder": output_folder,
            })
    return files


def metrics_already_exist(output_subdir: Path, prefix: str) -> bool:
    return all(
        (output_subdir / f"{prefix}_{m}.json").exists()
        for m in EMBEDDING_METRICS
    )


def process_file(file_info: dict, emb_model, emb_model_name: str):
    filepath = file_info["path"]
    filename = file_info["filename"]
    model_folder = file_info["model_folder"]

    eval_folder = MODEL_OUTPUT_MAP[emb_model_name]
    output_subdir = DATA_DIR / eval_folder / model_folder / filename
    prefix = filename

    if metrics_already_exist(output_subdir, prefix):
        return

    data = load_json(str(filepath))
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

    save_outputs(str(output_subdir), prefix, emb_data, metrics)
    print(f"  saved: {output_subdir.relative_to(DATA_DIR)}")


def run(model_names: list):
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")

    files = discover_files(INPUT_DIR)
    print(f"Device: {DEVICE}, Files: {len(files)}, Models: {model_names}")

    for model_name in model_names:
        print(f"嵌入模型: {model_name}")
        emb_model = get_embedding_model(model_name)

        for i, info in enumerate(files, 1):
            try:
                process_file(info, emb_model, model_name)
            except Exception as exc:
                print(f"  [ERROR] {info['filename']}: {exc}")
                traceback.print_exc()

    print("完成")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="使用 SciBERT 批量计算嵌入指标")
    p.add_argument("--models", nargs="+", default=["scibert"],
                   choices=list(MODEL_OUTPUT_MAP.keys()))
    args = p.parse_args()
    run(args.models)
