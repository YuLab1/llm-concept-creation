"""
批量评估脚本 — 遍历所有生成结果 JSON 文件，依次计算评估指标。

计算四个基于嵌入的指标（novelty, coverage, pairwise distance, entropy），
可选计算基于 LLM 的 feasibility 指标。

用法:
    python batch_evaluation.py --input_dir ../test/create生成结果_修复 --output_dir results/
    python batch_evaluation.py --input_dir ../test/create生成结果_修复 --output_dir results/ --feasibility --eval_model gpt-4o
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DOMAIN_MAPPING
from llm_utils import load_json, build_llm_client
from evaluation import (
    embed_terms,
    get_embedding_model,
    compute_novelty,
    compute_coverage,
    compute_pairwise_distance,
    compute_semantic_entropy,
    compute_feasibility,
    save_outputs,
)


class BatchEvaluator:
    def __init__(self, input_dir, output_dir, run_feasibility=False, eval_model="gpt-4o"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.run_feasibility = run_feasibility
        self.eval_model = eval_model

    # -----------------------------------------------------------------
    #  文件发现
    # -----------------------------------------------------------------
    def get_all_data_files(self):
        """获取所有待处理的 JSON 数据文件。"""
        files = []
        for model_folder in sorted(self.input_dir.iterdir()):
            if not model_folder.is_dir():
                continue
            for json_file in sorted(model_folder.glob("*.json")):
                files.append({
                    "path": str(json_file),
                    "filename": json_file.name,
                    "model_folder": model_folder.name,
                })
        return files

    # -----------------------------------------------------------------
    #  领域提取
    # -----------------------------------------------------------------
    @staticmethod
    def extract_domain(filename):
        """从文件名前缀推断领域标签。"""
        for prefix, domain in DOMAIN_MAPPING.items():
            if filename.startswith(prefix + "_"):
                return domain
        return None

    # -----------------------------------------------------------------
    #  单文件处理
    # -----------------------------------------------------------------
    def process_file(self, file_info):
        """处理单个数据文件：加载、嵌入、计算指标、保存。"""
        filepath = file_info["path"]
        filename = file_info["filename"]
        model_folder = file_info["model_folder"]

        print(f"\n{'=' * 70}")
        print(f"正在处理: {filepath}")
        print(f"{'=' * 70}")

        data = load_json(filepath)
        print(f"  已加载 {len(data)} 个概念")

        emb_model = get_embedding_model()
        emb_data = embed_terms(data, emb_model)
        print("  嵌入计算完成")

        print("  计算基础指标...")
        metrics = {
            "novelty": compute_novelty(emb_data),
            "coverage": compute_coverage(data, emb_model),
            "pairwise_distance": compute_pairwise_distance(emb_data),
            "entropy": compute_semantic_entropy(emb_data),
        }

        if self.run_feasibility:
            domain = self.extract_domain(filename)
            if domain:
                print(f"  计算 Feasibility（领域={domain}）...")
                llm_call = build_llm_client(self.eval_model)
                metrics["feasibility"] = compute_feasibility(data, domain, llm_call)
            else:
                print("  警告: 无法从文件名推断领域，跳过 Feasibility 计算")

        out_subdir = self.output_dir / model_folder / filename
        prefix = filename
        save_outputs(str(out_subdir), prefix, emb_data, metrics)
        print(f"  已保存至 {out_subdir}")

    # -----------------------------------------------------------------
    #  主循环
    # -----------------------------------------------------------------
    def run(self):
        """执行批量评估。"""
        files = self.get_all_data_files()
        print(f"共找到 {len(files)} 个待评估文件\n")
        for i, info in enumerate(files, 1):
            print(f"进度: [{i}/{len(files)}]")
            try:
                self.process_file(info)
            except Exception as exc:
                print(f"处理 {info['filename']} 时出错: {exc}")
                traceback.print_exc()
                print("继续处理下一个文件...")
        print(f"\n{'=' * 70}\n批量评估全部完成。\n{'=' * 70}")


# ====================================================================
#  命令行入口
# ====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="批量评估概念创造生成结果")
    p.add_argument("--input_dir", required=True,
                   help="包含各模型子文件夹的根目录")
    p.add_argument("--output_dir", required=True,
                   help="评估结果输出根目录")
    p.add_argument("--feasibility", action="store_true",
                   help="是否计算基于 LLM 的 Feasibility 指标")
    p.add_argument("--eval_model", default="gpt-4o",
                   help="用于 Feasibility 打分的 LLM 预设")
    return p.parse_args()


def main():
    args = parse_args()
    evaluator = BatchEvaluator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        run_feasibility=args.feasibility,
        eval_model=args.eval_model,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
