"""
评估流程 — 概念创造实验的指标计算。

已实现的评估指标:
  1. Novelty Score     — 原始描述与生成描述之间的余弦距离
  2. Coverage Score    — 基于关键词的语义覆盖率
  3. Pairwise Distance — 生成描述之间的平均余弦距离
  4. Semantic Entropy  — 基于层次聚类的香农熵
  5. Feasibility Score — 基于 LLM 的正向推理 + 反向质询评分

用法:
    python evaluation.py --input data.json --output_dir results/ --domain "Communication"
    python evaluation.py --input data.json --output_dir results/ --domain "Communication" --feasibility
"""

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from keybert import KeyBERT

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    EMBEDDING_MODEL_PATH,
    SEMANTIC_ENTROPY_MAX_K,
    SEMANTIC_ENTROPY_PCA_DIM,
    COVERAGE_TOP_N_ORIGINAL,
    COVERAGE_TOP_N_GENERATED,
    FEASIBILITY_LAMBDA,
    MAX_RETRIES,
    MAX_WORKERS,
)
from llm_utils import (
    build_llm_client,
    generate_with_retry as _generate_with_retry,
    load_json,
    save_json,
)


# ====================================================================
#  嵌入模型（两种可选：minilm / scibert）
# ====================================================================

EMBEDDING_MODEL_CHOICES = ["minilm", "scibert"]

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_LOCAL_SCIBERT = _MODELS_DIR / "scibert_scivocab_uncased"

EMBEDDING_MODEL_REGISTRY = {
    "minilm": {
        "type": "sentence_transformer",
        "path": EMBEDDING_MODEL_PATH,  # "all-MiniLM-L6-v2"
    },
    "scibert": {
        "type": "transformers",
        "path": str(_LOCAL_SCIBERT) if _LOCAL_SCIBERT.exists() else "allenai/scibert_scivocab_uncased",
        "pooling": "mean",
        "adapter": None,
    },
}


class TransformersEmbedder:
    """
    Wrapper around HuggingFace AutoModel / AutoAdapterModel,
    providing an .encode() interface compatible with SentenceTransformer,
    and an .embed() interface compatible with KeyBERT BaseEmbedder.

    - SciBERT:   mask-aware mean pooling
    - SPECTER2:  CLS token pooling (with adapter)
    """

    def __init__(self, model_path: str, pooling: str = "mean",
                 adapter: Optional[str] = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pooling = pooling
        self._torch = torch

        print(f"  Loading {model_path} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        if adapter is not None:
            from adapters import AutoAdapterModel
            self._model = AutoAdapterModel.from_pretrained(model_path)
            print(f"  Loading adapter: {adapter}")
            self._model.load_adapter(adapter, source="hf",
                                     load_as="specter2", set_active=True)
            self._model.to(self._device)
            print("  Adapter loaded and activated.")
        else:
            self._model = AutoModel.from_pretrained(model_path).to(self._device)

        self._model.eval()
        print(f"  Pooling: {pooling}  |  Device: {self._device}")


    @property
    def _torch_no_grad(self):
        return self._torch.no_grad

    def encode(self, sentences: Union[str, List[str]],
               normalize_embeddings: bool = False,
               batch_size: int = 32, **kwargs) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
            squeeze = True
        else:
            sentences = [str(s) for s in sentences]
            squeeze = False

        all_embs = []
        with self._torch_no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                enc = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=512, return_tensors="pt",
                ).to(self._device)
                out = self._model(**enc)

                if self._pooling == "cls":
                    embs = out.last_hidden_state[:, 0, :]
                else:
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    embs = ((out.last_hidden_state * mask).sum(1)
                            / mask.sum(1).clamp(min=1e-9))

                if normalize_embeddings:
                    embs = self._torch.nn.functional.normalize(embs, p=2, dim=1)

                all_embs.append(embs.cpu().numpy())

        result = np.vstack(all_embs)
        return result[0] if squeeze else result

    def embed(self, documents, verbose=False):
        """KeyBERT BaseEmbedder 兼容接口。"""
        if isinstance(documents, str):
            documents = [documents]
        return self.encode(documents)


_emb_model = None
_emb_model_name = None


def get_embedding_model(model_choice: str = "minilm"):
    """
    懒加载嵌入模型。支持三种选择:
      - "minilm"   : all-MiniLM-L6-v2 (SentenceTransformer)
      - "scibert"  : allenai/scibert_scivocab_uncased (mean pooling)
      - "specter2" : allenai/specter2_base + adapter (CLS pooling)
    """
    global _emb_model, _emb_model_name
    if _emb_model is not None and _emb_model_name == model_choice:
        return _emb_model

    cfg = EMBEDDING_MODEL_REGISTRY[model_choice]
    print(f"\n初始化嵌入模型: {model_choice}")

    if cfg["type"] == "sentence_transformer":
        _emb_model = SentenceTransformer(cfg["path"])
    else:
        _emb_model = TransformersEmbedder(
            model_path=cfg["path"],
            pooling=cfg["pooling"],
            adapter=cfg.get("adapter"),
        )

    _emb_model_name = model_choice
    return _emb_model


# ====================================================================
#  嵌入计算
# ====================================================================

def embed_terms(term_data, emb_model=None):
    """为所有概念计算归一化嵌入向量。"""
    if emb_model is None:
        emb_model = get_embedding_model()

    embedded = []
    for content in term_data:
        rec = {}
        name_ori = content["original_concept"]
        desc_ori = content["original_description"]
        rec["desc_ori_embs"] = emb_model.encode(desc_ori, normalize_embeddings=True)
        rec["name_ori_embs"] = emb_model.encode(name_ori, normalize_embeddings=True)

        base_names = [item["redefined_term"] for item in content["baseline"]]
        base_descs = [item["redefined_concept"] for item in content["baseline"]]
        rec["baseline"] = {
            "name_embs": emb_model.encode(base_names, normalize_embeddings=True),
            "desc_embs": emb_model.encode(base_descs, normalize_embeddings=True),
        }

        derived_result = []
        for group in content["derived"]:
            names = [item["redefined_term"] for item in group]
            descs = [item["redefined_concept"] for item in group]
            derived_result.append({
                "name_embs": emb_model.encode(names, normalize_embeddings=True),
                "desc_embs": emb_model.encode(descs, normalize_embeddings=True),
            })
        rec["derived"] = derived_result
        embedded.append(rec)

    return embedded


# ====================================================================
#  指标 1 — Novelty Score（新颖性）
# ====================================================================

def _cosine_distance(emb_ori, emb_list):
    """计算 emb_ori 与 emb_list 中各向量的平均余弦距离。"""
    if isinstance(emb_list[0], (list, np.ndarray)) and np.ndim(emb_list) == 2:
        sims = cosine_similarity([emb_ori], emb_list)[0]
    else:
        sims = cosine_similarity([emb_ori], [emb_list])[0]
    return float(1.0 - np.mean(sims))


def compute_novelty(embedding_dict):
    """计算所有概念的 Novelty Score。"""
    results = []
    for content in embedding_dict:
        emb_ori = content["desc_ori_embs"]
        base_avg = _cosine_distance(emb_ori, content["baseline"]["desc_embs"])
        derived_embs = [e for g in content["derived"] for e in g["desc_embs"]]
        derived_avg = _cosine_distance(emb_ori, derived_embs)
        results.append({
            "baseline_avg": round(base_avg, 5),
            "derived_avg": round(derived_avg, 5),
        })
    return results


# ====================================================================
#  指标 2 — Coverage Score（语义覆盖率）
# ====================================================================

def _make_keybert(emb_model):
    """创建与当前嵌入模型兼容的 KeyBERT 实例。"""
    if isinstance(emb_model, TransformersEmbedder):
        from keybert.backend import BaseEmbedder

        class _Wrapper(BaseEmbedder):
            def __init__(self, model):
                super().__init__()
                self.embedding_model = model

            def embed(self, documents, verbose=False):
                if isinstance(documents, str):
                    documents = [documents]
                return self.embedding_model.encode(documents)

        return KeyBERT(model=_Wrapper(emb_model))
    return KeyBERT(model=emb_model)


def _coverage_by_desc(original_desc, generated_desc, emb_model,
                      top_n1=COVERAGE_TOP_N_ORIGINAL,
                      top_n2=COVERAGE_TOP_N_GENERATED):
    """计算单对描述之间的关键词语义覆盖得分。"""
    kw_model = _make_keybert(emb_model)

    original_kws = kw_model.extract_keywords(original_desc, top_n=top_n1, stop_words="english")
    if not original_kws:
        return 0.0

    keywords = [kw[0] for kw in original_kws]
    weights = np.array([kw[1] for kw in original_kws])
    weights = weights / weights.sum()

    gen_tokens = [kw[0] for kw in
                  kw_model.extract_keywords(generated_desc, top_n=top_n2, stop_words="english")]
    if not gen_tokens:
        return 0.0

    kw_embs = emb_model.encode(keywords)
    tok_embs = emb_model.encode(gen_tokens)

    score = 0.0
    for i, emb_kw in enumerate(kw_embs):
        max_sim = max(cosine_similarity([emb_kw], [t])[0][0] for t in tok_embs)
        score += weights[i] * max_sim
    return round(float(score), 5)


def compute_coverage(term_dict, emb_model=None):
    """计算所有概念的 Coverage Score。"""
    if emb_model is None:
        emb_model = get_embedding_model()

    results = []
    for content in term_dict:
        desc_ori = content["original_description"]
        base_descs = [item["redefined_concept"] for item in content["baseline"]]
        derived_descs = [item["redefined_concept"]
                         for group in content["derived"] for item in group]

        base_avg = float(np.mean([_coverage_by_desc(desc_ori, d, emb_model) for d in base_descs]))
        der_avg = float(np.mean([_coverage_by_desc(desc_ori, d, emb_model) for d in derived_descs]))
        results.append({
            "baseline_avg": round(base_avg, 5),
            "derived_avg": round(der_avg, 5),
        })
    return results


# ====================================================================
#  指标 3 — Pairwise Distance（多样性）
# ====================================================================

def compute_pairwise_distance(embedding_dict):
    """计算生成描述之间的平均余弦距离。"""
    results = []
    for content in embedding_dict:
        base_embs = content["baseline"]["desc_embs"]
        pairs_b = list(combinations(range(len(base_embs)), 2))
        dist_b = [cosine_distances([base_embs[i]], [base_embs[j]])[0][0] for i, j in pairs_b]

        derived_all = [e for g in content["derived"] for e in g["desc_embs"]]
        pairs_d = list(combinations(range(len(derived_all)), 2))
        dist_d = [cosine_distances([derived_all[i]], [derived_all[j]])[0][0] for i, j in pairs_d]

        results.append({
            "baseline_avg": round(float(np.mean(dist_b)), 5) if dist_b else 0.0,
            "derived_avg": round(float(np.mean(dist_d)), 5) if dist_d else 0.0,
        })
    return results


# ====================================================================
#  指标 4 — Semantic Entropy（语义熵）
# ====================================================================

def _semantic_entropy(embeddings, max_k=SEMANTIC_ENTROPY_MAX_K,
                      pca_dim=SEMANTIC_ENTROPY_PCA_DIM):
    """计算嵌入向量集合的语义熵（基于层次聚类 + 香农熵）。"""
    if len(embeddings) < 2:
        return 0.0

    embs = np.array(embeddings)
    pca = PCA(n_components=min(pca_dim, embs.shape[0], embs.shape[1]))
    reduced = pca.fit_transform(embs)

    best_k, best_score = 2, -1
    for k in range(2, min(max_k, len(embs)) + 1):
        try:
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(reduced)
            score = silhouette_score(reduced, labels)
            if score > best_score:
                best_k, best_score = k, score
        except Exception:
            continue

    labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(reduced)
    entropy = 0.0
    for lbl in set(labels):
        p = np.sum(labels == lbl) / len(labels)
        entropy -= p * math.log(p)
    return round(entropy, 5)


def compute_semantic_entropy(embedding_dict):
    """计算所有概念的语义熵。"""
    results = []
    for content in embedding_dict:
        domain_embs = list(content["baseline"]["desc_embs"])
        element_embs = []
        for group in content["derived"]:
            element_embs.extend(group["desc_embs"])
            domain_embs.extend(group["desc_embs"])

        results.append({
            "element_entropy": round(float(_semantic_entropy(element_embs, 9)), 5),
            "domain_entropy": round(float(_semantic_entropy(domain_embs, 12)), 5),
        })
    return results


# ====================================================================
#  指标 5 — Feasibility（可行性，基于 LLM 打分）
# ====================================================================

def _feasibility_single(llm_call, original_term, generated_term, domain,
                         lambd=FEASIBILITY_LAMBDA):
    """对单个术语对进行正向推理 + 反向质询评分。"""
    forward_prompt = f"""
You are a Principal Engineer in the field of {domain}. The original term is "{original_term}", and a newly proposed concept is "{generated_term}".

Please reason in detail how this new concept could have evolved from the original one, considering technical background, principles, and potential innovations in the field of {domain} technologies.

Finally, rate the plausibility of this reasoning on a scale from 1 to 5, and briefly explain your reasoning.
Respond in JSON with keys "score" (int 1-5) and "rationale" (string).

Example(JSON):
{{
  "score": 4,
  "rationale": "Because ... "
}}
""".strip()

    backward_prompt = f"""
You are a Principal Engineer in the field of {domain}. Please critically evaluate the feasibility of the new concept "{generated_term}". From a technical, resource, or engineering standpoint, what challenges or limitations might arise in implementing this concept?

Identify potential issues and rate its overall feasibility on a scale from 1 to 5. The more severe or numerous the issues, the lower the feasibility score. Briefly explain your reasoning.
Respond in JSON with keys "score" (1-5) and "rationale" (string).

Example(JSON):
{{
  "score": 2,
  "rationale": "Potential issues are ... "
}}
""".strip()

    try:
        f1 = _generate_with_retry(llm_call, forward_prompt, {"score", "rationale"}).get("score", 3)
    except Exception:
        f1 = 3
    try:
        f2 = _generate_with_retry(llm_call, backward_prompt, {"score", "rationale"}).get("score", 3)
    except Exception:
        f2 = 3

    return f1, f2, lambd * f1 + (1 - lambd) * f2


def compute_feasibility(term_dict, domain, llm_call):
    """计算所有概念的 Feasibility Score（线程池并发打分）。"""
    results = []
    for content in term_dict:
        desc_ori = content["original_description"]
        base_descs = [item["redefined_concept"] for item in content["baseline"]]
        derived_descs = [item["redefined_concept"]
                         for group in content["derived"] for item in group]

        all_descs = base_descs + derived_descs
        scores = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            future_map = {
                pool.submit(_feasibility_single, llm_call, desc_ori, desc, domain): i
                for i, desc in enumerate(all_descs)
            }
            for fut in as_completed(future_map):
                idx = future_map[fut]
                scores[idx] = fut.result()

        b1, b2, bf = [], [], []
        for i in range(len(base_descs)):
            f1, f2, f = scores[i]
            b1.append(f1); b2.append(f2); bf.append(f)

        d1, d2, df = [], [], []
        for i in range(len(base_descs), len(all_descs)):
            f1, f2, f = scores[i]
            d1.append(f1); d2.append(f2); df.append(f)

        results.append({
            "base_F1_avg": round(float(np.mean(b1)), 2),
            "base_F2_avg": round(float(np.mean(b2)), 2),
            "base_F_avg": round(float(np.mean(bf)), 2),
            "der_F1_avg": round(float(np.mean(d1)), 2),
            "der_F2_avg": round(float(np.mean(d2)), 2),
            "der_F_avg": round(float(np.mean(df)), 2),
        })
    return results


# ====================================================================
#  保存结果
# ====================================================================

def save_outputs(output_dir, prefix, emb_data, metrics_dict):
    """保存嵌入数据和各项评估指标。"""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"{prefix}_emb_data.pkl"), "wb") as f:
        pickle.dump(emb_data, f)

    for key, value in metrics_dict.items():
        path = os.path.join(output_dir, f"{prefix}_{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)


# ====================================================================
#  命令行入口
# ====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="评估概念创造生成结果")
    p.add_argument("--input", required=True, help="生成结果 JSON 文件路径")
    p.add_argument("--output_dir", required=True, help="评估结果输出目录")
    p.add_argument("--prefix", default=None,
                   help="输出文件前缀（默认使用输入文件名）")
    p.add_argument("--domain", default="Communication",
                   help="用于 Feasibility 评估的领域标签")
    p.add_argument("--feasibility", action="store_true",
                   help="是否计算基于 LLM 的 Feasibility 指标")
    p.add_argument("--eval_model", default="gpt-4o",
                   help="用于 Feasibility 打分的 LLM 预设（默认 gpt-4o）")
    p.add_argument("--embedding_model", default="minilm",
                   choices=EMBEDDING_MODEL_CHOICES,
                   help="嵌入模型选择: minilm (默认), scibert, specter2")
    return p.parse_args()


def main():
    args = parse_args()
    prefix = args.prefix or os.path.splitext(os.path.basename(args.input))[0]

    print("加载数据...")
    data = load_json(args.input)

    print(f"计算嵌入向量 (模型: {args.embedding_model})...")
    emb_model = get_embedding_model(args.embedding_model)
    emb_data = embed_terms(data, emb_model)

    print("计算 Novelty...")
    novelty = compute_novelty(emb_data)

    print("计算 Coverage...")
    coverage = compute_coverage(data, emb_model)

    print("计算 Pairwise Distance...")
    pairwise = compute_pairwise_distance(emb_data)

    print("计算 Semantic Entropy...")
    entropy = compute_semantic_entropy(emb_data)

    metrics = {
        "novelty": novelty,
        "coverage": coverage,
        "pairwise_distance": pairwise,
        "entropy": entropy,
    }

    if args.feasibility:
        print("计算 Feasibility（LLM 打分）...")
        llm_call = build_llm_client(args.eval_model)
        feasibility = compute_feasibility(data, args.domain, llm_call)
        metrics["feasibility"] = feasibility

    save_outputs(args.output_dir, prefix, emb_data, metrics)
    print(f"\n所有结果已保存至 {args.output_dir}")


if __name__ == "__main__":
    main()
