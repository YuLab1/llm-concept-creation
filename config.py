"""
统一配置文件 — RQ1,2 与 RQ3 的所有可调参数集中管理。

包含路径设置、LLM 模型预设、流程超参数、评估指标参数、可视化参数等。
敏感凭据（API Key、Endpoint）应通过环境变量或 .env 文件配置，
严禁在源码中硬编码。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 显式指定 .env 路径，确保从任意目录运行都能加载
CODE_DIR = Path(__file__).resolve().parent
_env_path = CODE_DIR / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()

# ============================================================
#  公共 API 凭据（通过 .env 或环境变量配置）
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ============================================================
#  嵌入模型路径（可选，默认从 HuggingFace 下载）
# ============================================================
EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    "all-MiniLM-L6-v2",
)

# ============================================================
#  RQ1, 2 专用配置
# ============================================================
RQ12_DIR = CODE_DIR / "RQ1, 2"
RQ12_BASE_DIR = CODE_DIR.parent                           # camelready终稿/
RQ12_DATA_DIR = RQ12_BASE_DIR / "Data" / "DATA" / "数据集"
RQ12_INPUT_CONCEPTS_DIR = RQ12_DATA_DIR / "generation concepts"
RQ12_OUTPUT_DIR = RQ12_BASE_DIR / "output"
RQ12_CACHE_DIR = RQ12_DIR / ".llm_cache"
RQ12_ENABLE_CACHE = True

# --- LLM 模型预设 ---
MODEL_PRESETS = {
    "gpt-4o": {
        "backend": "openai",
        "model_name": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "temperature": 0.7,
        "timeout": 60,
    },
    "gpt-4o-mini": {
        "backend": "openai",
        "model_name": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "temperature": 0.7,
        "timeout": 60,
    },
    "qwen-max": {
        "backend": "dashscope",
        "model_name": "qwen-max",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "temperature": 0.7,
        "timeout": 60,
    },
}

# --- LLM 调用参数 ---
MAX_RETRIES = 15              # 最大重试次数
RETRY_SLEEP = 1               # 普通重试间隔（秒）
ERROR_SLEEP = 5               # API / 未知错误后的等待时间（秒）
MAX_WORKERS = 5               # RQ1,2 线程池并发数

# --- 概念创造流程参数 ---
TOP_K_ELEMENTS = 3            # 解构后保留的 top-k 要素数量
N_REPEAT = 3                  # 每个要素 / 基线的重复生成次数
DOMAINS = ["Communication", "Electromagnetism", "Artificial Intelligence"]

# --- 评估指标计算参数 ---
SEMANTIC_ENTROPY_MAX_K = 12
SEMANTIC_ENTROPY_PCA_DIM = 8
COVERAGE_TOP_N_ORIGINAL = 5
COVERAGE_TOP_N_GENERATED = 7
FEASIBILITY_LAMBDA = 0.5

# --- 领域映射（文件名前缀 → 领域标签）---
DOMAIN_MAPPING = {
    "ai":  "Artificial Intelligence",
    "com": "Communication",
    "ele": "Electromagnetism",
}

# ============================================================
#  RQ3 专用配置
# ============================================================
RQ3_DIR = CODE_DIR / "RQ3"
RQ3_DATA_DIR = RQ3_DIR / "data"
RQ3_OUTPUT_DIR = RQ3_DIR / "output"

# --- 输入数据文件 ---
COM_CONCEPTS_FILE = RQ3_DATA_DIR / "communication_concepts.json"
AI_CONCEPTS_FILE = RQ3_DATA_DIR / "ai_concepts.json"
BASELINE_FILE = RQ3_DATA_DIR / "combine.json"
GENERATED_FILE = RQ3_DATA_DIR / "combined_concepts_generated.json"
MATCH_RESULTS_FILE = RQ3_DATA_DIR / "exact_match_results.json"
EVAL_SCORES_FILE = RQ3_DATA_DIR / "evaluation_scores_1to4.json"

# --- 模型设置 ---
GENERATION_MODEL = "gpt-3.5-turbo-0125"   # 步骤一：组合生成
EVALUATION_MODEL = "gpt-4o"                # 步骤二、三：评估

# --- 多线程与重试 ---
RQ3_MAX_WORKERS = 10          # RQ3 线程池并发数
GENERATION_MAX_RETRY = 7      # 步骤一重试次数
EVALUATION_MAX_RETRY = 5      # 步骤二、三重试次数
EVAL_TIMES = 3                # 步骤三每个概念评估次数

# --- 可视化（步骤四）---
BASELINE_TOTAL = 20           # 基准概念总数

SCIENTIFIC_COLORS = {
    'high': '#2E7D32',         # 深绿（高分）
    'medium': '#F9A825',       # 金黄（中分）
    'low': '#C62828',          # 深红（低分）
    'baseline': '#1565C0',     # 深蓝（基准匹配）
}

PLOT_PARAMS = {
    'font.size': 22,
    'font.family': 'sans-serif',
    'axes.labelsize': 26,
    'axes.titlesize': 28,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 22,
    'figure.dpi': 300,
    'pdf.fonttype': 42,        # TrueType 字体嵌入
    'ps.fonttype': 42,
}
