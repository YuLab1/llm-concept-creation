"""
公共 LLM 工具模块 — 供 RQ1,2 与 RQ3 共同使用。

提供:
  - create_client()          : 构建 OpenAI 兼容客户端
  - build_llm_client()       : 根据预设构造 LLM 调用器（支持 langchain / dashscope）
  - extract_json_from_text() : 多策略鲁棒 JSON 提取
  - extract_json_response()  : 轻量 JSON 提取
  - validate_json()          : 带字段校验的 JSON 解析
  - generate_with_retry()    : 带重试与缓存的 LLM 调用
  - load_json() / save_json(): 文件读写辅助函数
"""

import hashlib
import json
import os
import re
import threading
import time
from typing import Dict, Optional

import openai
from openai import OpenAI

import config

# ====================================================================
#  langchain 按需导入（仅 RQ1,2 使用，RQ3 不依赖）
# ====================================================================
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import SystemMessage
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


# ====================================================================
#  OpenAI 客户端（RQ3 使用）
# ====================================================================

def create_client() -> OpenAI:
    """根据 config 中的 API 凭据创建 OpenAI 客户端。"""
    return OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )


# ====================================================================
#  JSON 提取
# ====================================================================

def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    多策略鲁棒 JSON 提取。

    依次尝试：
      1. 直接 json.loads
      2. 去除 Markdown 代码块标记
      3. 第一个 '{' 到最后一个 '}'
      4. 移除注释和尾随逗号后再解析
    """
    try:
        return json.loads(text)
    except Exception:
        pass

    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{.*\}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except Exception:
                continue

    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(text[start_idx:end_idx + 1])
        except Exception:
            pass

    text_clean = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text_clean = re.sub(r',\s*}', '}', text_clean)
    text_clean = re.sub(r',\s*]', ']', text_clean)
    try:
        return json.loads(text_clean)
    except Exception:
        pass

    return None


def extract_json_response(text: str) -> Optional[Dict]:
    """
    轻量 JSON 提取，处理纯 JSON 和 Markdown 包裹的 JSON。
    """
    try:
        return json.loads(text)
    except Exception:
        pass

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return None


def get_cache_key(key_str: str) -> str:
    """根据字符串生成 MD5 缓存键。"""
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()


# ====================================================================
#  以下为 RQ1,2 使用的工具函数
# ====================================================================

_cache_lock = threading.Lock()


def _cache_key_rq12(prompt: str, required_keys: frozenset) -> str:
    """根据 prompt 和 required_keys 生成唯一缓存键。"""
    raw = prompt.strip() + "|" + ",".join(sorted(required_keys))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str):
    """从磁盘缓存读取，命中返回 dict，未命中返回 None。"""
    if not config.RQ12_ENABLE_CACHE:
        return None
    path = config.RQ12_CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _cache_put(key: str, value: dict):
    """将结果写入磁盘缓存。"""
    if not config.RQ12_ENABLE_CACHE:
        return
    with _cache_lock:
        config.RQ12_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = config.RQ12_CACHE_DIR / f"{key}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
        except OSError:
            pass


def build_llm_client(preset_name: str):
    """
    根据预设名称构建 LLM 调用函数 ``llm_call(prompt) -> str``。

    支持两种后端:
      * "openai"    — 使用 langchain ChatOpenAI
      * "dashscope" — 使用 OpenAI 兼容客户端
    """
    if not _HAS_LANGCHAIN:
        raise ImportError(
            "langchain 未安装。请运行 pip install langchain langchain-openai")

    preset = config.MODEL_PRESETS[preset_name]
    api_key = os.getenv(preset["api_key_env"], "")
    base_url = os.getenv(preset["base_url_env"], "")

    if preset["backend"] == "openai":
        llm = ChatOpenAI(
            openai_api_key=api_key,
            base_url=base_url or None,
            model=preset["model_name"],
            temperature=preset["temperature"],
            timeout=preset["timeout"],
        )

        def _call_langchain(prompt: str) -> str:
            response = llm.invoke([SystemMessage(content=prompt)])
            return response.content.strip()

        return _call_langchain

    elif preset["backend"] == "dashscope":
        client = OpenAI(
            api_key=api_key,
            base_url=base_url or config.DASHSCOPE_BASE_URL,
            timeout=preset["timeout"],
        )

        def _call_dashscope(prompt: str) -> str:
            response = client.chat.completions.create(
                model=preset["model_name"],
                messages=[
                    {"role": "system",
                     "content": "You are a domain expert tasked with inventing "
                                "technically sound and semantically meaningful "
                                "new concepts."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )
            return response.choices[0].message.content.strip()

        return _call_dashscope

    else:
        raise ValueError(f"未知后端: {preset['backend']}")


# ====================================================================
#  带字段校验的 JSON 解析（RQ1,2 使用）
# ====================================================================

def normalize_field_names(obj):
    """将 LLM 返回的常见字段名变体映射为标准字段名。"""
    if not isinstance(obj, dict):
        return obj

    field_mapping = {
        "mapped_element": "mapped",
        "ai_concept": "mapped",
        "mapped_concept": "mapped",
        "mapped_term": "mapped",
        "target_concept": "mapped",
        "corresponding_concept": "mapped",
        "explanation": "justification",
        "reason": "justification",
        "rationale": "justification",
    }

    normalized = {}
    for key, value in obj.items():
        norm_key = field_mapping.get(key.lower(), key)
        if isinstance(value, dict):
            normalized[norm_key] = normalize_field_names(value)
        elif isinstance(value, list):
            normalized[norm_key] = [
                normalize_field_names(v) if isinstance(v, dict) else v
                for v in value
            ]
        else:
            normalized[norm_key] = value
    return normalized


def find_required_keys_in_dict(obj, required_keys, max_depth=3, _depth=0):
    """递归查找包含所有 required_keys 的扁平字典。"""
    if _depth >= max_depth:
        return None

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                item = normalize_field_names(item)
                result = find_required_keys_in_dict(
                    item, required_keys, max_depth, _depth)
                if result is not None:
                    return result
        return None

    if isinstance(obj, dict):
        obj = normalize_field_names(obj)
        if required_keys.issubset(obj.keys()):
            if all(not isinstance(v, (dict, list)) for v in obj.values()):
                return obj
        for value in obj.values():
            if isinstance(value, (dict, list)):
                result = find_required_keys_in_dict(
                    value, required_keys, max_depth, _depth + 1)
                if result is not None:
                    return result
    return None


def validate_json(response_text: str, required_keys: set):
    """
    从 LLM 原始输出中提取并校验 JSON 对象。

    成功返回解析后的字典，失败返回 False。
    """
    try:
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1:
            start = text.find("[")
            end = text.rfind("]")

        if start != -1 and end != -1:
            cleaned = text[start:end + 1]
        else:
            cleaned = text

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            brace = 0
            s = -1
            for i, ch in enumerate(text):
                if ch == "{":
                    if s == -1:
                        s = i
                    brace += 1
                elif ch == "}":
                    brace -= 1
                    if brace == 0 and s != -1:
                        try:
                            parsed = json.loads(text[s:i + 1])
                            break
                        except json.JSONDecodeError:
                            continue
            else:
                return False

        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]

        if required_keys.issubset(parsed.keys()):
            if all(not isinstance(v, dict) for v in parsed.values()):
                return parsed
            found = find_required_keys_in_dict(parsed, required_keys)
            return found if found else False

        found = find_required_keys_in_dict(parsed, required_keys)
        return found if found else False

    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        print(f"JSON 解析错误: {exc}")
        return False


# ====================================================================
#  带重试和缓存的 LLM 调用（RQ1,2 使用）
# ====================================================================

def generate_with_retry(llm_call, prompt: str, required_keys: set):
    """
    调用 llm_call(prompt)，校验 JSON 格式，失败时自动重试。

    相同 prompt + required_keys 的成功结果会缓存到磁盘，重复调用直接返回。
    """
    frozen_keys = frozenset(required_keys)
    key = _cache_key_rq12(prompt, frozen_keys)

    cached = _cache_get(key)
    if cached is not None:
        return cached

    for attempt in range(config.MAX_RETRIES):
        try:
            raw = llm_call(prompt)
            result = validate_json(raw, required_keys)
            if result:
                _cache_put(key, result)
                return result
            print(f"第 {attempt + 1} 次尝试: JSON 格式错误，重新请求...")

        except openai.APIError as exc:
            time.sleep(config.ERROR_SLEEP)
            print(f"API 错误: {exc}  ({attempt + 1}/{config.MAX_RETRIES})")

        except Exception as exc:
            time.sleep(config.ERROR_SLEEP)
            print(f"未知错误: {exc}  ({attempt + 1}/{config.MAX_RETRIES})")

        time.sleep(config.RETRY_SLEEP)

    print(f"已达最大重试次数 ({config.MAX_RETRIES})，返回空字典")
    return {}


# ====================================================================
#  文件读写辅助
# ====================================================================

def load_json(file_path):
    """读取 JSON 文件。"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path):
    """保存 JSON 文件（自动创建父目录）。"""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
