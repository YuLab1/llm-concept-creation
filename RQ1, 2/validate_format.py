"""
格式验证脚本 — 验证概念创造生成结果文件的 JSON 格式是否符合预期。

遍历指定目录下的所有 JSON 文件，按照单领域 / 跨领域的 schema 进行校验。

单领域（single）derived 字段要求:
    new_element, element_description, redefined_term, redefined_concept

跨领域（cross）derived 字段要求:
    mapped, justification, redefined_term, redefined_concept

用法:
    python validate_format.py --input_dir ../test/create生成结果_修复
    python validate_format.py --input_dir ../test/create生成结果_修复 --report report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_utils import save_json


# ====================================================================
#  单条记录验证
# ====================================================================

def _validate_baseline(item, errors):
    """验证 baseline 字段"""
    if not isinstance(item.get("baseline"), list):
        errors.append("baseline 必须是列表")
        return
    if len(item["baseline"]) != 3:
        errors.append(f"baseline 应包含 3 个元素，实际 {len(item['baseline'])} 个")
        return
    for i, b in enumerate(item["baseline"]):
        if not isinstance(b, dict):
            errors.append(f"baseline[{i}] 必须是字典")
        else:
            for k in ("redefined_term", "redefined_concept"):
                if k not in b:
                    errors.append(f"baseline[{i}] 缺少 '{k}'")


def _validate_top_elements(item, errors):
    """验证 top_elements 字段"""
    if not isinstance(item.get("top_elements"), list):
        errors.append("top_elements 必须是列表")
        return
    if len(item["top_elements"]) != 3:
        errors.append(f"top_elements 应包含 3 个元素，实际 {len(item['top_elements'])} 个")
        return
    for i, e in enumerate(item["top_elements"]):
        if not isinstance(e, dict):
            errors.append(f"top_elements[{i}] 必须是字典")
        else:
            for k in ("element_name", "innovativeness_score"):
                if k not in e:
                    errors.append(f"top_elements[{i}] 缺少 '{k}'")


def _validate_derived(item, errors, required_keys):
    """验证 derived 字段"""
    if not isinstance(item.get("derived"), list):
        errors.append("derived 必须是列表")
        return
    if len(item["derived"]) != 3:
        errors.append(f"derived 应包含 3 个子列表，实际 {len(item['derived'])} 个")
        return
    for i, sub in enumerate(item["derived"]):
        if not isinstance(sub, list):
            errors.append(f"derived[{i}] 必须是列表")
            continue
        if len(sub) != 3:
            errors.append(f"derived[{i}] 应包含 3 个元素，实际 {len(sub)} 个")
            continue
        for j, d in enumerate(sub):
            if not isinstance(d, dict):
                errors.append(f"derived[{i}][{j}] 必须是字典")
            else:
                for k in required_keys:
                    if k not in d:
                        errors.append(f"derived[{i}][{j}] 缺少 '{k}'")


def validate_single_domain_item(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """验证单领域格式的一个条目"""
    errors = []
    for key in ("original_concept", "original_description", "baseline",
                "top_elements", "derived"):
        if key not in item:
            errors.append(f"缺少顶层字段: {key}")
    if errors:
        return False, errors

    if not isinstance(item["original_concept"], str):
        errors.append("original_concept 必须是字符串")
    if not isinstance(item["original_description"], str):
        errors.append("original_description 必须是字符串")

    _validate_baseline(item, errors)
    _validate_top_elements(item, errors)
    _validate_derived(item, errors,
                      ["new_element", "element_description", "redefined_term", "redefined_concept"])
    return len(errors) == 0, errors


def validate_cross_domain_item(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """验证跨领域格式的一个条目"""
    errors = []
    for key in ("original_concept", "original_description", "baseline",
                "top_elements", "derived"):
        if key not in item:
            errors.append(f"缺少顶层字段: {key}")
    if errors:
        return False, errors

    if not isinstance(item["original_concept"], str):
        errors.append("original_concept 必须是字符串")
    if not isinstance(item["original_description"], str):
        errors.append("original_description 必须是字符串")

    _validate_baseline(item, errors)
    _validate_top_elements(item, errors)
    _validate_derived(item, errors,
                      ["mapped", "justification", "redefined_term", "redefined_concept"])
    return len(errors) == 0, errors


# ====================================================================
#  自动检测领域类型（单领域 / 跨领域）
# ====================================================================

def detect_domain_type(item: Dict[str, Any]) -> str:
    """根据 derived 字段内容判断是单领域还是跨领域，返回 'single'、'cross' 或 'unknown'。"""
    derived = item.get("derived")
    if not isinstance(derived, list):
        return "unknown"
    for sub in derived:
        if isinstance(sub, list) and sub:
            first = sub[0]
            if isinstance(first, dict):
                if "new_element" in first and "element_description" in first:
                    return "single"
                if "mapped" in first and "justification" in first:
                    return "cross"
    return "unknown"


# ====================================================================
#  文件 / 目录验证
# ====================================================================

def validate_file(file_path: str) -> Tuple[bool, List[Dict]]:
    """验证单个 JSON 文件"""
    issues = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return False, [{"file": file_path, "issue": str(exc)}]

    if not isinstance(data, list):
        return False, [{"file": file_path, "issue": "根元素必须是列表"}]

    expected = 150
    if len(data) != expected:
        issues.append({"file": file_path,
                        "issue": f"期望 {expected} 条数据，实际 {len(data)} 条"})

    filename_hint = "single" if "single" in os.path.basename(file_path).lower() else "cross"

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            issues.append({"file": file_path, "index": idx, "issue": "条目不是字典类型"})
            continue

        dtype = detect_domain_type(item)
        if dtype == "unknown":
            dtype = filename_hint

        if dtype == "single":
            ok, errs = validate_single_domain_item(item)
        else:
            ok, errs = validate_cross_domain_item(item)

        if not ok:
            issues.append({
                "file": file_path,
                "index": idx,
                "original_concept": item.get("original_concept", "N/A"),
                "errors": errs,
            })

    return len(issues) == 0, issues


def validate_directory(directory: str) -> Dict[str, Any]:
    """验证目录下的所有 JSON 文件"""
    results = {"total_files": 0, "valid_files": 0, "invalid_files": 0, "issues": []}
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"目录不存在: {directory}")
        return results

    json_files = sorted(dir_path.glob("*.json"))
    results["total_files"] = len(json_files)
    print(f"\n正在验证: {directory}  ({len(json_files)} 个文件)")

    for jf in json_files:
        ok, issues = validate_file(str(jf))
        if ok:
            results["valid_files"] += 1
            print(f"  [OK]  {jf.name}")
        else:
            results["invalid_files"] += 1
            print(f"  [ERR] {jf.name}  ({len(issues)} ge wenti)")
            results["issues"].extend(issues)

    return results


# ====================================================================
#  命令行入口
# ====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="验证概念创造生成结果文件的格式")
    p.add_argument("--input_dir", required=True,
                   help="包含各模型子文件夹的根目录")
    p.add_argument("--report", default="format_validation_report.json",
                   help="验证报告输出路径")
    return p.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_dir)

    all_results = {}
    all_issues = []

    for sub in sorted(input_root.iterdir()):
        if sub.is_dir():
            res = validate_directory(str(sub))
            all_results[sub.name] = res
            all_issues.extend(res["issues"])

    report = {
        "summary": {
            "total_files": sum(r["total_files"] for r in all_results.values()),
            "valid_files": sum(r["valid_files"] for r in all_results.values()),
            "invalid_files": sum(r["invalid_files"] for r in all_results.values()),
        },
        "directory_results": all_results,
        "all_issues": all_issues,
    }

    save_json(report, args.report)

    print(f"\n{'=' * 50}")
    print(f"总文件数: {report['summary']['total_files']}")
    print(f"格式正确: {report['summary']['valid_files']}")
    print(f"格式错误: {report['summary']['invalid_files']}")
    print(f"报告已保存: {args.report}")


if __name__ == "__main__":
    main()
