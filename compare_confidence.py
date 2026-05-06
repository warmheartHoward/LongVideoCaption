"""对比同一视频在两次打标中的 confidence 指标。

用法:
    python compare_confidence.py --dir-a D:/outputs_v1 --dir-b D:/outputs_v2 --out D:/analysis
"""

import argparse
import json
import os
import sys
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 文件发现
# ---------------------------------------------------------------------------

def _list_video_names(output_dir: str) -> set:
    if not os.path.isdir(output_dir):
        return set()
    names = set()
    for entry in os.scandir(output_dir):
        if entry.is_dir():
            names.add(entry.name)
    return names


def _find_confidence_file(video_dir: str, filename: str) -> Optional[str]:
    """在 video_dir 下任意 hyper_sig 子目录中查找 filename，返回完整路径。"""
    candidates = []
    try:
        for entry in os.scandir(video_dir):
            if not entry.is_dir():
                continue
            path = os.path.join(entry.path, filename)
            if os.path.isfile(path):
                candidates.append(path)
    except OSError:
        pass

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(f"  ⚠️ {video_dir} 下存在多个 {filename}，将使用第一个: {candidates[0]}",
              file=sys.stderr)
        return candidates[0]
    return None


# ---------------------------------------------------------------------------
# 值类型判定
# ---------------------------------------------------------------------------

def _is_detail_list(value: Any) -> bool:
    """判断是否为明细字段（应跳过统计）。

    明细字段特征：list，且元素为 dict 或非数字的 str。
    纯数字 list / 空 list 不算明细。
    """
    if not isinstance(value, list):
        return False
    if not value:  # 空 list 当作可统计
        return False
    # 若所有元素都是数字，不是明细
    if all(isinstance(v, (int, float)) for v in value):
        return False
    # 若所有元素都是 bool，不是明细
    if all(isinstance(v, bool) for v in value):
        return False
    return True


# ---------------------------------------------------------------------------
# 递归收集叶子值
# ---------------------------------------------------------------------------

def _collect_leaves(
    data: dict,
    prefix: str = "",
) -> Dict[str, List[float]]:
    """递归遍历 dict，收集所有可统计的叶子字段值。

    跳过：
      - list 类型的明细字段
      - None 值
      - 非数值类型（str 等）
    bool 转为 0/1。
    """
    result: Dict[str, List[float]] = {}

    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if value is None:
            continue

        if isinstance(value, bool):
            result[full_key] = [1.0 if value else 0.0]
        elif isinstance(value, (int, float)):
            result[full_key] = [float(value)]
        elif isinstance(value, dict):
            result.update(_collect_leaves(value, prefix=full_key))
        elif _is_detail_list(value):
            continue  # 跳过明细字段
        # str / 其他类型跳过

    return result


# ---------------------------------------------------------------------------
# 统计计算
# ---------------------------------------------------------------------------

def _compute_stats(
    values_a: List[float],
    values_b: List[float],
) -> dict:
    """输入两组等长数值列表，输出统计 dict。"""
    n = len(values_a)
    if n == 0:
        return {
            "mean_a": 0.0,
            "mean_b": 0.0,
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "sample_count": 0,
        }

    mean_a = sum(values_a) / n
    mean_b = sum(values_b) / n
    diffs = [b - a for a, b in zip(values_a, values_b)]
    mean_diff = sum(diffs) / n

    if n > 1:
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        std_diff = sqrt(variance)
    else:
        std_diff = 0.0

    return {
        "mean_a": round(mean_a, 6),
        "mean_b": round(mean_b, 6),
        "mean_diff": round(mean_diff, 6),
        "std_diff": round(std_diff, 6),
        "sample_count": n,
    }


# ---------------------------------------------------------------------------
# 构建结构一致的输出 dict
# ---------------------------------------------------------------------------

def _build_output_structure(original: dict, stats_map: Dict[str, dict]) -> dict:
    """按 original 的结构递归重建，叶子值替换为 stats_map 中的统计 dict。"""

    def _walk(value: Any, prefix: str) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            return stats_map.get(prefix)
        if isinstance(value, (int, float)):
            return stats_map.get(prefix)
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                child_key = f"{prefix}.{k}" if prefix else k
                result = _walk(v, child_key)
                if result is not None:
                    out[k] = result
            return out if out else None
        if _is_detail_list(value):
            return None  # 明细字段直接去掉
        return None

    return _walk(original, "") or {}


# ---------------------------------------------------------------------------
# 单类 confidence 对比
# ---------------------------------------------------------------------------

def _compare_one_confidence(
    video_name: str,
    dir_a: str,
    dir_b: str,
    filename: str,
    extract_stage3: bool = False,
) -> Optional[Tuple[dict, dict]]:
    """加载并收集单个 confidence 文件的叶子值。

    返回 (leaves_a_dict, leaves_b_dict)，即该 video 的全部叶子字段 → [value]。
    """
    path_a = _find_confidence_file(os.path.join(dir_a, video_name), filename)
    path_b = _find_confidence_file(os.path.join(dir_b, video_name), filename)

    if not path_a or not path_b:
        return None

    try:
        with open(path_a, "r", encoding="utf-8") as f:
            data_a = json.load(f)
        with open(path_b, "r", encoding="utf-8") as f:
            data_b = json.load(f)
    except Exception as e:
        print(f"  ⚠️ [{video_name}] 读取 {filename} 失败: {e}", file=sys.stderr)
        return None

    if extract_stage3:
        data_a = data_a.get("stage3_confidence", {}) or {}
        data_b = data_b.get("stage3_confidence", {}) or {}

    leaves_a = _collect_leaves(data_a)
    leaves_b = _collect_leaves(data_b)

    return leaves_a, leaves_b


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def compare_confidence(dir_a: str, dir_b: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    videos_a = _list_video_names(dir_a)
    videos_b = _list_video_names(dir_b)
    common = sorted(videos_a & videos_b)

    if not common:
        print(f"❌ 两个目录中没有共同视频。")
        print(f"   dir_a 视频数: {len(videos_a)}")
        print(f"   dir_b 视频数: {len(videos_b)}")
        return

    print(f"🔍 发现 {len(common)} 个共同视频")

    # 三类 confidence 的配置
    targets = [
        {
            "label": "pass1",
            "filename": "pass1_confidence.json",
            "out_name": "pass1_compare.json",
            "extract_stage3": False,
        },
        {
            "label": "pass3",
            "filename": "pass3_confidence.json",
            "out_name": "pass3_compare.json",
            "extract_stage3": False,
        },
        {
            "label": "stage3",
            "filename": "stage3_polished.json",
            "out_name": "stage3_compare.json",
            "extract_stage3": True,
        },
    ]

    for target in targets:
        label = target["label"]
        print(f"\n{'=' * 60}")
        print(f"📊 对比 {label} confidence ...")

        # 收集所有视频的叶子值
        # collector[field_key] = (values_a[], values_b[])
        collector: Dict[str, Tuple[List[float], List[float]]] = {}
        matched = 0

        for vname in common:
            result = _compare_one_confidence(
                vname, dir_a, dir_b,
                target["filename"],
                extract_stage3=target["extract_stage3"],
            )
            if result is None:
                continue

            leaves_a, leaves_b = result
            matched += 1

            # 合并字段名（取并集）
            all_keys = set(leaves_a.keys()) | set(leaves_b.keys())
            for key in all_keys:
                if key not in collector:
                    collector[key] = ([], [])
                va_list, vb_list = collector[key]
                va_list.extend(leaves_a.get(key, []))
                vb_list.extend(leaves_b.get(key, []))

        if matched == 0:
            print(f"  ⚠️ 没有成功匹配的视频，跳过 {label}")
            continue

        print(f"  ✅ 成功匹配 {matched} 个视频")

        # 计算统计
        stats_map: Dict[str, dict] = {}
        for key, (va, vb) in collector.items():
            stats_map[key] = _compute_stats(va, vb)

        # 重建输出结构 —— 用第一个匹配视频的数据作为模板
        template = None
        for vname in common:
            path = _find_confidence_file(
                os.path.join(dir_a, vname), target["filename"]
            )
            if path:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        template = json.load(f)
                    if target["extract_stage3"]:
                        template = template.get("stage3_confidence", {}) or {}
                    break
                except Exception:
                    pass

        if template is None:
            print(f"  ⚠️ 无法加载模板结构，跳过 {label}")
            continue

        output = _build_output_structure(template, stats_map)
        output["_meta"] = {
            "sample_count": matched,
            "dir_a": os.path.abspath(dir_a),
            "dir_b": os.path.abspath(dir_b),
        }

        out_path = os.path.join(out_dir, target["out_name"])
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"  💾 已保存: {out_path}")

    print(f"\n🎉 完成。结果保存在: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="对比同一视频在两次打标中的 confidence 指标"
    )
    p.add_argument("--dir-a", required=True, help="第一次打标的 output 目录")
    p.add_argument("--dir-b", required=True, help="第二次打标的 output 目录")
    p.add_argument("--out", required=True, help="分析结果保存目录")
    args = p.parse_args()

    compare_confidence(args.dir_a, args.dir_b, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
