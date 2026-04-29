import json
import os
import threading
from typing import Optional, Tuple

from .config import PipelineConfig
from .frame_extractor import (
    detect_scenes,
    get_base64_frames,
    get_base64_frames_qwen,
    get_raw_chunk_video_base64,
    get_target_timestamps,
    get_video_duration,
)
from .llm_client import request_llm_with_retry
from .prompts.pass1_v3 import build_sys_prompt, build_usr_prompt
from .token_tracker import TokenTracker
from .utils import format_timestamp, format_timestamp_sec, parse_timestamp_to_seconds, sanitize_filename


PASS_NAME = "pass1_perception"

_log_context = threading.local()


def _log(video_tag: str, msg: str) -> None:
    line = f"[{video_tag}] {msg}" if video_tag else msg
    print(line)
    log_path = getattr(_log_context, "log_file", None)
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def _save_chunk_prompt(
    run_dir: str,
    chunk_idx: int,
    chunk_name: str,
    chunk_start: float,
    chunk_end: float,
    usr_prompt: str,
    previous_context: str,
    timestamps_str_list: list,
) -> None:
    prompts_dir = os.path.join(run_dir, "_pass1_prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    safe_chunk = sanitize_filename(chunk_name.replace(" ", "")).strip("[]")
    path = os.path.join(prompts_dir, f"chunk_{chunk_idx:03d}_{safe_chunk}.json")
    payload = {
        "chunk_index": chunk_idx,
        "chunk_name": chunk_name,
        "chunk_start_sec": chunk_start,
        "chunk_end_sec": chunk_end,
        "previous_context": previous_context,
        "timestamps_whitelist": timestamps_str_list,
        "usr_prompt": usr_prompt,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 保存 chunk prompt 失败: {e}")


def _format_overlap_events(events_slice: list) -> str:
    """把重叠区间内的 event 完整字段展平成可读文本，供下一段模型参考。"""
    if not events_slice:
        return "（无）"
    blocks = []
    for idx, ev in enumerate(events_slice, 1):
        blocks.append(
            f"--- 上段重叠 Event #{idx} [{ev.get('start_time', '')} ~ {ev.get('end_time', '')}] ---\n"
            f"step1_objective_visual:\n{ev.get('step1_objective_visual', '')}\n"
            f"step2_contextual_reasoning:\n{ev.get('step2_contextual_reasoning', '')}\n"
            f"step3_synthesized_dense_caption:\n{ev.get('step3_synthesized_dense_caption', '')}"
        )
    return "\n\n".join(blocks)


def _build_previous_context(
    history_summaries: list,
    last_action: str,
    last_end_str: str,
    next_chunk_start_sec: float,
    overlap_active: bool,
    overlap_events: Optional[list] = None,
) -> str:
    accumulated_story = "\n".join(history_summaries) if history_summaries else "暂无"
    if overlap_active:
        overlap_text = _format_overlap_events(overlap_events or [])
        return (
            f"【全局剧情脉络】:\n{accumulated_story}\n\n"
            f"【⚠️ 视觉重叠回顾区间】：本片段画面 [{format_timestamp(next_chunk_start_sec)} ~ {last_end_str}] "
            f"是上一片段最后若干个 event 的视觉回顾，仅作为前情上下文展示，**严禁对此区间重新创建 event**。\n\n"
            f"【⚠️ 跨chunk截断判定任务】：\n"
            f"上段最后一个 event 的 end_time 为 {last_end_str}，恰好落在上段 chunk 边界。"
            f"请结合本段开头画面，判断「上段重叠区间中 end_time 为 {last_end_str} 的那个 event」"
            f"的动作是否在上段已真正完成：\n"
            f"  • 若已完成 → 顶层字段 `prev_event_revision` 填 null，events[0] 从 {last_end_str} 之后开始；\n"
            f"  • 若动作延续至本段（未完成） → 顶层字段 `prev_event_revision` 填修订对象"
            f"（结构见下方 JSON schema 说明），且 events[0] 必须从修订后的 end_time 之后开始，"
            f"严禁把延续部分再次作为独立 event 重复打标。\n\n"
            f"【上段重叠区间·完整 event 输出（step1/step2/step3 全量）】"
            f"（仅作为上下文参考，禁止复写、改写或重新切分；新 event 必须从 {last_end_str} 之后开始）:\n"
            f"{overlap_text}\n\n"
            f"【当前接力要求】: 上一幕最后一个事件是：'{last_action}'，结束于 {last_end_str}。"
            # f"请从 {last_end_str} 开始切分新事件，紧接着这个动作和时间点继续描述。"
        )
    return (
        f"【全局剧情脉络】:\n{accumulated_story}\n\n"
        f"【当前无缝接力要求】: 上一幕的最后一个事件是：'{last_action}'，结束于 {last_end_str}。"
        f"请紧接着这个动作和时间点继续描述。"
    )


def _pick_next_start(
    events: list,
    cfg: PipelineConfig,
    current_chunk_start: float,
    current_chunk_end: float,
) -> Optional[Tuple[float, str, str, bool, int]]:
    """根据 prev_event_overlap_count 选择下一个 chunk_start。

    返回 (next_start_sec, last_end_str, last_action, overlap_active, k_used)。
    若 events 为空或末尾时间不合理（接力异常），返回 None，由调用方走 80% 兜底。
    """
    if not events:
        return None

    last_ev = events[-1]
    last_end_str = last_ev.get("end_time", "")
    last_end_sec = parse_timestamp_to_seconds(last_end_str)
    last_action = last_ev.get("step3_synthesized_dense_caption", "")

    if not (current_chunk_start + 5 <= last_end_sec <= current_chunk_end + 10):
        return None

    n = max(0, int(cfg.prev_event_overlap_count))
    if n == 0:
        return last_end_sec, last_end_str, last_action, False, 0

    k = min(n, len(events))
    anchor_ev = events[-k]
    proposed = parse_timestamp_to_seconds(anchor_ev.get("start_time", ""))

    if proposed <= current_chunk_start + 1.0:
        return last_end_sec, last_end_str, last_action, False, 0

    return proposed, last_end_str, last_action, True, k


def _record_large_snap_delta(stats: Optional[dict], delta: float, label: str) -> None:
    if stats is None or delta <= 1.0:
        return
    stats["large_snap_delta_sum"] = round(float(stats.get("large_snap_delta_sum", 0.0)) + float(delta), 3)
    details = stats.setdefault("large_snap_deltas", [])
    details.append({"label": label, "delta_sec": round(float(delta), 3)})


def _validate_and_snap_event_times(
    events: list,
    whitelist_str_list: list,
    chunk_start: float,
    chunk_end: float,
    video_tag: str,
    stats: Optional[dict] = None,
) -> None:
    """就地校验 events 的 start_time/end_time，把非白名单值 snap 到最近的合法项。

    - 白名单为空 → 直接跳过
    - 非白名单 → snap 到最近项（Δ>1.5s 记 WARN，否则 INFO）
    - start > end → swap
    - end - start < 0.1s → 丢弃该 event
    - 越出 chunk 边界 > 0.5s → clamp 到 chunk 内最近白名单
    - key_frame_times 仅 WARN 不 snap
    """
    if not whitelist_str_list or not events:
        return

    whitelist_map = {s: parse_timestamp_to_seconds(s) for s in whitelist_str_list}
    sorted_items = sorted(whitelist_map.items(), key=lambda p: p[1])

    def _snap(ts_str: str, label: str) -> Tuple[str, float]:
        target = parse_timestamp_to_seconds(ts_str)
        for wl_str, wl_sec in sorted_items:
            if abs(wl_sec - target) < 0.01:
                return wl_str, wl_sec
        nearest_str, nearest_sec = min(sorted_items, key=lambda p: abs(p[1] - target))
        delta = abs(nearest_sec - target)
        _record_large_snap_delta(stats, delta, label)
        _log(video_tag, f"  ⚠️ [时间戳校准] {label}={ts_str} → {nearest_str} (Δ={delta:.2f}s)")
        return nearest_str, nearest_sec

    def _clamp_left(cur_str: str, cur_sec: float) -> Tuple[str, float]:
        if cur_sec >= chunk_start - 0.5:
            return cur_str, cur_sec
        for s, sv in sorted_items:
            if sv >= chunk_start - 0.1:
                _log(video_tag, f"  ⚠️ [时间戳校准] start={cur_str} 越出 chunk 左界 {chunk_start:.2f}s，clamp 至 {s}")
                return s, sv
        return cur_str, cur_sec

    def _clamp_right(cur_str: str, cur_sec: float) -> Tuple[str, float]:
        if cur_sec <= chunk_end + 0.5:
            return cur_str, cur_sec
        for s, sv in reversed(sorted_items):
            if sv <= chunk_end + 0.1:
                _log(video_tag, f"  ⚠️ [时间戳校准] end={cur_str} 越出 chunk 右界 {chunk_end:.2f}s，clamp 至 {s}")
                return s, sv
        return cur_str, cur_sec

    drop_indices = []
    for idx, ev in enumerate(events):
        start_str, start_sec = _snap(ev.get("start_time", ""), f"event[{idx}].start")
        end_str, end_sec = _snap(ev.get("end_time", ""), f"event[{idx}].end")

        if start_sec > end_sec:
            _log(video_tag, f"  ⚠️ [时间戳校准] event[{idx}] start>end，自动交换 ({start_str} ↔ {end_str})")
            start_str, end_str = end_str, start_str
            start_sec, end_sec = end_sec, start_sec

        if end_sec - start_sec < 0.1:
            _log(video_tag, f"  ⚠️ [时间戳校准] event[{idx}] Δ={end_sec - start_sec:.3f}s < 0.1s，丢弃")
            drop_indices.append(idx)
            continue

        start_str, start_sec = _clamp_left(start_str, start_sec)
        end_str, end_sec = _clamp_right(end_str, end_sec)

        ev["start_time"] = start_str
        ev["end_time"] = end_str

        for kft in ev.get("key_frame_times", []) or []:
            if not isinstance(kft, str):
                continue
            kft_sec = parse_timestamp_to_seconds(kft)
            if not (start_sec - 0.1 <= kft_sec <= end_sec + 0.1):
                _log(
                    video_tag,
                    f"  ⚠️ [时间戳校准] event[{idx}].key_frame_times={kft} "
                    f"越出 [{start_str}, {end_str}] 区间",
                )

    for di in reversed(drop_indices):
        events.pop(di)


def _validate_revision_end_time(
    revision,
    whitelist_str_list: list,
    video_tag: str,
    stats: Optional[dict] = None,
) -> None:
    """校验 prev_event_revision.end_time 是否在白名单（按秒数比对，容差 0.01s）。"""
    if not isinstance(revision, dict) or not revision.get("need_merge"):
        return
    if not whitelist_str_list:
        return
    end_str = revision.get("end_time", "")
    if not end_str:
        return

    whitelist_map = {s: parse_timestamp_to_seconds(s) for s in whitelist_str_list}
    target = parse_timestamp_to_seconds(end_str)
    for wl_str, wl_sec in whitelist_map.items():
        if abs(wl_sec - target) < 0.01:
            revision["end_time"] = wl_str
            return

    nearest_str, nearest_sec = min(whitelist_map.items(), key=lambda p: abs(p[1] - target))
    delta = abs(nearest_sec - target)
    _record_large_snap_delta(stats, delta, "prev_event_revision.end_time")
    _log(video_tag, f"  ⚠️ [时间戳校准] revision.end={end_str} → {nearest_str} (Δ={delta:.2f}s)")
    revision["end_time"] = nearest_str


def _flatten_events(global_results: list) -> list:
    flat = []
    for chunk_idx, chunk in enumerate(global_results):
        chunk_range = chunk.get("chunk_time_range", "")
        for event_idx, ev in enumerate(chunk.get("data", {}).get("events", []) or []):
            start_sec = parse_timestamp_to_seconds(ev.get("start_time", ""))
            end_sec = parse_timestamp_to_seconds(ev.get("end_time", ""))
            flat.append({
                "chunk_index": chunk_idx,
                "event_index": event_idx,
                "chunk_time_range": chunk_range,
                "start_time": ev.get("start_time", ""),
                "end_time": ev.get("end_time", ""),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "key_frame_times": ev.get("key_frame_times", []) or [],
            })
    flat.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    return flat


def _compute_pass1_confidence(global_results: list, total_duration: float, stats: Optional[dict] = None) -> dict:
    flat_events = _flatten_events(global_results)
    gap_segments = []
    cursor = 0.0
    eps = 0.01

    for item in flat_events:
        start_sec = item["start_sec"]
        end_sec = item["end_sec"]
        if end_sec <= cursor + eps:
            cursor = max(cursor, end_sec)
            continue
        if start_sec > cursor + eps:
            gap_segments.append({
                "start_time": format_timestamp(cursor),
                "end_time": format_timestamp(start_sec),
                "gap_sec": round(start_sec - cursor, 3),
            })
        cursor = max(cursor, end_sec)

    if total_duration > cursor + eps:
        gap_segments.append({
            "start_time": format_timestamp(cursor),
            "end_time": format_timestamp(total_duration),
            "gap_sec": round(total_duration - cursor, 3),
        })

    invalid_key_frames = []
    total_key_frames = 0
    invalid_key_frames_count = 0
    for item in flat_events:
        start_sec = item["start_sec"]
        end_sec = item["end_sec"]
        for kft in item["key_frame_times"]:
            if not isinstance(kft, str):
                continue
            total_key_frames += 1
            kft_sec = parse_timestamp_to_seconds(kft)
            if start_sec - 0.1 <= kft_sec <= end_sec + 0.1:
                continue
            invalid_key_frames_count += 1
            invalid_key_frames.append({
                "chunk_index": item["chunk_index"],
                "event_index": item["event_index"],
                "event_range": f"{item['start_time']} - {item['end_time']}",
                "key_frame_time": kft,
            })

    large_delta_sum = round(float((stats or {}).get("large_snap_delta_sum", 0.0) or 0.0), 3)
    return {
        "video_duration_sec": round(float(total_duration), 3),
        "event_count": len(flat_events),
        "event_time_coverage": {
            "is_fully_covered": len(gap_segments) == 0,
            "gap_count": len(gap_segments),
            "total_gap_sec": round(sum(seg["gap_sec"] for seg in gap_segments), 3),
            "gaps": gap_segments,
        },
        "key_frame_time_validity": {
            "total_key_frame_times": total_key_frames,
            "invalid_count": invalid_key_frames_count,
            "all_within_event_ranges": invalid_key_frames_count == 0,
            "invalid_items": invalid_key_frames,
        },
        "timestamp_calibration": {
            "large_delta_threshold_sec": 1.0,
            "large_delta_sum_sec": large_delta_sum,
            "large_delta_count": len((stats or {}).get("large_snap_deltas", [])),
            "large_deltas": list((stats or {}).get("large_snap_deltas", [])),
        },
    }


def _write_pass1_confidence(
    run_dir: str,
    global_results: list,
    total_duration: float,
    stats: Optional[dict],
) -> str:
    confidence_path = os.path.join(run_dir, "pass1_confidence.json")
    payload = _compute_pass1_confidence(global_results, total_duration, stats)
    with open(confidence_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return confidence_path


def _load_existing_confidence_stats(run_dir: str) -> dict:
    confidence_path = os.path.join(run_dir, "pass1_confidence.json")
    default = {"large_snap_delta_sum": 0.0, "large_snap_deltas": []}
    if not os.path.exists(confidence_path):
        return default
    try:
        with open(confidence_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        calibration = payload.get("timestamp_calibration", {}) or {}
        details = calibration.get("large_deltas", []) or []
        return {
            "large_snap_delta_sum": float(calibration.get("large_delta_sum_sec", 0.0) or 0.0),
            "large_snap_deltas": list(details),
        }
    except Exception:
        return default


def _apply_prev_event_revision(chunk_data: dict, global_results: list, video_tag: str) -> None:
    """若模型在本段输出中携带 prev_event_revision=need_merge，就地修订上一段 chunk 的 events[-1]。

    - 始终从 chunk_data 中 pop 该字段（避免污染当前 chunk 产物）。
    - 仅当 need_merge=True 且 revision.start_time 与上段末事件匹配（按秒数比对，容差 0.01s）时才覆盖。
    - 与当前 chunk events[0] 冲突时夹紧 revision.end_time（避免跨段 overlap）。
    - 非法或匹配失败时记日志跳过，不影响主流程。
    """
    revision = chunk_data.pop("prev_event_revision", None)
    if revision is None:
        return
    if not isinstance(revision, dict):
        _log(video_tag, f"⚠️ [修订跳过] prev_event_revision 非字典类型: {type(revision).__name__}")
        return
    if not revision.get("need_merge"):
        return
    if not global_results:
        _log(video_tag, "⚠️ [修订跳过] 无上段 chunk 可供修订，忽略 prev_event_revision")
        return

    prev_events = global_results[-1].get("data", {}).get("events", [])
    if not prev_events:
        _log(video_tag, "⚠️ [修订跳过] 上段 chunk 无 events，忽略 prev_event_revision")
        return

    last_ev = prev_events[-1]
    last_start_raw = last_ev.get("start_time", "")
    rev_start_raw = revision.get("start_time", "")
    last_start_sec = parse_timestamp_to_seconds(last_start_raw)
    rev_start_sec = parse_timestamp_to_seconds(rev_start_raw)

    if abs(last_start_sec - rev_start_sec) > 0.01:
        _log(
            video_tag,
            f"⚠️ [修订跳过] revision.start_time={rev_start_raw!r} ({rev_start_sec:.2f}s) "
            f"与上段末事件 start_time={last_start_raw!r} ({last_start_sec:.2f}s) 不匹配",
        )
        return

    old_end_raw = last_ev.get("end_time", "")
    old_end_sec = parse_timestamp_to_seconds(old_end_raw)
    rev_end_raw = revision.get("end_time", old_end_raw)
    rev_end_sec = parse_timestamp_to_seconds(rev_end_raw)

    current_events = chunk_data.get("events", []) or []
    if current_events:
        first_start_raw = current_events[0].get("start_time", "")
        first_start_sec = parse_timestamp_to_seconds(first_start_raw)
        if rev_end_sec > first_start_sec + 0.01:
            if first_start_sec <= old_end_sec + 0.01:
                _log(
                    video_tag,
                    f"⚠️ [修订冲突拒绝] revision.end={rev_end_raw} ({rev_end_sec:.2f}s) 超出当前 events[0].start="
                    f"{first_start_raw} ({first_start_sec:.2f}s)，且夹紧后 ≤ 原 end_time {old_end_raw}，跳过 revision",
                )
                return
            _log(
                video_tag,
                f"⚠️ [修订冲突夹紧] revision.end={rev_end_raw} ({rev_end_sec:.2f}s) → "
                f"{first_start_raw} ({first_start_sec:.2f}s)（避免与当前 events[0] 重叠）",
            )
            rev_end_raw = first_start_raw
            rev_end_sec = first_start_sec

    new_end = rev_end_raw
    prev_events[-1] = {
        "start_time": last_start_raw,
        "end_time": new_end,
        "step1_objective_visual": revision.get("step1_objective_visual", last_ev.get("step1_objective_visual", "")),
        "step2_contextual_reasoning": revision.get("step2_contextual_reasoning", last_ev.get("step2_contextual_reasoning", "")),
        "step3_synthesized_dense_caption": revision.get("step3_synthesized_dense_caption", last_ev.get("step3_synthesized_dense_caption", "")),
        "key_frame_times": revision.get("key_frame_times", last_ev.get("key_frame_times", [])),
    }
    _log(
        video_tag,
        f"🔧 [跨chunk合并] 修订上段末尾 event: end_time {old_end_raw} → {new_end}",
    )


def _enforce_event_continuity(events: list, video_tag: str) -> None:
    """按 start_time 排序后扫一遍，修平相邻 event 的 overlap / gap。

    必须在 _validate_and_snap_event_times 之后调用（事件时间已 snap 到白名单）。
    维护 last_valid 指针，确保被丢弃事件后续的 event 跟"前一个保留事件"对齐，避免
    drop 引入新的 non-adjacent overlap。
    - nxt.start < last_valid.end → overlap，吸附 nxt.start = last_valid.end
    - nxt.start > last_valid.end + 0.01 → gap，吸附 nxt.start = last_valid.end
    - 调整后 nxt.start >= nxt.end → 丢弃该 nxt（last_valid 不变）。
    """
    if not events or len(events) < 2:
        return

    events.sort(key=lambda ev: parse_timestamp_to_seconds(ev.get("start_time", "")))

    drop_indices = set()
    last_valid_idx = None
    for i in range(len(events)):
        if last_valid_idx is None:
            last_valid_idx = i
            continue
        cur = events[last_valid_idx]
        nxt = events[i]
        cur_end_str = cur.get("end_time", "")
        cur_end_sec = parse_timestamp_to_seconds(cur_end_str)
        nxt_start_str = nxt.get("start_time", "")
        nxt_start_sec = parse_timestamp_to_seconds(nxt_start_str)
        nxt_end_sec = parse_timestamp_to_seconds(nxt.get("end_time", ""))

        if abs(nxt_start_sec - cur_end_sec) > 0.01:
            if nxt_start_sec < cur_end_sec - 0.01:
                _log(
                    video_tag,
                    f"  ⚠️ [事件连续性] events[{i}].start={nxt_start_str} 早于 events[{last_valid_idx}].end={cur_end_str}，"
                    f"overlap={cur_end_sec - nxt_start_sec:.2f}s，吸附",
                )
            else:
                gap = nxt_start_sec - cur_end_sec
                _log(
                    video_tag,
                    f"  ⚠️ [事件连续性] events[{last_valid_idx}].end={cur_end_str} 与 events[{i}].start={nxt_start_str} "
                    f"存在 {gap:.2f}s gap，吸附",
                )
            nxt["start_time"] = cur_end_str
            nxt_start_sec = cur_end_sec

        if nxt_start_sec >= nxt_end_sec - 0.01:
            _log(
                video_tag,
                f"  ⚠️ [事件连续性] events[{i}] 调整后 start>=end "
                f"({nxt_start_sec:.2f}s >= {nxt_end_sec:.2f}s)，丢弃",
            )
            drop_indices.add(i)
        else:
            last_valid_idx = i

    for di in sorted(drop_indices, reverse=True):
        events.pop(di)


def _enforce_cross_chunk_continuity(
    current_events: list,
    prev_events: list,
    video_tag: str,
) -> None:
    """跨 chunk 兜底：若当前 chunk events[0] 起点早于上段末 event 终点，吸附或丢弃。

    必须在 _apply_prev_event_revision 之后调用（prev_events[-1].end_time 可能已被 revision 覆盖）。
    """
    if not prev_events or not current_events:
        return
    prev_end_str = prev_events[-1].get("end_time", "")
    prev_end_sec = parse_timestamp_to_seconds(prev_end_str)
    if prev_end_sec <= 0:
        return

    drop_indices = []
    for idx, ev in enumerate(current_events):
        start_sec = parse_timestamp_to_seconds(ev.get("start_time", ""))
        end_sec = parse_timestamp_to_seconds(ev.get("end_time", ""))
        if end_sec <= prev_end_sec + 0.01:
            _log(
                video_tag,
                f"  ⚠️ [跨段连续性] events[{idx}]={ev.get('start_time')}-{ev.get('end_time')} "
                f"完全在上段末事件 [..,{prev_end_str}] 内，丢弃",
            )
            drop_indices.append(idx)
            continue
        if start_sec < prev_end_sec - 0.01:
            _log(
                video_tag,
                f"  ⚠️ [跨段连续性] events[{idx}].start={ev.get('start_time')} 早于上段末 "
                f"{prev_end_str}，吸附",
            )
            ev["start_time"] = prev_end_str
        else:
            break

    for di in sorted(set(drop_indices), reverse=True):
        current_events.pop(di)


def _build_scene_whitelist(
    precomputed_scenes: list,
    chunk_start: float,
    chunk_end: float,
) -> list:
    """从全片 scene 列表里提取落在 [chunk_start, chunk_end] 内的镜头切换时间点（秒）。

    pyscenedetect 返回的 scene = (s_start, s_end)，相邻 scene 的 s_end / s_start 重合，
    set 去重后即为该 chunk 内的镜头切换点集合。完全由 scenedetect 决定，**不强加
    chunk_start / chunk_end**。模型只能在这些镜头切换点中选取 event 起止时间。
    """
    boundaries: set = set()
    for s_start_raw, s_end_raw in precomputed_scenes:
        for ts in (s_start_raw, s_end_raw):
            if chunk_start - 0.01 <= ts <= chunk_end + 0.01:
                boundaries.add(round(float(ts), 3))
    return sorted(boundaries)


def _resume_from_progress(
    events: list,
    cfg: PipelineConfig,
) -> Tuple[float, str, str, bool, list]:
    """断点恢复时按 overlap 超参计算下一段 chunk_start。

    返回 (chunk_start_sec, last_end_str, last_action, overlap_active, overlap_events)。
    """
    last_ev = events[-1]
    last_end_str = last_ev.get("end_time", "")
    last_end_sec = parse_timestamp_to_seconds(last_end_str)
    last_action = last_ev.get("step3_synthesized_dense_caption", "")

    n = max(0, int(cfg.prev_event_overlap_count))
    if n == 0:
        return last_end_sec, last_end_str, last_action, False, []

    k = min(n, len(events))
    proposed = parse_timestamp_to_seconds(events[-k].get("start_time", ""))
    if 0 < proposed < last_end_sec:
        return proposed, last_end_str, last_action, True, list(events[-k:])
    return last_end_sec, last_end_str, last_action, False, []


def run_pass1(
    cfg: PipelineConfig,
    video_path: str,
    run_dir: str,
    client,
    token_tracker: Optional[TokenTracker] = None,
    video_tag: str = "",
) -> str:
    os.makedirs(run_dir, exist_ok=True)
    pass1_output_path = os.path.join(run_dir, "pass1_progress.json")
    temp_dir = os.path.join(run_dir, "_tmp")
    confidence_stats = _load_existing_confidence_stats(run_dir)

    _log_context.log_file = os.path.join(run_dir, "pass1.log")

    total_duration = float(int(get_video_duration(video_path)))

    precomputed_scenes = None
    if cfg.frame_extraction_strategy == "scenedetect":
        precomputed_scenes = detect_scenes(video_path, cfg.scene_detect_threshold)
        _log(
            video_tag,
            f"📐 [pyscenedetect] 整片场景检测完成，共 {len(precomputed_scenes)} 个场景"
            f"（阈值={cfg.scene_detect_threshold}），将作为各 chunk event 起止白名单。",
        )

    global_results = []
    chunk_start = 0.0
    history_summaries = []
    previous_context = "无前情提要，这是视频的开篇。"

    overlap_active = False
    last_end_str = ""

    if os.path.exists(pass1_output_path):
        try:
            with open(pass1_output_path, 'r', encoding='utf-8') as f:
                global_results = json.load(f)

            if global_results:
                last_chunk_range = global_results[-1].get("chunk_time_range", "")
                last_chunk_end_sec = None
                if isinstance(last_chunk_range, str) and " - " in last_chunk_range:
                    last_chunk_end_sec = parse_timestamp_to_seconds(
                        last_chunk_range.split(" - ", 1)[1]
                    )

                if last_chunk_end_sec is not None and last_chunk_end_sec >= total_duration - 0.01:
                    _log(video_tag, "\n=========================================")
                    _log(
                        video_tag,
                        f"✅ 检测到 Pass 1 历史产物已覆盖到视频末尾 "
                        f"({format_timestamp(total_duration)})，跳过 Pass 1。",
                    )
                    _write_pass1_confidence(run_dir, global_results, total_duration, confidence_stats)
                    return pass1_output_path

                _log(video_tag, "\n=========================================")
                _log(video_tag, "🔄 检测到历史运行记录，尝试恢复断点...")

                for idx, res in enumerate(global_results):
                    summ = res.get("data", {}).get("chunk_summary", "")
                    if summ:
                        history_summaries.append(f"第{idx+1}段: {summ}")

                last_chunk = global_results[-1].get("data", {})
                last_events = last_chunk.get("events", [])

                if last_events:
                    chunk_start, last_end_str, last_action, overlap_active, overlap_events = _resume_from_progress(last_events, cfg)
                    previous_context = _build_previous_context(
                        history_summaries, last_action, last_end_str, chunk_start, overlap_active, overlap_events
                    )
                    _log(
                        video_tag,
                        f"⏩ 成功加载进度，将从 {format_timestamp(chunk_start)} 继续执行 "
                        f"(overlap N={cfg.prev_event_overlap_count}, 实际生效={overlap_active})。",
                    )
                else:
                    _log(video_tag, "⚠️ 历史记录格式异常，将从头开始。")
                    global_results = []
                    chunk_start = 0.0
                    history_summaries = []
        except Exception as e:
            _log(video_tag, f"⚠️ 读取断点文件失败: {e}，将从头开始。")
            global_results = []
            chunk_start = 0.0
            history_summaries = []

    while chunk_start < total_duration:
        chunk_end = min(chunk_start + cfg.chunk_duration_sec, total_duration)
        chunk_name = f"[{format_timestamp(chunk_start)} - {format_timestamp(chunk_end)}]"
        log_tag = f"[{video_tag}] {chunk_name}" if video_tag else chunk_name
        _log(video_tag, f"\n=========================================")
        _log(video_tag, f"🚀 Pass 1 - 正在处理视频块: {chunk_name} | 模式: {cfg.input_payload_format}")

        user_content = []
        timestamps_str_list = []

        if cfg.input_payload_format == "video_base64":
            valid_timestamps, video_b64 = get_raw_chunk_video_base64(
                video_path, chunk_start, chunk_end, cfg.frame_max_width, cfg.target_fps, temp_dir=temp_dir
            )
            if not video_b64:
                chunk_start = chunk_end
                continue
            frame_timestamps_str = [format_timestamp_sec(t) for t in valid_timestamps]
            user_content.append({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}})
        else:
            target_timestamps = get_target_timestamps(
                video_path, chunk_start, chunk_end,
                cfg.frame_extraction_strategy, cfg.scene_detect_threshold, cfg.max_frames_per_chunk,
                log_prefix=f"[{video_tag}] " if video_tag else "",
                precomputed_scenes=precomputed_scenes,
            )
            if 'qwen' in cfg.model_name:
                valid_timestamps, base64_frames = get_base64_frames_qwen(
                    video_path, target_timestamps, cfg.max_total_pixels, cfg.frame_jpg_quality,
                )
            else:
                valid_timestamps, base64_frames = get_base64_frames(
                    video_path, target_timestamps, cfg.frame_max_width, cfg.frame_jpg_quality,
                )
            if not base64_frames:
                chunk_start = chunk_end
                continue
            frame_timestamps_str = [format_timestamp_sec(t) for t in valid_timestamps]
            if 'qwen' in cfg.model_name:
                for t_str, b64 in zip(valid_timestamps, base64_frames):
                    user_content.append({"type": "text", "text": f"<{t_str:.1f} seconds>"})
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            else:
                for t_str, b64 in zip(frame_timestamps_str, base64_frames):
                    user_content.append({"type": "text", "text": f"画面时间 {t_str}:"})
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})

        # event 起止白名单：scenedetect 模式仅使用本 chunk 内的镜头切换点；
        # 其它模式保持原有行为（抽帧时间戳 + chunk_start 兜底）。
        if precomputed_scenes is not None:
            boundaries_sec = _build_scene_whitelist(precomputed_scenes, chunk_start, chunk_end)
            if not boundaries_sec:
                _log(
                    video_tag,
                    f"⚠️ [pyscenedetect] 本段无镜头切换点，回落到 [chunk_start, chunk_end] 兜底白名单。",
                )
                boundaries_sec = [chunk_start, chunk_end]
            timestamps_str_list = sorted({format_timestamp_sec(t) for t in boundaries_sec})
            _log(
                video_tag,
                f"🎬 [白名单] 本段镜头切换点 {len(timestamps_str_list)} 个: {timestamps_str_list}",
            )
        else:
            timestamps_str_list = sorted(set(frame_timestamps_str))
            start_str = format_timestamp_sec(chunk_start)
            if start_str not in timestamps_str_list:
                timestamps_str_list.insert(0, start_str)

        if precomputed_scenes is not None:
            scene_boundaries_sec = _build_scene_whitelist(precomputed_scenes, chunk_start, chunk_end)
            boundaries_sec = set(scene_boundaries_sec)
            boundaries_sec.update([chunk_start, chunk_end])
            if overlap_active and last_end_str:
                boundaries_sec.add(parse_timestamp_to_seconds(last_end_str))
            timestamps_str_list = sorted({format_timestamp_sec(round(float(t), 3)) for t in boundaries_sec})

        timestamps_str = ", ".join(timestamps_str_list)

        sys_prompt = build_sys_prompt(chunk_name, timestamps_str)
        usr_prompt = build_usr_prompt(previous_context)

        user_content.append({"type": "text", "text": usr_prompt})

        _save_chunk_prompt(
            run_dir=run_dir,
            chunk_idx=len(global_results),
            chunk_name=chunk_name,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            usr_prompt=usr_prompt,
            previous_context=previous_context,
            timestamps_str_list=timestamps_str_list,
        )

        next_start = chunk_start + (cfg.chunk_duration_sec * 0.8)

        try:
            chunk_data = request_llm_with_retry(
                client=client, model=cfg.model_name,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}],
                max_tokens=cfg.llm_max_tokens, temperature=cfg.llm_temperature,
                max_retries=cfg.max_retries, chunk_name=log_tag,
                token_tracker=token_tracker, stage=PASS_NAME,
            )

            _validate_and_snap_event_times(
                chunk_data.get("events", []),
                timestamps_str_list,
                chunk_start, chunk_end,
                video_tag,
                confidence_stats,
            )
            _enforce_event_continuity(chunk_data.get("events", []), video_tag)

            _validate_revision_end_time(
                chunk_data.get("prev_event_revision"),
                timestamps_str_list,
                video_tag,
                confidence_stats,
            )

            prev_events_before = (
                global_results[-1].get("data", {}).get("events", [])
                if global_results else []
            )
            _apply_prev_event_revision(chunk_data, global_results, video_tag)

            _enforce_cross_chunk_continuity(
                chunk_data.get("events", []),
                prev_events_before,
                video_tag,
            )
            _enforce_event_continuity(chunk_data.get("events", []), video_tag)

            global_results.append({"chunk_time_range": chunk_name, "data": chunk_data})

            with open(pass1_output_path, 'w', encoding='utf-8') as f:
                json.dump(global_results, f, ensure_ascii=False, indent=2)
            _write_pass1_confidence(run_dir, global_results, total_duration, confidence_stats)

            events = chunk_data.get("events", [])

            if events:
                new_summary = chunk_data.get('chunk_summary', '').strip()
                if new_summary:
                    history_summaries.append(f"第{len(history_summaries)+1}段: {new_summary}")

                pick = _pick_next_start(events, cfg, chunk_start, chunk_end)
                if pick is not None:
                    next_start, last_end_str, last_action, overlap_active, k_used = pick
                    overlap_events = list(events[-k_used:]) if (overlap_active and k_used > 0) else []
                    if overlap_active:
                        _log(
                            video_tag,
                            f"🔁 [视觉重叠] N={cfg.prev_event_overlap_count}，下段 chunk_start 后退至 "
                            f"{format_timestamp(next_start)}（含 {k_used} 个 event 回顾），打标起点 {last_end_str}。",
                        )
                    else:
                        _log(video_tag, f"🔗 [动态接力] 本段动作自然结束于 {format_timestamp(next_start)}，以此为下段起点。")
                    previous_context = _build_previous_context(
                        history_summaries, last_action, last_end_str, next_start, overlap_active, overlap_events
                    )
                else:
                    last_ev = events[-1]
                    last_action = last_ev.get("step3_synthesized_dense_caption", "")
                    last_end_str = last_ev.get("end_time", "")
                    _log(video_tag, f"⚠️ [接力异常] 末尾时间不合理，启动 80% 安全重叠兜底推进。")
                    accumulated_story = "\n".join(history_summaries) if history_summaries else "暂无"
                    previous_context = (
                        f"【全局剧情脉络】:\n{accumulated_story}\n\n"
                        f"【当前无缝接力要求】: 上一幕的最后一个事件是：'{last_action}'，结束于 {format_timestamp(last_end_str)}。"
                        f"请紧接着这个动作和时间点继续描述。"
                    )
            else:
                _log(video_tag, "⚠️ [接力异常] 未提取到事件，启动 80% 安全重叠兜底推进。")
                previous_context = f"【系统提示】: 上一片段解析异常，请直接从 {format_timestamp(next_start)} 开始重新捕捉动作。"

        except Exception as e:
            if cfg.strict_failure:
                _log(video_tag, f"💥 [严格失败] Chunk {chunk_name} 多次尝试均失败: {e} → 终止本视频 Pass 1，跳过下游阶段。")
                raise
            _log(video_tag, f"❌ [严重跳过] Chunk {chunk_name} 多次尝试均失败: {e}（strict_failure=False，继续下一段）")

        if chunk_end >= total_duration - 0.01:
            _log(video_tag, f"🏁 已处理到视频末尾 ({format_timestamp(total_duration)})，Pass 1 结束。")
            break

        chunk_start = next_start

    _write_pass1_confidence(run_dir, global_results, total_duration, confidence_stats)
    return pass1_output_path
