import json
import os
import threading
from typing import Optional, Tuple

from .config import PipelineConfig
from .frame_extractor import (
    get_base64_frames,
    get_raw_chunk_video_base64,
    get_target_timestamps,
    get_video_duration,
)
from .llm_client import request_llm_with_retry
from .prompts.stage1_v3 import build_sys_prompt, build_usr_prompt
from .token_tracker import TokenTracker
from .utils import format_timestamp, format_timestamp_sec, parse_timestamp_to_seconds, sanitize_filename


STAGE_NAME = "stage1_perception"

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
    prompts_dir = os.path.join(run_dir, "_stage1_prompts")
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
            f"请从 {last_end_str} 开始切分新事件，紧接着这个动作和时间点继续描述。"
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


def _validate_and_snap_event_times(
    events: list,
    whitelist_str_list: list,
    chunk_start: float,
    chunk_end: float,
    video_tag: str,
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


def _validate_revision_end_time(revision, whitelist_str_list: list, video_tag: str) -> None:
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
    _log(video_tag, f"  ⚠️ [时间戳校准] revision.end={end_str} → {nearest_str} (Δ={delta:.2f}s)")
    revision["end_time"] = nearest_str


def _apply_prev_event_revision(chunk_data: dict, global_results: list, video_tag: str) -> None:
    """若模型在本段输出中携带 prev_event_revision=need_merge，就地修订上一段 chunk 的 events[-1]。

    - 始终从 chunk_data 中 pop 该字段（避免污染当前 chunk 产物）。
    - 仅当 need_merge=True 且 revision.start_time 与上段末事件匹配（按秒数比对，容差 0.01s）时才覆盖。
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

    old_end = last_ev.get("end_time", "")
    new_end = revision.get("end_time", old_end)
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
        f"🔧 [跨chunk合并] 修订上段末尾 event: end_time {old_end} → {new_end}",
    )


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


def run_stage1(
    cfg: PipelineConfig,
    video_path: str,
    run_dir: str,
    client,
    token_tracker: Optional[TokenTracker] = None,
    video_tag: str = "",
) -> str:
    os.makedirs(run_dir, exist_ok=True)
    pass1_output_path = os.path.join(run_dir, "stage1_progress.json")
    temp_dir = os.path.join(run_dir, "_tmp")

    _log_context.log_file = os.path.join(run_dir, "stage1.log")

    total_duration = float(int(get_video_duration(video_path)))

    global_results = []
    chunk_start = 0.0
    history_summaries = []
    previous_context = "无前情提要，这是视频的开篇。"

    if os.path.exists(pass1_output_path):
        try:
            with open(pass1_output_path, 'r', encoding='utf-8') as f:
                global_results = json.load(f)

            if global_results:
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
            timestamps_str_list = sorted({format_timestamp_sec(t) for t in valid_timestamps})
            user_content.append({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}})
        else:
            target_timestamps = get_target_timestamps(
                video_path, chunk_start, chunk_end,
                cfg.frame_extraction_strategy, cfg.scene_detect_threshold, cfg.max_frames_per_chunk,
                log_prefix=f"[{video_tag}] " if video_tag else "",
            )
            valid_timestamps, base64_frames = get_base64_frames(
                video_path, target_timestamps, cfg.frame_max_width, cfg.frame_jpg_quality,
            )
            if not base64_frames:
                chunk_start = chunk_end
                continue
            frame_timestamps_str = [format_timestamp_sec(t) for t in valid_timestamps]
            timestamps_str_list = sorted(set(frame_timestamps_str))
            for t_str, b64 in zip(frame_timestamps_str, base64_frames):
                user_content.append({"type": "text", "text": f"画面时间 {t_str}:"})
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})

        start_str = format_timestamp_sec(chunk_start)
        if start_str not in timestamps_str_list:
            timestamps_str_list.insert(0, start_str)

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
                token_tracker=token_tracker, stage=STAGE_NAME,
            )

            _validate_and_snap_event_times(
                chunk_data.get("events", []),
                timestamps_str_list,
                chunk_start, chunk_end,
                video_tag,
            )
            _validate_revision_end_time(
                chunk_data.get("prev_event_revision"),
                timestamps_str_list,
                video_tag,
            )

            _apply_prev_event_revision(chunk_data, global_results, video_tag)

            global_results.append({"chunk_time_range": chunk_name, "data": chunk_data})

            with open(pass1_output_path, 'w', encoding='utf-8') as f:
                json.dump(global_results, f, ensure_ascii=False, indent=2)

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
                    _log(video_tag, f"⚠️ [接力异常] 末尾时间不合理，启动 80% 安全重叠兜底推进。")
                    accumulated_story = "\n".join(history_summaries) if history_summaries else "暂无"
                    previous_context = (
                        f"【全局剧情脉络】:\n{accumulated_story}\n\n"
                        f"【当前无缝接力要求】: 上一幕的最后一个事件是：'{last_action}'，结束于 {format_timestamp(next_start)}。"
                        f"请紧接着这个动作和时间点继续描述。"
                    )
            else:
                _log(video_tag, "⚠️ [接力异常] 未提取到事件，启动 80% 安全重叠兜底推进。")
                previous_context = f"【系统提示】: 上一片段解析异常，请直接从 {format_timestamp(next_start)} 开始重新捕捉动作。"

        except Exception as e:
            _log(video_tag, f"❌ [严重跳过] Chunk {chunk_name} 多次尝试均失败: {e}")

        if chunk_end >= total_duration - 0.01:
            _log(video_tag, f"🏁 已处理到视频末尾 ({format_timestamp(total_duration)})，Stage 1 结束。")
            break

        chunk_start = next_start

    return pass1_output_path
