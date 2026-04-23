import json
import os
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
from .utils import format_timestamp, format_timestamp_sec, parse_timestamp_to_seconds


STAGE_NAME = "stage1_perception"


def _log(video_tag: str, msg: str) -> None:
    print(f"[{video_tag}] {msg}" if video_tag else msg)


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

        next_start = chunk_start + (cfg.chunk_duration_sec * 0.8)

        try:
            chunk_data = request_llm_with_retry(
                client=client, model=cfg.model_name,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}],
                max_tokens=cfg.llm_max_tokens, temperature=cfg.llm_temperature,
                max_retries=cfg.max_retries, chunk_name=log_tag,
                token_tracker=token_tracker, stage=STAGE_NAME,
            )
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

        chunk_start = next_start

    return pass1_output_path
