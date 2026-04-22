import json
import os
from typing import Optional

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

    total_duration = get_video_duration(video_path)

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

                if "events" in last_chunk and len(last_chunk["events"]) > 0:
                    last_ev = last_chunk["events"][-1]
                    last_end_str = last_ev.get("end_time", "")
                    chunk_start = parse_timestamp_to_seconds(last_end_str)

                    last_action = last_ev.get("dense_caption", "")
                    accumulated_story = "\n".join(history_summaries) if history_summaries else "暂无"
                    previous_context = (
                        f"【全局剧情脉络】:\n{accumulated_story}\n\n"
                        f"【当前无缝接力要求】: 上一幕的最后一个动作是：'{last_action}'，结束于 {last_end_str}。请紧接着这个动作和时间点继续描述。"
                    )
                    _log(video_tag, f"⏩ 成功加载进度，将从 {format_timestamp(chunk_start)} 继续执行。")
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

            if "events" in chunk_data and len(chunk_data["events"]) > 0:
                last_ev = chunk_data["events"][-1]
                last_end_str = last_ev.get("end_time", "")
                last_end_sec = parse_timestamp_to_seconds(last_end_str)

                if chunk_start + 5 <= last_end_sec <= chunk_end + 10:
                    next_start = last_end_sec
                    _log(video_tag, f"🔗 [动态接力] 本段动作自然结束于 {format_timestamp(next_start)}，以此为下段起点。")
                else:
                    _log(video_tag, f"⚠️ [接力异常] 返回末尾时间 {format_timestamp(last_end_sec)} 不合理，启动 80% 安全重叠兜底推进。")

                new_summary = chunk_data.get('chunk_summary', '').strip()
                if new_summary:
                    history_summaries.append(f"第{len(history_summaries)+1}段: {new_summary}")

                accumulated_story = "\n".join(history_summaries)
                last_action = last_ev.get("dense_caption", "")

                previous_context = (
                    f"【全局剧情脉络】:\n{accumulated_story}\n\n"
                    f"【当前无缝接力要求】: 上一幕的最后一个事件是：'{last_action}'，结束于 {format_timestamp(next_start)}。请紧接着这个动作和时间点继续描述。"
                )
            else:
                _log(video_tag, "⚠️ [接力异常] 未提取到事件，启动 80% 安全重叠兜底推进。")
                previous_context = f"【系统提示】: 上一片段解析异常，请直接从 {format_timestamp(next_start)} 开始重新捕捉动作。"

        except Exception as e:
            _log(video_tag, f"❌ [严重跳过] Chunk {chunk_name} 多次尝试均失败: {e}")

        chunk_start = next_start

    return pass1_output_path
