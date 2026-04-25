"""Stage 2 — 事件级帧精修（Frame Inspection）。

输入：stage1 终产物（pass3_final.json，含 chapters/events/step3_synthesized_dense_caption）。
处理：按 event 串行抽帧 + VL 调用，把上一 event 的 frame_caption 作为前序上下文（chapter 首事件不传）。
输出：stage2_refined.json，结构完全克隆 stage1 终产物，每个 event 增加 frame_caption / frame_timestamps 字段。
断点续跑：每完成一个 event 即整体落盘；启动时若文件已存在，跳过已有 frame_caption 的 event。
"""

import json
import os
from typing import List

from .config import PipelineConfig
from .frame_extractor import get_event_frames_base64
from .llm_client import request_llm_text_with_retry
from .prompts.stage2_v1 import SYS_PROMPT_STAGE2, build_stage2_user_prompt
from .token_tracker import TokenTracker
from .utils import parse_timestamp_to_seconds


STAGE_NAME = "stage2_frame_inspection"


def _log(video_tag: str, msg: str) -> None:
    print(f"[{video_tag}] {msg}" if video_tag else msg)


def _clone_stage1_to_stage2(stage1_data: dict) -> dict:
    cloned = {
        "video_path": stage1_data.get("video_path", ""),
        "video_summary": stage1_data.get("video_summary", ""),
        "chapters": [],
    }
    for ch in stage1_data.get("chapters", []):
        new_ch = {
            "chapter_id": ch.get("chapter_id", ""),
            "title": ch.get("title", ""),
            "chapter_summary": ch.get("chapter_summary", ""),
            "start_time": ch.get("start_time", ""),
            "end_time": ch.get("end_time", ""),
            "events": [],
        }
        for ev in ch.get("events", []):
            new_ev = dict(ev)
            new_ev.setdefault("frame_caption", "")
            new_ev.setdefault("frame_timestamps", [])
            new_ch["events"].append(new_ev)
        cloned["chapters"].append(new_ch)
    return cloned


def _build_user_content(
    base64_frames: List[str],
    timestamps: List[float],
    user_text: str,
) -> list:
    content: list = []
    for ts, b64 in zip(timestamps, base64_frames):
        content.append({"type": "text", "text": f"<{ts:.1f} seconds>"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": user_text})
    return content


def run_stage2(
    cfg: PipelineConfig,
    video_path: str,
    stage1_final_path: str,
    run_dir: str,
    client,
    token_tracker: TokenTracker = None,
    video_tag: str = "",
) -> str:
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "stage2_refined.json")

    with open(stage1_final_path, 'r', encoding='utf-8') as f:
        stage1_data = json.load(f)

    if os.path.exists(out_path):
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            _log(video_tag, f"\n🔄 [Stage 2] 检测到 stage2_refined.json，按 event 续跑。")
        except Exception as e:
            _log(video_tag, f"\n⚠️ [Stage 2] stage2_refined.json 损坏，重建：{e}")
            state = _clone_stage1_to_stage2(stage1_data)
    else:
        state = _clone_stage1_to_stage2(stage1_data)

    chapters = state.get("chapters", [])
    total_events = sum(len(ch.get("events", [])) for ch in chapters)
    done_count = sum(
        1 for ch in chapters for ev in ch.get("events", []) if ev.get("frame_caption", "")
    )

    _log(video_tag, "\n" + "=" * 50)
    _log(video_tag, f"🎞️  Stage 2 启动：共 {total_events} 个 event，已完成 {done_count}，待处理 {total_events - done_count}")
    _log(video_tag, "=" * 50)

    if done_count >= total_events and total_events > 0:
        _log(video_tag, "⏭️  Stage 2 全部 event 已完成，跳过。")
        return out_path

    flat_events: list = []
    for ch_idx, ch in enumerate(chapters):
        for ev_idx, ev in enumerate(ch.get("events", [])):
            flat_events.append((ch_idx, ev_idx, ev))

    previous_caption = ""
    last_chapter_idx = -1

    for global_idx, (ch_idx, ev_idx, ev) in enumerate(flat_events):
        if ch_idx != last_chapter_idx:
            previous_caption = ""
            last_chapter_idx = ch_idx

        event_id = ev.get("event_id", f"ch{ch_idx+1}_ev{ev_idx+1}")

        if ev.get("frame_caption", ""):
            previous_caption = ev["frame_caption"]
            continue

        start_sec = parse_timestamp_to_seconds(ev.get("start_time", ""))
        end_sec = parse_timestamp_to_seconds(ev.get("end_time", ""))
        if end_sec <= start_sec:
            _log(video_tag, f"  ⚠️ [{event_id}] 时间范围非法 ({ev.get('start_time')} → {ev.get('end_time')})，跳过。")
            continue

        adjust_start = False
        if global_idx > 0:
            _, _, prev_ev = flat_events[global_idx - 1]
            prev_end_sec = parse_timestamp_to_seconds(prev_ev.get("end_time", ""))
            if abs(start_sec - prev_end_sec) < 0.01:
                adjust_start = True

        actual_start_sec = start_sec
        if adjust_start and start_sec > 0:
            actual_start_sec = min(start_sec + cfg.stage2_adjust_start_offset, max(start_sec, end_sec - 0.01))

        timestamps, base64_frames = get_event_frames_base64(
            video_path=video_path,
            start_sec=actual_start_sec,
            end_sec=end_sec,
            fps=cfg.stage2_fps,
            max_frames=cfg.stage2_max_frames,
            max_width=cfg.stage2_frame_max_width,
            jpg_quality=cfg.stage2_frame_jpg_quality,
        )

        if not base64_frames:
            _log(video_tag, f"  ⚠️ [{event_id}] 抽帧失败 ({actual_start_sec:.2f}s → {end_sec:.2f}s)，跳过。")
            continue

        prev_block = previous_caption if previous_caption else "无"
        initial_caption = ev.get("step3_synthesized_dense_caption", "")
        user_text = build_stage2_user_prompt(prev_block, initial_caption)
        user_content = _build_user_content(base64_frames, timestamps, user_text)

        log_tag = f"[{video_tag}] Stage2 {event_id} ({len(base64_frames)}帧)" if video_tag else f"Stage2 {event_id}"
        try:
            refined_text = request_llm_text_with_retry(
                client=client, model=cfg.model_name,
                messages=[
                    {"role": "system", "content": SYS_PROMPT_STAGE2},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=cfg.stage2_max_tokens,
                temperature=cfg.stage2_temperature,
                max_retries=cfg.max_retries,
                chunk_name=log_tag,
                token_tracker=token_tracker,
                stage=STAGE_NAME,
            )
        except Exception as e:
            _log(video_tag, f"  ❌ [{event_id}] Stage 2 调用失败：{e}")
            continue

        ev["frame_caption"] = refined_text
        ev["frame_timestamps"] = [round(t, 3) for t in timestamps]
        previous_caption = refined_text

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    final_done = sum(
        1 for ch in chapters for ev in ch.get("events", []) if ev.get("frame_caption", "")
    )
    _log(video_tag, f"\n🎉 Stage 2 完成：成功 {final_done}/{total_events} 个 event → {out_path}")
    return out_path
