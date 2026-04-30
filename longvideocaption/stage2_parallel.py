"""Stage 2 Parallel — 事件级帧精修并行版。

输出结构与 stage2.py 保持兼容：每个 event 增加/更新
frame_caption、frame_timestamps、frame_caption_output_tokens。
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from .config import PipelineConfig
from .frame_extractor import get_event_frames_base64, get_event_frames_base64_qwen
from .llm_client import request_llm_text_with_retry
from .prompts.stage2_v3 import SYS_PROMPT_STAGE2, build_stage2_user_prompt
from .token_tracker import TokenTracker
from .utils import format_timestamp_for_mode, parse_timestamp_to_seconds


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
            new_ev.setdefault("frame_caption_output_tokens", 0)
            new_ch["events"].append(new_ev)
        cloned["chapters"].append(new_ch)
    return cloned


def _ensure_stage2_event_fields(state: dict) -> None:
    for ch in state.get("chapters", []):
        for ev in ch.get("events", []):
            ev.setdefault("frame_caption", "")
            ev.setdefault("frame_timestamps", [])
            ev.setdefault("frame_caption_output_tokens", 0)


def _build_user_content(
    cfg: PipelineConfig,
    base64_frames: List[str],
    timestamps: List[float],
    user_text: str,
) -> list:
    content: list = []
    for ts, b64 in zip(timestamps, base64_frames):
        if "qwen" in cfg.model_name.lower():
            content.append({"type": "text", "text": _format_qwen_frame_timestamp(ts)})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        else:
            label = _format_stage2_frame_label(ts, cfg)
            content.append({"type": "text", "text": f"画面时间 {label}:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            })
    content.append({"type": "text", "text": user_text})
    return content


def _round_qwen_timestamp_seconds(seconds: float) -> float:
    return round(float(seconds), 1)


def _format_qwen_frame_timestamp(seconds: float) -> str:
    return f"<{_round_qwen_timestamp_seconds(seconds):.1f} seconds>"


def _format_stage2_frame_label(seconds: float, cfg: PipelineConfig) -> str:
    if cfg.pass1_timestamp_mode == "qwen_millisecond":
        return _format_qwen_frame_timestamp(seconds)
    return format_timestamp_for_mode(seconds, cfg.pass1_timestamp_mode)


def _write_state(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class _VisualTokenBudget:
    def __init__(self, budget: int) -> None:
        self.budget = max(1, int(budget))
        self.in_flight = 0
        self._cond = threading.Condition()

    def acquire(self, visual_tokens: int) -> int:
        charge = min(max(1, int(visual_tokens or 1)), self.budget)
        with self._cond:
            while self.in_flight + charge > self.budget:
                self._cond.wait()
            self.in_flight += charge
        return charge

    def release(self, charge: int) -> None:
        with self._cond:
            self.in_flight = max(0, self.in_flight - int(charge))
            self._cond.notify_all()


def _run_one_event(
    cfg: PipelineConfig,
    video_path: str,
    client,
    token_tracker: Optional[TokenTracker],
    video_tag: str,
    item: dict,
    qwen_budget: Optional[_VisualTokenBudget],
) -> dict:
    ev = item["event"]
    event_id = item["event_id"]
    start_sec = item["start_sec"]
    end_sec = item["end_sec"]
    actual_start_sec = item["actual_start_sec"]

    is_qwen = "qwen" in cfg.model_name.lower()
    visual_tokens = 0
    if is_qwen:
        timestamps, base64_frames, visual_tokens = get_event_frames_base64_qwen(
            video_path=video_path,
            start_sec=actual_start_sec,
            end_sec=end_sec,
            fps=cfg.stage2_fps,
            max_frames=cfg.stage2_max_frames,
            max_total_pixels=cfg.stage2_max_total_pixels,
            jpg_quality=cfg.stage2_frame_jpg_quality,
            return_visual_tokens=True,
        )
    else:
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
        return {
            "status": "skipped",
            "event_id": event_id,
            "reason": f"抽帧失败 ({actual_start_sec:.2f}s → {end_sec:.2f}s)",
        }

    initial_caption = ev.get("step3_synthesized_dense_caption", "")
    characters_in_event = ev.get("characters_in_event", [])
    user_text = build_stage2_user_prompt(initial_caption, characters_in_event)
    user_content = _build_user_content(cfg, base64_frames, timestamps, user_text)
    log_tag = f"[{video_tag}] Stage2 {event_id} ({len(base64_frames)}帧)" if video_tag else f"Stage2 {event_id}"

    budget_charge = 0
    try:
        if qwen_budget is not None:
            budget_charge = qwen_budget.acquire(visual_tokens)
        refined_text, usage = request_llm_text_with_retry(
            client=client,
            model=cfg.model_name,
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
            return_usage=True,
        )
    finally:
        if qwen_budget is not None and budget_charge:
            qwen_budget.release(budget_charge)

    return {
        "status": "ok",
        "event_id": event_id,
        "chapter_index": item["chapter_index"],
        "event_index": item["event_index"],
        "frame_caption": refined_text,
        "frame_timestamps": [round(t, 3) for t in timestamps],
        "frame_caption_output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "frame_visual_tokens": int(visual_tokens or 0),
        "start_sec": start_sec,
        "end_sec": end_sec,
    }


def run_stage2_parallel(
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

    with open(stage1_final_path, "r", encoding="utf-8") as f:
        stage1_data = json.load(f)

    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            _log(video_tag, "\n🔄 [Stage 2 Parallel] 检测到 stage2_refined.json，按 event 续跑。")
        except Exception as e:
            _log(video_tag, f"\n⚠️ [Stage 2 Parallel] stage2_refined.json 损坏，重建：{e}")
            state = _clone_stage1_to_stage2(stage1_data)
    else:
        state = _clone_stage1_to_stage2(stage1_data)

    _ensure_stage2_event_fields(state)
    chapters = state.get("chapters", [])

    flat_events: list = []
    for ch_idx, ch in enumerate(chapters):
        for ev_idx, ev in enumerate(ch.get("events", [])):
            flat_events.append((ch_idx, ev_idx, ev))

    pending: list = []
    for global_idx, (ch_idx, ev_idx, ev) in enumerate(flat_events):
        if ev.get("frame_caption", ""):
            continue

        event_id = ev.get("event_id", f"ch{ch_idx+1}_ev{ev_idx+1}")
        start_sec = parse_timestamp_to_seconds(ev.get("start_time", ""))
        end_sec = parse_timestamp_to_seconds(ev.get("end_time", ""))
        if end_sec <= start_sec:
            _log(video_tag, f"  ⚠️ [{event_id}] 时间范围非法 ({ev.get('start_time')} → {ev.get('end_time')})，跳过。")
            continue

        actual_start_sec = start_sec
        if global_idx > 0:
            _, _, prev_ev = flat_events[global_idx - 1]
            prev_end_sec = parse_timestamp_to_seconds(prev_ev.get("end_time", ""))
            if abs(start_sec - prev_end_sec) < 0.01 and start_sec > 0:
                actual_start_sec = min(
                    start_sec + cfg.stage2_adjust_start_offset,
                    max(start_sec, end_sec - 0.01),
                )

        pending.append({
            "chapter_index": ch_idx,
            "event_index": ev_idx,
            "event": ev,
            "event_id": event_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "actual_start_sec": actual_start_sec,
        })

    total_events = len(flat_events)
    done_count = sum(
        1 for _ch_idx, _ev_idx, ev in flat_events if ev.get("frame_caption", "")
    )
    _log(video_tag, "\n" + "=" * 50)
    _log(video_tag, f"🎞️  Stage 2 Parallel 启动：共 {total_events} 个 event，已完成 {done_count}，待处理 {len(pending)}")
    if "qwen" in cfg.model_name.lower():
        _log(video_tag, f"🧮 qwen 视觉 token 并发预算: {cfg.stage2_qwen_parallel_visual_token_budget}")
        _log(video_tag, f"🧵 qwen Stage 2 线程池上限: {cfg.stage2_qwen_parallel_max_workers}")
    else:
        _log(video_tag, f"🧵 Stage 2 最大并行请求数: {cfg.stage2_parallel_max_workers}")
    _log(video_tag, "=" * 50)

    if not pending:
        _write_state(out_path, state)
        _log(video_tag, "⏭️  Stage 2 Parallel 全部 event 已完成，跳过。")
        return out_path

    is_qwen = "qwen" in cfg.model_name.lower()
    max_workers_cfg = (
        cfg.stage2_qwen_parallel_max_workers
        if is_qwen
        else cfg.stage2_parallel_max_workers
    )
    max_workers = max(1, int(max_workers_cfg))
    max_workers = min(max_workers, len(pending))
    qwen_budget = (
        _VisualTokenBudget(cfg.stage2_qwen_parallel_visual_token_budget)
        if is_qwen
        else None
    )

    completed_now = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(
                _run_one_event,
                cfg,
                video_path,
                client,
                token_tracker,
                video_tag,
                item,
                qwen_budget,
            ): item
            for item in pending
        }

        for fut in as_completed(future_to_item):
            item = future_to_item[fut]
            event_id = item["event_id"]
            try:
                result = fut.result()
            except Exception as e:
                if cfg.strict_failure:
                    _log(video_tag, f"  💥 [严格失败] [{event_id}] Stage 2 Parallel 调用失败：{e} → 终止本视频，跳过 Stage 3。")
                    raise
                _log(video_tag, f"  ❌ [{event_id}] Stage 2 Parallel 调用失败：{e}（strict_failure=False，跳过该 event）")
                continue

            if result.get("status") != "ok":
                _log(video_tag, f"  ⚠️ [{event_id}] {result.get('reason', '跳过')}")
                continue

            ev = chapters[result["chapter_index"]]["events"][result["event_index"]]
            ev["frame_caption"] = result["frame_caption"]
            ev["frame_timestamps"] = result["frame_timestamps"]
            ev["frame_caption_output_tokens"] = result["frame_caption_output_tokens"]
            if result.get("frame_visual_tokens", 0):
                ev["frame_visual_tokens"] = result["frame_visual_tokens"]

            completed_now += 1
            _write_state(out_path, state)
            _log(
                video_tag,
                f"  ✅ [{event_id}] 完成 "
                f"output_tokens={ev['frame_caption_output_tokens']} "
                f"visual_tokens={ev.get('frame_visual_tokens', 0)}",
            )

    final_done = sum(
        1 for ch in chapters for ev in ch.get("events", []) if ev.get("frame_caption", "")
    )
    _log(video_tag, f"\n🎉 Stage 2 Parallel 完成：本次成功 {completed_now} 个，累计 {final_done}/{total_events} → {out_path}")
    return out_path
