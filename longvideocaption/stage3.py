"""Stage 3 — 全局精修与指代统一（Global Polish）。

输入：stage2_refined.json（含每个 event 的 frame_caption）。
处理：将所有 chapters/events 拼成 {chapter_id, chapter_title, events:[{event_id, caption}]} 结构，
     一次性喂给纯文本 LLM，回填到每个 event 的 final_caption 字段。
输出：stage3_polished.json（克隆 stage2 结构，新增 final_caption；未被模型回填的 event 退回 frame_caption）。
"""

import json
import os

from .config import PipelineConfig
from .llm_client import request_llm_with_retry
from .prompts.stage3_v1 import SYS_PROMPT_STAGE3, build_stage3_user_prompt
from .token_tracker import TokenTracker


STAGE_NAME = "stage3_global_polish"


def _log(video_tag: str, msg: str) -> None:
    print(f"[{video_tag}] {msg}" if video_tag else msg)


def _build_input_payload(stage2_data: dict) -> dict:
    chapters_payload = []
    for ch in stage2_data.get("chapters", []):
        events_payload = []
        for ev in ch.get("events", []):
            cap = ev.get("frame_caption", "") or ev.get("step3_synthesized_dense_caption", "")
            if not cap:
                continue
            events_payload.append({
                "event_id": ev.get("event_id", ""),
                "caption": cap,
            })
        if not events_payload:
            continue
        chapters_payload.append({
            "chapter_id": ch.get("chapter_id", ""),
            "chapter_title": ch.get("title", ""),
            "events": events_payload,
        })
    return {"chapters": chapters_payload}


def _clone_with_final_caption(stage2_data: dict, polished_map: dict) -> dict:
    cloned = {
        "video_path": stage2_data.get("video_path", ""),
        "video_summary": stage2_data.get("video_summary", ""),
        "chapters": [],
    }
    for ch in stage2_data.get("chapters", []):
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
            ev_id = ev.get("event_id", "")
            polished = polished_map.get(ev_id, "")
            new_ev["final_caption"] = polished or ev.get("frame_caption", "") or ev.get("step3_synthesized_dense_caption", "")
            new_ch["events"].append(new_ev)
        cloned["chapters"].append(new_ch)
    return cloned


def run_stage3(
    cfg: PipelineConfig,
    stage2_path: str,
    run_dir: str,
    client,
    token_tracker: TokenTracker = None,
    video_tag: str = "",
) -> str:
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "stage3_polished.json")

    if os.path.exists(out_path):
        _log(video_tag, "\n" + "=" * 50)
        _log(video_tag, "⏭️  Stage 3 终产物已存在，跳过。")
        _log(video_tag, "=" * 50)
        return out_path

    with open(stage2_path, 'r', encoding='utf-8') as f:
        stage2_data = json.load(f)

    payload = _build_input_payload(stage2_data)
    if not payload["chapters"]:
        _log(video_tag, "⚠️ Stage 3：stage2 结果为空，无可润色 event，跳过 LLM 调用。")
        cloned = _clone_with_final_caption(stage2_data, {})
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cloned, f, ensure_ascii=False, indent=2)
        return out_path

    user_prompt = build_stage3_user_prompt(payload)

    _log(video_tag, "\n" + "=" * 50)
    _log(video_tag, f"✨ Stage 3 启动：全局润色 {sum(len(ch['events']) for ch in payload['chapters'])} 个 event")
    _log(video_tag, "=" * 50)

    log_tag = f"[{video_tag}] Stage3 全局润色" if video_tag else "Stage3 全局润色"
    polished_json = request_llm_with_retry(
        client=client, model=cfg.model_name,
        messages=[
            {"role": "system", "content": SYS_PROMPT_STAGE3},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=cfg.stage3_max_tokens,
        temperature=cfg.stage3_temperature,
        max_retries=cfg.max_retries,
        chunk_name=log_tag,
        token_tracker=token_tracker,
        stage=STAGE_NAME,
    )

    polished_map: dict = {}
    for ch in polished_json.get("chapters", []):
        for ev in ch.get("events", []):
            ev_id = ev.get("event_id", "")
            cap = ev.get("caption", "")
            if ev_id and cap:
                polished_map[ev_id] = cap

    cloned = _clone_with_final_caption(stage2_data, polished_map)

    total_events = sum(len(ch.get("events", [])) for ch in cloned["chapters"])
    polished_count = sum(
        1 for ch in cloned["chapters"] for ev in ch.get("events", []) if ev.get("event_id", "") in polished_map
    )

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(cloned, f, ensure_ascii=False, indent=2)

    _log(video_tag, f"\n🎉 Stage 3 完成：润色 {polished_count}/{total_events} 个 event → {out_path}")
    return out_path
