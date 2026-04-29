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


def _event_count(chapters: list) -> int:
    count = 0
    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        events = ch.get("events", [])
        if isinstance(events, list):
            count += len(events)
    return count


def _count_duplicates(values: list) -> list:
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _build_stage3_confidence(input_payload: dict, polished_json: dict, polished_map: dict) -> dict:
    input_chapters = input_payload.get("chapters", [])
    output_chapters = polished_json.get("chapters", []) if isinstance(polished_json, dict) else []
    if not isinstance(input_chapters, list):
        input_chapters = []
    if not isinstance(output_chapters, list):
        output_chapters = []

    input_chapter_count = len(input_chapters)
    output_chapter_count = len(output_chapters)
    input_event_count = _event_count(input_chapters)
    output_event_count = _event_count(output_chapters)

    chapter_mismatches = []
    for idx in range(min(input_chapter_count, output_chapter_count)):
        expected = input_chapters[idx]
        actual = output_chapters[idx]
        expected_title = expected.get("chapter_title", "")
        actual_title = actual.get("chapter_title", actual.get("title", ""))
        if expected.get("chapter_id", "") != actual.get("chapter_id", "") or expected_title != actual_title:
            chapter_mismatches.append({
                "index": idx,
                "expected_chapter_id": expected.get("chapter_id", ""),
                "actual_chapter_id": actual.get("chapter_id", ""),
                "expected_title": expected_title,
                "actual_title": actual_title,
            })

    event_mismatches = []
    missing_events = []
    extra_events = []
    empty_caption_events = []
    input_keys = []
    output_keys = []

    for ch_idx, expected_ch in enumerate(input_chapters):
        actual_ch = output_chapters[ch_idx] if ch_idx < output_chapter_count else {}
        expected_events = expected_ch.get("events", [])
        actual_events = actual_ch.get("events", []) if isinstance(actual_ch, dict) else []
        if not isinstance(actual_events, list):
            actual_events = []
        chapter_id = expected_ch.get("chapter_id", "")
        actual_chapter_id = actual_ch.get("chapter_id", "") if isinstance(actual_ch, dict) else ""

        for ev_idx, expected_ev in enumerate(expected_events):
            expected_event_id = expected_ev.get("event_id", "")
            input_keys.append((chapter_id, expected_event_id))
            if ev_idx >= len(actual_events):
                missing_events.append({
                    "chapter_index": ch_idx,
                    "event_index": ev_idx,
                    "chapter_id": chapter_id,
                    "event_id": expected_event_id,
                })
                continue

            actual_ev = actual_events[ev_idx]
            if not isinstance(actual_ev, dict):
                actual_ev = {}
            actual_event_id = actual_ev.get("event_id", "")
            actual_key = (actual_chapter_id, actual_event_id)
            output_keys.append(actual_key)
            if actual_event_id != expected_event_id:
                event_mismatches.append({
                    "chapter_index": ch_idx,
                    "event_index": ev_idx,
                    "chapter_id": chapter_id,
                    "expected_event_id": expected_event_id,
                    "actual_event_id": actual_event_id,
                })
            if not (actual_ev.get("caption", "") or "").strip():
                empty_caption_events.append({
                    "chapter_index": ch_idx,
                    "event_index": ev_idx,
                    "chapter_id": actual_chapter_id,
                    "event_id": actual_event_id,
                })

        for ev_idx in range(len(expected_events), len(actual_events)):
            actual_ev = actual_events[ev_idx]
            if not isinstance(actual_ev, dict):
                actual_ev = {}
            actual_event_id = actual_ev.get("event_id", "")
            output_keys.append((actual_chapter_id, actual_event_id))
            extra_events.append({
                "chapter_index": ch_idx,
                "event_index": ev_idx,
                "chapter_id": actual_chapter_id,
                "event_id": actual_event_id,
            })
            if not (actual_ev.get("caption", "") or "").strip():
                empty_caption_events.append({
                    "chapter_index": ch_idx,
                    "event_index": ev_idx,
                    "chapter_id": actual_chapter_id,
                    "event_id": actual_event_id,
                })

    for ch_idx in range(input_chapter_count, output_chapter_count):
        actual_ch = output_chapters[ch_idx]
        if not isinstance(actual_ch, dict):
            actual_ch = {}
        actual_chapter_id = actual_ch.get("chapter_id", "")
        actual_events = actual_ch.get("events", [])
        if not isinstance(actual_events, list):
            actual_events = []
        for ev_idx, actual_ev in enumerate(actual_events):
            if not isinstance(actual_ev, dict):
                actual_ev = {}
            actual_event_id = actual_ev.get("event_id", "")
            output_keys.append((actual_chapter_id, actual_event_id))
            extra_events.append({
                "chapter_index": ch_idx,
                "event_index": ev_idx,
                "chapter_id": actual_chapter_id,
                "event_id": actual_event_id,
            })
            if not (actual_ev.get("caption", "") or "").strip():
                empty_caption_events.append({
                    "chapter_index": ch_idx,
                    "event_index": ev_idx,
                    "chapter_id": actual_chapter_id,
                    "event_id": actual_event_id,
                })

    chapter_count_matches = input_chapter_count == output_chapter_count
    event_count_matches = input_event_count == output_event_count
    chapter_id_title_matches = chapter_count_matches and not chapter_mismatches
    event_id_matches = event_count_matches and not event_mismatches and not missing_events and not extra_events
    polished_event_count = sum(1 for key in input_keys if key in polished_map)
    fallback_event_count = input_event_count - polished_event_count

    return {
        "ok": chapter_count_matches and event_count_matches and chapter_id_title_matches and event_id_matches and not empty_caption_events,
        "chapter_count_matches": chapter_count_matches,
        "event_count_matches": event_count_matches,
        "chapter_id_title_matches": chapter_id_title_matches,
        "event_id_matches": event_id_matches,
        "input_chapter_count": input_chapter_count,
        "output_chapter_count": output_chapter_count,
        "input_event_count": input_event_count,
        "output_event_count": output_event_count,
        "polished_event_count": polished_event_count,
        "fallback_event_count": fallback_event_count,
        "chapter_mismatches": chapter_mismatches,
        "event_mismatches": event_mismatches,
        "missing_events": missing_events,
        "extra_events": extra_events,
        "empty_caption_events": empty_caption_events,
        "duplicate_input_event_keys": [
            {"chapter_id": chapter_id, "event_id": event_id}
            for chapter_id, event_id in _count_duplicates(input_keys)
        ],
        "duplicate_output_event_keys": [
            {"chapter_id": chapter_id, "event_id": event_id}
            for chapter_id, event_id in _count_duplicates(output_keys)
        ],
    }


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
            ch_id = ch.get("chapter_id", "")
            ev_id = ev.get("event_id", "")
            polished = polished_map.get((ch_id, ev_id), "")
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
        cloned["stage3_confidence"] = _build_stage3_confidence(payload, {"chapters": []}, {})
        cloned["stage3_confidence"]["skipped"] = True
        cloned["stage3_confidence"]["skip_reason"] = "no_caption_events"
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
    polished_chapters = polished_json.get("chapters", []) if isinstance(polished_json, dict) else []
    if not isinstance(polished_chapters, list):
        polished_chapters = []
    for ch in polished_chapters:
        if not isinstance(ch, dict):
            continue
        ch_id = ch.get("chapter_id", "")
        events = ch.get("events", [])
        if not isinstance(events, list):
            continue
        for ev in events:
            if not isinstance(ev, dict):
                continue
            ev_id = ev.get("event_id", "")
            cap = (ev.get("caption", "") or "").strip()
            if ev_id and cap:
                polished_map[(ch_id, ev_id)] = cap

    cloned = _clone_with_final_caption(stage2_data, polished_map)
    cloned["stage3_confidence"] = _build_stage3_confidence(payload, polished_json, polished_map)

    total_events = sum(len(ch.get("events", [])) for ch in cloned["chapters"])
    polished_count = sum(
        1
        for ch in cloned["chapters"]
        for ev in ch.get("events", [])
        if (ch.get("chapter_id", ""), ev.get("event_id", "")) in polished_map
    )

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(cloned, f, ensure_ascii=False, indent=2)

    _log(video_tag, f"\n🎉 Stage 3 完成：润色 {polished_count}/{total_events} 个 event → {out_path}")
    if not cloned["stage3_confidence"]["ok"]:
        _log(video_tag, "⚠️ Stage 3 校验未完全通过，详情见 stage3_polished.json 的 stage3_confidence 字段。")
    return out_path
