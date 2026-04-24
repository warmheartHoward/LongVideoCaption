import json
import os
import re
from typing import Tuple

from .config import PipelineConfig
from .llm_client import request_llm_with_retry
from .token_tracker import TokenTracker
from .utils import parse_timestamp_to_seconds


PASS_NAME = "pass3_aggregation"

_ROLE_PATTERN = re.compile(r'\[[^\[\]]+\]')


def _log(video_tag: str, msg: str) -> None:
    print(f"[{video_tag}] {msg}" if video_tag else msg)


def _extract_event_characters(step3_text: str, name_to_desc: dict) -> list:
    if not step3_text:
        return []
    seen = []
    for name in _ROLE_PATTERN.findall(step3_text):
        if name not in seen:
            seen.append(name)
    return [{"name": n, "desc": name_to_desc.get(n, "")} for n in seen]

SYS_PROMPT = """你是一个资深的电影剧本统筹。
你需要阅读一份带有时间戳的视频底层动作流水账，将其聚合并切分为几个符合叙事逻辑的「大章节（Chapters）」。

【核心要求】
1. 切分逻辑：基于“起因、经过、转折、高潮、结局”等叙事节拍来切分章节，切忌按时间机械平分。
2. 时间边界：新章节的 start_time 必须严格等于它包含的【第一个动作】的时间；end_time 必须等于它包含的【最后一个动作】的时间。时间格式必须保留原样（如 [00:01:20.000]）。
3. 文本要求：输出丰富自然的剧情总结，严禁在最终的 summary 文本中遗留时间戳或事件ID。

请严格输出以下 JSON 结构：
{
  "video_summary": "<全片核心剧情深度总结，交代故事的起承转合>",
  "chapters": [
    {
      "chapter_id": "ch_01",
      "title": "<极具概括性和文学性的章节标题>",
      "chapter_summary": "<本章节剧情深度复盘，写清前因后果和人物核心动作>",
      "start_time": "<本章第一个动作的 start_time>",
      "end_time": "<本章最后一个动作的 end_time>"
    }
  ]
}"""


def _run_chapter_aggregation(
    cfg: PipelineConfig,
    aligned_results: list,
    client,
    token_tracker: TokenTracker = None,
    video_tag: str = "",
) -> Tuple[dict, list]:
    _log(video_tag, "\n" + "=" * 50)
    _log(video_tag, "🎬 [Pass 3 · 子步骤 A] 宏观章节 (Chapter) 逻辑切分...")
    _log(video_tag, "=" * 50)

    all_events = []
    timeline_text = ""

    for chunk in aligned_results:
        for ev in chunk.get("data", {}).get("events", []):
            all_events.append(ev)
            start_str = ev.get("start_time", "")
            end_str = ev.get("end_time", "")
            desc = ev.get("step3_synthesized_dense_caption", "")
            timeline_text += f"{start_str} - {end_str} : {desc}\n"

    user_prompt = f"【完整底层时间轴】:\n{timeline_text}\n\n请严格基于时间轴进行章节切分，输出 JSON。"

    log_tag = f"[{video_tag}] 章节切分" if video_tag else "章节切分"
    chapter_agg_result = request_llm_with_retry(
        client=client, model=cfg.model_name,
        messages=[{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt}],
        max_tokens=cfg.pass3_max_tokens, temperature=cfg.pass3_temperature,
        max_retries=cfg.max_retries, chunk_name=log_tag,
        token_tracker=token_tracker, stage=PASS_NAME,
    )
    return chapter_agg_result, all_events


def _assemble_final(video_path: str, chapter_agg_result: dict, all_events: list, name_to_desc: dict, video_tag: str = "") -> dict:
    _log(video_tag, "\n" + "=" * 50)
    _log(video_tag, "🛠️ [Pass 3 · 子步骤 B] Chapter-Event 层级数据物理挂载...")
    _log(video_tag, "=" * 50)

    final_json = {
        "video_path": video_path,
        "video_summary": chapter_agg_result.get("video_summary", "全片剧情总结提取失败"),
        "chapters": [],
    }

    chapters_def = chapter_agg_result.get("chapters", [])
    if not chapters_def:
        _log(video_tag, "❌ 未获取到章节定义，将采用兜底单章节模式。")
        chapters_def = [{
            "chapter_id": "ch_01", "title": "完整视频", "chapter_summary": "兜底聚合",
            "start_time": all_events[0]["start_time"] if all_events else "[00:00:00.000]",
            "end_time": all_events[-1]["end_time"] if all_events else "[00:00:00.000]",
        }]

    for ch_idx, ch_def in enumerate(chapters_def):
        ch_start_str = ch_def.get("start_time", "")
        ch_end_str = ch_def.get("end_time", "")

        ch_start_sec = parse_timestamp_to_seconds(ch_start_str)
        ch_end_sec = parse_timestamp_to_seconds(ch_end_str)

        if ch_idx == len(chapters_def) - 1:
            ch_end_sec += 9999.0

        chapter_obj = {
            "chapter_id": ch_def.get("chapter_id", f"ch_{ch_idx+1:02d}"),
            "title": ch_def.get("title", f"第 {ch_idx+1} 章"),
            "chapter_summary": ch_def.get("chapter_summary", ""),
            "start_time": ch_start_str,
            "end_time": ch_end_str,
            "events": [],
        }

        for ev in all_events:
            ev_start_sec = parse_timestamp_to_seconds(ev.get("start_time", ""))

            if ch_start_sec - 0.5 <= ev_start_sec < ch_end_sec:
                step3_text = ev.get("step3_synthesized_dense_caption", "")
                chapter_obj["events"].append({
                    "event_id": f"ev_{(ch_idx+1):02d}_{len(chapter_obj['events'])+1:03d}",
                    "start_time": ev.get("start_time", ""),
                    "end_time": ev.get("end_time", ""),
                    "step1_objective_visual": ev.get("step1_objective_visual", ""),
                    "step2_contextual_reasoning": ev.get("step2_contextual_reasoning", ""),
                    "step3_synthesized_dense_caption": step3_text,
                    "characters_in_event": _extract_event_characters(step3_text, name_to_desc),
                    "key_frame_times": ev.get("key_frame_times", []),
                })

        final_json["chapters"].append(chapter_obj)

    assigned_event_count = sum(len(ch["events"]) for ch in final_json["chapters"])
    if assigned_event_count < len(all_events):
        _log(video_tag, f"⚠️ 警告: 有 {len(all_events) - assigned_event_count} 个事件越界未挂载，已强制追加至最终章。")
        last_chapter = final_json["chapters"][-1]
        for ev in all_events:
            ev_start_sec = parse_timestamp_to_seconds(ev.get("start_time", ""))
            if ev_start_sec >= parse_timestamp_to_seconds(last_chapter["end_time"]):
                step3_text = ev.get("step3_synthesized_dense_caption", "")
                last_chapter["events"].append({
                    "event_id": f"ev_fallback_{len(last_chapter['events'])+1:03d}",
                    "start_time": ev.get("start_time", ""),
                    "end_time": ev.get("end_time", ""),
                    "step1_objective_visual": ev.get("step1_objective_visual", ""),
                    "step2_contextual_reasoning": ev.get("step2_contextual_reasoning", ""),
                    "step3_synthesized_dense_caption": step3_text,
                    "characters_in_event": _extract_event_characters(step3_text, name_to_desc),
                    "key_frame_times": ev.get("key_frame_times", []),
                })

    return final_json


def run_pass3(
    cfg: PipelineConfig,
    video_path: str,
    aligned_json_path: str,
    run_dir: str,
    client,
    token_tracker: TokenTracker = None,
    video_tag: str = "",
) -> str:
    os.makedirs(run_dir, exist_ok=True)
    final_output_path = os.path.join(run_dir, "pass3_final.json")

    if os.path.exists(final_output_path):
        _log(video_tag, "\n" + "=" * 50)
        _log(video_tag, "⏭️  Pass 3 终产物已存在，跳过装配。")
        _log(video_tag, "=" * 50)
        return final_output_path

    with open(aligned_json_path, 'r', encoding='utf-8') as f:
        aligned_results = json.load(f)

    bank_path = os.path.join(run_dir, "pass2_global_bank.json")
    name_to_desc = {}
    if os.path.exists(bank_path):
        try:
            with open(bank_path, 'r', encoding='utf-8') as f:
                lite_bank = json.load(f)
            name_to_desc = {entry["角色名"]: entry["外貌特征"] for entry in lite_bank}
            _log(video_tag, f"📖 已加载全局图鉴，收录 {len(name_to_desc)} 名角色，用于 event 级角色映射")
        except Exception as e:
            _log(video_tag, f"⚠️ 全局图鉴加载失败，event 角色字段将仅含 name 而无 desc: {e}")

    chapter_agg_result, all_events = _run_chapter_aggregation(cfg, aligned_results, client, token_tracker, video_tag)
    final_storyboard = _assemble_final(video_path, chapter_agg_result, all_events, name_to_desc, video_tag)

    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_storyboard, f, ensure_ascii=False, indent=2)

    _log(video_tag, f"\n🎉 大功告成！最终的剧本级结构化数据已保存至: {final_output_path}")
    return final_output_path
