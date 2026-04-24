import copy
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

from .config import PipelineConfig
from .frame_extractor import extract_single_frame_base64
from .token_tracker import TokenTracker
from .utils import clean_json_response, safe_replace


PASS_NAME = "pass2_alignment"
REVIEW_STAGE_NAME = "pass2_review"


def _log(video_tag: str, msg: str) -> None:
    print(f"[{video_tag}] {msg}" if video_tag else msg)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


SYS_PROMPT_ALIGNMENT = """你是一个顶级的电影多模态视觉统筹。
当前视频正在分段处理，同一个角色可能会被赋予不同的【临时名称】。
你的任务是：接收【已确立的全局聚类库】与【当前片段的新角色图片】，进行严谨的身份同一性判决。
（⚠️ 本阶段只做**身份是否相同**的判决。最终的权威命名会在所有 chunk 处理完后由终审环节统一决定，此处不要纠结名字，只要判同一性）

【⚠️ 严禁发生的错误（Anti-Patterns）】
1. 严禁"望文生义"：绝对不要因为临时名字里都有"公主"或"勇士"就判定为同一人。必须 100% 依赖图像特征！
2. 警惕"同类不同体"：如果两只猫的颜色、眼睛形状有显著差异，即使都是猫，也必须判定为 NEW。
3. 宁缺毋滥原则：当且仅当你对两者的视觉特征有 80% 以上的把握吻合时，才进行合并。如果有严重疑虑，坚决判定为 NEW。

【💡 优秀推理范例学习】
假设聚类库里有 cluster_0003（参考名：皮皮，特征：黑色圆润，巨大空洞眼）。当前片段出现 [大眼鼠勇士]（特征：身披铠甲，巨大空洞眼）。
✅ 正确的 step3_comparison_logic："虽然穿了铠甲且被称为鼠，但其核心面部特征（占据半脸的巨大空洞眼）与聚类 cluster_0003 完全一致。结合剧情，判定为同一聚类。"

请严格输出 JSON：
{
  "multimodal_reasoning_process": [
    {
      "temp_name": "<当前评估的临时角色>",
      "step1_bank_visual_features": "<它最像聚类库里哪个 cluster？详细列出那个 cluster 的核心视觉特征。若完全不像任何人，填无>",
      "step2_current_visual_features": "<仔细观察当前角色的截图，列出其客观视觉特征>",
      "step3_comparison_logic": "<对比 step1 和 step2，结合动作逻辑，给出判定理由>"
    }
  ],
  "chunk_identity_mapping": [
    {
      "temp_name_in_chunk": "<当前片段中的临时名称>",
      "match_result": "<若匹配到已有聚类，填其 cluster_id（例如 cluster_0003）；若是新角色，填 NEW>",
      "confidence_score": <0-100的整数，表示你对这次判定的视觉把握有多大>
    }
  ]
}"""


SYS_PROMPT_REVIEW = """你是终审级电影多模态视觉统筹。系统已通过滚动匹配完成初步的角色聚类，现在你收到其中**一个聚类**的全部证据（高清视觉帧 + 该角色在各 chunk 里被赋予的临时名称 + 每次出现时的外貌描述）。

你要做三件事：
1. 【权威命名】综合所有证据，为该聚类确定一个**权威全局标准名**。命名权威度排序：
   - 对白/字幕点名的专名（如"周太安"） > 有辨识度的绰号 > 描述性命名（如"红衣男子"） > 泛化编号（"男人1"）
   - 候选名里都不够权威，可基于视觉特征自拟更准确的名字
   - 输出格式：方括号包裹，例如 [周太安]
2. 【描述重写】基于**高清帧**重写该角色的客观视觉描述。滚动阶段使用的是较低分辨率的帧，可能有错误，以高清帧为准。
3. 【拆分判断】判断这个聚类是否应该被**拆分** —— 即滚动阶段是否把多个不同角色误合并了。高清帧下若发现明显是不同的人，给出拆分建议；不确定就不拆。

请严格输出 JSON：
{
  "cluster_id": "<传入的 cluster_id 原样返回>",
  "final_global_name": "[xxx]",
  "refined_visual_description": "基于高清帧重写的客观外貌描述",
  "chosen_reason": "选择该名字的理由（源自哪个 sighting 的 temp_name / 为什么权威）",
  "split_suggestion": null
}

若需要拆分，将 split_suggestion 填为数组：
[
  {
    "sub_cluster_label": "sub_A",
    "sighting_indices": [0, 2, 4],
    "inferred_name": "[aaa]",
    "inferred_description": "...",
    "reason": "为什么认为这些 sighting 和其他的不是同一人"
  },
  ...
]
拆分时 sighting_indices 必须**覆盖全部** sightings 且互不重叠（总数 = 传入的 sighting 数）。"""


# =========================
#   持久化
# =========================

def _load_progress(progress_path: str):
    if not os.path.exists(progress_path):
        return None
    try:
        with open(progress_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 读取 pass2 进度失败: {e}")
        return None


def _save_progress(progress_path: str, payload: dict) -> None:
    tmp = progress_path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, progress_path)


def _build_progress_payload(global_bank, chunk_mappings, processed_count, accumulated_story, low_conf_flags):
    return {
        "schema_version": 2,
        "processed_chunk_count": processed_count,
        "accumulated_story": accumulated_story,
        "global_bank": global_bank,
        "chunk_mappings": chunk_mappings,
        "low_conf_flags": low_conf_flags,
    }


def _save_review_log(path: str, review_log: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({"reviewed_clusters": review_log}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# =========================
#   工具
# =========================

def _next_cluster_id(global_bank: list) -> str:
    return f"cluster_{len(global_bank) + 1:04d}"


_NAME_STRIP_CHARS = "[]【】《》〈〉<>「」『』 \t\r\n"


def _normalize_name(name) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().strip(_NAME_STRIP_CHARS).strip().lower()


def _prime_cluster_frame_cache(video_path: str, global_bank: list, cfg: PipelineConfig) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    for cluster in global_bank:
        first = cluster["sightings"][0] if cluster.get("sightings") else None
        if first:
            cache[cluster["cluster_id"]] = extract_single_frame_base64(
                video_path, first.get("anchor_ts", ""), max_width=cfg.frame_max_width
            )
        else:
            cache[cluster["cluster_id"]] = ""
    return cache


def _select_review_sighting_indices(n: int, max_frames: int) -> List[int]:
    if n <= max_frames:
        return list(range(n))
    if max_frames == 1:
        return [0]
    if max_frames == 2:
        return [0, n - 1]
    out = {0, n - 1}
    step = (n - 1) / (max_frames - 1)
    for k in range(1, max_frames - 1):
        out.add(int(round(k * step)))
    return sorted(out)


# =========================
#   Phase A: 滚动聚类
# =========================

def _phase_a_rolling(cfg, video_path, pass1_results, global_bank, chunk_mappings,
                    processed_count, accumulated_story, low_conf_flags,
                    progress_path, client, token_tracker, video_tag) -> Tuple[int, str]:
    cluster_frame_cache = _prime_cluster_frame_cache(video_path, global_bank, cfg)

    for i in range(processed_count, len(pass1_results)):
        chunk = pass1_results[i]
        chunk_range = chunk.get("chunk_time_range", f"Chunk_{i+1}")
        data = chunk.get("data", {})

        current_story = data.get("chunk_summary", "")
        current_chars = (
            data.get("characters_in_chunk")
            or data.get("new_characters_in_chunk")
            or []
        )

        _log(video_tag, f"\n🔍 [Phase A] 分析 {chunk_range} | 发现待定面孔: {len(current_chars)} 个")

        chunk_map: Dict[str, str] = {}

        if not current_chars:
            accumulated_story += f"第{i+1}段: {current_story}\n"
            _log(video_tag, "  ⏭️ 无需对齐，跳过")
            chunk_mappings.append(chunk_map)
            processed_count = i + 1
            _save_progress(progress_path, _build_progress_payload(
                global_bank, chunk_mappings, processed_count, accumulated_story, low_conf_flags
            ))
            continue

        user_content: List[dict] = [
            {"type": "text", "text": f"【前情提要】\n{accumulated_story if accumulated_story else '无前情，视频开篇。'}\n"},
            {"type": "text", "text": f"【当前片段剧情】\n{current_story}\n"},
        ]

        if global_bank:
            user_content.append({"type": "text", "text": "====================\n【已确立的全局聚类库】 (老熟人列表):"})
            for g in global_bank:
                cid = g["cluster_id"]
                ref = g["sightings"][0]
                user_content.append({"type": "text", "text": (
                    f"cluster_id: {cid}\n"
                    f"参考临时名: {ref.get('temp_name', '')}\n"
                    f"参考外貌描述: {ref.get('desc', '')}"
                )})
                b64 = cluster_frame_cache.get(cid, "")
                if b64:
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        else:
            user_content.append({"type": "text", "text": "====================\n【已确立的全局聚类库】: 当前为空。"})

        user_content.append({"type": "text", "text": "====================\n【当前片段出现的疑似新角色】 (等待判决):"})

        current_b64s: Dict[str, str] = {}
        for c_char in current_chars:
            temp_name = c_char.get("temp_name", "")
            anchor_ts = c_char.get("anchor_timestamp", "")
            b64 = extract_single_frame_base64(video_path, anchor_ts, max_width=cfg.frame_max_width)
            current_b64s[temp_name] = b64

            user_content.append({"type": "text", "text": (
                f"临时名称: {temp_name}\n"
                f"阶段描述: {c_char.get('visual_description', '')}"
            )})
            if b64:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        user_content.append({"type": "text", "text": "请输出 JSON 判决结果。"})

        align_result = None
        for attempt in range(1, cfg.max_retries + 1):
            try:
                start_time = time.time()
                response = client.chat.completions.create(
                    model=cfg.model_name,
                    max_tokens=cfg.pass2_max_tokens,
                    temperature=cfg.pass2_temperature,
                    messages=[
                        {"role": "system", "content": SYS_PROMPT_ALIGNMENT},
                        {"role": "user", "content": user_content},
                    ],
                )
                if response.usage and token_tracker is not None:
                    token_tracker.record(PASS_NAME, response.usage)
                    _log(video_tag, f"  📊 [Token] Prompt: {response.usage.prompt_tokens} | Total: {response.usage.total_tokens}")

                align_result = json.loads(clean_json_response(response.choices[0].message.content))
                _log(video_tag, f"  ✅ 判决完成 (耗时: {time.time() - start_time:.2f}s)")
                break
            except Exception as e:
                _log(video_tag, f"  ⚠️ API 请求异常 (尝试 {attempt}/{cfg.max_retries}): {e}")
                if attempt == cfg.max_retries:
                    _log(video_tag, "  ❌ 跳过该 Chunk 对齐。")
                time.sleep(2)

        if align_result is not None:
            valid_cluster_ids = {g["cluster_id"] for g in global_bank}
            norm_char_lookup: Dict[str, dict] = {}
            for c in current_chars:
                norm_key = _normalize_name(c.get("temp_name", ""))
                if norm_key and norm_key not in norm_char_lookup:
                    norm_char_lookup[norm_key] = c

            for mapping in align_result.get("chunk_identity_mapping", []):
                temp_name_raw = mapping.get("temp_name_in_chunk")
                match_result = (mapping.get("match_result") or "").strip()
                conf_score = mapping.get("confidence_score", 100)

                if not temp_name_raw:
                    continue
                norm_key = _normalize_name(temp_name_raw)
                matched_char = norm_char_lookup.get(norm_key)
                if not matched_char:
                    _log(video_tag, f"    ⚠️ 模型输出的 temp_name 不在当前片段: {temp_name_raw!r}（归一化后={norm_key!r}），忽略")
                    continue

                temp_name = matched_char.get("temp_name", "") or temp_name_raw
                if temp_name_raw != temp_name:
                    _log(video_tag, f"    ℹ️ temp_name 规范化: 模型输出 {temp_name_raw!r} → 使用原始 {temp_name!r}")

                force_new = False
                if match_result != "NEW":
                    if match_result not in valid_cluster_ids:
                        _log(video_tag, f"    ⚠️ 模型输出的 cluster_id 无效: {match_result}，强制判 NEW")
                        force_new = True
                    elif conf_score < cfg.pass2_hard_split_threshold:
                        _log(video_tag, f"    🚫 置信度极低 ({conf_score} < {cfg.pass2_hard_split_threshold})，强制判 NEW: {temp_name}")
                        force_new = True

                if match_result == "NEW" or force_new:
                    new_cid = _next_cluster_id(global_bank)
                    new_cluster = {
                        "cluster_id": new_cid,
                        "sightings": [{
                            "chunk_i": i,
                            "temp_name": temp_name,
                            "desc": matched_char.get("visual_description", ""),
                            "anchor_ts": matched_char.get("anchor_timestamp", ""),
                        }],
                    }
                    global_bank.append(new_cluster)
                    valid_cluster_ids.add(new_cid)
                    cluster_frame_cache[new_cid] = current_b64s.get(temp_name, "")
                    chunk_map[temp_name] = new_cid
                    _log(video_tag, f"    🌟 新聚类: {new_cid} <- {temp_name} (置信度: {conf_score})")
                else:
                    is_low_conf = conf_score < cfg.pass2_confidence_threshold
                    cluster = next((c for c in global_bank if c["cluster_id"] == match_result), None)
                    if cluster is not None:
                        cluster["sightings"].append({
                            "chunk_i": i,
                            "temp_name": temp_name,
                            "desc": matched_char.get("visual_description", ""),
                            "anchor_ts": matched_char.get("anchor_timestamp", ""),
                        })
                    chunk_map[temp_name] = match_result
                    if is_low_conf:
                        low_conf_flags.append({
                            "chunk_i": i,
                            "temp_name": temp_name,
                            "cluster_id": match_result,
                            "confidence_score": conf_score,
                        })
                        _log(video_tag, f"    ⚠️ 低置信合并 ({conf_score}): {temp_name} -> {match_result}（打标待终审复核）")
                    else:
                        _log(video_tag, f"    🔗 合并: {temp_name} -> {match_result} (置信度: {conf_score})")
        else:
            _log(video_tag, "  ⚠️ 本 chunk 无有效判决，当前片段角色将保留为临时名（Phase C 会兜底）")

        accumulated_story += f"第{i+1}段: {current_story}\n"
        chunk_mappings.append(chunk_map)
        processed_count = i + 1
        _save_progress(progress_path, _build_progress_payload(
            global_bank, chunk_mappings, processed_count, accumulated_story, low_conf_flags
        ))

    return processed_count, accumulated_story


# =========================
#   Phase B: 高清终审
# =========================

def _phase_b_review(cfg, video_path, global_bank, review_log,
                    reviewed_cluster_ids, review_log_path,
                    client, token_tracker, video_tag) -> Dict[str, dict]:
    final_info: Dict[str, dict] = {}
    for entry in review_log:
        cid = entry.get("cluster_id")
        if cid:
            final_info[cid] = {
                "final_global_name": entry.get("final_global_name", ""),
                "refined_visual_description": entry.get("refined_visual_description", ""),
                "chosen_reason": entry.get("chosen_reason", ""),
            }

    idx = 0
    while idx < len(global_bank):
        cluster = global_bank[idx]
        cid = cluster["cluster_id"]
        idx += 1

        if cid in reviewed_cluster_ids:
            continue

        sightings = cluster.get("sightings", [])
        if not sightings:
            _log(video_tag, f"  ⚠️ [Phase B] {cid} 无 sightings，跳过")
            reviewed_cluster_ids.add(cid)
            continue

        sel_indices = _select_review_sighting_indices(len(sightings), cfg.pass2_review_max_frames_per_cluster)
        _log(video_tag, f"\n🔎 [Phase B] 终审 {cid} | sightings={len(sightings)} | 高清帧采样索引={sel_indices}")

        temp_name_list = [s.get("temp_name", "") for s in sightings]
        user_content: List[dict] = [
            {"type": "text", "text": f"cluster_id: {cid}"},
            {"type": "text", "text": f"该聚类共 {len(sightings)} 次出现。临时名列表（按出现顺序）: {temp_name_list}"},
        ]

        for i_s, s in enumerate(sightings):
            user_content.append({"type": "text", "text": (
                f"--- sighting #{i_s} ---\n"
                f"所在 chunk 索引: {s.get('chunk_i')}\n"
                f"临时名: {s.get('temp_name', '')}\n"
                f"滚动阶段描述: {s.get('desc', '')}\n"
                f"锚定时间: {s.get('anchor_ts', '')}"
            )})
            if i_s in sel_indices:
                b64 = extract_single_frame_base64(
                    video_path,
                    s.get("anchor_ts", ""),
                    max_width=cfg.pass2_review_frame_max_width,
                    jpg_quality=cfg.pass2_review_jpg_quality,
                )
                if b64:
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                else:
                    _log(video_tag, f"    ⚠️ sighting #{i_s} 高清帧提取失败: {s.get('anchor_ts', '')}")

        user_content.append({"type": "text", "text": "请输出 JSON 终审结果。"})

        review_result = None
        for attempt in range(1, cfg.max_retries + 1):
            try:
                start_time = time.time()
                response = client.chat.completions.create(
                    model=cfg.model_name,
                    max_tokens=cfg.pass2_review_max_tokens,
                    temperature=cfg.pass2_review_temperature,
                    messages=[
                        {"role": "system", "content": SYS_PROMPT_REVIEW},
                        {"role": "user", "content": user_content},
                    ],
                )
                if response.usage and token_tracker is not None:
                    token_tracker.record(REVIEW_STAGE_NAME, response.usage)
                    _log(video_tag, f"  📊 [Token] Prompt: {response.usage.prompt_tokens} | Total: {response.usage.total_tokens}")

                review_result = json.loads(clean_json_response(response.choices[0].message.content))
                _log(video_tag, f"  ✅ 终审完成 (耗时: {time.time() - start_time:.2f}s)")
                break
            except Exception as e:
                _log(video_tag, f"  ⚠️ 终审 API 异常 (尝试 {attempt}/{cfg.max_retries}): {e}")
                if attempt == cfg.max_retries:
                    _log(video_tag, f"  ❌ {cid} 终审失败，将回退为首个 sighting 的 temp_name 作为全局名")
                time.sleep(2)

        if review_result is None:
            fallback_name = f"[{sightings[0].get('temp_name', cid)}]"
            fallback_desc = sightings[0].get("desc", "")
            final_info[cid] = {
                "final_global_name": fallback_name,
                "refined_visual_description": fallback_desc,
                "chosen_reason": "FALLBACK: 终审调用失败，回退到首个 sighting 的 temp_name",
            }
            review_log.append({
                "cluster_id": cid,
                "final_global_name": fallback_name,
                "refined_visual_description": fallback_desc,
                "chosen_reason": "FALLBACK: 终审调用失败",
                "split_suggestion": None,
                "reviewed_at": _now_iso(),
                "status": "fallback",
            })
            reviewed_cluster_ids.add(cid)
            _save_review_log(review_log_path, review_log)
            continue

        final_name = review_result.get("final_global_name") or f"[{sightings[0].get('temp_name', cid)}]"
        refined_desc = review_result.get("refined_visual_description") or sightings[0].get("desc", "")
        chosen_reason = review_result.get("chosen_reason", "")
        split_suggestion = review_result.get("split_suggestion")

        _log(video_tag, f"  📝 {cid} -> {final_name}")
        _log(video_tag, f"     选名理由: {chosen_reason}")

        entry = {
            "cluster_id": cid,
            "final_global_name": final_name,
            "refined_visual_description": refined_desc,
            "chosen_reason": chosen_reason,
            "split_suggestion": split_suggestion,
            "reviewed_at": _now_iso(),
            "status": "ok",
        }

        if isinstance(split_suggestion, list) and len(split_suggestion) >= 2:
            all_idx: List[int] = []
            for sub in split_suggestion:
                all_idx.extend(sub.get("sighting_indices") or [])
            expected = list(range(len(sightings)))
            if sorted(all_idx) != expected:
                _log(video_tag, f"  ⚠️ 拆分 indices 不完整或重叠 (got={sorted(all_idx)}, expected={expected})，拆分被拒绝")
                entry["status"] = "split_rejected_invalid_indices"
                final_info[cid] = {
                    "final_global_name": final_name,
                    "refined_visual_description": refined_desc,
                    "chosen_reason": chosen_reason,
                }
            else:
                _log(video_tag, f"  ✂️ 执行拆分: {cid} -> {len(split_suggestion)} 个子聚类")
                sub0 = split_suggestion[0]
                sub0_indices = sub0.get("sighting_indices") or []
                cluster["sightings"] = [sightings[k] for k in sub0_indices]
                if sub0_indices:
                    first_s = sightings[sub0_indices[0]]
                    sub0_name = sub0.get("inferred_name") or f"[{first_s.get('temp_name', cid)}]"
                    sub0_desc = sub0.get("inferred_description") or first_s.get("desc", "")
                else:
                    sub0_name = final_name
                    sub0_desc = refined_desc
                final_info[cid] = {
                    "final_global_name": sub0_name,
                    "refined_visual_description": sub0_desc,
                    "chosen_reason": f"SPLIT sub_A (由 {cid} 拆出，原因: {sub0.get('reason', '')})",
                }
                _log(video_tag, f"    └─ {cid} 保留 sightings={sub0_indices} -> {sub0_name}")

                for sub in split_suggestion[1:]:
                    sub_indices = sub.get("sighting_indices") or []
                    new_cid = _next_cluster_id(global_bank)
                    if sub_indices:
                        first_s = sightings[sub_indices[0]]
                        sub_name = sub.get("inferred_name") or f"[{first_s.get('temp_name', new_cid)}]"
                        sub_desc = sub.get("inferred_description") or first_s.get("desc", "")
                    else:
                        sub_name = f"[{new_cid}]"
                        sub_desc = ""
                    new_cluster = {
                        "cluster_id": new_cid,
                        "sightings": [sightings[k] for k in sub_indices],
                    }
                    global_bank.append(new_cluster)
                    final_info[new_cid] = {
                        "final_global_name": sub_name,
                        "refined_visual_description": sub_desc,
                        "chosen_reason": f"SPLIT {sub.get('sub_cluster_label', '')} (由 {cid} 拆出，原因: {sub.get('reason', '')})",
                    }
                    reviewed_cluster_ids.add(new_cid)
                    review_log.append({
                        "cluster_id": new_cid,
                        "final_global_name": sub_name,
                        "refined_visual_description": sub_desc,
                        "chosen_reason": f"SPLIT {sub.get('sub_cluster_label', '')} from {cid}",
                        "split_suggestion": None,
                        "reviewed_at": _now_iso(),
                        "status": "split_child",
                    })
                    _log(video_tag, f"    └─ 新增 {new_cid} 接管 sightings={sub_indices} -> {sub_name}")

                entry["status"] = "split_applied"
        else:
            final_info[cid] = {
                "final_global_name": final_name,
                "refined_visual_description": refined_desc,
                "chosen_reason": chosen_reason,
            }

        review_log.append(entry)
        reviewed_cluster_ids.add(cid)
        _save_review_log(review_log_path, review_log)

    return final_info


# =========================
#   Phase C: caption 批量改写
# =========================

def _build_sighting_to_cluster_map(global_bank) -> Dict[Tuple[int, str], str]:
    m: Dict[Tuple[int, str], str] = {}
    for cluster in global_bank:
        cid = cluster["cluster_id"]
        for s in cluster.get("sightings", []):
            m[(s.get("chunk_i"), s.get("temp_name", ""))] = cid
    return m


def _phase_c_rewrite(pass1_results, global_bank, chunk_mappings, final_info, video_tag) -> list:
    aligned_results = copy.deepcopy(pass1_results)
    authoritative = _build_sighting_to_cluster_map(global_bank)
    total_replacements = 0

    for i, chunk in enumerate(aligned_results):
        data = chunk.get("data", {})
        events = data.get("events", []) or []

        chunk_map = dict(chunk_mappings[i]) if i < len(chunk_mappings) else {}
        for temp_name in list(chunk_map.keys()):
            key = (i, temp_name)
            if key in authoritative:
                chunk_map[temp_name] = authoritative[key]

        if not chunk_map:
            continue

        for ev in events:
            caption = ev.get("step3_synthesized_dense_caption", "")
            if not caption:
                continue
            new_caption = caption
            for temp_name, cid in chunk_map.items():
                info = final_info.get(cid)
                if not info:
                    continue
                final_name = info.get("final_global_name", "")
                if not final_name or not temp_name:
                    continue
                before = new_caption
                stripped = temp_name.strip()
                if not (stripped.startswith("[") and stripped.endswith("]")):
                    new_caption = safe_replace(new_caption, f"[{stripped}]", final_name)
                new_caption = safe_replace(new_caption, temp_name, final_name)
                if new_caption != before:
                    total_replacements += 1
            ev["step3_synthesized_dense_caption"] = new_caption

        chars = data.get("characters_in_chunk") or data.get("new_characters_in_chunk") or []
        for c in chars:
            tn = c.get("temp_name")
            cid = chunk_map.get(tn)
            if cid and cid in final_info:
                c["resolved_cluster_id"] = cid
                c["resolved_global_name"] = final_info[cid].get("final_global_name", "")

    _log(video_tag, f"  🔄 Phase C 完成：共执行 {total_replacements} 处 caption 改写")
    return aligned_results


# =========================
#   主入口
# =========================

def run_pass2(
    cfg: PipelineConfig,
    video_path: str,
    pass1_json_path: str,
    run_dir: str,
    client,
    token_tracker: TokenTracker = None,
    video_tag: str = "",
) -> Tuple[str, str]:
    os.makedirs(run_dir, exist_ok=True)
    aligned_out_path = os.path.join(run_dir, "pass2_aligned.json")
    bank_out_path = os.path.join(run_dir, "pass2_global_bank.json")
    progress_path = os.path.join(run_dir, "pass2_progress.json")
    review_log_path = os.path.join(run_dir, "pass2_review_log.json")

    if os.path.exists(aligned_out_path) and os.path.exists(bank_out_path):
        _log(video_tag, "\n" + "=" * 60)
        _log(video_tag, "⏭️  Pass 2 终产物已存在，跳过对齐。")
        _log(video_tag, "=" * 60)
        return aligned_out_path, bank_out_path

    with open(pass1_json_path, 'r', encoding='utf-8') as f:
        pass1_results = json.load(f)

    _log(video_tag, "\n" + "=" * 60)
    _log(video_tag, "🚀 启动 Pass 2: 滚动聚类 + 终审命名 + 批量改写")
    _log(video_tag, "=" * 60)

    payload = _load_progress(progress_path)
    global_bank: list = []
    chunk_mappings: List[Dict[str, str]] = []
    processed_count = 0
    accumulated_story = ""
    low_conf_flags: list = []

    if payload and payload.get("schema_version") == 2:
        global_bank = payload.get("global_bank", []) or []
        chunk_mappings = payload.get("chunk_mappings", []) or []
        processed_count = payload.get("processed_chunk_count", 0)
        accumulated_story = payload.get("accumulated_story", "")
        low_conf_flags = payload.get("low_conf_flags", []) or []
        _log(video_tag, f"🔄 检测到 pass2 断点: 已完成 {processed_count} 个 chunk / 聚类 {len(global_bank)} 个")
    elif payload:
        _log(video_tag, "⚠️ 检测到旧版 pass2 进度文件（schema 不匹配），将重新开始滚动聚类")

    _log(video_tag, "\n---------- Phase A: 滚动聚类 ----------")
    processed_count, accumulated_story = _phase_a_rolling(
        cfg, video_path, pass1_results, global_bank, chunk_mappings,
        processed_count, accumulated_story, low_conf_flags,
        progress_path, client, token_tracker, video_tag,
    )
    _log(video_tag, f"\n✅ Phase A 完成: 共聚类 {len(global_bank)} 个角色 | 低置信软合并 {len(low_conf_flags)} 条")

    review_log: list = []
    reviewed_cluster_ids: set = set()
    if os.path.exists(review_log_path):
        try:
            with open(review_log_path, 'r', encoding='utf-8') as f:
                review_log = json.load(f).get("reviewed_clusters", []) or []
            reviewed_cluster_ids = {e.get("cluster_id") for e in review_log if e.get("cluster_id")}
            _log(video_tag, f"🔄 检测到终审断点: 已复核 {len(reviewed_cluster_ids)} 个聚类")
        except Exception as e:
            _log(video_tag, f"⚠️ 读取 pass2_review_log.json 失败: {e}（从头复核）")
            review_log = []
            reviewed_cluster_ids = set()

    if cfg.pass2_review_enable:
        _log(video_tag, "\n---------- Phase B: 高清终审 ----------")
        final_info = _phase_b_review(
            cfg, video_path, global_bank, review_log,
            reviewed_cluster_ids, review_log_path,
            client, token_tracker, video_tag,
        )
    else:
        _log(video_tag, "\n⏭️  Phase B 被配置关闭 (pass2_review_enable=False)，使用首个 sighting 的 temp_name 作为全局名")
        final_info = {}
        for cluster in global_bank:
            cid = cluster["cluster_id"]
            first = cluster["sightings"][0] if cluster.get("sightings") else {}
            final_info[cid] = {
                "final_global_name": f"[{first.get('temp_name', cid)}]",
                "refined_visual_description": first.get("desc", ""),
                "chosen_reason": "REVIEW_DISABLED",
            }

    _log(video_tag, "\n---------- Phase C: 批量 Caption 改写 ----------")
    aligned_results = _phase_c_rewrite(pass1_results, global_bank, chunk_mappings, final_info, video_tag)

    with open(aligned_out_path, 'w', encoding='utf-8') as f:
        json.dump(aligned_results, f, ensure_ascii=False, indent=2)

    lite_bank = []
    for cluster in global_bank:
        cid = cluster["cluster_id"]
        info = final_info.get(cid, {})
        lite_bank.append({
            "cluster_id": cid,
            "角色名": info.get("final_global_name", ""),
            "外貌特征": info.get("refined_visual_description", ""),
            "出现次数": len(cluster.get("sightings", [])),
            "选名理由": info.get("chosen_reason", ""),
        })
    with open(bank_out_path, 'w', encoding='utf-8') as f:
        json.dump(lite_bank, f, ensure_ascii=False, indent=2)

    _log(video_tag, "\n" + "=" * 60)
    _log(video_tag, "🎉 Pass 2 完成！最终角色名单:")
    for entry in lite_bank:
        _log(video_tag, f"  - {entry['cluster_id']}: {entry['角色名']} ({entry['出现次数']} 次)")
    _log(video_tag, "=" * 60)
    _log(video_tag, f"💾 对齐后 JSON: {aligned_out_path}")
    _log(video_tag, f"💾 全局图鉴:   {bank_out_path}")
    _log(video_tag, f"💾 终审日志:   {review_log_path}")

    return aligned_out_path, bank_out_path
