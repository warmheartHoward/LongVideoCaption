import copy
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple

from .config import PipelineConfig
from .frame_extractor import extract_single_frame_base64
from .token_tracker import TokenTracker
from .utils import clean_json_response


PASS_NAME = "pass2_alignment"
REVIEW_STAGE_NAME = "pass2_review"


def _log(video_tag: str, msg: str) -> None:
    print(f"[{video_tag}] {msg}" if video_tag else msg)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


SYS_PROMPT_ALIGNMENT = """你是电影多模态视觉统筹。视频被切成多个 chunk 处理，同一角色在不同 chunk 中可能被赋予不同的临时名。请判定当前片段里出现的候选角色是否等同于【全局聚类库】中某个已有角色。本阶段只判"身份是否相同"，不命名（权威命名在后续终审环节）。

【综合四类证据做判断】
1. 视觉：对比候选帧与聚类库参考帧的核心不变特征（面部、体型、服饰、物种、标志性外观）。镜头角度、距离、光照、姿态变化属正常差异，聚焦不变特征。
2. 历史临时名：聚类库展示了该角色在过去 chunk 里被叫过的所有临时名（去重列表）。候选的临时名若出现在该列表里，是支持合并的较强线索；但仅靠名字重合不足以合并，需视觉/描述/剧情至少再有一项支持。
3. 外貌描述：候选的描述与库中各 chunk 描述是否吻合。明显矛盾（物种、颜色、显著体征不同）→ 支持 NEW。
4. 剧情上下文：结合前情提要与当前片段剧情，该候选的出场是否与某 cluster 的剧情自然衔接。

【判定结果】
- 多类证据支持合并 → 填 cluster_id，confidence 视证据强度给 60-95
- 视觉/描述/剧情都无决定性证据，或视觉明显冲突 → NEW

请严格输出 JSON：
{
  "multimodal_reasoning_process": [
    {
      "temp_name": "<当前评估的临时角色>",
      "comparison": "<对比了聚类库里哪个 cluster、用了哪些证据（视觉/名字/描述/剧情）、结论>"
    }
  ],
  "chunk_identity_mapping": [
    {
      "temp_name_in_chunk": "<当前片段中的临时名称>",
      "match_result": "<匹配则填 cluster_id（例如 cluster_0003）；新角色填 NEW>",
      "confidence_score": <0-100 的整数>
    }
  ]
}"""


SYS_PROMPT_REVIEW = """你是终审级电影多模态视觉统筹。系统已通过滚动匹配完成初步的角色聚类，现在你收到**全部待终审聚类**的证据（高清视觉帧 + 各角色在不同 chunk 里被赋予的临时名称 + 每次出现时的外貌描述）。请在一次回复中为**每一个**聚类完成下面三件事。**不要做拆分判断**，滚动阶段的聚类结果保持不变。

1. 【权威命名】综合所有证据，为该聚类确定一个**权威全局标准名**。命名权威度排序：
   - 对白/字幕点名的专名（如"周太安"） > 有辨识度的绰号 > 描述性命名（如"红衣男子"） > 泛化编号（"男人1"）
   - 候选名里都不够权威，可基于视觉特征自拟更准确的名字
   - 输出格式：方括号包裹，例如 [周太安]
2. 【描述重写】基于**高清帧**重写该角色的客观视觉描述。滚动阶段使用较低分辨率的帧，可能有错误，以高清帧为准。
3. 【保留名单 preserve_temp_names】**逐 temp_name 决定是否保留原样**。系统提供了该聚类的临时名列表（按出现顺序），请挑出其中**不能**被权威名覆盖的临时名，其余默认替换为权威名。
   - 默认 []（空列表）：所有临时名都可安全替换为权威名（消除别名漂移）。
   - 列入 preserve_temp_names 的典型情形：该临时名承载了**重要的状态/形态/阶段信息**，权威名替换会让 caption 失真。常见模式：
     * 形态切换：临时名捕捉了当前的存在状态（不同状态的临时名分别保留各自形象信息）。
     * 装扮 / 重伤 / 化身 / 变形 / 年龄跨度极大：临时名描述的是当下视觉特征，权威名抹平后失去剧情信息。
     * 临时名本身参与叙事：如"蒙面人"在揭面前后分别出现，统一抹平会丢失"揭面"这一节点的语义。
   - 列入的临时名必须**逐字**来自系统给出的临时名列表（不要自创、不要带方括号变体——原样照写）。
   - 同一聚类中其他临时名仍会被替换为权威名；只有列入此名单的字符串保留原样。
   - 若聚类下**全部**临时名都不该被替换，则把全部临时名都列入。

请严格输出 JSON，`reviews` 数组必须**覆盖全部传入的聚类**，每个聚类一项：
{
  "reviews": [
    {
      "cluster_id": "<传入的 cluster_id 原样返回>",
      "final_global_name": "[xxx]",
      "refined_visual_description": "基于高清帧重写的客观外貌描述",
      "chosen_reason": "选择该名字的理由（源自哪个 sighting 的 temp_name / 为什么权威）",
      "preserve_temp_names": [],
      "preserve_reason": "若 preserve_temp_names 非空，必须解释每个被保留的临时名为何不能用权威名覆盖；为空则填空字符串"
    }
  ]
}"""


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
        raw_chars = (
            data.get("characters_in_chunk")
            or data.get("new_characters_in_chunk")
            or []
        )

        seen_norm: set = set()
        current_chars: list = []
        dup_dropped = 0
        for c in raw_chars:
            norm = _normalize_name(c.get("temp_name", ""))
            if not norm:
                continue
            if norm in seen_norm:
                dup_dropped += 1
                continue
            seen_norm.add(norm)
            current_chars.append(c)

        if dup_dropped > 0:
            _log(video_tag, f"  🧹 [去重] 合并同名 temp_name 条目 {dup_dropped} 个（同 chunk 内视为同一角色）")

        _log(video_tag, f"\n🔍 [Phase A] 分析 {chunk_range} | 发现待定面孔: {len(current_chars)} 个（原始 {len(raw_chars)} 条）")

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
                sightings = g.get("sightings", [])

                seen_norm: set = set()
                name_history: List[str] = []
                for s in sightings:
                    tn = s.get("temp_name", "")
                    norm = _normalize_name(tn)
                    if not norm or norm in seen_norm:
                        continue
                    seen_norm.add(norm)
                    name_history.append(tn)

                desc_lines: List[str] = []
                for s in sightings:
                    desc = s.get("desc", "")
                    if not desc:
                        continue
                    chunk_i = s.get("chunk_i")
                    chunk_label = f"第{chunk_i + 1}段" if isinstance(chunk_i, int) else "?"
                    desc_lines.append(f"  - ({chunk_label}) {desc}")
                desc_block = "\n".join(desc_lines) if desc_lines else "  (无)"

                user_content.append({"type": "text", "text": (
                    f"cluster_id: {cid}\n"
                    f"出现次数: {len(sightings)}\n"
                    f"历史临时名（按出现顺序去重）: {name_history}\n"
                    f"历史外貌描述:\n{desc_block}"
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

            processed_temp_norm: set = set()
            for mapping in align_result.get("chunk_identity_mapping", []):
                temp_name_raw = mapping.get("temp_name_in_chunk")
                match_result = (mapping.get("match_result") or "").strip()
                conf_score = mapping.get("confidence_score", 100)

                if not temp_name_raw:
                    continue
                norm_key = _normalize_name(temp_name_raw)
                if norm_key in processed_temp_norm:
                    _log(video_tag, f"    ⚠️ 模型对 {temp_name_raw!r} 输出了重复映射，忽略重复项")
                    continue
                matched_char = norm_char_lookup.get(norm_key)
                if not matched_char:
                    _log(video_tag, f"    ⚠️ 模型输出的 temp_name 不在当前片段: {temp_name_raw!r}（归一化后={norm_key!r}），忽略")
                    continue
                processed_temp_norm.add(norm_key)

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
                "preserve_temp_names": entry.get("preserve_temp_names", []) or [],
                "preserve_reason": entry.get("preserve_reason", ""),
            }

    pending = [
        c for c in global_bank
        if c["cluster_id"] not in reviewed_cluster_ids and c.get("sightings")
    ]
    skipped_empty = [
        c["cluster_id"] for c in global_bank
        if c["cluster_id"] not in reviewed_cluster_ids and not c.get("sightings")
    ]
    for cid in skipped_empty:
        _log(video_tag, f"  ⚠️ [Phase B] {cid} 无 sightings，跳过")
        reviewed_cluster_ids.add(cid)

    if not pending:
        _log(video_tag, "  ⏭️ [Phase B] 无待复核聚类")
        return final_info

    batch_size = cfg.pass2_review_batch_size
    if batch_size and batch_size > 0:
        batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
        size_label = str(batch_size)
    else:
        batches = [pending]
        size_label = "ALL"

    _log(video_tag, f"\n🔎 [Phase B] 终审 {len(pending)} 个聚类，分 {len(batches)} 批 (batch_size={size_label})")

    for batch_idx, batch in enumerate(batches, 1):
        _log(video_tag, f"\n  ── 批次 {batch_idx}/{len(batches)} | 本批 {len(batch)} 个聚类 ──")

        user_content: List[dict] = [
            {"type": "text", "text": (
                f"以下是 {len(batch)} 个待终审聚类的全部证据。请按 system 指令在一次回复中为每个聚类输出一项。"
            )}
        ]
        for cluster in batch:
            cid = cluster["cluster_id"]
            sightings = cluster["sightings"]
            sel_indices = _select_review_sighting_indices(len(sightings), cfg.pass2_review_max_frames_per_cluster)
            temp_name_list = [s.get("temp_name", "") for s in sightings]
            _log(video_tag, f"  ➕ 装载 {cid} | sightings={len(sightings)} | 高清帧采样索引={sel_indices}")

            user_content.append({"type": "text", "text": (
                f"\n========== {cid} ==========\n"
                f"该聚类共 {len(sightings)} 次出现。临时名列表（按出现顺序）: {temp_name_list}"
            )})

            for i_s, s in enumerate(sightings):
                user_content.append({"type": "text", "text": (
                    f"--- {cid} sighting #{i_s} ---\n"
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
                        _log(video_tag, f"    ⚠️ {cid} sighting #{i_s} 高清帧提取失败: {s.get('anchor_ts', '')}")

        user_content.append({"type": "text", "text": "请输出 JSON 终审结果，reviews 必须覆盖本批传入的全部聚类。"})

        batch_result = None
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

                batch_result = json.loads(clean_json_response(response.choices[0].message.content))
                _log(video_tag, f"  ✅ 批次 {batch_idx} 终审完成 (耗时: {time.time() - start_time:.2f}s)")
                break
            except Exception as e:
                _log(video_tag, f"  ⚠️ 终审 API 异常 (尝试 {attempt}/{cfg.max_retries}): {e}")
                if attempt == cfg.max_retries:
                    _log(video_tag, f"  ❌ 批次 {batch_idx} 调用失败，本批聚类将回退到首个 sighting 的 temp_name")
                time.sleep(2)

        items_by_cid: Dict[str, dict] = {}
        if batch_result is not None:
            items = batch_result.get("reviews") if isinstance(batch_result, dict) else batch_result
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        cid = it.get("cluster_id")
                        if cid:
                            items_by_cid[cid] = it

        for cluster in batch:
            cid = cluster["cluster_id"]
            sightings = cluster["sightings"]
            item = items_by_cid.get(cid)

            cluster_temp_names = [s.get("temp_name", "") for s in sightings]
            cluster_temp_norm_to_raw: Dict[str, str] = {}
            for tn in cluster_temp_names:
                norm = _normalize_name(tn)
                if norm and norm not in cluster_temp_norm_to_raw:
                    cluster_temp_norm_to_raw[norm] = tn

            if item is None:
                fallback_name = f"[{sightings[0].get('temp_name', cid)}]"
                fallback_desc = sightings[0].get("desc", "")
                fallback_preserve = list(dict.fromkeys(cluster_temp_names))
                final_info[cid] = {
                    "final_global_name": fallback_name,
                    "refined_visual_description": fallback_desc,
                    "chosen_reason": "FALLBACK: 批量终审未返回该聚类",
                    "preserve_temp_names": fallback_preserve,
                    "preserve_reason": "FALLBACK 状态下保留全部临时名，避免误改",
                }
                review_log.append({
                    "cluster_id": cid,
                    "final_global_name": fallback_name,
                    "refined_visual_description": fallback_desc,
                    "chosen_reason": "FALLBACK: 批量终审未返回该聚类",
                    "preserve_temp_names": fallback_preserve,
                    "preserve_reason": "FALLBACK 状态下保留全部临时名，避免误改",
                    "reviewed_at": _now_iso(),
                    "status": "fallback",
                })
                reviewed_cluster_ids.add(cid)
                _log(video_tag, f"  ❌ {cid} 未在批量结果中，fallback 到首个 sighting 名（保留全部临时名不替换）")
                continue

            final_name = item.get("final_global_name") or f"[{sightings[0].get('temp_name', cid)}]"
            refined_desc = item.get("refined_visual_description") or sightings[0].get("desc", "")
            chosen_reason = item.get("chosen_reason", "")
            raw_preserve = item.get("preserve_temp_names") or []
            preserve_reason = item.get("preserve_reason", "")

            preserve_validated: List[str] = []
            preserve_dropped: List[str] = []
            seen_norm: set = set()
            if isinstance(raw_preserve, list):
                for tn in raw_preserve:
                    if not isinstance(tn, str):
                        continue
                    norm = _normalize_name(tn)
                    if not norm:
                        continue
                    if norm in seen_norm:
                        continue
                    if norm not in cluster_temp_norm_to_raw:
                        preserve_dropped.append(tn)
                        continue
                    seen_norm.add(norm)
                    preserve_validated.append(cluster_temp_norm_to_raw[norm])

            if preserve_validated:
                replace_tag = (
                    "🚫全部保留" if len(preserve_validated) == len(cluster_temp_norm_to_raw)
                    else f"⚠️部分保留 {len(preserve_validated)}/{len(cluster_temp_norm_to_raw)}"
                )
            else:
                replace_tag = "✅全部替换"
            _log(video_tag, f"  📝 {cid} -> {final_name} | {replace_tag}")
            if chosen_reason:
                _log(video_tag, f"     选名理由: {chosen_reason}")
            if preserve_validated:
                _log(video_tag, f"     保留临时名: {preserve_validated}")
                if preserve_reason:
                    _log(video_tag, f"     保留理由: {preserve_reason}")
            if preserve_dropped:
                _log(video_tag, f"     ⚠️ 模型输出了不在该聚类临时名列表中的保留项，已忽略: {preserve_dropped}")

            final_info[cid] = {
                "final_global_name": final_name,
                "refined_visual_description": refined_desc,
                "chosen_reason": chosen_reason,
                "preserve_temp_names": preserve_validated,
                "preserve_reason": preserve_reason,
            }
            review_log.append({
                "cluster_id": cid,
                "final_global_name": final_name,
                "refined_visual_description": refined_desc,
                "chosen_reason": chosen_reason,
                "preserve_temp_names": preserve_validated,
                "preserve_reason": preserve_reason,
                "reviewed_at": _now_iso(),
                "status": "ok",
            })
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

        pairs: Dict[str, str] = {}
        for temp_name, cid in chunk_map.items():
            info = final_info.get(cid)
            if not info:
                continue
            final_name = info.get("final_global_name", "")
            if not final_name or not temp_name:
                continue
            preserve_set = {_normalize_name(t) for t in info.get("preserve_temp_names") or []}
            if _normalize_name(temp_name) in preserve_set:
                continue
            stripped = temp_name.strip()
            if stripped and not (stripped.startswith("[") and stripped.endswith("]")):
                pairs.setdefault(f"[{stripped}]", final_name)
            pairs.setdefault(temp_name, final_name)

        if pairs:
            sorted_olds = sorted(pairs.keys(), key=len, reverse=True)
            pattern = re.compile("|".join(re.escape(o) for o in sorted_olds))
            for ev in events:
                caption = ev.get("step3_synthesized_dense_caption", "")
                if not caption:
                    continue
                hits = pattern.findall(caption)
                if hits:
                    total_replacements += len(hits)
                    ev["step3_synthesized_dense_caption"] = pattern.sub(
                        lambda m: pairs[m.group(0)], caption
                    )

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

    with open(pass1_json_path, 'r', encoding='utf-8') as f:
        pass1_results = json.load(f)

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
    elif payload:
        _log(video_tag, "⚠️ 检测到旧版 pass2 进度文件（schema 不匹配），将重新开始滚动聚类")

    review_log: list = []
    reviewed_cluster_ids: set = set()
    if os.path.exists(review_log_path):
        try:
            with open(review_log_path, 'r', encoding='utf-8') as f:
                review_log = json.load(f).get("reviewed_clusters", []) or []
            reviewed_cluster_ids = {e.get("cluster_id") for e in review_log if e.get("cluster_id")}
        except Exception as e:
            _log(video_tag, f"⚠️ 读取 pass2_review_log.json 失败: {e}（从头复核）")
            review_log = []
            reviewed_cluster_ids = set()

    bank_cluster_ids = {c["cluster_id"] for c in global_bank}
    phase_a_done = processed_count >= len(pass1_results)
    phase_b_done = phase_a_done and (not bank_cluster_ids or bank_cluster_ids.issubset(reviewed_cluster_ids))
    phase_c_done = os.path.exists(aligned_out_path) and os.path.exists(bank_out_path)

    _log(video_tag, "\n" + "=" * 60)
    _log(video_tag, f"🚀 Pass 2 状态: "
                    f"A={'✅' if phase_a_done else '⏳'} ({processed_count}/{len(pass1_results)} chunk) | "
                    f"B={'✅' if phase_b_done else '⏳'} ({len(reviewed_cluster_ids)}/{len(bank_cluster_ids)} cluster) | "
                    f"C={'✅' if phase_c_done else '⏳'}")
    _log(video_tag, "   断点续跑（删文件触发对应阶段重跑）:")
    _log(video_tag, "     删 pass2_aligned.json / pass2_global_bank.json → 仅重跑 Phase C")
    _log(video_tag, "     删 pass2_review_log.json                       → 重跑 Phase B + C")
    _log(video_tag, "     删 pass2_progress.json                         → 重跑 Phase A + B + C")
    _log(video_tag, "=" * 60)

    if phase_a_done and phase_b_done and phase_c_done:
        _log(video_tag, "⏭️  全部阶段已完成，跳过 Pass 2。")
        return aligned_out_path, bank_out_path

    if not phase_a_done:
        _log(video_tag, "\n---------- Phase A: 滚动聚类 ----------")
        if processed_count > 0:
            _log(video_tag, f"🔄 续跑 Phase A: 已完成 {processed_count}/{len(pass1_results)} chunk | 当前聚类 {len(global_bank)} 个")
        processed_count, accumulated_story = _phase_a_rolling(
            cfg, video_path, pass1_results, global_bank, chunk_mappings,
            processed_count, accumulated_story, low_conf_flags,
            progress_path, client, token_tracker, video_tag,
        )
        _log(video_tag, f"\n✅ Phase A 完成: 共聚类 {len(global_bank)} 个角色 | 低置信软合并 {len(low_conf_flags)} 条")
    else:
        _log(video_tag, f"\n⏭️  Phase A 已完成（{len(global_bank)} 个聚类），跳过")

    if cfg.pass2_review_enable:
        _log(video_tag, "\n---------- Phase B: 高清终审 ----------")
        if reviewed_cluster_ids:
            _log(video_tag, f"🔄 已有终审日志: 已复核 {len(reviewed_cluster_ids)} 个聚类")
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
                "preserve_temp_names": [],
                "preserve_reason": "",
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
            "保留原样的临时名": info.get("preserve_temp_names", []),
            "保留理由": info.get("preserve_reason", ""),
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
