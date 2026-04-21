import copy
import json
import os
import time
from typing import Tuple

from .config import PipelineConfig
from .frame_extractor import extract_single_frame_base64
from .token_tracker import TokenTracker
from .utils import clean_json_response


STAGE_NAME = "stage2_alignment"

SYS_PROMPT_ALIGNMENT = """你是一个顶级的电影多模态视觉统筹。
当前视频正在分段处理，同一个角色可能会被赋予不同的【临时名称】。
你的任务是：接收【已确立的全局图鉴】与【当前片段的新角色图片】，进行严谨的身份同一性判决。

【⚠️ 严禁发生的错误（Anti-Patterns）】
1. 严禁“望文生义”：绝对不要因为临时名字里都有“公主”或“勇士”就判定为同一人。必须 100% 依赖图像特征！
2. 警惕“同类不同体”：如果两只猫的颜色、眼睛形状有显著差异，即使都是猫，也必须判定为 NEW。
3. 宁缺毋滥原则：当且仅当你对两者的视觉特征有 80% 以上的把握吻合时，才进行合并。如果有严重疑虑，坚决判定为 NEW。

【💡 优秀推理范例学习】
假设图鉴中有 [皮皮]（特征：黑色圆润，巨大空洞眼）。当前片段出现 [大眼鼠勇士]（特征：身披铠甲，巨大空洞眼）。
✅ 正确的 step3_comparison_logic：“虽然穿了铠甲且被称为鼠，但其核心面部特征（占据半脸的巨大空洞眼）与图鉴中的 [皮皮] 完全一致。结合剧情，判定为同一人。”

请严格输出 JSON：
{
  "multimodal_reasoning_process": [
    {
      "temp_name": "<当前评估的临时角色>",
      "step1_bank_visual_features": "<它最像全局图鉴里的谁？详细列出图鉴里那个角色的核心视觉特征。若完全不像任何人，填无>",
      "step2_current_visual_features": "<仔细观察当前角色的截图，列出其客观视觉特征>",
      "step3_comparison_logic": "<对比 step1 和 step2，结合动作逻辑，给出判定理由>"
    }
  ],
  "chunk_identity_mapping": [
    {
      "temp_name_in_chunk": "<当前片段中的临时名称>",
      "match_result": "<若是老角色，填其 global_standard_name；若是新角色，填 NEW>",
      "assigned_global_name": "<最终分配的全局标准名（带方括号）>",
      "confidence_score": <0-100的整数，表示你对这次判定的视觉把握有多大>
    }
  ]
}"""


def _load_progress(progress_path: str, bank_path: str):
    aligned_results = None
    bank_with_b64 = []
    processed_count = 0
    accumulated_story = ""

    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            aligned_results = payload.get("aligned_results")
            processed_count = payload.get("processed_chunk_count", 0)
            accumulated_story = payload.get("accumulated_story", "")
        except Exception as e:
            print(f"⚠️ 读取 stage2 进度失败: {e}")

    if os.path.exists(bank_path):
        try:
            with open(bank_path, 'r', encoding='utf-8') as f:
                bank_with_b64 = json.load(f)
        except Exception as e:
            print(f"⚠️ 读取 stage2 图鉴进度失败: {e}")

    return aligned_results, bank_with_b64, processed_count, accumulated_story


def _save_progress(progress_path: str, bank_path: str,
                   aligned_results, bank_with_b64, processed_count, accumulated_story):
    tmp_progress = progress_path + ".tmp"
    with open(tmp_progress, 'w', encoding='utf-8') as f:
        json.dump({
            "aligned_results": aligned_results,
            "processed_chunk_count": processed_count,
            "accumulated_story": accumulated_story,
        }, f, ensure_ascii=False, indent=2)
    os.replace(tmp_progress, progress_path)

    tmp_bank = bank_path + ".tmp"
    with open(tmp_bank, 'w', encoding='utf-8') as f:
        json.dump(bank_with_b64, f, ensure_ascii=False)
    os.replace(tmp_bank, bank_path)


def run_stage2(
    cfg: PipelineConfig,
    video_path: str,
    stage1_json_path: str,
    run_dir: str,
    client,
    token_tracker: TokenTracker = None,
) -> Tuple[str, str]:
    os.makedirs(run_dir, exist_ok=True)
    aligned_out_path = os.path.join(run_dir, "stage2_aligned.json")
    bank_out_path = os.path.join(run_dir, "stage2_global_bank.json")
    progress_path = os.path.join(run_dir, "stage2_progress.json")
    bank_progress_path = os.path.join(run_dir, "stage2_bank_progress.json")

    if os.path.exists(aligned_out_path) and os.path.exists(bank_out_path):
        print("\n" + "=" * 60)
        print("⏭️  Stage 2 终产物已存在，跳过对齐。")
        print("=" * 60)
        return aligned_out_path, bank_out_path

    with open(stage1_json_path, 'r', encoding='utf-8') as f:
        stage1_results = json.load(f)

    print("\n" + "=" * 60)
    print("🚀 启动 Stage 2: 多模态滚动实体消解 (带置信度拦截)")
    print("=" * 60)

    aligned_results, global_bank, processed_count, accumulated_story = _load_progress(progress_path, bank_progress_path)
    if aligned_results is None:
        aligned_results = copy.deepcopy(stage1_results)
    if processed_count > 0:
        print(f"🔄 检测到 stage2 断点：已完成 {processed_count} 个 chunk 的对齐，图鉴规模 {len(global_bank)}")

    for i in range(processed_count, len(aligned_results)):
        chunk = aligned_results[i]
        chunk_range = chunk.get("chunk_time_range", f"Chunk_{i+1}")
        data = chunk.get("data", {})

        current_story = data.get("chunk_summary", "")
        current_chars = data.get("new_characters_in_chunk", []) or []

        print(f"\n🔍 分析 {chunk_range} | 发现待定面孔: {len(current_chars)} 个")

        if not current_chars:
            accumulated_story += f"第{i+1}段: {current_story}\n"
            print("  ⏭️ 无需对齐，跳过")
            processed_count = i + 1
            _save_progress(progress_path, bank_progress_path, aligned_results, global_bank, processed_count, accumulated_story)
            continue

        user_content = [
            {"type": "text", "text": f"【前情提要】\n{accumulated_story if accumulated_story else '无前情，视频开篇。'}\n"},
            {"type": "text", "text": f"【当前片段剧情】\n{current_story}\n"},
        ]

        if global_bank:
            user_content.append({"type": "text", "text": "====================\n【已确立的全局图鉴】 (老熟人列表):"})
            for g_char in global_bank:
                user_content.append({"type": "text", "text": f"全局标准名: {g_char['global_standard_name']}\n外貌描述: {g_char['desc']}"})
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{g_char['b64']}"}})
        else:
            user_content.append({"type": "text", "text": "====================\n【已确立的全局图鉴】: 当前为空。"})

        user_content.append({"type": "text", "text": "====================\n【当前片段出现的疑似新角色】 (等待判决):"})
        for c_char in current_chars:
            b64_img = extract_single_frame_base64(
                video_path, c_char.get("anchor_timestamp", ""), max_width=cfg.frame_max_width
            )
            c_char["_temp_b64"] = b64_img

            user_content.append({"type": "text", "text": f"临时名称: {c_char['temp_name']}\n阶段描述: {c_char['visual_description']}"})
            if b64_img:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})

        user_content.append({"type": "text", "text": "请输出 JSON 判决结果。"})

        mapping_dict = {}
        for attempt in range(1, cfg.max_retries + 1):
            try:
                start_time = time.time()
                response = client.chat.completions.create(
                    model=cfg.model_name,
                    max_tokens=cfg.stage2_max_tokens,
                    temperature=cfg.stage2_temperature,
                    messages=[
                        {"role": "system", "content": SYS_PROMPT_ALIGNMENT},
                        {"role": "user", "content": user_content},
                    ],
                )
                if response.usage and token_tracker is not None:
                    token_tracker.record(STAGE_NAME, response.usage)
                    print(f"  📊 [Token] Prompt: {response.usage.prompt_tokens} | Total: {response.usage.total_tokens}")

                align_result = json.loads(clean_json_response(response.choices[0].message.content))
                print(f"  ✅ VLM 推理成功 (耗时: {time.time() - start_time:.2f}s)")

                for mapping in align_result.get("chunk_identity_mapping", []):
                    temp_name = mapping.get("temp_name_in_chunk")
                    assigned_name = mapping.get("assigned_global_name")
                    match_result = mapping.get("match_result")
                    conf_score = mapping.get("confidence_score", 100)

                    if match_result != "NEW" and conf_score < cfg.stage2_confidence_threshold:
                        print(f"    ⚠️ 置信度过低 ({conf_score})，拦截合并: {temp_name} 被强制判定为 NEW")
                        match_result = "NEW"
                        assigned_name = temp_name

                    if temp_name and assigned_name and match_result != "NEW":
                        mapping_dict[temp_name] = assigned_name
                        print(f"    🔗 映射: {temp_name} -> {assigned_name} (置信度: {conf_score})")

                    if match_result == "NEW":
                        print(f"    🌟 新角色: {assigned_name} (置信度: {conf_score})")
                        matched_char = next((c for c in current_chars if c["temp_name"] == temp_name), None)
                        if matched_char:
                            global_bank.append({
                                "global_standard_name": assigned_name,
                                "desc": matched_char["visual_description"],
                                "b64": matched_char["_temp_b64"],
                            })
                break

            except Exception as e:
                print(f"  ⚠️ API 请求异常 (尝试 {attempt}/{cfg.max_retries}): {e}")
                if attempt == cfg.max_retries:
                    print("  ❌ 跳过该 Chunk 对齐。")
                time.sleep(2)

        if mapping_dict:
            replace_count = 0
            for ev in data.get("events", []):
                old_caption = ev.get("step3_synthesized_dense_caption", "")
                new_caption = old_caption
                for old_name, global_name in mapping_dict.items():
                    if old_name in new_caption:
                        new_caption = new_caption.replace(old_name, global_name)
                        replace_count += 1
                ev["step3_synthesized_dense_caption"] = new_caption
            print(f"  🔄 文本已更新: 修正了 {replace_count} 处身份漂移。")

        for c_char in current_chars:
            c_char.pop("_temp_b64", None)

        accumulated_story += f"第{i+1}段: {current_story}\n"
        processed_count = i + 1
        _save_progress(progress_path, bank_progress_path, aligned_results, global_bank, processed_count, accumulated_story)

    print("\n" + "=" * 60)
    print("🎉 Stage 2 全局对齐完毕！图鉴收录: ", [c['global_standard_name'] for c in global_bank])
    print("=" * 60)

    with open(aligned_out_path, 'w', encoding='utf-8') as f:
        json.dump(aligned_results, f, ensure_ascii=False, indent=2)

    lite_bank = [{"角色名": c["global_standard_name"], "外貌特征": c["desc"]} for c in global_bank]
    with open(bank_out_path, 'w', encoding='utf-8') as f:
        json.dump(lite_bank, f, ensure_ascii=False, indent=2)

    print(f"💾 [核心数据] 对齐后的纯净 JSON 已保存至: {aligned_out_path}")
    print(f"💾 [附属数据] 全局图鉴文本设定集 已保存至: {bank_out_path}")

    return aligned_out_path, bank_out_path
