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
from .token_tracker import TokenTracker
from .utils import format_timestamp, parse_timestamp_to_seconds


STAGE_NAME = "stage1_perception"


def run_stage1(
    cfg: PipelineConfig,
    video_path: str,
    run_dir: str,
    client,
    token_tracker: Optional[TokenTracker] = None,
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
                print("\n=========================================")
                print("🔄 检测到历史运行记录，尝试恢复断点...")

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
                    print(f"⏩ 成功加载进度，将从 {format_timestamp(chunk_start)} 继续执行。")
                else:
                    print("⚠️ 历史记录格式异常，将从头开始。")
                    global_results = []
                    chunk_start = 0.0
                    history_summaries = []
        except Exception as e:
            print(f"⚠️ 读取断点文件失败: {e}，将从头开始。")
            global_results = []
            chunk_start = 0.0
            history_summaries = []

    while chunk_start < total_duration:
        chunk_end = min(chunk_start + cfg.chunk_duration_sec, total_duration)
        chunk_name = f"[{format_timestamp(chunk_start)} - {format_timestamp(chunk_end)}]"
        print(f"\n=========================================")
        print(f"🚀 Pass 1 - 正在处理视频块: {chunk_name} | 模式: {cfg.input_payload_format}")

        user_content = []
        timestamps_str_list = []

        if cfg.input_payload_format == "video_base64":
            valid_timestamps, video_b64 = get_raw_chunk_video_base64(
                video_path, chunk_start, chunk_end, cfg.frame_max_width, cfg.target_fps, temp_dir=temp_dir
            )
            if not video_b64:
                chunk_start = chunk_end
                continue
            timestamps_str_list = [format_timestamp(t) for t in valid_timestamps]
            user_content.append({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}})
        else:
            target_timestamps = get_target_timestamps(
                video_path, chunk_start, chunk_end,
                cfg.frame_extraction_strategy, cfg.scene_detect_threshold, cfg.max_frames_per_chunk,
            )
            valid_timestamps, base64_frames = get_base64_frames(
                video_path, target_timestamps, cfg.frame_max_width, cfg.frame_jpg_quality,
            )
            if not base64_frames:
                chunk_start = chunk_end
                continue
            timestamps_str_list = [format_timestamp(t) for t in valid_timestamps]
            for t_str, b64 in zip(timestamps_str_list, base64_frames):
                user_content.append({"type": "text", "text": f"画面时间 {t_str}:"})
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})

        start_str = format_timestamp(chunk_start)
        if start_str not in timestamps_str_list:
            timestamps_str_list.insert(0, start_str)

        timestamps_str = ", ".join(timestamps_str_list)

        sys_prompt = f"""你是一个资深的影视内容分析家与专业的视频数据打标人员。当前正在分析视频片段：{chunk_name}。

            【🔴 核心分析维度与思维链要求 🔴】
            1. 强制思维链（CoT）：必须先在 `scene_analysis_scratchpad` 中打草稿，完成时间轴规划与核心事件梳理，再输出具体分析。
            2. 客观画面描述：专注于肉眼可见的物理细节（动作、神态、空间位置与环境交互），绝不能脑补看不清的微观细节或添加主观色彩。
            3. 剧情与情绪归因：基于【前情提要】和客观动作，推导人物真实的内心动机、情绪状态及对剧情的推动作用。
            4. 高密度综合描述（Dense Caption）：将客观画面与主观归因自然融合，输出高质量的剧本级描述。

            【🟡 角色命名与视觉建档（为后期身份对齐做准备） 🟡】
            5. 代号符号化：给判定为推动剧情的核心角色起个简短自然的代号（如已知真名用真名，无真名用自然代号）。**极其重要：在 step3 中提及他们时，必须使用方括号包裹代号（例如：[红衣女人]、[李雷]）**。背景路人甲无需括号。
            6. 新面孔建档：如果本片段中出现了【新的/重要的】核心面孔，请在 `new_characters_in_chunk` 中为他/她建立视觉档案，并给出一个最能看清他/她的时间戳。不要去猜测他是不是之前出现过的人，只管记录当前画面的特征。

            【🟢 时间与格式强制约束（绝对服从）】
            7. 时间轴切分：事件的 `start_time`、`end_time` 以及角色建档的 `anchor_timestamp` 必须严格从以下列表中选取（禁止自我捏造）：
            可用时间戳白名单：{timestamps_str}

            请严格输出 JSON，禁止用 markdown 代码块包裹。"""

        usr_prompt = f"""【前情提要】\n{previous_context}

            请严格按以下字段顺序输出 JSON（注意：生成顺序即推理顺序，绝对不可乱序）：
            {{
            "scene_analysis_scratchpad": "<思维链草稿：1. 从白名单中规划时间切分。2. 梳理核心事件和出现的角色代号。>",
            "new_characters_in_chunk": [
                {{
                "temp_name": "<带方括号的代号，例如 [大眼小恶魔]>",
                "visual_description": "<纯客观的视觉特征描述：衣着颜色、发型、面部特征、特殊配饰等。严禁写剧情动机。>",
                "anchor_timestamp": "<从可用时间戳白名单中，选一个该角色面部/特征最清晰的时间点>"
                }}
            ],
            "events": [
                {{
                "start_time": "<从白名单列表选取>",
                "end_time": "<从白名单列表选取>",
                "step1_objective_visual": "<客观画面描述：他/她实际上做了什么？环境如何？描述真实可见的动作、神态和位置交互。>",
                "step2_contextual_reasoning": "<剧情与情绪归因：结合【前情提要】，推导角色此时的心理动因、情绪状态及情节张力。>",
                "step3_synthesized_dense_caption": ""<剧本级高密度融合：拒绝机械拼接！请用充满文学张力的笔触，将 step2 的心理动机、情绪状态作为驱动力，自然地融入到 step1 的物理动作描写中。确保包含[带括号的代号]，字数充实，画面感极强。>"",
                "key_frame_times": ["<提取最能代表该事件的时间戳>"]
                }}
            ],
            "chunk_summary": "<宏观提炼：总结本片段核心剧情脉络、关键动作与情绪转折，作为下一段的前情提要。>"
            }}"""

        user_content.append({"type": "text", "text": usr_prompt})

        next_start = chunk_start + (cfg.chunk_duration_sec * 0.8)

        try:
            chunk_data = request_llm_with_retry(
                client=client, model=cfg.model_name,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}],
                max_tokens=cfg.llm_max_tokens, temperature=cfg.llm_temperature,
                max_retries=cfg.max_retries, chunk_name=chunk_name,
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
                    print(f"🔗 [动态接力] 本段动作自然结束于 {format_timestamp(next_start)}，以此为下段起点。")
                else:
                    print(f"⚠️ [接力异常] 返回末尾时间 {format_timestamp(last_end_sec)} 不合理，启动 80% 安全重叠兜底推进。")

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
                print("⚠️ [接力异常] 未提取到事件，启动 80% 安全重叠兜底推进。")
                previous_context = f"【系统提示】: 上一片段解析异常，请直接从 {format_timestamp(next_start)} 开始重新捕捉动作。"

        except Exception as e:
            print(f"❌ [严重跳过] Chunk {chunk_name} 多次尝试均失败: {e}")

        chunk_start = next_start

    return pass1_output_path
