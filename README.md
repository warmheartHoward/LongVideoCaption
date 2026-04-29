# Long Video Caption v2

长视频结构化打标流水线，五个阶段串行：**视觉感知 → 身份对齐 → 章节聚合 → 帧级精修 → 全局润色**。
支持文件夹批量、多线程并发、按内容稳定路径的细粒度断点续打、分阶段 Token 统计。

---

## 整体架构

```mermaid
flowchart TB
    CLI["CLI main.py<br/>--input、--output、--workers"] --> Runner

    subgraph Runner["runner.run_batch"]
        Disc["discover_videos<br/>递归扫描 + mtime 排序"]
        Pool{"ThreadPoolExecutor<br/>max_workers 并发"}
        Disc --> Pool
    end

    Pool -.->|"每视频一个 worker 线程"| Pipe1["process_single_video"]
    Pool -.->|"..."| PipeN["process_single_video"]

    subgraph PerVideo["单视频流水线 pipeline.py"]
        direction TB
        P1["Pass 1<br/>视觉感知"]
        P2["Pass 2<br/>身份对齐"]
        P3["Pass 3<br/>章节聚合"]
        S2["Stage 2<br/>帧级精修"]
        S3["Stage 3<br/>全局润色"]
        P1 --> P2 --> P3 --> S2 --> S3
    end

    Pipe1 --> PerVideo

    subgraph Artifacts["磁盘产物 output/video/hyper_sig"]
        F1["pass1_progress.json<br/>pass1_confidence.json"]
        F2["pass2_progress.json<br/>pass2_review_log.json"]
        F3["pass2_aligned.json<br/>pass2_global_bank.json"]
        F4["pass3_final.json<br/>pass3_confidence.json"]
        F5["stage2_refined.json"]
        F6["★ stage3_polished.json"]
    end

    P1 -.->|"chunk 级增量写 + 健康度报告"| F1
    P2 -.->|"Phase A、B 增量写"| F2
    P2 -.->|"Phase C 终产物"| F3
    P3 -.->|"单次调用 + 健康度报告"| F4
    S2 -.->|"per-event 增量写"| F5
    S3 -.->|"单次调用"| F6

    PerVideo --> Agg["聚合<br/>_aggregate_token_usage.json<br/>_run_summary.json"]
```

> ★ `stage3_polished.json` 是最终交付产物；`pass3_final.json` 是不带 frame-level 精修的中间稳定版本（也可单独消费）。

---

## 各阶段细节

### Pass 1 — 视觉感知（chunk 级 LLM 多模态调用）

```mermaid
flowchart TB
    Init(["video 入口<br/>整片跑一次 pyscenedetect<br/>得到全片 scene 列表"]) --> Loop{"chunk_start 小于 video_duration?"}

    Loop -->|"yes"| Whitelist["从全片 scene 列表抽取<br/>落在 [chunk_start, chunk_end] 内的镜头切换点<br/>并加入 chunk_start/chunk_end<br/>作为 timestamps_whitelist"]
    Whitelist --> Extract["抽帧<br/>scenedetect 复用全片结果 + get_base64_frames<br/>或 video_base64 整段"]
    Extract --> Build["拼 prompt<br/>sys + 前情提要 + 角色滚动史 + 帧序列 + 白名单"]
    Build --> LLM["LLM 多模态调用<br/>JSON 严格输出，event 起止必须取自白名单"]
    LLM --> Parse["解析 events、chunk_summary、characters_in_chunk"]
    Parse --> Snap["_validate_and_snap_event_times<br/>把模型输出 snap 到最近的白名单点"]
    Snap --> Revise["prev_event_revision<br/>跨 chunk 截断动作合并"]
    Revise --> Relay{"动态接力<br/>5 ≤ last_end_sec ≤ chunk_end+10 ?"}
    Relay -->|"yes"| NextEnd["chunk_start = last_end"]
    Relay -->|"no"| Overlap["chunk_start = chunk_end - 20%重叠"]
    NextEnd --> Append["追加到 pass1_progress.json"]
    Overlap --> Append
    Append --> Loop

    Loop -->|"done"| Out(["pass1_progress.json<br/>chunks 数组、每个含 data.events 列表<br/>pass1_confidence.json"])
```

**关键不变量**：
- events **首尾相连**逐字相等（`events[i+1].start_time == events[i].end_time`），由 prompt 强制约束。
- 整片只跑一次 `pyscenedetect.detect`（`frame_extractor.detect_scenes`），结果在所有 chunk 间复用 —— 避免 N× 重复扫描。
- `timestamps_whitelist` 来自当前 chunk 内的镜头切换时间点，并显式包含 `chunk_start` / `chunk_end`；重叠接力时还会加入上一段 `last_end`，确保模型只能选择可校验的边界。
- `pass1_confidence.json` 记录事件时间轴覆盖、相邻 event 连续性、后处理修补/丢弃统计、event 时长分布、内容缺失、关键帧越界和时间戳吸附偏移。

---

### Pass 2 — 滚动身份对齐（三阶段）

```mermaid
flowchart TB
    Read["读 pass1_progress.json"] --> PA

    subgraph PhaseA["Phase A · 滚动聚类 per-chunk 断点"]
        PA["chunk 内的临时角色"] --> Compare["与已有 cluster 多模态对比<br/>多 sighting 高清帧"]
        Compare --> Conf{"confidence ≥ pass2_confidence_threshold ?"}
        Conf -->|"是"| Merge["加入已有 cluster"]
        Conf -->|"否"| New["新建 cluster"]
        Merge --> WriteA["pass2_progress.json"]
        New --> WriteA
    end

    WriteA --> PB

    subgraph PhaseB["Phase B · 终审 high-res review"]
        PB["对每个 cluster<br/>收集多个 sighting"] --> HiRes["高清帧 + 临时名候选"]
        HiRes --> Review["LLM 批量复核<br/>final_global_name + refined_desc"]
        Review --> Preserve["逐 temp_name 决定<br/>preserve_temp_names<br/>强默认全替换"]
        Preserve --> WriteB["pass2_review_log.json"]
    end

    WriteB --> PC

    subgraph PhaseC["Phase C · 批量 caption 改写"]
        PC["遍历每个 chunk 的 events"] --> Regex["长度优先正则替换<br/>跳过 preserve_temp_names"]
        Regex --> WriteC["pass2_aligned.json<br/>pass2_global_bank.json lite"]
    end
```

**关键约束**：
- Phase A 决策一旦写盘即定；Phase B **不再做拆分**，只决定是否替换。
- `preserve_temp_names` 只接受 cluster 实际包含的 temp_name；模型幻觉的临时名会被丢弃并打印 warn。
- Phase B/C 都会逐 sighting 应用 preserve 名单，避免"同一角色不同形态"被错误统一（如活体↔骸骨、揭面前↔揭面后）。

---

### Pass 3 — 章节聚合 + 层级装配（单次 LLM 调用）

输入由三层组合而成（详见 [pass3.py](longvideocaption/pass3.py)）：

```
【全局角色图鉴】           ← 来自 pass2_global_bank.json
- [李雷]: 二十多岁男性...
- [韩梅梅]: ...

【分段叙事摘要（按 chunk 顺序）】 ← 来自每个 chunk 的 chunk_summary
第1段 [00:00:00-00:01:00]: ...
第2段 [00:01:00-00:02:00]: ...

【完整底层时间轴】          ← 每个 event 的 step3_synthesized_dense_caption
[00:00:01.250] - [00:00:08.500] : ...
...
```

模型输出 chapters 切分后，本地 `_validate_chapters` 会先把章节边界吸附到真实 event 边界、修正相邻章节 gap/overlap 并删除非法空章节；随后 `_assemble_final` 按章节时间区间把 events 物理挂载到对应 chapter，最后一章使用开放右边界兜底，越界事件追加为 `ev_fallback_*`。

每次运行都会落 `_debug_pass3_input.txt` 与 `_debug_pass3_chapter_response.json` 用于排错，并生成 `pass3_confidence.json` 记录章节结构、时间有效性、边界吸附、event 挂载和章节分布。

**`characters_in_event` 抽取**：从 `step3_synthesized_dense_caption` 用 `\[[^\[\]]+\]` 正则抓 `[xxx]`，并丢弃形如 `[00:04:41]` / `[3.14]` 这类纯数字/冒号/小数点构成的"伪角色"（即时间戳），避免把时间戳错当成角色名落到 `event.characters` 里。

---

### Stage 2 — 事件级帧精修（per-event 串行）

```mermaid
flowchart TB
    Start["读 pass3_final.json"] --> Exists{"stage2_refined.json 存在 ?"}
    Exists -->|"是"| Resume["载入状态<br/>跳过已有 frame_caption 的 event"]
    Exists -->|"否"| Clone["克隆结构<br/>每个 event 加 frame_caption 空"]

    Resume --> Loop{"遍历 event"}
    Clone --> Loop
    Loop -->|"chapter 切换"| Reset["previous_caption 置空"]
    Reset --> ExtFrames["get_event_frames_base64<br/>区间内按 stage2_fps 抽帧"]
    Loop --> ExtFrames

    ExtFrames --> AdjStart{"abs start - prev.end 小于 0.01s ?"}
    AdjStart -->|"是"| Shift["start += stage2_adjust_start_offset<br/>避免抽到重复帧"]
    AdjStart -->|"否"| BuildMsg["messages = sys + user_text<br/>+ N 张 image_url 带秒数"]
    Shift --> BuildMsg

    BuildMsg --> CallVL["request_llm_text_with_retry<br/>纯文本输出"]
    CallVL --> Update["ev.frame_caption = result<br/>previous_caption = result"]
    Update --> Persist["整体落盘<br/>stage2_refined.json"]
    Persist --> Loop

    Loop -->|"done"| End(["stage2_refined.json<br/>每 event 多 frame_caption 与 frame_timestamps"])
```

**关键约束**：
- **串行处理** —— 当前 event 必须用上一个 event 的 `frame_caption` 作为前序上下文。
- **chapter 边界**：首事件不传 `previous_caption`，避免章节叙事污染。
- **失败容错**：单 event 抽帧/调用失败仅跳过该 event，不阻断后续。

---

### Stage 3 — 全局精修（单次纯文本 LLM 调用）

把所有 chapter/event 拼成 `{chapters:[{chapter_id, chapter_title, events:[{event_id, caption}]}]}` 一次性喂给模型，回填每个 event 的 `final_caption`。模型未回填的 event 退回到 `frame_caption`/`step3_synthesized_dense_caption`。

主要做：跨章节叙事缝合、指代统一去冗余、消除 "视频开头/纠正：xxx" 之类元描述与负向纠错痕迹。

---

## 断点续打矩阵

路径按内容（视频名 + 超参签名）稳定 → **重跑同样的命令即自动续打**，不需要任何特殊参数。

| 删除（或不存在）的文件                                 | 重跑范围                                            |
|--------------------------------------------------------|-----------------------------------------------------|
| `pass1_progress.json`                                  | Pass 1 → Pass 2 → Pass 3 → Stage 2 → Stage 3        |
| `pass2_progress.json`                                  | Pass 2 (A+B+C) → Pass 3 → Stage 2 → Stage 3         |
| `pass2_review_log.json`                                | Pass 2 (B+C) → Pass 3 → Stage 2 → Stage 3           |
| `pass2_aligned.json` 或 `pass2_global_bank.json`       | Pass 2 (C) → Pass 3 → Stage 2 → Stage 3             |
| `pass3_final.json`                                     | Pass 3 → Stage 2 → Stage 3                          |
| `stage2_refined.json`                                  | Stage 2（已有 `frame_caption` 的 event 仍跳过）→ Stage 3 |
| `stage3_polished.json`                                 | 仅 Stage 3                                          |

---

## 目录结构

```
LongVideoCaption_v2/
├── main.py                       # CLI 入口
├── .vscode/launch.json           # VSCode 调试配置
└── longvideocaption/
    ├── config.py                 # PipelineConfig + hyper_signature
    ├── utils.py                  # 时间戳 / JSON 清洗 / 文件名安全化
    ├── token_tracker.py          # per-video Tracker + 全局聚合器（带锁）
    ├── llm_client.py             # JSON / 纯文本两种 LLM 调用，统一重试 + token hook
    ├── frame_extractor.py        # scenedetect 抽帧 / chunk 视频 / 单帧 / 区间抽帧
    ├── pass1.py                  # Pass 1 视觉感知
    ├── pass2.py                  # Pass 2 三阶段身份对齐
    ├── pass3.py                  # Pass 3 章节聚合 + 装配
    ├── stage2.py                 # Stage 2 事件级帧精修
    ├── stage3.py                 # Stage 3 全局润色
    ├── pipeline.py               # 单视频串联 5 阶段
    ├── runner.py                 # 文件夹扫描 + ThreadPoolExecutor
    └── prompts/                  # pass1_v3 / stage2_v1 / stage3_v1 prompts
```

---

## 依赖

```bash
pip install openai httpx opencv-python numpy scenedetect
```

---

## 快速上手

### 单视频

```bash
python main.py \
  --input  D:/videos/I5cFBi02O34.mp4 \
  --output D:/outputs_v2 \
  --api-key YOUR_KEY \
  --base-url https://az.gptplus5.com/v1
```

### 文件夹批量 + 3 并发

```bash
python main.py \
  --input  D:/videos \
  --output D:/outputs_v2 \
  --workers 3 \
  --api-key YOUR_KEY \
  --base-url https://az.gptplus5.com/v1
```

### VSCode 调试

`.vscode/launch.json` 里已预置三条配置：
- **单视频打标** — 1 并发，适合 debug 单条视频。
- **文件夹批量打标 (3 并发)** — 批量 + 并发。
- **自定义超参 (video_base64 / 90s chunk)** — 演示 payload / chunk / fps 等参数覆盖。

使用前请把 `YOUR_API_KEY_HERE` 和输入输出路径替换成你本地的实际值。

---

## CLI 参数

| 参数              | 说明                                     | 默认                         |
|-------------------|------------------------------------------|------------------------------|
| `--input`         | 视频文件 **或** 文件夹（必填）           | —                            |
| `--output`        | 输出根目录（必填）                       | —                            |
| `--workers`       | 并发视频数                               | `2`                          |
| `--api-key`       | OpenAI 兼容 API key                      | 空（必须传或改 config.py）   |
| `--base-url`      | API base URL                             | 空（必须传或改 config.py）   |
| `--model`         | 模型名                                   | `gemini-3.1-pro-preview`     |
| `--chunk`         | `chunk_duration_sec`                     | `90`                         |
| `--payload`       | `image_list` / `video_base64`            | `image_list`                 |
| `--max-frames`    | 每 chunk 最大帧数                        | `360`                        |
| `--scene-thresh`  | scenedetect 阈值                         | `15.0`                       |
| `--frame-width`   | 帧宽（缩放上限）                         | `960`                        |
| `--target-fps`    | video_base64 采样帧率                    | `1.0`                        |
| `--pass1-timestamp-mode` | Pass 1 时间戳白名单格式：`second` / `millisecond` | `second`                     |
| `--conf-thresh`   | Pass 2 身份对齐置信度拦截阈值            | `80`                         |

进阶超参（Stage 2/3 的 fps、max_frames、temperature、max_tokens 等）在 `longvideocaption/config.py` 的 `PipelineConfig` 里改默认值。

---

## 输出结构

```
{output}/
├── _aggregate_token_usage.json     # 所有视频 per-stage + 总量汇总
├── _run_summary.json               # 每个视频的 success/failed + 产物路径
│
└── {video_basename}/
    └── {hyper_sig}/                # 例: gemini-3.1-pro-preview__chk90s__image_list__mf360__sc15_0__fw960__tssecond__ovlp0
        ├── pass1_progress.json     # Pass 1 事件流（chunk 级增量写）
        ├── pass1_confidence.json   # Pass 1 健康度报告：覆盖/连续性/修补/内容完整度
        ├── pass2_progress.json     # Pass 2 Phase A 断点（global_bank/chunk_mappings/低置信标记）
        ├── pass2_review_log.json   # Pass 2 Phase B 终审日志
        ├── pass2_aligned.json      # Pass 2 终产物：身份对齐后的事件流
        ├── pass2_global_bank.json  # 全局角色图鉴（lite，含名字 + 外貌）
        ├── pass3_final.json        # Pass 3 终产物：层级章节 JSON
        ├── pass3_confidence.json   # Pass 3 健康度报告：章节边界/挂载/分布
        ├── _debug_pass3_input.txt        # Pass 3 实际喂给模型的完整 prompt
        ├── _debug_pass3_chapter_response.json  # Pass 3 模型原始返回
        ├── stage2_refined.json     # Stage 2 终产物：每 event 多 frame_caption
        ├── stage3_polished.json    # ★ Stage 3 终产物：最终 final_caption
        ├── token_usage.json        # 本视频分阶段 token 消耗
        └── run_meta.json           # 运行时间戳 + 配置快照 + status
```

### `stage3_polished.json` 结构（最终交付）

```json
{
  "video_path": "...",
  "video_summary": "全片总结",
  "chapters": [
    {
      "chapter_id": "ch_01",
      "title": "章节标题",
      "chapter_summary": "本章总结",
      "start_time": "[00:00:00.000]",
      "end_time": "[00:05:30.000]",
      "events": [
        {
          "event_id": "ev_01_001",
          "start_time": "...", "end_time": "...",
          "step1_objective_visual": "Pass 1 客观画面",
          "step2_contextual_reasoning": "Pass 1 剧情与情绪归因",
          "step3_synthesized_dense_caption": "Pass 1 融合描述（含 [全局角色名]）",
          "frame_caption": "Stage 2 事件级帧精修结果",
          "frame_timestamps": [12.5, 13.5, ...],
          "final_caption": "★ Stage 3 全局润色后的最终描述",
          "characters_in_event": [{"name": "[李雷]", "desc": "..."}],
          "key_frame_times": ["..."]
        }
      ]
    }
  ]
}
```

---

## Confidence 健康度报告

详细字段解释见 [confidence.md](confidence.md)。当前会随视频运行目录生成两份报告：

| 文件 | 关注点 | 典型用途 |
|------|--------|----------|
| `pass1_confidence.json` | event 时间覆盖、相邻连续性、时间戳吸附、后处理修补/丢弃、event 时长分布、内容缺失、关键帧越界 | 判断 Pass 1 的事件切分是否完整、是否靠大量修补才变得连续 |
| `pass3_confidence.json` | LLM 章节结构、无效章节时间、章节边界吸附、未挂载事件、章节 event 分布 | 判断章节切分和 event 挂载是否稳定可靠 |

几个高信号字段：

- `pass1_confidence.event_time_coverage.is_fully_covered`：是否完整覆盖整片时间轴。
- `pass1_confidence.event_time_continuity.adjacency_break_count`：相邻 event 是否仍存在 gap/overlap。
- `pass1_confidence.event_validation_summary`：Pass 1 为修平时间轴做了多少交换、吸附和丢弃。
- `pass1_confidence.timestamp_calibration.large_abs_delta_sum_sec`：模型时间戳被吸附到白名单时的大偏移绝对值总和。
- `pass3_confidence.chapter_boundary_calibration.total_abs_snap_delta_sec`：章节边界被吸附到 event 边界的绝对偏移总和。
- `pass3_confidence.event_mounting.unbound_event_count`：未能自然落入章节、最终被追加到末章的 event 数量。

---

## Token 统计

每次 LLM 调用都会通过 `llm_client.request_llm_with_retry` / `request_llm_text_with_retry` 读取 `completion.usage`，按 stage 名累加到 per-video `TokenTracker`：

| Stage 名                       | 来源                              |
|--------------------------------|-----------------------------------|
| `pass1_perception`             | Pass 1 chunk 级多模态调用         |
| `pass2_alignment`              | Pass 2 Phase A 滚动聚类           |
| `pass2_review`                 | Pass 2 Phase B 高清终审           |
| `pass3_aggregation`            | Pass 3 章节切分                   |
| `stage2_frame_inspection`      | Stage 2 per-event 帧精修          |
| `stage3_global_polish`         | Stage 3 全局润色                  |

- **单视频级**：`{run_dir}/token_usage.json`
- **批量聚合**：`{output}/_aggregate_token_usage.json`（含 `per_video` / `per_stage_totals` / `grand_total`）

---

## 并发模型

- 每个视频独立 worker 线程 → 独立 `OpenAI` client / `httpx.Client` / `cv2.VideoCapture`，互不干扰。
- 视频内部各阶段**严格串行**（Pass 2 Phase A 需要 chunk 顺序、Stage 2 依赖前一 event 的精修结果）。
- `GlobalTokenAggregator` 是唯一跨线程共享对象，带 `threading.Lock`，每视频收尾时调用一次。
- 建议 `--workers` 别调太大，受 LLM 厂商 QPS / TPM 限制，2–4 一般足够。

---

## 已知注意事项

- `config.py` 里 `api_key` / `base_url` 默认空串，必须通过 CLI 或修改默认值来提供。
- scenedetect 多线程可用，但每个视频内部是同步 CPU 计算，大批量并发需留意 CPU。
- Stage 2 所有 event 串行 → 长视频耗时较长（事件数 × 单调用耗时），可以根据需要在 `config.py` 调小 `stage2_max_frames` 来提速。
- Pass 3 的 `_debug_*` 文件每次都会覆盖写入，仅供当次排错使用，不要直接消费。
