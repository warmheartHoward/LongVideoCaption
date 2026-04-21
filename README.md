# Long Video Caption v2

长视频三阶段打标流水线（视觉感知 → 身份对齐 → 章节装配），支持文件夹批量、多线程并发、断点续打、分阶段 Token 统计。

## 目录结构

```
LongVideoCaption_v2/
├── main.py                       # CLI 入口
├── .vscode/launch.json           # VSCode 调试配置
└── longvideocaption/
    ├── config.py                 # PipelineConfig + hyper_signature
    ├── utils.py                  # 时间戳 / JSON 清洗 / 文件名安全化
    ├── token_tracker.py          # per-video Tracker + 全局聚合器（带锁）
    ├── llm_client.py             # LLM 调用 + 重试 + token hook
    ├── frame_extractor.py        # scenedetect 抽帧 / chunk 视频 / 单帧
    ├── stage1.py                 # Pass 1 视觉感知（chunk 级断点）
    ├── stage2.py                 # 滚动身份对齐（per-chunk 断点）
    ├── stage3.py                 # 章节聚合 + 层级装配
    ├── pipeline.py               # 单视频串联 stage1 → 2 → 3
    └── runner.py                 # 文件夹扫描 + ThreadPoolExecutor
```

## 依赖

```bash
pip install openai httpx opencv-python numpy scenedetect
```

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

## CLI 参数

| 参数              | 说明                                     | 默认                         |
|-------------------|------------------------------------------|------------------------------|
| `--input`         | 视频文件 **或** 文件夹（必填）           | —                            |
| `--output`        | 输出根目录（必填）                       | —                            |
| `--workers`       | 并发视频数                               | `2`                          |
| `--api-key`       | OpenAI 兼容 API key                      | 空（必须传或改 config.py）   |
| `--base-url`      | API base URL                             | 空（必须传或改 config.py）   |
| `--model`         | 模型名                                   | `gemini-3.1-pro-preview`     |
| `--chunk`         | `chunk_duration_sec`                     | `60`                         |
| `--payload`       | `image_list` / `video_base64`            | `image_list`                 |
| `--max-frames`    | 每 chunk 最大帧数                        | `240`                        |
| `--scene-thresh`  | scenedetect 阈值                         | `27.0`                       |
| `--frame-width`   | 帧宽（缩放上限）                         | `960`                        |
| `--target-fps`    | video_base64 采样帧率                    | `1.0`                        |
| `--conf-thresh`   | stage2 身份对齐置信度拦截阈值            | `80`                         |

进阶超参（LLM temperature / max_tokens / retries 等）在 `longvideocaption/config.py` 的 `PipelineConfig` 里改默认值。

## 输出结构

每个视频按 **视频名 + 超参签名** 稳定分目录，相同配置 + 相同视频始终落到同一路径，断点续打开箱即用：

```
{output}/
├── _aggregate_token_usage.json     # 所有视频 per-stage + 总量汇总
├── _run_summary.json               # 每个视频的 success/failed + 产物路径
│
└── {video_basename}/
    └── {hyper_sig}/                # 例如 gemini-3.1-pro-preview__chk60s__image_list__mf240__sc27_0__fw960
        ├── stage1_progress.json    # Pass 1 事件流（chunk 级增量写）
        ├── stage2_progress.json    # Stage 2 per-chunk 断点（跑完可删）
        ├── stage2_bank_progress.json
        ├── stage2_aligned.json     # 身份对齐后的事件流（终产物）
        ├── stage2_global_bank.json # 全局角色图鉴（lite，含名字+外貌）
        ├── stage3_final.json       # ★ 最终层级章节 JSON
        ├── token_usage.json        # 本视频分阶段 token 消耗
        └── run_meta.json           # 运行时间戳 + 配置快照 + status
```

### `stage3_final.json` 结构

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
          "step1_objective_visual": "客观画面",
          "step2_contextual_reasoning": "剧情与情绪归因",
          "step3_synthesized_dense_caption": "融合后的剧本级描述（含 [全局角色名]）",
          "key_frame_times": ["..."]
        }
      ]
    }
  ]
}
```

## 断点续打

路径按内容稳定 → **重跑完全相同的命令即自动续打**，不需要额外参数。

- **Stage 1** — `stage1_progress.json` 记录每个 chunk 的事件；重跑时从最后一个 event 的 `end_time` 继续。
- **Stage 2** — `stage2_progress.json` 记录已处理 chunk 索引；重跑时跳过并从图鉴（含 base64）续跑。
- **Stage 3** — `stage3_final.json` 存在即跳过（单次 LLM 调用，无子粒度）。

想强制重跑？删 `run_dir` 或对应阶段的 `*_progress.json` / 终产物即可。

## Token 统计

每次 LLM 调用都会走 `llm_client.request_llm_with_retry`（或 stage2 里的直接调用），读取 `completion.usage` 并按 stage 名（`stage1_perception` / `stage2_alignment` / `stage3_aggregation`）累加到 per-video `TokenTracker`。

- **单视频级**：`{run_dir}/token_usage.json`
  ```json
  {
    "per_stage": {
      "stage1_perception": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ..., "calls": ...},
      "stage2_alignment": {...},
      "stage3_aggregation": {...}
    },
    "grand_total": {...}
  }
  ```
- **批量聚合**：`{output}/_aggregate_token_usage.json`
  ```json
  {
    "per_video": { "video1.mp4": {...}, "video2.mkv": {...} },
    "per_stage_totals": { "stage1_perception": {...}, ... },
    "grand_total": {...}
  }
  ```

## 并发模型

- 每个视频独立 worker 线程 → 独立 `OpenAI` client / `httpx.Client` / `cv2.VideoCapture`，互不干扰。
- `GlobalTokenAggregator` 是唯一跨线程共享对象，带 `threading.Lock`，每个视频收尾时调用一次。
- 建议 `--workers` 别调太大，受 LLM 厂商的并发/QPS 限制，2–4 一般足够。

## 不改的核心业务逻辑

三个原脚本的 prompts、LLM 温度/max_tokens/超时/重试、scenedetect 抽帧算法、stage1 动态接力规则（`chunk_start+5 ≤ last_end ≤ chunk_end+10` 用 end 作为下段起点，否则 80% 重叠兜底）、stage2 置信度 < 80 强制判 NEW、stage3 浮点容差 `-0.5` / 最后一章 `+9999.0` / `ev_fallback_` 兜底追加 —— 全部原样搬进新模块。

## 已知注意事项

- `config.py` 里 `api_key` / `base_url` 默认空串，必须通过 CLI 或修改默认值来提供。
- scenedetect 多线程可用，但每个视频内部是同步 CPU 计算，大批量并发需留意 CPU。
- stage1 断点续跑时的 `previous_context` 里会丢失"上一个动作描述"（原脚本遗留，读取了不存在的 `dense_caption` 字段 —— 未修，保持行为一致）。
