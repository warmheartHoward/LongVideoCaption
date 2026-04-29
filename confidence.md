# 置信度报告说明

本文档说明流水线中生成的置信度/健康度报告文件。

## pass1_confidence.json

`pass1_confidence.json` 会写入当前视频的运行目录。它是 Pass 1 感知与事件切分阶段的健康度报告，用来检查 event 时间轴覆盖、连续性、时间戳校准、关键帧合法性、event 粒度和内容完整度。

### 顶层字段

- `video_duration_sec`：视频总时长，单位为秒。
- `event_count`：Pass 1 当前累计产出的 event 数量。

### event_time_coverage

检查所有 event 按时间排序后，是否覆盖了从视频开头到视频结尾的完整时间轴。

- `is_fully_covered`：是否没有发现时间空白。
- `gap_count`：发现的空白段数量。
- `total_gap_sec`：所有空白段的累计时长，单位为秒。
- `gaps`：空白段明细，每项包含 `start_time`、`end_time`、`gap_sec`。

### event_time_continuity

检查相邻 event 是否严格首尾相连，并显式统计 overlap。

- `overlap_count`：相邻 event 之间存在重叠的次数。
- `total_overlap_sec`：所有 overlap 的累计时长，单位为秒。
- `max_overlap_sec`：最大单次 overlap 时长，单位为秒。
- `overlaps`：overlap 明细，包含前后 event 的时间范围、chunk/event 下标和重叠秒数。
- `adjacency_break_count`：相邻 event 不满足严格首尾相连的次数。gap 和 overlap 都会计入。

### event_validation_summary

统计 Pass 1 在后处理阶段为了修正时间轴而做过的操作。这个部分用于判断最终时间轴是否是“自然干净”，还是靠大量修补得到。

- `dropped_invalid_time_count`：因 start/end 时间格式无法解析而丢弃的 event 数量。
- `dropped_too_short_count`：因事件时长过短而丢弃的 event 数量。
- `dropped_overlap_count`：因 overlap 修正后无有效时长、或完全落在上段末事件内而丢弃的 event 数量。
- `continuity_snap_count`：chunk 内为了修平 gap/overlap 执行的 start_time 吸附次数。
- `cross_chunk_snap_count`：跨 chunk 为避免与上段末事件重叠执行的 start_time 吸附次数。
- `swap_count`：发现 `start_time > end_time` 后自动交换起止时间的次数。

### event_duration_distribution

统计 event 时长分布，用来判断切分粒度是否异常。

- `min_sec`：最短 event 时长。
- `max_sec`：最长 event 时长。
- `avg_sec`：平均 event 时长。
- `median_sec`：event 时长中位数。
- `too_short_threshold_sec`：过短 event 阈值，目前为 `1.0` 秒。
- `too_short_count`：短于阈值的 event 数量。
- `too_long_threshold_sec`：过长 event 阈值，目前为 `60.0` 秒。
- `too_long_count`：长于阈值的 event 数量。

### event_content_health

检查 event 和 chunk 的结构化内容是否为空或明显缺失。

- `missing_step1_count`：缺失 `step1_objective_visual` 的 event 数量。
- `missing_step2_count`：缺失 `step2_contextual_reasoning` 的 event 数量。
- `missing_step3_count`：缺失 `step3_synthesized_dense_caption` 的 event 数量。
- `missing_key_frame_times_count`：缺失 `key_frame_times` 的 event 数量。
- `empty_characters_in_chunk_count`：`characters_in_chunk` 为空的 chunk 数量。无角色视频不一定异常，但剧情类视频中这个数值较高时需要关注。

### key_frame_time_validity

检查每个 event 的 `key_frame_times` 是否落在该 event 的时间范围附近。

- `total_key_frame_times`：关键帧时间戳总数。
- `invalid_count`：越界关键帧数量。
- `all_within_event_ranges`：所有关键帧是否都落在对应 event 时间范围附近。
- `invalid_items`：越界关键帧明细，包含 chunk 下标、event 下标、event 时间范围和越界时间戳。

### timestamp_calibration

统计模型输出时间戳被吸附到白名单时的较大偏移。

- `large_delta_threshold_sec`：大偏移阈值，目前为 `1.0` 秒。
- `large_abs_delta_sum_sec`：大偏移绝对值累计秒数。
- `large_delta_count`：大偏移次数。
- `large_deltas`：大偏移明细，包含字段标签和绝对偏移秒数。

## pass3_confidence.json

`pass3_confidence.json` 会和 `pass3_final.json` 一起写入当前视频的运行目录。它是 Pass 3 的轻量健康度报告，用来检查章节聚合、章节边界校准和 event 挂载是否可靠。

### 顶层字段

- `event_count`：Pass 3 收集到的合法输入 event 数量。非法时间范围的 event 会在进入 LLM 聚合前被丢弃。
- `chapter_count`：最终输出中实际保留的章节数量。

### llm_output_shape

检查 LLM 返回的章节结构是否可用。

- `raw_chapter_count`：LLM 原始返回的章节数量，尚未经过 Pass 3 校验。
- `validated_chapter_count`：经过 Pass 3 校验、删除非法章节或兜底处理后的章节数量。
- `fallback_used`：是否启用了兜底单章节模式。
- `missing_required_field_count`：LLM 原始章节中缺失必需字段的总数。必需字段包括 `title`、`chapter_summary`、`start_time`、`end_time`。

### chapter_time_validity

统计章节时间字段在边界修复前的非法情况。

- `invalid_start_count`：无法解析的章节 `start_time` 数量。
- `invalid_end_count`：无法解析的章节 `end_time` 数量。
- `invalid_range_count`：解析后满足 `end_time <= start_time` 的章节数量。

### chapter_boundary_calibration

衡量 Pass 3 为了把章节边界吸附到 event 边界上，做了多少修正。

- `snap_count`：边界吸附次数，只统计绝对偏移大于 `0.01` 秒的修正。
- `total_abs_snap_delta_sec`：所有边界吸附绝对偏移的总和，单位为秒。
- `max_abs_snap_delta_sec`：单次边界吸附的最大绝对偏移，单位为秒。

这些数值越大，通常说明 LLM 没有严格逐字保留 event 边界，或者返回的章节边界离真实事件时间轴较远。

### event_mounting

检查输入 event 是否都进入了最终章节树。

- `mounted_event_count`：最终章节树中挂载的 event 数量。包含正常挂载和兜底追加的 event。
- `unbound_event_count`：正常按章节时间范围挂载时没有命中任何章节、最后被强制追加到最终章的 event 数量。
- `event_count_matches_input`：`mounted_event_count` 是否等于 `event_count`。

如果 `unbound_event_count > 0`，说明章节时间范围没有自然覆盖所有 event，需要重点检查章节边界。

### chapter_distribution

检查章节内 event 分布是否明显异常。

- `empty_chapter_count`：最终没有挂载任何 event 的章节数量。
- `single_event_chapter_count`：只包含一个 event 的章节数量。
- `dominant_chapter_event_ratio`：event 数最多的章节占全部输入 event 的比例。

`dominant_chapter_event_ratio` 较高不一定代表错误，但可能说明章节切分不均衡，大部分视频内容被压进了同一个章节。

## stage3_confidence

`stage3_confidence` 会写入 `stage3_polished.json` 的顶层字段。它是 Stage 3 全局润色阶段的结构一致性报告，用来检查 LLM 返回的章节和 event 是否与输入 payload 保持一致，以及最终有多少 event 成功使用了润色结果。

Stage 3 的输入 payload 来自 `stage2_refined.json` 中有可用 caption 的 event。caption 优先使用 `frame_caption`，没有时回退到 `step3_synthesized_dense_caption`。因此这里的输入数量指“送入 Stage 3 LLM 的 chapter/event 数量”，不一定等于 `stage2_refined.json` 中所有 event 的总数。

### 顶层字段

- `ok`：Stage 3 结构校验是否完全通过。需要章节数量、event 数量、chapter id/title、event id 全部一致，并且输出 caption 不为空。
- `skipped`：是否跳过了 Stage 3 LLM 调用。仅在没有可润色 event 时出现。
- `skip_reason`：跳过原因。当前可能值为 `no_caption_events`。

### 数量一致性

检查 LLM 输出和 Stage 3 输入 payload 的 chapter/event 数量是否一致。

- `chapter_count_matches`：输出 chapter 数量是否等于输入 chapter 数量。
- `event_count_matches`：输出 event 总数是否等于输入 event 总数。
- `input_chapter_count`：送入 Stage 3 LLM 的 chapter 数量。
- `output_chapter_count`：LLM 返回的 chapter 数量。
- `input_event_count`：送入 Stage 3 LLM 的 event 数量。
- `output_event_count`：LLM 返回的 event 数量。

### ID 与标题一致性

检查 LLM 是否保持了输入结构中的章节标识、章节标题和事件标识。

- `chapter_id_title_matches`：每个输出 chapter 的 `chapter_id` 和 `chapter_title` 是否与同位置输入 chapter 一致。
- `event_id_matches`：每个输出 event 的 `event_id` 是否与同位置输入 event 一致，并且没有缺失或额外 event。
- `chapter_mismatches`：chapter 不一致明细，包含 chapter 下标、期望/实际 `chapter_id` 和期望/实际标题。
- `event_mismatches`：event id 不一致明细，包含 chapter/event 下标、chapter id、期望 event id 和实际 event id。
- `missing_events`：输入中存在但输出中缺失的 event 明细。
- `extra_events`：输出中多出来的 event 明细。

### 润色覆盖情况

统计最终有多少 event 实际使用了 LLM 的 Stage 3 润色文本。

- `polished_event_count`：输出中成功匹配到 `(chapter_id, event_id)` 且 caption 非空的 event 数量。
- `fallback_event_count`：未成功使用 Stage 3 润色结果、最终回退到 `frame_caption` 或 `step3_synthesized_dense_caption` 的 event 数量。
- `empty_caption_events`：LLM 输出中 caption 为空的 event 明细。

如果 `fallback_event_count > 0`，说明部分 event 没有拿到有效的 Stage 3 润色结果；最终 caption 仍可用，但需要检查是否是模型漏回、id 变化或空 caption 导致。

### 重复键检查

Stage 3 使用 `(chapter_id, event_id)` 作为回填键，避免不同章节里相同 event id 互相串写。

- `duplicate_input_event_keys`：输入 payload 中重复的 `(chapter_id, event_id)`。
- `duplicate_output_event_keys`：LLM 输出中重复的 `(chapter_id, event_id)`。

如果出现重复键，后续回填可能无法区分同一 chapter 内的同名 event，需要检查上游 event id 生成逻辑。
