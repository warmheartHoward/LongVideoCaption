from dataclasses import dataclass, field, asdict
from typing import List, Optional

from .utils import sanitize_filename


STAGE_NAMES = ("stage1", "stage2", "stage3")


def normalize_stages(stages: Optional[List[str]]) -> Optional[List[str]]:
    """归一化 + 校验 stages 选择。

    - None / 空 → None（跑完整 pipeline）
    - 必须是 STAGE_NAMES 子集
    - 必须连续（stage1 / stage1,stage2 / stage1,stage2,stage3 / stage2 /
      stage2,stage3 / stage3），不允许 stage1+stage3 跳段
    """
    if not stages:
        return None
    norm = []
    seen = set()
    for s in stages:
        key = (s or "").strip().lower()
        if not key:
            continue
        if key not in STAGE_NAMES:
            raise ValueError(f"未知 stage: {s!r}，可选: {STAGE_NAMES}")
        if key in seen:
            continue
        seen.add(key)
        norm.append(key)
    if not norm:
        return None
    norm.sort(key=STAGE_NAMES.index)
    indices = [STAGE_NAMES.index(s) for s in norm]
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(
            f"stages 必须是连续区间，不允许跳段: {norm}（合法: stage1 / stage1,stage2 / "
            f"stage1,stage2,stage3 / stage2 / stage2,stage3 / stage3）"
        )
    return norm


@dataclass
class PipelineConfig:
    api_key: str = ""
    base_url: str = ""
    model_name: str = "gemini-3.1-pro-preview"
    llm_timeout: float = 6000.0
    max_retries: int = 3

    chunk_duration_sec: int = 90
    input_payload_format: str = "image_list"
    frame_extraction_strategy: str = "scenedetect"
    max_frames_per_chunk: int = 360
    scene_detect_threshold: float = 15.0
    target_fps: float = 1.0
    frame_max_width: int = 960
    max_total_pixels: int = 128 * 1024 * 32 * 32
    frame_jpg_quality: int = 95
    pass1_timestamp_mode: str = "second"  # second / millisecond / qwen_millisecond
    llm_temperature: float = 0.2
    llm_max_tokens: int = 32768
    prev_event_overlap_count: int = 0

    pass2_temperature: float = 0.1
    pass2_max_tokens: int = 32768
    pass2_confidence_threshold: int = 80
    pass2_hard_split_threshold: int = 50

    pass2_review_enable: bool = True
    pass2_review_frame_max_width: int = 0
    pass2_review_jpg_quality: int = 95
    pass2_review_max_frames_per_cluster: int = 3
    pass2_review_temperature: float = 0.1
    pass2_review_max_tokens: int = 8192
    pass2_review_batch_size: int = 0  # 0 或负数 = 一次性全部；>0 = 每批最多 N 个聚类

    pass3_temperature: float = 0.4
    pass3_max_tokens: int = 32768

    stage2_fps: float = 4.0
    stage2_max_frames: int = 1200
    stage2_temperature: float = 0.2
    stage2_max_tokens: int = 16384
    stage2_adjust_start_offset: float = 0.5
    stage2_frame_max_width: int = 960
    stage2_max_total_pixels: int = 128 * 1024 * 32 * 32
    stage2_frame_jpg_quality: int = 95
    stage2_parallel_max_workers: int = 4
    stage2_qwen_parallel_max_workers: int = 32
    stage2_qwen_parallel_visual_token_budget: int = 128 * 1024

    stage3_temperature: float = 0.4
    stage3_max_tokens: int = 65536

    max_workers: int = 2

    # 限制只跑指定 stage 子集；None 表示完整 pipeline。
    # 合法取值：stage1 (Pass1+2+3 → pass3_final.json) / stage2 (帧精修 → stage2_refined.json) /
    # stage3 (全局润色 → stage3_polished.json)。仅允许连续区间。
    stages: Optional[List[str]] = None

    # 显式覆盖 run_dir 的 hyper_sig 段。设置后 run_dir = {output}/{video}/{resume_hyper_sig}/，
    # 不再调 hyper_signature(cfg)。典型用途：stage2/stage3 想换模型续跑（模型变了 hyper_signature
    # 会变，导致找不到上游 pass3_final.json/stage2_refined.json），手工传入 stage1 时的 hyper_sig
    # 即可继续。批量场景下对每个视频都生效。
    resume_hyper_sig: Optional[str] = None

    strict_failure: bool = True
    video_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"]
    )

    def to_dict(self) -> dict:
        return asdict(self)


def hyper_signature(cfg: PipelineConfig) -> str:
    model = sanitize_filename(cfg.model_name)
    scene = str(cfg.scene_detect_threshold).replace(".", "_")
    return (
        f"{model}"
        f"__chk{cfg.chunk_duration_sec}s"
        f"__{cfg.input_payload_format}"
        f"__mf{cfg.max_frames_per_chunk}"
        f"__sc{scene}"
        f"__fw{cfg.frame_max_width}"
        f"__ts{cfg.pass1_timestamp_mode}"
        f"__ovlp{cfg.prev_event_overlap_count}"
    )
