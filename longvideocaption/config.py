from dataclasses import dataclass, field, asdict
from typing import List

from .utils import sanitize_filename


@dataclass
class PipelineConfig:
    api_key: str = ""
    base_url: str = ""
    model_name: str = "gemini-3.1-pro-preview"
    llm_timeout: float = 600.0
    max_retries: int = 3

    chunk_duration_sec: int = 60
    input_payload_format: str = "image_list"
    frame_extraction_strategy: str = "scenedetect"
    max_frames_per_chunk: int = 240
    scene_detect_threshold: float = 27.0
    target_fps: float = 1.0
    frame_max_width: int = 960
    frame_jpg_quality: int = 90
    llm_temperature: float = 0.2
    llm_max_tokens: int = 16384
    prev_event_overlap_count: int = 0

    pass2_temperature: float = 0.1
    pass2_max_tokens: int = 16384
    pass2_confidence_threshold: int = 80
    pass2_hard_split_threshold: int = 50

    pass2_review_enable: bool = True
    pass2_review_frame_max_width: int = 0
    pass2_review_jpg_quality: int = 92
    pass2_review_max_frames_per_cluster: int = 3
    pass2_review_temperature: float = 0.1
    pass2_review_max_tokens: int = 8192
    pass2_review_batch_size: int = 0  # 0 或负数 = 一次性全部；>0 = 每批最多 N 个聚类

    pass3_temperature: float = 0.3
    pass3_max_tokens: int = 16384

    stage2_fps: float = 1.0
    stage2_max_frames: int = 100
    stage2_temperature: float = 0.2
    stage2_max_tokens: int = 8192
    stage2_adjust_start_offset: float = 0.5
    stage2_frame_max_width: int = 960
    stage2_frame_jpg_quality: int = 90

    stage3_temperature: float = 0.4
    stage3_max_tokens: int = 16384

    max_workers: int = 2

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
        f"__ovlp{cfg.prev_event_overlap_count}"
    )
