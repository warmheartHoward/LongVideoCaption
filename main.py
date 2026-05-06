import argparse
import sys
import warnings

from longvideocaption.config import PipelineConfig, normalize_stages
from longvideocaption.runner import run_batch


warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Long Video Caption — 三遍扫描式长视频打标（支持文件夹批量 + 多线程 + 断点续打）"
    )
    p.add_argument("--input", required=True, help="视频文件或文件夹路径")
    p.add_argument("--output", required=True, help="输出根目录")
    p.add_argument("--workers", type=int, default=2, help="并发视频数（默认 2）")

    p.add_argument("--model", default=None, help="覆盖 model_name")
    p.add_argument("--api-key", default=None, help="覆盖 api_key")
    p.add_argument("--base-url", default=None, help="覆盖 base_url")

    p.add_argument("--chunk", type=int, default=None, help="覆盖 chunk_duration_sec")
    p.add_argument("--payload", choices=["image_list", "video_base64"], default=None, help="覆盖 input_payload_format")
    p.add_argument("--max-frames", type=int, default=None, help="覆盖 max_frames_per_chunk")
    p.add_argument("--scene-thresh", type=float, default=None, help="覆盖 scene_detect_threshold")
    p.add_argument("--frame-width", type=int, default=None, help="覆盖 frame_max_width")
    p.add_argument("--target-fps", type=float, default=None, help="覆盖 target_fps")
    p.add_argument(
        "--pass1-timestamp-mode",
        choices=["second", "millisecond", "qwen_millisecond"],
        default=None,
        help=(
            "Pass1 给模型的 timestamp 白名单格式：second=hh:mm:ss，"
            "millisecond=hh:mm:ss.fff，qwen_millisecond=x.x seconds 提示词格式、毫秒格式落盘"
        ),
    )

    p.add_argument("--conf-thresh", type=int, default=None, help="覆盖 pass2_confidence_threshold")
    p.add_argument(
        "--resume-hyper-sig",
        default=None,
        help=(
            "显式指定 run_dir 的 hyper_sig 段（即 {output}/{video}/<这一段>/）。"
            "设置后将绕过 hyper_signature(cfg) 计算。"
            "典型用途：stage2/stage3 想换模型续跑时，传入 stage1 跑时的 hyper_sig 以复用上游产物。"
            "批量模式下对所有视频生效。"
        ),
    )
    p.add_argument(
        "--stages",
        default=None,
        help=(
            "限定只跑哪些 stage，逗号分隔，必须是连续区间。"
            "可选值: stage1 (Pass1+2+3 → pass3_final.json) / "
            "stage2 (帧精修 → stage2_refined.json) / stage3 (全局润色 → stage3_polished.json)。"
            "示例: --stages stage2 / --stages stage2,stage3。"
            "不传则跑完整 pipeline。"
        ),
    )
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = PipelineConfig()
    cfg.max_workers = max(1, args.workers)

    if args.model is not None:
        cfg.model_name = args.model
    if args.api_key is not None:
        cfg.api_key = args.api_key
    if args.base_url is not None:
        cfg.base_url = args.base_url

    if args.chunk is not None:
        cfg.chunk_duration_sec = args.chunk
    if args.payload is not None:
        cfg.input_payload_format = args.payload
    if args.max_frames is not None:
        cfg.max_frames_per_chunk = args.max_frames
    if args.scene_thresh is not None:
        cfg.scene_detect_threshold = args.scene_thresh
    if args.frame_width is not None:
        cfg.frame_max_width = args.frame_width
    if args.target_fps is not None:
        cfg.target_fps = args.target_fps
    if args.pass1_timestamp_mode is not None:
        cfg.pass1_timestamp_mode = args.pass1_timestamp_mode

    if args.conf_thresh is not None:
        cfg.pass2_confidence_threshold = args.conf_thresh

    if args.stages is not None:
        cfg.stages = normalize_stages([s for s in args.stages.split(",") if s.strip()])

    if args.resume_hyper_sig is not None:
        cfg.resume_hyper_sig = args.resume_hyper_sig.strip() or None

    run_batch(cfg, args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
