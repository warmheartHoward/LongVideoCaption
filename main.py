import argparse
import sys
import warnings

from longvideocaption.config import PipelineConfig
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

    p.add_argument("--conf-thresh", type=int, default=None, help="覆盖 pass2_confidence_threshold")
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

    if args.conf_thresh is not None:
        cfg.pass2_confidence_threshold = args.conf_thresh

    run_batch(cfg, args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
