import json
import os
import traceback
from datetime import datetime

from .config import PipelineConfig, hyper_signature
from .llm_client import build_client
from .pass1 import run_pass1
from .pass2 import run_pass2
from .pass3 import run_pass3
from .stage2 import run_stage2
from .stage3 import run_stage3
from .token_tracker import TokenTracker
from .utils import sanitize_filename


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_meta(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _write_meta(path: str, meta: dict) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def resolve_run_dir(cfg: PipelineConfig, video_path: str, output_root: str) -> str:
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(
        output_root,
        sanitize_filename(video_basename),
        hyper_signature(cfg),
    )


def process_single_video(cfg: PipelineConfig, video_path: str, output_root: str) -> dict:
    video_tag = sanitize_filename(os.path.splitext(os.path.basename(video_path))[0])
    run_dir = resolve_run_dir(cfg, video_path, output_root)
    os.makedirs(run_dir, exist_ok=True)

    meta_path = os.path.join(run_dir, "run_meta.json")
    token_path = os.path.join(run_dir, "token_usage.json")

    meta = _load_meta(meta_path)
    if not meta:
        meta = {
            "video_path": video_path,
            "hyper_sig": hyper_signature(cfg),
            "config": cfg.to_dict(),
            "runs": [],
        }
    this_run = {"started_at": _now_iso(), "ended_at": None, "status": "running"}
    meta["runs"].append(this_run)
    _write_meta(meta_path, meta)

    tracker = TokenTracker()
    client = build_client(cfg)

    result = {
        "video_path": video_path,
        "run_dir": run_dir,
        "status": "running",
        "token_usage": None,
        "artifacts": {},
        "error": None,
    }

    try:
        print(f"\n{'#' * 70}\n# [{video_tag}] 处理视频: {video_path}\n# [{video_tag}] 运行目录: {run_dir}\n{'#' * 70}")

        pass1_path = run_pass1(cfg, video_path, run_dir, client, tracker, video_tag=video_tag)
        result["artifacts"]["pass1_progress"] = pass1_path

        aligned_path, bank_path = run_pass2(cfg, video_path, pass1_path, run_dir, client, tracker, video_tag=video_tag)
        result["artifacts"]["pass2_aligned"] = aligned_path
        result["artifacts"]["pass2_global_bank"] = bank_path

        final_path = run_pass3(cfg, video_path, aligned_path, run_dir, client, tracker, video_tag=video_tag)
        result["artifacts"]["pass3_final"] = final_path

        stage2_path = run_stage2(cfg, video_path, final_path, run_dir, client, tracker, video_tag=video_tag)
        result["artifacts"]["stage2_refined"] = stage2_path

        stage3_path = run_stage3(cfg, stage2_path, run_dir, client, tracker, video_tag=video_tag)
        result["artifacts"]["stage3_polished"] = stage3_path

        this_run["status"] = "completed"
        result["status"] = "completed"

    except Exception as e:
        err_text = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"\n💥 [{video_tag}] 视频处理失败: {video_path}\n{err_text}")
        this_run["status"] = "failed"
        this_run["error"] = err_text
        result["status"] = "failed"
        result["error"] = err_text

    finally:
        this_run["ended_at"] = _now_iso()
        token_dict = tracker.to_dict()
        result["token_usage"] = token_dict
        try:
            with open(token_path, 'w', encoding='utf-8') as f:
                json.dump(token_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ [{video_tag}] token_usage.json 写入失败: {e}")
        _write_meta(meta_path, meta)

    return result
