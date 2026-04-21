import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List

from .config import PipelineConfig
from .pipeline import process_single_video
from .token_tracker import GlobalTokenAggregator


def discover_videos(input_path: str, extensions: List[str]) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    ext_set = {e.lower() for e in extensions}
    videos = []
    for root, _dirs, files in os.walk(input_path):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in ext_set:
                videos.append(os.path.join(root, fname))
    videos.sort(key=lambda p: os.path.getmtime(p))
    return videos


def run_batch(cfg: PipelineConfig, input_path: str, output_root: str) -> dict:
    os.makedirs(output_root, exist_ok=True)
    videos = discover_videos(input_path, cfg.video_extensions)
    if not videos:
        print(f"⚠️ 在 {input_path} 未发现任何视频文件。")
        return {"videos": [], "aggregate_tokens": {}}

    print(f"\n🎯 发现 {len(videos)} 个视频，并发度 = {cfg.max_workers}")
    for v in videos:
        print(f"  - {v}")

    aggregator = GlobalTokenAggregator()
    summary = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "input": input_path,
        "output_root": output_root,
        "max_workers": cfg.max_workers,
        "videos": [],
    }

    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as executor:
        future_to_video = {
            executor.submit(process_single_video, cfg, video_path, output_root): video_path
            for video_path in videos
        }
        for fut in as_completed(future_to_video):
            video_path = future_to_video[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"💥 {video_path} 线程级异常: {e}")
                result = {
                    "video_path": video_path,
                    "status": "failed",
                    "error": str(e),
                    "artifacts": {},
                    "token_usage": None,
                    "run_dir": None,
                }

            video_key = os.path.basename(video_path)
            if result.get("token_usage"):
                aggregator.add(video_key, result["token_usage"])

            summary["videos"].append({
                "video_path": video_path,
                "status": result.get("status"),
                "run_dir": result.get("run_dir"),
                "artifacts": result.get("artifacts", {}),
                "error": result.get("error"),
            })

    summary["ended_at"] = datetime.now().isoformat(timespec="seconds")
    summary["aggregate_tokens"] = aggregator.to_dict()

    agg_path = os.path.join(output_root, "_aggregate_token_usage.json")
    run_summary_path = os.path.join(output_root, "_run_summary.json")

    with open(agg_path, 'w', encoding='utf-8') as f:
        json.dump(summary["aggregate_tokens"], f, ensure_ascii=False, indent=2)
    with open(run_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "#" * 70)
    print(f"# 批量处理完成: {len(videos)} 个视频")
    grand = summary["aggregate_tokens"].get("grand_total", {})
    print(f"# 总 Token: prompt={grand.get('prompt_tokens', 0)}  "
          f"completion={grand.get('completion_tokens', 0)}  "
          f"total={grand.get('total_tokens', 0)}  calls={grand.get('calls', 0)}")
    print(f"# 聚合表: {agg_path}")
    print(f"# 运行摘要: {run_summary_path}")
    print("#" * 70)

    return summary
