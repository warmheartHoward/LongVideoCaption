import threading
from typing import Any, Dict


class TokenTracker:
    def __init__(self) -> None:
        self.stages: Dict[str, Dict[str, int]] = {}

    def _ensure(self, stage: str) -> Dict[str, int]:
        if stage not in self.stages:
            self.stages[stage] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            }
        return self.stages[stage]

    def record(self, stage: str, usage: Any) -> None:
        if usage is None:
            return
        slot = self._ensure(stage)
        slot["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        slot["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        slot["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
        slot["calls"] += 1

    def grand_total(self) -> Dict[str, int]:
        return {
            "prompt_tokens": sum(s["prompt_tokens"] for s in self.stages.values()),
            "completion_tokens": sum(s["completion_tokens"] for s in self.stages.values()),
            "total_tokens": sum(s["total_tokens"] for s in self.stages.values()),
            "calls": sum(s["calls"] for s in self.stages.values()),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"per_stage": self.stages, "grand_total": self.grand_total()}


class GlobalTokenAggregator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._per_video: Dict[str, Dict[str, Any]] = {}

    def add(self, video_key: str, tracker_dict: Dict[str, Any]) -> None:
        with self._lock:
            self._per_video[video_key] = tracker_dict

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            grand = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            }
            per_stage_totals: Dict[str, Dict[str, int]] = {}
            for v in self._per_video.values():
                gt = v.get("grand_total", {})
                for k in grand:
                    grand[k] += int(gt.get(k, 0) or 0)
                for stage_name, stage_data in v.get("per_stage", {}).items():
                    slot = per_stage_totals.setdefault(
                        stage_name,
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
                    )
                    for k in slot:
                        slot[k] += int(stage_data.get(k, 0) or 0)
            return {
                "per_video": dict(self._per_video),
                "per_stage_totals": per_stage_totals,
                "grand_total": grand,
            }
