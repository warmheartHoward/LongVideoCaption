import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator


def _format_seconds(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _empty_slot() -> Dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
        "seconds": 0.0,
    }


class TokenTracker:
    def __init__(self) -> None:
        self.stages: Dict[str, Dict[str, Any]] = {}
        self._start_time = time.time()
        self._previous_total_seconds = 0.0
        self._lock = threading.Lock()

    def _ensure(self, stage: str) -> Dict[str, Any]:
        if stage not in self.stages:
            self.stages[stage] = _empty_slot()
        return self.stages[stage]

    def record(self, stage: str, usage: Any) -> None:
        if usage is None:
            return
        with self._lock:
            slot = self._ensure(stage)
            slot["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            slot["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            slot["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
            slot["calls"] += 1

    def add_stage_seconds(self, stage: str, seconds: float) -> None:
        if seconds <= 0:
            return
        with self._lock:
            slot = self._ensure(stage)
            slot["seconds"] = round(float(slot.get("seconds", 0.0)) + float(seconds), 2)

    @contextmanager
    def stage_timer(self, stage: str) -> Iterator[None]:
        start = time.time()
        try:
            yield
        finally:
            self.add_stage_seconds(stage, time.time() - start)

    def load_from_dict(self, prev: Dict[str, Any]) -> None:
        """续跑时把上一份 token_usage.json 作为基线累加进来（token + 各 stage 秒数）。"""
        if not prev:
            return
        per_stage = prev.get("per_stage", {}) or {}
        with self._lock:
            for stage, data in per_stage.items():
                slot = self._ensure(stage)
                slot["prompt_tokens"] += int(data.get("prompt_tokens", 0) or 0)
                slot["completion_tokens"] += int(data.get("completion_tokens", 0) or 0)
                slot["total_tokens"] += int(data.get("total_tokens", 0) or 0)
                slot["calls"] += int(data.get("calls", 0) or 0)
                slot["seconds"] = round(
                    float(slot.get("seconds", 0.0)) + float(data.get("seconds", 0.0) or 0.0), 2
                )
            self._previous_total_seconds += float(prev.get("total_seconds", 0.0) or 0.0)

    def grand_total(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "prompt_tokens": sum(s["prompt_tokens"] for s in self.stages.values()),
                "completion_tokens": sum(s["completion_tokens"] for s in self.stages.values()),
                "total_tokens": sum(s["total_tokens"] for s in self.stages.values()),
                "calls": sum(s["calls"] for s in self.stages.values()),
                "stage_seconds_sum": round(
                    sum(float(s.get("seconds", 0.0)) for s in self.stages.values()), 2
                ),
            }

    def current_run_seconds(self) -> float:
        return max(0.0, time.time() - self._start_time)

    def total_seconds(self) -> float:
        return self._previous_total_seconds + self.current_run_seconds()

    def to_dict(self) -> Dict[str, Any]:
        cur = self.current_run_seconds()
        total = self._previous_total_seconds + cur
        with self._lock:
            stages_snapshot: Dict[str, Dict[str, Any]] = {}
            for name, slot in self.stages.items():
                stages_snapshot[name] = {
                    "prompt_tokens": slot["prompt_tokens"],
                    "completion_tokens": slot["completion_tokens"],
                    "total_tokens": slot["total_tokens"],
                    "calls": slot["calls"],
                    "seconds": round(float(slot.get("seconds", 0.0)), 2),
                    "seconds_hms": _format_seconds(float(slot.get("seconds", 0.0))),
                }
        return {
            "per_stage": stages_snapshot,
            "grand_total": self.grand_total(),
            "current_run_seconds": round(cur, 2),
            "current_run_seconds_hms": _format_seconds(cur),
            "previous_total_seconds": round(self._previous_total_seconds, 2),
            "previous_total_seconds_hms": _format_seconds(self._previous_total_seconds),
            "total_seconds": round(total, 2),
            "total_seconds_hms": _format_seconds(total),
        }


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
            per_stage_totals: Dict[str, Dict[str, Any]] = {}
            total_seconds_sum = 0.0
            for v in self._per_video.values():
                gt = v.get("grand_total", {})
                for k in grand:
                    grand[k] += int(gt.get(k, 0) or 0)
                for stage_name, stage_data in v.get("per_stage", {}).items():
                    slot = per_stage_totals.setdefault(
                        stage_name,
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                         "calls": 0, "seconds": 0.0},
                    )
                    slot["prompt_tokens"] += int(stage_data.get("prompt_tokens", 0) or 0)
                    slot["completion_tokens"] += int(stage_data.get("completion_tokens", 0) or 0)
                    slot["total_tokens"] += int(stage_data.get("total_tokens", 0) or 0)
                    slot["calls"] += int(stage_data.get("calls", 0) or 0)
                    slot["seconds"] = round(
                        float(slot["seconds"]) + float(stage_data.get("seconds", 0.0) or 0.0), 2
                    )
                total_seconds_sum += float(v.get("total_seconds", 0.0) or 0.0)

            for slot in per_stage_totals.values():
                slot["seconds_hms"] = _format_seconds(float(slot["seconds"]))

            return {
                "per_video": dict(self._per_video),
                "per_stage_totals": per_stage_totals,
                "grand_total": grand,
                "total_seconds": round(total_seconds_sum, 2),
                "total_seconds_hms": _format_seconds(total_seconds_sum),
            }
