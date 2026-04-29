import re
from typing import Optional


def format_timestamp(seconds: float) -> str:
    total_millis = int(round(float(seconds) * 1000))
    hours = total_millis // 3_600_000
    minutes = (total_millis % 3_600_000) // 60_000
    secs = (total_millis % 60_000) // 1000
    millis = total_millis % 1000
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}]"


def format_timestamp_sec(seconds: float) -> str:
    total = int(round(float(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def format_timestamp_for_mode(seconds: float, mode: str) -> str:
    if mode == "millisecond":
        return format_timestamp(seconds)
    if mode == "second":
        return format_timestamp_sec(seconds)
    raise ValueError(f"不支持的 timestamp 模式: {mode!r}，应为 'second' 或 'millisecond'")


_TIMESTAMP_RE = re.compile(r'^\[?(\d{2,}):([0-5]\d):([0-5]\d)(?:\.(\d{3}))?\]?$')


def parse_timestamp_to_seconds_strict(ts_str: str) -> Optional[float]:
    if not isinstance(ts_str, str):
        return None
    match = _TIMESTAMP_RE.fullmatch(ts_str.strip())
    if not match:
        return None

    hours, minutes, seconds, millis = match.groups()
    total = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    if millis is not None:
        total += int(millis) / 1000.0
    return float(total)


def parse_timestamp_to_seconds(ts_str: str) -> float:
    seconds = parse_timestamp_to_seconds_strict(ts_str)
    if seconds is None:
        return 0.0
    return seconds


def clean_json_response(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_filename(name: str) -> str:
    s = _INVALID_FS_CHARS.sub("_", name)
    s = s.strip(" .")
    return s or "unnamed"


def safe_replace(text: str, old: str, new: str) -> str:
    if not isinstance(text, str) or not old or old == new:
        return text
    pattern = r"(?<![\w一-鿿])" + re.escape(old) + r"(?![\w一-鿿])"
    return re.sub(pattern, new, text)
