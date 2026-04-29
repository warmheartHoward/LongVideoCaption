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
    total_millis = int(round(float(seconds) * 1000))
    if total_millis % 1000:
        return format_timestamp(seconds)
    total = total_millis // 1000
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def parse_timestamp_to_seconds_strict(ts_str: str) -> Optional[float]:
    if not isinstance(ts_str, str):
        return None
    clean_str = ts_str.strip('[] ')
    try:
        parts = clean_str.split(':')
        if len(parts) == 3:
            sec_parts = parts[2].split('.')
            seconds = float(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(sec_parts[0]))
            if len(sec_parts) > 1:
                seconds += float(f"0.{sec_parts[1]}")
            return seconds
        return None
    except Exception:
        return None


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
