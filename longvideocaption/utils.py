import re


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}]"


def parse_timestamp_to_seconds(ts_str: str) -> float:
    if not isinstance(ts_str, str):
        return 0.0
    clean_str = ts_str.strip('[] ')
    try:
        parts = clean_str.split(':')
        if len(parts) == 3:
            sec_parts = parts[2].split('.')
            seconds = float(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(sec_parts[0]))
            if len(sec_parts) > 1:
                seconds += float(f"0.{sec_parts[1]}")
            return seconds
        return 0.0
    except Exception:
        return 0.0


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
