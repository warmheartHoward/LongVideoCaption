import base64
import os
import time
from typing import List, Tuple

import cv2
import numpy as np

from .utils import format_timestamp


def get_target_timestamps(
    video_path: str,
    chunk_start: float,
    chunk_end: float,
    strategy: str,
    threshold: float,
    max_frames: int,
    log_prefix: str = "",
) -> List[float]:
    print(f"\n{log_prefix}[序列构建] 分析 {format_timestamp(chunk_start)} 到 {format_timestamp(chunk_end)}。策略: {strategy}")
    if strategy == "scenedetect":
        from scenedetect import detect, ContentDetector

        scene_list = detect(video_path, ContentDetector(threshold=threshold))
        valid_scenes = []
        for scene in scene_list:
            s_start = max(scene[0].get_seconds(), chunk_start)
            s_end = min(scene[1].get_seconds(), chunk_end)
            if s_start < s_end:
                valid_scenes.append((s_start, s_end))
        if not valid_scenes:
            valid_scenes = [(chunk_start, chunk_end)]

        total_duration = sum(s_end - s_start for s_start, s_end in valid_scenes) or 1e-5
        timestamps: List[float] = []
        remaining_frames = max_frames
        for i, (s_start, s_end) in enumerate(valid_scenes):
            duration = s_end - s_start
            alloc = (
                remaining_frames
                if i == len(valid_scenes) - 1
                else max(1, int(round((duration / total_duration) * max_frames)))
            )
            alloc = min(alloc, remaining_frames)
            remaining_frames -= alloc
            if alloc == 1:
                timestamps.append(s_start + duration / 2.0)
            elif alloc > 1:
                timestamps.extend(np.linspace(s_start, s_end, alloc + 2)[1:-1].tolist())
        return sorted(list(set(timestamps)))
    else:
        return np.linspace(chunk_start, chunk_end, max_frames).tolist()


def get_base64_frames(
    video_path: str,
    target_timestamps: List[float],
    max_width: int,
    jpg_quality: int,
) -> Tuple[List[float], List[str]]:
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    scale = max_width / orig_width if orig_width > max_width else 1.0
    target_w, target_h = int(orig_width * scale), int(orig_height * scale)

    valid_timestamps: List[float] = []
    base64_frames: List[str] = []
    for t in target_timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * orig_fps))
        ret, frame = cap.read()
        if ret:
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
            valid_timestamps.append(t)
    cap.release()
    return valid_timestamps, base64_frames


def get_raw_chunk_video_base64(
    video_path: str,
    chunk_start: float,
    chunk_end: float,
    max_width: int,
    target_fps: float,
    temp_dir: str = ".",
) -> Tuple[List[float], str]:
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    scale = max_width / orig_width if orig_width > max_width else 1.0
    target_w, target_h = int(orig_width * scale), int(orig_height * scale)

    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = os.path.join(temp_dir, f"temp_chunk_{int(time.time() * 1000)}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_filename, fourcc, target_fps, (target_w, target_h))

    start_frame = int(chunk_start * orig_fps)
    end_frame = int(chunk_end * orig_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_interval = max(1, int(round(orig_fps / target_fps)))
    valid_timestamps: List[float] = []
    curr_frame = start_frame

    while curr_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (curr_frame - start_frame) % frame_interval == 0:
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            out.write(resized)
            valid_timestamps.append(curr_frame / orig_fps)
        curr_frame += 1

    out.release()
    cap.release()

    video_b64 = ""
    try:
        with open(temp_filename, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode('utf-8')
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return valid_timestamps, video_b64


def extract_single_frame_base64(video_path: str, timestamp_str: str, max_width: int = 960) -> str:
    if not isinstance(timestamp_str, str):
        return ""
    clean_str = timestamp_str.strip('[] ')
    try:
        parts = clean_str.split(':')
        if len(parts) != 3:
            return ""
        sec_parts = parts[2].split('.')
        seconds = float(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(sec_parts[0]))
        if len(sec_parts) > 1:
            seconds += float(f"0.{sec_parts[1]}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ""

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        orig_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        orig_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        scale = max_width / orig_width if orig_width > max_width else 1.0

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * orig_fps))
        ret, frame = cap.read()
        b64_str = ""
        if ret:
            resized = cv2.resize(frame, (int(orig_width * scale), int(orig_height * scale)))
            _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            b64_str = base64.b64encode(buffer).decode('utf-8')
        cap.release()
        return b64_str
    except Exception as e:
        print(f"提取帧异常 {timestamp_str}: {e}")
        return ""


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(duration)
