import os, sys, types

# stubs
def _cv2():
    s = types.ModuleType("cv2")
    s.CAP_PROP_FPS=5; s.CAP_PROP_FRAME_WIDTH=3; s.CAP_PROP_FRAME_HEIGHT=4
    s.CAP_PROP_FRAME_COUNT=7; s.CAP_PROP_POS_FRAMES=1
    s.IMWRITE_JPEG_QUALITY=1; s.INTER_AREA=3
    class C:
        def __init__(s_, p): pass
        def get(s_, p): return {5:30.0,3:1920.0,4:1080.0,7:9000}.get(p,0)
        def set(s_, *a, **k): return True
        def read(s_): return False, None
        def isOpened(s_): return True
        def release(s_): pass
    s.VideoCapture = C
    s.VideoWriter = lambda *a, **k: type("W",(),{"write":lambda s,f:None,"release":lambda s:None})()
    s.VideoWriter_fourcc = lambda *a, **k: 0
    s.imencode = lambda *a, **k: (True, b"x")
    s.resize = lambda img, sz, interpolation=None: img
    return s

def _sd():
    s = types.ModuleType("scenedetect")
    s.detect = lambda *a, **k: []
    s.ContentDetector = type("CD", (), {"__init__": lambda *a, **k: None})
    return s

def _httpx():
    s = types.ModuleType("httpx")
    s.Client = type("C", (), {"__init__": lambda *a, **k: None})
    return s

def _openai():
    s = types.ModuleType("openai")
    s.OpenAI = type("O", (), {"__init__": lambda *a, **k: None})
    return s

sys.modules.setdefault("cv2", _cv2())
sys.modules.setdefault("scenedetect", _sd())
sys.modules.setdefault("httpx", _httpx())
sys.modules.setdefault("openai", _openai())
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from longvideocaption.pass1 import _build_scene_whitelist
from longvideocaption.frame_extractor import get_target_timestamps

P = F = 0
def chk(name, cond, detail=""):
    global P, F
    if cond:
        P += 1; print(f"PASS  {name}")
    else:
        F += 1; print(f"FAIL  {name}  {detail}")

# Scenes: full video has 4 scenes
scenes = [(0.0, 12.4), (12.4, 30.0), (30.0, 58.7), (58.7, 90.0)]

# chunk [0, 60]: boundaries within = {0.0, 12.4, 30.0, 58.7} (90.0 outside)
out = _build_scene_whitelist(scenes, 0.0, 60.0)
chk("chunk[0,60] picks 4 boundaries", out == [0.0, 12.4, 30.0, 58.7], f"got {out}")

# chunk [12.4, 60]: 12.4, 30.0, 58.7 (boundary on edge included)
out = _build_scene_whitelist(scenes, 12.4, 60.0)
chk("chunk left edge inclusive", out == [12.4, 30.0, 58.7], f"got {out}")

# chunk [60, 90]: only 90.0 ... wait 58.7 < 60, 90.0 == 90.0 → just [90.0]
out = _build_scene_whitelist(scenes, 60.0, 90.0)
chk("chunk[60,90] picks tail boundary only", out == [90.0], f"got {out}")

# Empty: chunk inside one scene
scenes2 = [(0.0, 100.0)]
out = _build_scene_whitelist(scenes2, 30.0, 60.0)
chk("chunk fully inside one scene → empty whitelist", out == [], f"got {out}")

# Dedup: same boundary appears as s_end and next s_start
scenes3 = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
out = _build_scene_whitelist(scenes3, 0.0, 30.0)
chk("dedup adjacent boundaries", out == [0.0, 10.0, 20.0, 30.0], f"got {out}")

# Float rounding (12.345678 → 12.346)
scenes4 = [(0.0, 12.345678), (12.345678, 25.0)]
out = _build_scene_whitelist(scenes4, 0.0, 25.0)
chk("rounds to 3 decimals", 12.346 in out, f"got {out}")

# get_target_timestamps with precomputed_scenes — does NOT call detect again
class CountingDetect:
    n = 0

import longvideocaption.frame_extractor as fe
orig = fe.detect_scenes
fe.detect_scenes = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called"))
try:
    ts = get_target_timestamps("dummy", 0.0, 60.0, "scenedetect", 27.0, 5,
                                precomputed_scenes=scenes)
    chk("get_target_timestamps reuses precomputed_scenes (no re-detect)", True)
except AssertionError as e:
    chk("get_target_timestamps reuses precomputed_scenes (no re-detect)", False, str(e))
finally:
    fe.detect_scenes = orig

# get_target_timestamps without precomputed → falls back to detect_scenes (returns [] in stub)
called = {"n": 0}
def fake_detect(*a, **k):
    called["n"] += 1
    return []
fe.detect_scenes = fake_detect
try:
    ts = get_target_timestamps("dummy", 0.0, 60.0, "scenedetect", 27.0, 5)
    chk("get_target_timestamps detects when precomputed is None",
        called["n"] == 1, f"called={called['n']}")
finally:
    fe.detect_scenes = orig

print(f"\nPASS={P} FAIL={F}")
sys.exit(0 if F == 0 else 1)
