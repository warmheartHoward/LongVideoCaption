__all__ = ["PipelineConfig", "hyper_signature", "process_single_video", "run_batch"]


def __getattr__(name):
    if name in ("PipelineConfig", "hyper_signature"):
        from .config import PipelineConfig, hyper_signature
        return {"PipelineConfig": PipelineConfig, "hyper_signature": hyper_signature}[name]
    if name == "process_single_video":
        from .pipeline import process_single_video
        return process_single_video
    if name == "run_batch":
        from .runner import run_batch
        return run_batch
    raise AttributeError(f"module 'longvideocaption' has no attribute {name!r}")
