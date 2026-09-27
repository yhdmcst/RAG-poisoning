import os
import shutil
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models"

REPOS = [
    ("BAAI/bge-base-en-v1.5", MODEL_ROOT / "BAAI" / "bge-base-en-v1.5"),
    ("BAAI/bge-reranker-base", MODEL_ROOT / "BAAI" / "bge-reranker-base"),
    ("Qwen/Qwen2.5-7B-Instruct", MODEL_ROOT / "Qwen" / "Qwen2.5-7B-Instruct"),
]


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            copy_tree(item, target)
        elif not target.exists() or item.stat().st_size != target.stat().st_size:
            shutil.copy2(item, target)


def main() -> int:
    os.environ.setdefault("MODELSCOPE_HOME", str(MODEL_ROOT / "modelscope_home"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(MODEL_ROOT / "modelscope_cache"))
    os.environ.setdefault("MODELSCOPE_SDK_CACHE", str(MODEL_ROOT / "modelscope_home"))
    os.environ.setdefault("XDG_CACHE_HOME", str(MODEL_ROOT / "xdg_cache"))
    for model_id, local_dir in REPOS:
        print(f"Downloading {model_id}", flush=True)
        cache_path = Path(snapshot_download(model_id, cache_dir=str(MODEL_ROOT / "modelscope_cache")))
        print(f"Cached at {cache_path}", flush=True)
        print(f"Copying to {local_dir}", flush=True)
        copy_tree(cache_path, local_dir)
        print(f"Done {model_id} -> {local_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
