import os
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models"

REPOS = [
    ("BAAI/bge-base-en-v1.5", MODEL_ROOT / "BAAI" / "bge-base-en-v1.5"),
    ("BAAI/bge-reranker-base", MODEL_ROOT / "BAAI" / "bge-reranker-base"),
    ("Qwen/Qwen2.5-7B-Instruct", MODEL_ROOT / "Qwen" / "Qwen2.5-7B-Instruct"),
]


def main() -> int:
    os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(MODEL_ROOT / "hf_cache"))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    for repo_id, local_dir in REPOS:
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} -> {local_dir}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=2,
        )
        print(f"Done {repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
