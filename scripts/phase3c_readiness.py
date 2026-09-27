#!/usr/bin/env python3
"""Phase 3C environment, model, data, and corpus readiness checks.

This script intentionally does not run the formal Phase 3B R1-R8 matrix.
It prepares and audits the prerequisites needed to restart that matrix with
a compliant raw NQ passage corpus.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


OUT_DIR = Path("results/phase3c_infra_data_readiness")
DATA_DIR = OUT_DIR / "data"
PARTS_DIR = DATA_DIR / "parts"
EXTRACT_DIR = DATA_DIR / "beir_nq"
ZIP_PATH = DATA_DIR / "nq.zip"
CHUNKS_PATH = DATA_DIR / "chunks_c128_s32.jsonl.gz"
CHUNK_MANIFEST_PATH = DATA_DIR / "chunk_manifest.json"
SPLIT_PATH = DATA_DIR / "splits_seed20260905.json"

BEIR_NQ_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
EXPECTED_NQ_ZIP_BYTES = 498_307_926
EXPECTED_NQ_ZIP_ETAG = "6075b282-1db39356"

NQ_OPEN_TRAIN = Path("/data/modelscope/hub/media_resources/evalscope/data/nq-open/nq-open-train.jsonl")
NQ_OPEN_VALID = Path("/data/modelscope/hub/media_resources/evalscope/data/nq-open/nq-open-validation.jsonl")
OPENCOMPASS_NQ_DEV = Path("/data/modelscope/hub/media_resources/evalscope/data/nq/nq-dev.qa.csv")
OPENCOMPASS_NQ_TEST = Path("/data/modelscope/hub/media_resources/evalscope/data/nq/nq-test.qa.csv")
OPENCOMPASS_NQ_DEV_ALT = Path("/data/lzh/opencompass/data/nq/nq-dev.qa.csv")
OPENCOMPASS_NQ_TEST_ALT = Path("/data/lzh/opencompass/data/nq/nq-test.qa.csv")

CONFIG = {
    "dataset": "Natural Questions / NQ",
    "calibration_queries": 100,
    "eval_queries": 100,
    "split_seed": 20260905,
    "attack_seed": 42,
    "chunking": "C128-S32",
    "chunk_size": 128,
    "stride": 32,
    "min_chunk_length": 20,
    "retriever": "BAAI/bge-base-en-v1.5",
    "top_m": 50,
    "reranker": "BAAI/bge-reranker-base",
    "K_gen": 5,
    "generator": "Qwen/Qwen2.5-7B-Instruct",
    "decoding": {
        "temperature": 0,
        "do_sample": False,
        "top_p": 1.0,
        "max_new_tokens": 64,
    },
    "attack": "targeted_template_poison",
    "poison_budget": 5,
    "safegate_signal": "query_overlap_anomaly",
    "alpha": 0.05,
}

REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "sentence-transformers",
    "FlagEmbedding",
    "faiss-cpu",
    "numpy",
    "scikit-learn",
    "PyYAML",
    "tqdm",
]

IMPORT_NAMES = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "sentence-transformers": "sentence_transformers",
    "FlagEmbedding": "FlagEmbedding",
    "faiss-cpu": "faiss",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "PyYAML": "yaml",
    "tqdm": "tqdm",
}

BGE_RETRIEVER_CANDIDATES = [
    OUT_DIR / "modelscope_cache" / "BAAI" / "bge-base-en-v1___5",
    Path("results/phase3_minipilot_seed42/modelscope_cache/BAAI/bge-base-en-v1___5"),
    Path("/data/modelscope/hub/BAAI/bge-base-en-v1___5"),
    Path("/data/modelscope/hub/BAAI/bge-base-en-v1.5"),
]

BGE_RERANKER_CANDIDATES = [
    OUT_DIR / "modelscope_cache" / "BAAI" / "bge-reranker-base",
    Path("results/phase3_minipilot_seed42/modelscope_cache/BAAI/bge-reranker-base"),
    Path("/data/modelscope/hub/BAAI/bge-reranker-base"),
]

QWEN_CANDIDATES = [
    Path("/data/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"),
    Path("/data/modelscope/hub/qwen/Qwen2___5-7B-Instruct"),
    Path("results/phase3_minipilot_seed42/modelscope_cache/Qwen/Qwen2___5-7B-Instruct"),
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}


def ensure_dirs() -> None:
    for path in [OUT_DIR, DATA_DIR, PARTS_DIR, EXTRACT_DIR, OUT_DIR / "logs", OUT_DIR / "cache"]:
        path.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_cmd(args: Sequence[str], timeout: Optional[int] = None) -> dict:
    started = time.time()
    try:
        proc = subprocess.run(
            list(args),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "args": list(args),
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "elapsed_s": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            "args": list(args),
            "returncode": -1,
            "stdout": "",
            "stderr": repr(exc),
            "elapsed_s": round(time.time() - started, 3),
        }


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def line_count(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return sum(1 for _ in f)


def package_version(dist_name: str, import_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        return False, None, None
    version = None
    for candidate in [dist_name, import_name, import_name.replace("_", "-")]:
        try:
            version = importlib.metadata.version(candidate)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    return True, version, getattr(spec, "origin", None)


def check_environment() -> dict:
    ensure_dirs()
    nvidia = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,driver_version",
            "--format=csv,noheader",
        ],
        timeout=30,
    )
    pip_freeze = run_cmd([sys.executable, "-m", "pip", "freeze"], timeout=120)
    if pip_freeze["returncode"] == 0:
        write_text(OUT_DIR / "requirements_freeze.txt", pip_freeze["stdout"] + "\n")
    else:
        write_text(
            OUT_DIR / "requirements_freeze.txt",
            "# pip freeze failed\n" + pip_freeze["stderr"] + "\n",
        )

    package_rows = []
    for dist_name in REQUIRED_PACKAGES:
        ok, version, origin = package_version(dist_name, IMPORT_NAMES[dist_name])
        package_rows.append(
            {
                "package": dist_name,
                "import_name": IMPORT_NAMES[dist_name],
                "installed": ok,
                "version": version,
                "origin": origin,
            }
        )

    torch_info = {"import_ok": False}
    try:
        import torch

        torch_info = {
            "import_ok": True,
            "version": torch.__version__,
            "cuda_is_available": bool(torch.cuda.is_available()),
            "torch_version_cuda": torch.version.cuda,
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        torch_info = {"import_ok": False, "error": repr(exc)}

    env = {
        "created_at": now(),
        "os": {
            "platform": platform.platform(),
            "uname": " ".join(platform.uname()),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
        },
        "gpu": {
            "nvidia_smi_returncode": nvidia["returncode"],
            "nvidia_smi_stdout": nvidia["stdout"],
            "nvidia_smi_stderr": nvidia["stderr"],
        },
        "torch": torch_info,
        "packages": package_rows,
        "pip_freeze_returncode": pip_freeze["returncode"],
    }
    write_json(OUT_DIR / "environment_manifest.json", env)
    write_environment_report(env)
    return env


def write_environment_report(env: dict) -> None:
    missing = [p["package"] for p in env["packages"] if not p["installed"]]
    pkg_lines = [
        f"- {p['package']}: {'installed' if p['installed'] else 'missing'}"
        + (f" ({p['version']})" if p.get("version") else "")
        for p in env["packages"]
    ]
    text = f"""# ENVIRONMENT_READINESS

Status: {'READY' if not missing and env['torch'].get('cuda_is_available') else 'BLOCKED_OR_PARTIAL'}

## OS

- Platform: `{env['os']['platform']}`
- Uname: `{env['os']['uname']}`

## Python

- Executable: `{env['python']['executable']}`
- Version: `{env['python']['version']}`
- Prefix: `{env['python']['prefix']}`
- CONDA_PREFIX: `{env['python'].get('conda_prefix')}`
- VIRTUAL_ENV: `{env['python'].get('virtual_env')}`

## GPU

`nvidia-smi` return code: `{env['gpu']['nvidia_smi_returncode']}`

```text
{env['gpu']['nvidia_smi_stdout'] or env['gpu']['nvidia_smi_stderr']}
```

## PyTorch

- Import OK: `{env['torch'].get('import_ok')}`
- Version: `{env['torch'].get('version')}`
- torch.cuda.is_available(): `{env['torch'].get('cuda_is_available')}`
- torch.version.cuda: `{env['torch'].get('torch_version_cuda')}`
- CUDA device count: `{env['torch'].get('cuda_device_count')}`
- CUDA device 0: `{env['torch'].get('cuda_device_0')}`

## Required Packages

{chr(10).join(pkg_lines)}

## Requirements Freeze

Saved to `requirements_freeze.txt`.
"""
    write_text(OUT_DIR / "ENVIRONMENT_READINESS.md", text)


def model_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def find_incomplete_files(path: Path) -> List[str]:
    if not path.exists():
        return []
    bad = []
    for pattern in ["*.safetensors.incomplete", "*.incomplete", "*.tmp"]:
        bad.extend(str(p) for p in path.rglob(pattern))
    return sorted(set(bad))


def first_existing_model(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if (path / "config.json").exists():
            return path
    return None


def resolve_model_paths() -> dict:
    paths = {
        "retriever": first_existing_model(BGE_RETRIEVER_CANDIDATES),
        "reranker": first_existing_model(BGE_RERANKER_CANDIDATES),
        "generator": first_existing_model(QWEN_CANDIDATES),
    }
    return {k: str(v.resolve()) if v else None for k, v in paths.items()}


def check_models(qwen_smoke: bool = True) -> dict:
    ensure_dirs()
    started = time.time()
    paths = resolve_model_paths()
    manifest = {
        "created_at": now(),
        "fixed_models": {
            "retriever": CONFIG["retriever"],
            "reranker": CONFIG["reranker"],
            "generator": CONFIG["generator"],
        },
        "paths": paths,
        "models": {},
        "smoke": {},
    }

    for key, path_str in paths.items():
        path = Path(path_str) if path_str else None
        manifest["models"][key] = {
            "path": path_str,
            "exists": bool(path and path.exists()),
            "config_json": bool(path and (path / "config.json").exists()),
            "size_bytes": model_size_bytes(path) if path else 0,
            "incomplete_files": find_incomplete_files(path) if path else [],
        }

    try:
        import numpy as np
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        manifest["smoke"]["torch_cuda_available"] = bool(torch.cuda.is_available())
        manifest["smoke"]["device_used"] = device

        if paths["retriever"]:
            t0 = time.time()
            retriever = SentenceTransformer(paths["retriever"], device=device)
            q_emb = retriever.encode(["what is the capital of france"], normalize_embeddings=True)
            d_emb = retriever.encode(
                ["Paris is the capital of France.", "Berlin is in Germany."],
                normalize_embeddings=True,
            )
            scores = np.matmul(q_emb, np.asarray(d_emb).T).tolist()[0]
            manifest["smoke"]["retriever"] = {
                "status": "ok",
                "elapsed_s": round(time.time() - t0, 3),
                "scores": scores,
                "top_index": int(np.argmax(scores)),
            }
            del retriever
            cleanup_cuda()
        else:
            manifest["smoke"]["retriever"] = {"status": "missing_path"}

        if paths["reranker"]:
            t0 = time.time()
            reranker = CrossEncoder(paths["reranker"], device=device)
            scores = reranker.predict(
                [
                    ("what is the capital of france", "Paris is the capital of France."),
                    ("what is the capital of france", "The Pacific Ocean is large."),
                ]
            )
            scores_list = [float(x) for x in list(scores)]
            manifest["smoke"]["reranker"] = {
                "status": "ok",
                "elapsed_s": round(time.time() - t0, 3),
                "scores": scores_list,
                "top_index": int(max(range(len(scores_list)), key=scores_list.__getitem__)),
            }
            del reranker
            cleanup_cuda()
        else:
            manifest["smoke"]["reranker"] = {"status": "missing_path"}

        if qwen_smoke and paths["generator"]:
            manifest["smoke"]["generator"] = qwen_generation_smoke(paths["generator"])
        elif not paths["generator"]:
            manifest["smoke"]["generator"] = {"status": "missing_path"}
        else:
            manifest["smoke"]["generator"] = {"status": "skipped"}
    except Exception as exc:
        manifest["smoke"]["exception"] = repr(exc)

    manifest["elapsed_s"] = round(time.time() - started, 3)
    manifest["ready"] = models_ready(manifest)
    write_json(OUT_DIR / "model_manifest.json", manifest)
    write_model_report(manifest)
    return manifest


def qwen_generation_smoke(model_path: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    messages = [{"role": "user", "content": "Answer with one word: What is the capital of France?"}]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
    except Exception:
        encoded = tokenizer("Answer with one word: What is the capital of France?", return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=16,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, input_ids.shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    del model
    cleanup_cuda()
    return {
        "status": "ok" if text else "empty_output",
        "elapsed_s": round(time.time() - t0, 3),
        "device": device,
        "output": text,
    }


def cleanup_cuda() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def models_ready(manifest: dict) -> bool:
    for key in ["retriever", "reranker", "generator"]:
        row = manifest["models"].get(key, {})
        if not row.get("exists") or not row.get("config_json") or row.get("incomplete_files"):
            return False
    smoke = manifest.get("smoke", {})
    return (
        smoke.get("retriever", {}).get("status") == "ok"
        and smoke.get("reranker", {}).get("status") == "ok"
        and smoke.get("generator", {}).get("status") == "ok"
    )


def write_model_report(manifest: dict) -> None:
    lines = ["# model_readiness_report", ""]
    lines.append(f"Status: {'READY' if manifest.get('ready') else 'BLOCKED_OR_PARTIAL'}")
    lines.append("")
    for key in ["retriever", "reranker", "generator"]:
        row = manifest["models"].get(key, {})
        lines.extend(
            [
                f"## {key}",
                "",
                f"- Required model: `{manifest['fixed_models'][key]}`",
                f"- Resolved path: `{row.get('path')}`",
                f"- Exists: `{row.get('exists')}`",
                f"- config.json: `{row.get('config_json')}`",
                f"- Size bytes: `{row.get('size_bytes')}`",
                f"- Incomplete files: `{len(row.get('incomplete_files') or [])}`",
                "",
            ]
        )
    lines.append("## Smoke")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(manifest.get("smoke", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    if manifest.get("gpu_visible_smoke"):
        lines.append("")
        lines.append("## GPU-Visible Smoke")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(manifest.get("gpu_visible_smoke", {}), ensure_ascii=False, indent=2))
        lines.append("```")
    write_text(OUT_DIR / "model_readiness_report.md", "\n".join(lines) + "\n")


def curl_head() -> dict:
    result = run_cmd(["curl", "-I", "--connect-timeout", "60", BEIR_NQ_URL], timeout=120)
    headers = result["stdout"] + "\n" + result["stderr"]
    content_length = None
    etag = None
    accept_ranges = None
    for line in headers.splitlines():
        lower = line.lower()
        if lower.startswith("content-length:"):
            content_length = int(line.split(":", 1)[1].strip())
        elif lower.startswith("etag:"):
            etag = line.split(":", 1)[1].strip().strip('"')
        elif lower.startswith("accept-ranges:"):
            accept_ranges = line.split(":", 1)[1].strip()
    return {
        "url": BEIR_NQ_URL,
        "returncode": result["returncode"],
        "content_length": content_length,
        "etag": etag,
        "accept_ranges": accept_ranges,
        "raw": result,
    }


def seed_first_part_from_partial(part_size: int, partial_path: Optional[str]) -> dict:
    if not partial_path:
        return {"seeded": False, "reason": "no_partial_path"}
    src = Path(partial_path)
    part0 = PARTS_DIR / "part_000.bin"
    if not src.exists():
        return {"seeded": False, "reason": "partial_missing", "partial_path": str(src)}
    if part0.exists() and part0.stat().st_size >= min(part_size, src.stat().st_size):
        return {"seeded": False, "reason": "part0_already_seeded", "part0_bytes": part0.stat().st_size}
    to_copy = min(part_size, src.stat().st_size)
    with src.open("rb") as fin, part0.open("wb") as fout:
        shutil.copyfileobj(_LimitedReader(fin, to_copy), fout)
    return {
        "seeded": True,
        "partial_path": str(src),
        "copied_bytes": to_copy,
        "part0": str(part0),
    }


class _LimitedReader:
    def __init__(self, f, remaining: int):
        self.f = f
        self.remaining = remaining

    def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        if size < 0 or size > self.remaining:
            size = self.remaining
        data = self.f.read(size)
        self.remaining -= len(data)
        return data


def download_part(index: int, start: int, end: int) -> dict:
    part_path = PARTS_DIR / f"part_{index:03d}.bin"
    expected = end - start + 1
    have = part_path.stat().st_size if part_path.exists() else 0
    if have == expected:
        return {"index": index, "status": "already_complete", "bytes": have}
    if have > expected:
        part_path.unlink()
        have = 0
    tail_start = start + have
    tmp_path = PARTS_DIR / f"part_{index:03d}.tail"
    if tmp_path.exists():
        tmp_path.unlink()
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "20",
        "--retry-delay",
        "10",
        "--connect-timeout",
        "60",
        "--range",
        f"{tail_start}-{end}",
        "-o",
        str(tmp_path),
        BEIR_NQ_URL,
    ]
    t0 = time.time()
    result = run_cmd(cmd, timeout=None)
    elapsed = time.time() - t0
    if result["returncode"] != 0:
        return {
            "index": index,
            "status": "failed",
            "returncode": result["returncode"],
            "stderr": result["stderr"][-2000:],
            "stdout": result["stdout"][-1000:],
            "elapsed_s": round(elapsed, 3),
        }
    got = tmp_path.stat().st_size if tmp_path.exists() else 0
    expected_tail = expected - have
    if got != expected_tail:
        return {
            "index": index,
            "status": "bad_size",
            "expected_tail": expected_tail,
            "got_tail": got,
            "elapsed_s": round(elapsed, 3),
        }
    with part_path.open("ab") as fout, tmp_path.open("rb") as fin:
        shutil.copyfileobj(fin, fout)
    tmp_path.unlink()
    final_size = part_path.stat().st_size
    return {
        "index": index,
        "status": "complete" if final_size == expected else "partial",
        "bytes": final_size,
        "expected": expected,
        "elapsed_s": round(elapsed, 3),
    }


def download_beir_nq(workers: int = 12, parts: int = 16, seed_partial: Optional[str] = None) -> dict:
    ensure_dirs()
    t0 = time.time()
    head = curl_head()
    total = head.get("content_length") or EXPECTED_NQ_ZIP_BYTES
    part_size = (total + parts - 1) // parts
    seed = seed_first_part_from_partial(part_size, seed_partial)
    ranges = []
    for i in range(parts):
        start = i * part_size
        end = min(total - 1, start + part_size - 1)
        if start <= end:
            ranges.append((i, start, end))

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(download_part, i, start, end): i for i, start, end in ranges}
        for fut in as_completed(future_map):
            results.append(fut.result())
            write_json(
                OUT_DIR / "download_status.json",
                {
                    "created_at": now(),
                    "head": head,
                    "total_expected_bytes": total,
                    "workers": workers,
                    "parts": parts,
                    "seed": seed,
                    "results": sorted(results, key=lambda x: x["index"]),
                    "zip_exists": ZIP_PATH.exists(),
                    "zip_bytes": ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0,
                },
            )

    complete = all(r.get("status") in {"complete", "already_complete"} for r in results)
    merged = False
    zip_test = None
    sha256 = None
    if complete:
        with ZIP_PATH.open("wb") as fout:
            for i, start, end in ranges:
                part_path = PARTS_DIR / f"part_{i:03d}.bin"
                with part_path.open("rb") as fin:
                    shutil.copyfileobj(fin, fout)
        merged = ZIP_PATH.stat().st_size == total
        if merged:
            sha256 = file_sha256(ZIP_PATH)
            zip_test = test_zip(ZIP_PATH)

    manifest = {
        "created_at": now(),
        "status": "complete" if complete and merged and zip_test and zip_test.get("ok") else "partial_or_failed",
        "url": BEIR_NQ_URL,
        "head": head,
        "total_expected_bytes": total,
        "workers": workers,
        "parts": parts,
        "seed": seed,
        "part_results": sorted(results, key=lambda x: x["index"]),
        "zip_path": str(ZIP_PATH),
        "zip_bytes": ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0,
        "zip_sha256": sha256,
        "zip_test": zip_test,
        "elapsed_s": round(time.time() - t0, 3),
    }
    write_json(OUT_DIR / "download_manifest.json", manifest)
    return manifest


def test_zip(path: Path) -> dict:
    t0 = time.time()
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            names = zf.namelist()
        return {
            "ok": bad is None,
            "bad_file": bad,
            "num_entries": len(names),
            "first_entries": names[:20],
            "elapsed_s": round(time.time() - t0, 3),
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "elapsed_s": round(time.time() - t0, 3)}


def extract_beir_nq() -> dict:
    ensure_dirs()
    if not ZIP_PATH.exists():
        return {"status": "missing_zip", "zip_path": str(ZIP_PATH)}
    zip_test = test_zip(ZIP_PATH)
    if not zip_test.get("ok"):
        return {"status": "bad_zip", "zip_test": zip_test}
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(EXTRACT_DIR)
    files = find_beir_files(EXTRACT_DIR)
    result = {
        "status": "ok" if files.get("corpus") and files.get("queries") and files.get("qrels") else "missing_beir_files",
        "extract_dir": str(EXTRACT_DIR),
        "elapsed_s": round(time.time() - t0, 3),
        "files": files,
    }
    write_json(OUT_DIR / "extract_manifest.json", result)
    return result


def find_beir_files(root: Path) -> dict:
    corpus = sorted(root.rglob("corpus.jsonl"))
    queries = sorted(root.rglob("queries.jsonl"))
    qrels = sorted([p for p in root.rglob("*") if p.is_file() and "qrels" in str(p).lower()])
    qrels = [p for p in qrels if p.suffix in {".tsv", ".txt", ".csv"} or p.name in {"test.tsv", "dev.tsv", "train.tsv"}]
    return {
        "corpus": str(corpus[0]) if corpus else None,
        "queries": str(queries[0]) if queries else None,
        "qrels": str(qrels[0]) if qrels else None,
        "all_qrels_candidates": [str(p) for p in qrels[:10]],
    }


def open_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_queries(path: Path) -> Dict[str, str]:
    queries = {}
    for row in open_jsonl(path):
        qid = str(row.get("_id") or row.get("id") or row.get("query_id"))
        text = row.get("text") or row.get("query") or row.get("question")
        if qid and text:
            queries[qid] = text
    return queries


def read_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        first = True
        for row in reader:
            if first and row and row[0].lower() in {"query-id", "query_id", "qid"}:
                first = False
                continue
            first = False
            if len(row) < 3:
                continue
            qid, docid, score = row[0], row[1], row[2]
            try:
                rel = int(float(score))
            except ValueError:
                rel = 0
            if rel > 0:
                qrels.setdefault(str(qid), {})[str(docid)] = rel
    return qrels


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalize_question(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def clean_answers(raw) -> List[str]:
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            raw = [raw]
    answers = []
    for item in raw or []:
        text = str(item).strip()
        if text and normalize_answer(text):
            answers.append(text)
    return answers


def load_answer_labels() -> Tuple[Dict[str, List[str]], List[str]]:
    mapping: Dict[str, List[str]] = {}
    sources = []
    for path in [NQ_OPEN_TRAIN, NQ_OPEN_VALID]:
        if not path.exists():
            continue
        sources.append(str(path))
        for row in open_jsonl(path):
            question = row.get("question")
            answers = clean_answers(row.get("answer", []))
            if question and answers:
                mapping.setdefault(normalize_question(question), answers)
    for path in [OPENCOMPASS_NQ_DEV, OPENCOMPASS_NQ_TEST, OPENCOMPASS_NQ_DEV_ALT, OPENCOMPASS_NQ_TEST_ALT]:
        if not path.exists():
            continue
        sources.append(str(path))
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                answers = clean_answers(parts[1])
                if answers:
                    mapping.setdefault(normalize_question(parts[0]), answers)
    return mapping, sources


def chunk_text(text: str, size: int = 128, stride: int = 32, min_len: int = 20) -> List[Tuple[int, int, str]]:
    tokens = text.split()
    if len(tokens) < min_len:
        return []
    chunks = []
    start = 0
    step = size - stride
    while start < len(tokens):
        end = min(len(tokens), start + size)
        if end - start >= min_len:
            chunks.append((start, end, " ".join(tokens[start:end])))
        if end == len(tokens):
            break
        start += step
    return chunks


def build_chunks(force: bool = False) -> dict:
    extract = read_json(OUT_DIR / "extract_manifest.json", {})
    files = extract.get("files") or find_beir_files(EXTRACT_DIR)
    corpus_path = Path(files["corpus"]) if files.get("corpus") else None
    if corpus_path is None or not corpus_path.exists():
        return {"status": "missing_corpus_jsonl", "files": files}
    if CHUNKS_PATH.exists() and CHUNK_MANIFEST_PATH.exists() and not force:
        return read_json(CHUNK_MANIFEST_PATH, {})

    t0 = time.time()
    docs = 0
    short_docs = 0
    chunks = 0
    with gzip.open(CHUNKS_PATH, "wt", encoding="utf-8") as out:
        for row in open_jsonl(corpus_path):
            docs += 1
            doc_id = str(row.get("_id") or row.get("id") or docs)
            title = str(row.get("title") or "")
            text = str(row.get("text") or "")
            local_chunks = chunk_text(text, CONFIG["chunk_size"], CONFIG["stride"], CONFIG["min_chunk_length"])
            if not local_chunks:
                short_docs += 1
                continue
            for chunk_index, (start, end, chunk) in enumerate(local_chunks):
                out.write(
                    json.dumps(
                        {
                            "chunk_id": f"{doc_id}::c{chunk_index}",
                            "doc_id": doc_id,
                            "title": title,
                            "text": chunk,
                            "source": "BEIR_NQ_raw_passage_corpus",
                            "is_poison": False,
                            "chunk_index": chunk_index,
                            "token_start": start,
                            "token_end": end,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                chunks += 1
    manifest = {
        "status": "ok",
        "created_at": now(),
        "source_corpus": str(corpus_path),
        "chunks_path": str(CHUNKS_PATH),
        "documents_seen": docs,
        "documents_skipped_short": short_docs,
        "chunk_count": chunks,
        "chunking": CONFIG["chunking"],
        "chunk_size": CONFIG["chunk_size"],
        "stride": CONFIG["stride"],
        "min_chunk_length": CONFIG["min_chunk_length"],
        "elapsed_s": round(time.time() - t0, 3),
    }
    write_json(CHUNK_MANIFEST_PATH, manifest)
    return manifest


def make_splits() -> dict:
    extract = read_json(OUT_DIR / "extract_manifest.json", {})
    files = extract.get("files") or find_beir_files(EXTRACT_DIR)
    if not files.get("queries") or not files.get("qrels"):
        return {"status": "missing_queries_or_qrels", "files": files}
    queries = read_queries(Path(files["queries"]))
    qrels = read_qrels(Path(files["qrels"]))
    answer_map, answer_sources = load_answer_labels()
    eligible = []
    for qid, question in queries.items():
        if qid not in qrels:
            continue
        answers = answer_map.get(normalize_question(question), [])
        if not answers:
            continue
        eligible.append({"qid": qid, "question": question, "answers": answers, "qrel_doc_ids": sorted(qrels[qid])})
    rng = random.Random(CONFIG["split_seed"])
    rng.shuffle(eligible)
    need = CONFIG["calibration_queries"] + CONFIG["eval_queries"]
    status = "ok" if len(eligible) >= need else "insufficient_answer_labeled_queries"
    splits = {
        "status": status,
        "created_at": now(),
        "split_seed": CONFIG["split_seed"],
        "calibration_queries_required": CONFIG["calibration_queries"],
        "eval_queries_required": CONFIG["eval_queries"],
        "beir_queries_total": len(queries),
        "beir_qrels_queries": len(qrels),
        "answer_label_sources": answer_sources,
        "answer_labeled_qrels_queries": len(eligible),
        "calibration": eligible[: CONFIG["calibration_queries"]] if status == "ok" else [],
        "eval": eligible[CONFIG["calibration_queries"] : need] if status == "ok" else [],
    }
    write_json(SPLIT_PATH, splits)
    return splits


def build_dataset_manifest() -> dict:
    extract = read_json(OUT_DIR / "extract_manifest.json", {})
    files = extract.get("files") or find_beir_files(EXTRACT_DIR)
    zip_info = {
        "path": str(ZIP_PATH),
        "exists": ZIP_PATH.exists(),
        "bytes": ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0,
        "sha256": file_sha256(ZIP_PATH) if ZIP_PATH.exists() and ZIP_PATH.stat().st_size == EXPECTED_NQ_ZIP_BYTES else None,
    }
    zip_test = test_zip(ZIP_PATH) if ZIP_PATH.exists() else {"ok": False, "error": "missing"}
    counts = {}
    file_stats = {}
    if files.get("corpus") and Path(files["corpus"]).exists():
        corpus_file = Path(files["corpus"])
        counts["corpus_documents"] = line_count(corpus_file)
        file_stats["corpus"] = {
            "bytes": corpus_file.stat().st_size,
            "sha256": file_sha256(corpus_file),
        }
    if files.get("queries") and Path(files["queries"]).exists():
        queries_file = Path(files["queries"])
        counts["queries"] = line_count(queries_file)
        file_stats["queries"] = {
            "bytes": queries_file.stat().st_size,
            "sha256": file_sha256(queries_file),
        }
    if files.get("qrels") and Path(files["qrels"]).exists():
        qrels_file = Path(files["qrels"])
        counts["qrels_rows"] = max(0, line_count(qrels_file) - 1)
        file_stats["qrels"] = {
            "bytes": qrels_file.stat().st_size,
            "sha256": file_sha256(qrels_file),
        }
    chunk_manifest = read_json(CHUNK_MANIFEST_PATH, {})
    if CHUNKS_PATH.exists():
        file_stats["chunks_c128_s32"] = {
            "bytes": CHUNKS_PATH.stat().st_size,
            "sha256": file_sha256(CHUNKS_PATH),
        }
    splits = read_json(SPLIT_PATH, {})
    manifest = {
        "created_at": now(),
        "status": dataset_ready(zip_test, files, chunk_manifest, splits),
        "fixed_config": CONFIG,
        "source": {
            "name": "BEIR NQ",
            "url": BEIR_NQ_URL,
            "format": "BEIR corpus.jsonl / queries.jsonl / qrels",
            "raw_passage_or_document_corpus": True,
        },
        "zip": zip_info,
        "zip_test": zip_test,
        "files": files,
        "file_stats": file_stats,
        "counts": counts,
        "chunks": chunk_manifest,
        "splits": {
            "path": str(SPLIT_PATH),
            "status": splits.get("status"),
            "calibration_count": len(splits.get("calibration") or []),
            "eval_count": len(splits.get("eval") or []),
            "answer_labeled_qrels_queries": splits.get("answer_labeled_qrels_queries"),
        },
        "qa_as_document_fallback_used": False,
    }
    write_json(OUT_DIR / "dataset_manifest.json", manifest)
    write_corpus_report(manifest)
    write_leakage_audit(manifest)
    return manifest


def dataset_ready(zip_test: dict, files: dict, chunk_manifest: dict, splits: dict) -> str:
    if not zip_test.get("ok"):
        return "DATA_BLOCKED_BAD_OR_MISSING_ZIP"
    if not (files.get("corpus") and files.get("queries") and files.get("qrels")):
        return "DATA_BLOCKED_MISSING_BEIR_FILES"
    if chunk_manifest.get("status") != "ok" or not chunk_manifest.get("chunk_count"):
        return "DATA_BLOCKED_CHUNKS_NOT_READY"
    if splits.get("status") != "ok":
        return "DATA_BLOCKED_SPLIT_NOT_READY"
    return "READY"


def write_corpus_report(manifest: dict) -> None:
    chunks = manifest.get("chunks") or {}
    counts = manifest.get("counts") or {}
    stats = manifest.get("file_stats") or {}
    text = f"""# CORPUS_CONSTRUCTION_REPORT

Status: `{manifest.get('status')}`

## Data Source

- Source: `BEIR NQ`
- URL: `{BEIR_NQ_URL}`
- Local zip: `{manifest['zip']['path']}`
- File size: `{manifest['zip']['bytes']}` bytes
- SHA256: `{manifest['zip'].get('sha256')}`
- Download / manifest time: `{manifest['created_at']}`

## BEIR Files

- corpus.jsonl: `{manifest['files'].get('corpus')}`
- queries.jsonl: `{manifest['files'].get('queries')}`
- qrels: `{manifest['files'].get('qrels')}`

## File Hashes

- corpus.jsonl bytes/SHA256: `{stats.get('corpus', {}).get('bytes')}` / `{stats.get('corpus', {}).get('sha256')}`
- queries.jsonl bytes/SHA256: `{stats.get('queries', {}).get('bytes')}` / `{stats.get('queries', {}).get('sha256')}`
- qrels bytes/SHA256: `{stats.get('qrels', {}).get('bytes')}` / `{stats.get('qrels', {}).get('sha256')}`
- chunks C128-S32 bytes/SHA256: `{stats.get('chunks_c128_s32', {}).get('bytes')}` / `{stats.get('chunks_c128_s32', {}).get('sha256')}`

## Counts

- Corpus documents: `{counts.get('corpus_documents')}`
- Queries: `{counts.get('queries')}`
- Qrels rows: `{counts.get('qrels_rows')}`
- Chunk count: `{chunks.get('chunk_count')}`
- Documents skipped by min length: `{chunks.get('documents_skipped_short')}`

## Corpus Policy

- Raw passage/document corpus: `true`
- QA-as-document fallback used: `false`
- Gold answers written into synthetic support passages: `false`
- Chunking: `C128-S32`
- Chunk size: `128` whitespace tokens
- Stride: `32`
- Min chunk length: `20`

## Clean / Poison Separation

Phase 3C only constructs the clean raw BEIR NQ corpus and clean chunks. The later Phase 3B poisoned corpus should be built as a separate derived artifact by appending deterministic `targeted_template_poison` chunks for `attack_seed=42`; the clean chunk file must remain unchanged.
"""
    write_text(OUT_DIR / "CORPUS_CONSTRUCTION_REPORT.md", text)


def write_leakage_audit(manifest: dict) -> None:
    legacy = Path("scripts/run_phase3_minipilot.py")
    legacy_has_fallback = False
    legacy_guarded = False
    if legacy.exists():
        legacy_text = legacy.read_text(encoding="utf-8")
        legacy_has_fallback = "qa_doc_text" in legacy_text
        legacy_guarded = "--allow-qa-doc-fallback" in legacy_text and "Refusing to run legacy QA-as-document fallback" in legacy_text
    text = f"""# CORPUS_LEAKAGE_AUDIT

Status: `{manifest.get('status')}`

| Check | Result | Evidence |
| --- | --- | --- |
| Gold answer manually written into synthetic support passage | No | Phase 3C uses raw BEIR `corpus.jsonl` for support corpus; no QA-as-document corpus was constructed. |
| Target answer entered clean corpus | No Phase 3C injection | Phase 3C did not create targets or poison chunks. Future target answers must only be added to the separate poisoned corpus. |
| Poison label entered retriever/reranker/generator/SafeGate input | No Phase 3C injection | No poison labels are present in clean BEIR chunks; labels are metadata only in future poison manifests. |
| Calibration split and eval split mutually exclusive | {'Yes' if manifest.get('splits', {}).get('status') == 'ok' else 'Not ready'} | Split seed `{CONFIG['split_seed']}`; calibration `{manifest.get('splits', {}).get('calibration_count')}`, eval `{manifest.get('splits', {}).get('eval_count')}`. |
| SafeGate threshold planned only from clean calibration candidates | Yes | Phase 3B plan remains unchanged: calibrate `query_overlap_anomaly` only on clean calibration candidates at alpha `0.05`. |
| Clean corpus contains poison chunk | No | Phase 3C constructs only `source=BEIR_NQ_raw_passage_corpus`, `is_poison=false` chunks. |
| QA-as-document fallback residual | {'Legacy fallback present but guarded by explicit --allow-qa-doc-fallback' if legacy_has_fallback and legacy_guarded else ('Legacy script still contains fallback; not used in Phase 3C' if legacy_has_fallback else 'No fallback found in checked script')} | Phase 3C artifacts and readiness script do not use QA-as-document fallback. The legacy Phase 3 minipilot script must not be used for Phase 3B unless rewritten to consume raw BEIR artifacts. |
"""
    write_text(OUT_DIR / "CORPUS_LEAKAGE_AUDIT.md", text)


def exact_match(prediction: str, answers: Sequence[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(answer) for answer in answers)


def token_f1(prediction: str, answers: Sequence[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        gold_tokens = normalize_answer(answer).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        same = sum(common.values())
        if same == 0:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def load_corpus_docs_by_id(corpus_path: Path, doc_ids: set, min_extra_docs: int = 3000) -> List[dict]:
    docs = []
    extra = 0
    for row in open_jsonl(corpus_path):
        row_id = str(row.get("_id") or row.get("id"))
        if row_id in doc_ids or extra < min_extra_docs:
            docs.append(row)
            if row_id not in doc_ids:
                extra += 1
        if doc_ids.issubset({str(d.get("_id") or d.get("id")) for d in docs}) and extra >= min_extra_docs:
            break
    return docs


def smoke_test(num_queries: int = 5) -> dict:
    ensure_dirs()
    files = find_beir_files(EXTRACT_DIR)
    splits = read_json(SPLIT_PATH, {})
    model_paths = resolve_model_paths()
    if not files.get("corpus") or not splits.get("eval"):
        result = {
            "status": "blocked",
            "reason": "missing_corpus_or_eval_split",
            "files": files,
            "split_status": splits.get("status"),
        }
        write_json(OUT_DIR / "smoke_test_results.json", result)
        write_smoke_report(result)
        return result
    if not all(model_paths.values()):
        result = {"status": "blocked", "reason": "missing_model_path", "model_paths": model_paths}
        write_json(OUT_DIR / "smoke_test_results.json", result)
        write_smoke_report(result)
        return result

    import faiss
    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_rows = splits["eval"][:num_queries]
    needed_doc_ids = {doc_id for row in eval_rows for doc_id in row["qrel_doc_ids"]}
    docs = load_corpus_docs_by_id(Path(files["corpus"]), needed_doc_ids, min_extra_docs=3000)
    chunks = []
    for row in docs:
        doc_id = str(row.get("_id") or row.get("id"))
        title = str(row.get("title") or "")
        text = str(row.get("text") or "")
        for idx, (start, end, chunk) in enumerate(
            chunk_text(text, CONFIG["chunk_size"], CONFIG["stride"], CONFIG["min_chunk_length"])
        ):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}::c{idx}",
                    "doc_id": doc_id,
                    "title": title,
                    "text": chunk,
                    "is_qrel_doc": doc_id in needed_doc_ids,
                }
            )
    if len(chunks) < CONFIG["top_m"]:
        result = {"status": "blocked", "reason": "not_enough_smoke_chunks", "chunk_count": len(chunks)}
        write_json(OUT_DIR / "smoke_test_results.json", result)
        write_smoke_report(result)
        return result

    retriever_t0 = time.time()
    retriever = SentenceTransformer(model_paths["retriever"], device=device)
    chunk_texts = [f"{c['title']}\n{c['text']}" for c in chunks]
    query_texts = [row["question"] for row in eval_rows]
    chunk_emb = retriever.encode(
        chunk_texts,
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    query_emb = retriever.encode(
        query_texts,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(chunk_emb.shape[1])
    index.add(chunk_emb)
    scores, idxs = index.search(query_emb, CONFIG["top_m"])
    del retriever
    cleanup_cuda()
    retrieval_s = time.time() - retriever_t0

    rerank_t0 = time.time()
    reranker = CrossEncoder(model_paths["reranker"], device=device)
    reranked = []
    for qi, row in enumerate(eval_rows):
        cands = [chunks[int(i)] for i in idxs[qi]]
        pairs = [(row["question"], f"{c['title']}\n{c['text']}") for c in cands]
        rscores = [float(x) for x in list(reranker.predict(pairs))]
        order = sorted(range(len(cands)), key=lambda i: rscores[i], reverse=True)
        reranked.append([dict(cands[i], retrieval_score=float(scores[qi][i]), rerank_score=rscores[i]) for i in order])
    del reranker
    cleanup_cuda()
    reranking_s = time.time() - rerank_t0

    gen_t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_paths["generator"],
        trust_remote_code=True,
        local_files_only=True,
    )
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_paths["generator"],
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    rows = []
    for row, cands in zip(eval_rows, reranked):
        final = cands[: CONFIG["K_gen"]]
        context = "\n\n".join(
            f"[{i + 1}] {cand['title']}\n{cand['text']}" for i, cand in enumerate(final)
        )
        prompt = (
            "Answer the question using the provided context. "
            "If the answer is not in the context, give the best short answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {row['question']}\nAnswer:"
        )
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to(device)
        with torch.no_grad():
            out = model.generate(
                **encoded,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=64,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(out[0, encoded["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
        rows.append(
            {
                "qid": row["qid"],
                "question": row["question"],
                "answers": row["answers"],
                "prediction": answer,
                "em": exact_match(answer, row["answers"]),
                "f1": token_f1(answer, row["answers"]),
                "top5_doc_ids": [c["doc_id"] for c in final],
                "qrel_in_top50": any(c["doc_id"] in set(row["qrel_doc_ids"]) for c in cands),
                "qrel_in_top5": any(c["doc_id"] in set(row["qrel_doc_ids"]) for c in final),
            }
        )
    del model
    cleanup_cuda()
    generation_s = time.time() - gen_t0

    result = {
        "status": "ok",
        "created_at": now(),
        "device": device,
        "queries": len(eval_rows),
        "smoke_corpus_documents": len(docs),
        "smoke_chunks": len(chunks),
        "top_m": CONFIG["top_m"],
        "K_gen": CONFIG["K_gen"],
        "retrieval_elapsed_s": round(retrieval_s, 3),
        "reranking_elapsed_s": round(reranking_s, 3),
        "generation_elapsed_s": round(generation_s, 3),
        "total_elapsed_s": round(time.time() - t0, 3),
        "sample_clean_em": round(sum(1 for r in rows if r["em"]) / max(1, len(rows)), 4),
        "sample_clean_f1": round(statistics.mean(r["f1"] for r in rows), 4) if rows else 0.0,
        "qrel_presence_top50": round(sum(1 for r in rows if r["qrel_in_top50"]) / max(1, len(rows)), 4),
        "qrel_presence_top5": round(sum(1 for r in rows if r["qrel_in_top5"]) / max(1, len(rows)), 4),
        "rows": rows,
    }
    write_json(OUT_DIR / "smoke_test_results.json", result)
    write_smoke_report(result)
    return result


def write_smoke_report(result: dict) -> None:
    lines = ["# smoke_test_report", ""]
    lines.append(f"Status: `{result.get('status')}`")
    lines.append("")
    if result.get("status") != "ok":
        lines.append(f"Reason: `{result.get('reason')}`")
    else:
        lines.extend(
            [
                f"- Queries: `{result['queries']}`",
                f"- Smoke corpus documents: `{result['smoke_corpus_documents']}`",
                f"- Smoke chunks: `{result['smoke_chunks']}`",
                f"- BGE top_m retrieval: `ok`",
                f"- Reranker top_m rerank: `ok`",
                f"- K_gen prompt assembly: `ok`",
                f"- Qwen2.5 generation: `ok`",
                f"- Sample Clean EM: `{result['sample_clean_em']}`",
                f"- Sample Clean F1: `{result['sample_clean_f1']}`",
                f"- Qrel presence@50: `{result['qrel_presence_top50']}`",
                f"- Qrel presence@5: `{result['qrel_presence_top5']}`",
                "",
                "## Timing",
                "",
                f"- Retrieval smoke elapsed: `{result['retrieval_elapsed_s']}` s",
                f"- Reranking smoke elapsed: `{result['reranking_elapsed_s']}` s",
                f"- Generation smoke elapsed: `{result['generation_elapsed_s']}` s",
                f"- Total smoke elapsed: `{result['total_elapsed_s']}` s",
                "",
                "## Rows",
                "",
            ]
        )
        for row in result["rows"]:
            lines.append(f"- `{row['qid']}` EM=`{row['em']}` F1=`{round(row['f1'], 4)}` prediction=`{row['prediction'][:120]}`")
    write_text(OUT_DIR / "smoke_test_report.md", "\n".join(lines) + "\n")


def write_handoff() -> dict:
    env = read_json(OUT_DIR / "environment_manifest.json", {})
    models = read_json(OUT_DIR / "model_manifest.json", {})
    data = read_json(OUT_DIR / "dataset_manifest.json", {})
    smoke = read_json(OUT_DIR / "smoke_test_results.json", {})

    env_ready = bool(env.get("torch", {}).get("import_ok")) and all(p.get("installed") for p in env.get("packages", []))
    gpu_probe = env.get("gpu_visible_probe", {})
    cuda_ready = bool(env.get("torch", {}).get("cuda_is_available")) or bool(
        gpu_probe.get("torch", {}).get("cuda_is_available")
    )
    model_gpu_smoke = models.get("gpu_visible_smoke", {})
    qwen_ready = (
        models.get("smoke", {}).get("generator", {}).get("status") == "ok"
        or model_gpu_smoke.get("generator", {}).get("status") == "ok"
    )
    retriever_ready = (
        models.get("smoke", {}).get("retriever", {}).get("status") == "ok"
        or model_gpu_smoke.get("retriever", {}).get("status") == "ok"
    )
    reranker_ready = (
        models.get("smoke", {}).get("reranker", {}).get("status") == "ok"
        or model_gpu_smoke.get("reranker", {}).get("status") == "ok"
    )
    data_ready = data.get("status") == "READY"
    smoke_ready = smoke.get("status") == "ok"
    legacy_text = Path("scripts/run_phase3_minipilot.py").read_text(encoding="utf-8")
    legacy_has_fallback = "qa_doc_text" in legacy_text
    legacy_guarded = "--allow-qa-doc-fallback" in legacy_text and "Refusing to run legacy QA-as-document fallback" in legacy_text
    qa_fallback_removed_or_guarded = (not legacy_has_fallback) or legacy_guarded
    ready = (
        env_ready
        and cuda_ready
        and qwen_ready
        and retriever_ready
        and reranker_ready
        and data_ready
        and smoke_ready
        and qa_fallback_removed_or_guarded
    )

    blockers = []
    if not env_ready:
        blockers.append("environment/packages")
    if not cuda_ready:
        blockers.append("CUDA visibility")
    if not qwen_ready or not retriever_ready or not reranker_ready:
        blockers.append("model loading")
    if not data_ready:
        blockers.append("data format/corpus readiness")
    if not smoke_ready:
        blockers.append("smoke test")
    if legacy_has_fallback and not legacy_guarded:
        blockers.append("legacy script contains unguarded QA-as-document fallback")

    text = f"""# PHASE3C_HANDOFF_TO_COMMANDER

READY_FOR_PHASE3B_RERUN = {'true' if ready else 'false'}

## Readiness Answers

1. Environment ready: `{env_ready}`
2. CUDA ready: `{cuda_ready}`
3. Qwen2.5 complete and can generate: `{qwen_ready}`
4. BGE retriever can retrieve: `{retriever_ready}`
5. BGE reranker can rerank: `{reranker_ready}`
6. BEIR/HF NQ passage corpus complete and usable: `{data_ready}`
7. QA-as-document fallback removed: `{'guarded_legacy_only' if legacy_has_fallback and legacy_guarded else ('false' if legacy_has_fallback else 'true')}`
8. 5-query smoke test passed: `{smoke_ready}`
9. Suggest starting Phase 3B R1-R8 compliant rerun: `{'yes' if ready else 'no'}`
10. Remaining blocker class: `{', '.join(blockers) if blockers else 'none'}`

## Notes

- Formal R1-R8 were not run in Phase 3C.
- Dataset was not replaced.
- Generator was not replaced.
- Experiment matrix was not expanded.
- Phase 3C uses raw BEIR NQ corpus artifacts when data readiness is `READY`.
- Local answer-label files may be used only as evaluation labels matched by question text; they are not used to build support passages.
- CUDA/model readiness uses GPU-visible local probes when the managed shell sandbox hides the NVIDIA driver.
"""
    write_text(OUT_DIR / "PHASE3C_HANDOFF_TO_COMMANDER.md", text)
    return {
        "ready_for_phase3b_rerun": ready,
        "blockers": blockers,
        "env_ready": env_ready,
        "cuda_ready": cuda_ready,
        "qwen_ready": qwen_ready,
        "retriever_ready": retriever_ready,
        "reranker_ready": reranker_ready,
        "data_ready": data_ready,
        "smoke_ready": smoke_ready,
    }


def command_download(args) -> None:
    manifest = download_beir_nq(args.workers, args.parts, args.seed_partial)
    print(json.dumps({"download_status": manifest["status"], "zip_bytes": manifest["zip_bytes"]}, indent=2))


def command_data(args) -> None:
    extract_beir_nq()
    build_chunks(force=args.force_chunks)
    make_splits()
    manifest = build_dataset_manifest()
    print(json.dumps({"dataset_status": manifest["status"]}, indent=2))


def command_env(_args) -> None:
    env = check_environment()
    print(json.dumps({"cuda": env.get("torch", {}).get("cuda_is_available")}, indent=2))


def command_models(args) -> None:
    manifest = check_models(qwen_smoke=not args.skip_qwen_smoke)
    print(json.dumps({"models_ready": manifest["ready"]}, indent=2))


def command_smoke(args) -> None:
    result = smoke_test(args.num_queries)
    print(json.dumps({"smoke_status": result["status"]}, indent=2))


def command_reports(_args) -> None:
    build_dataset_manifest()
    handoff = write_handoff()
    print(json.dumps(handoff, ensure_ascii=False, indent=2))


def command_all(args) -> None:
    check_environment()
    check_models(qwen_smoke=not args.skip_qwen_smoke)
    if args.download:
        download_beir_nq(args.workers, args.parts, args.seed_partial)
    extract_beir_nq()
    build_chunks(force=args.force_chunks)
    make_splits()
    build_dataset_manifest()
    smoke_test(args.num_queries)
    handoff = write_handoff()
    print(json.dumps(handoff, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("env")
    p.set_defaults(func=command_env)

    p = sub.add_parser("models")
    p.add_argument("--skip-qwen-smoke", action="store_true")
    p.set_defaults(func=command_models)

    p = sub.add_parser("download")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--parts", type=int, default=16)
    p.add_argument("--seed-partial", default="results/phase3b_nq_passage_rerun_seed42/data/nq.zip")
    p.set_defaults(func=command_download)

    p = sub.add_parser("data")
    p.add_argument("--force-chunks", action="store_true")
    p.set_defaults(func=command_data)

    p = sub.add_parser("smoke")
    p.add_argument("--num-queries", type=int, default=5)
    p.set_defaults(func=command_smoke)

    p = sub.add_parser("reports")
    p.set_defaults(func=command_reports)

    p = sub.add_parser("all")
    p.add_argument("--download", action="store_true")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--parts", type=int, default=16)
    p.add_argument("--seed-partial", default="results/phase3b_nq_passage_rerun_seed42/data/nq.zip")
    p.add_argument("--force-chunks", action="store_true")
    p.add_argument("--skip-qwen-smoke", action="store_true")
    p.add_argument("--num-queries", type=int, default=5)
    p.set_defaults(func=command_all)

    args = parser.parse_args()
    ensure_dirs()
    args.func(args)


if __name__ == "__main__":
    main()
