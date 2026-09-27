import csv
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase3_minipilot_seed42"


RUNS = [
    {
        "run_id": "R1",
        "pipeline": "simplified",
        "corpus": "clean",
        "gate": "none",
        "description": "simplified RAG clean no-gate",
    },
    {
        "run_id": "R2",
        "pipeline": "simplified",
        "corpus": "poisoned",
        "gate": "none",
        "description": "simplified RAG under targeted_template_poison no-gate",
    },
    {
        "run_id": "R3",
        "pipeline": "multi-stage",
        "corpus": "clean",
        "gate": "none",
        "description": "multi-stage RAG clean no-gate",
    },
    {
        "run_id": "R4",
        "pipeline": "multi-stage",
        "corpus": "poisoned",
        "gate": "none",
        "description": "multi-stage RAG under targeted_template_poison no-gate",
    },
    {
        "run_id": "R5",
        "pipeline": "multi-stage",
        "corpus": "clean",
        "gate": "P_ret",
        "description": "multi-stage RAG clean + P_ret",
    },
    {
        "run_id": "R6",
        "pipeline": "multi-stage",
        "corpus": "poisoned",
        "gate": "P_ret",
        "description": "multi-stage RAG poisoned + P_ret",
    },
    {
        "run_id": "R7",
        "pipeline": "multi-stage",
        "corpus": "clean",
        "gate": "P_gen",
        "description": "multi-stage RAG clean + P_gen",
    },
    {
        "run_id": "R8",
        "pipeline": "multi-stage",
        "corpus": "poisoned",
        "gate": "P_gen",
        "description": "multi-stage RAG poisoned + P_gen",
    },
]


FIXED_CONFIG = {
    "dataset": "Natural Questions / NQ",
    "calibration_queries": 100,
    "eval_queries": 100,
    "split_seed": 20260905,
    "attack_seed": 42,
    "chunking": "C128-S32",
    "chunk_size_whitespace_tokens": 128,
    "stride_whitespace_tokens": 32,
    "min_chunk_length_whitespace_tokens": 20,
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
    "poison_budget_chunks_per_query": 5,
    "safegate_signal": "query_overlap_anomaly",
    "threshold_calibration": "clean calibration candidates, chunk-level FPR alpha=0.05",
}


CSV_COLUMNS = [
    "run_id",
    "description",
    "pipeline",
    "corpus",
    "gate",
    "status",
    "output_path",
    "clean_em",
    "clean_f1",
    "asr",
    "asr_drop",
    "tpr",
    "fpr",
    "utility_drop_f1",
    "poison_presence_at_5",
    "poison_fraction_at_5",
    "latency_seconds",
    "blocking_reason",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def python_check(code: str, timeout: int = 30) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": repr(exc)}


def summarize_path(path: Path) -> dict:
    result = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    files = []
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        stat = item.stat()
        entry = {
            "relative_path": item.relative_to(path).as_posix(),
            "bytes": stat.st_size,
            "last_write_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        }
        if stat.st_size <= 10_000_000:
            entry["sha256"] = sha256_file(item)
        files.append(entry)
    result["file_count"] = len(files)
    result["total_bytes"] = sum(item["bytes"] for item in files)
    result["files"] = files
    return result


def model_resource_status() -> dict:
    paths = {
        "generator_protocol": ROOT / "models" / "Qwen" / "Qwen2.5-7B-Instruct",
        "generator_modelscope_cache": (
            ROOT
            / "models"
            / "modelscope_cache"
            / "models"
            / "Qwen--Qwen2.5-7B-Instruct"
            / "snapshots"
            / "master"
        ),
        "generator_fallback_not_used": Path("D:/local_qwen_models/Qwen3.5-4B/Qwen3.5-4B"),
        "dense_retriever": ROOT / "models" / "BAAI" / "bge-base-en-v1.5",
        "reranker": ROOT / "models" / "BAAI" / "bge-reranker-base",
    }
    status = {name: summarize_path(path) for name, path in paths.items()}

    dense = paths["dense_retriever"]
    reranker = paths["reranker"]
    qwen = paths["generator_protocol"]
    qwen_cache = paths["generator_modelscope_cache"]
    protocol_shards = [qwen / f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)]
    cache_incomplete = sorted(str(p) for p in qwen_cache.glob("*.incomplete")) if qwen_cache.exists() else []

    status["dense_retriever"]["usable_for_protocol"] = (
        dense.exists()
        and (dense / "config.json").exists()
        and ((dense / "model.safetensors").exists() or (dense / "pytorch_model.bin").exists())
        and ((dense / "tokenizer.json").exists() or (dense / "vocab.txt").exists())
    )
    status["reranker"]["usable_for_protocol"] = (
        reranker.exists()
        and (reranker / "config.json").exists()
        and ((reranker / "model.safetensors").exists() or (reranker / "pytorch_model.bin").exists())
        and ((reranker / "tokenizer.json").exists() or (reranker / "sentencepiece.bpe.model").exists())
    )
    status["generator_protocol"]["usable_for_protocol"] = (
        qwen.exists()
        and (qwen / "config.json").exists()
        and (qwen / "tokenizer.json").exists()
        and all(p.exists() for p in protocol_shards)
    )
    status["generator_protocol"]["expected_weight_shards"] = [str(p) for p in protocol_shards]
    status["generator_modelscope_cache"]["incomplete_files"] = cache_incomplete
    status["generator_modelscope_cache"]["has_incomplete_weights"] = bool(cache_incomplete)
    return status


def dataset_resource_status() -> dict:
    candidates = [
        ROOT / "data" / "raw" / "nq",
        ROOT / "data" / "processed" / "nq",
        ROOT / "models" / "hf_home" / "datasets",
        ROOT / "models" / "hf_cache" / "datasets",
        Path.home() / ".cache" / "huggingface" / "datasets",
    ]
    checked = []
    usable = False
    for path in candidates:
        entry = {"path": str(path), "exists": path.exists()}
        matches = []
        if path.exists():
            for item in path.rglob("*"):
                if not item.is_file():
                    continue
                name = item.name.lower()
                full = str(item).lower()
                if any(marker in name or marker in full for marker in ["nq", "natural", "beir/nq"]):
                    matches.append(str(item))
                    if len(matches) >= 50:
                        break
            usable = usable or bool(matches)
        entry["matching_files"] = matches
        checked.append(entry)
    return {
        "dataset": "Natural Questions / NQ",
        "usable_local_candidate_found": usable,
        "checked_paths": checked,
    }


def preflight() -> dict:
    required_imports = python_check(
        "import importlib.util as u; "
        "mods=['torch','transformers','sentence_transformers','faiss','datasets','modelscope','numpy','sklearn','yaml']; "
        "print({m: bool(u.find_spec(m)) for m in mods})",
        timeout=60,
    )
    torch_status = python_check(
        "import torch; "
        "print({'version': torch.__version__, 'cuda': torch.cuda.is_available(), "
        "'cuda_count': torch.cuda.device_count() if torch.cuda.is_available() else 0})",
        timeout=60,
    )
    model_status = model_resource_status()
    dataset_status = dataset_resource_status()

    missing = []
    if not model_status["generator_protocol"]["usable_for_protocol"]:
        missing.append("Qwen/Qwen2.5-7B-Instruct generator not complete locally")
    if not model_status["dense_retriever"]["usable_for_protocol"]:
        missing.append("BAAI/bge-base-en-v1.5 retriever not usable locally")
    if not model_status["reranker"]["usable_for_protocol"]:
        missing.append("BAAI/bge-reranker-base reranker not usable locally")
    if not dataset_status["usable_local_candidate_found"]:
        missing.append("Natural Questions / BEIR NQ data not found locally")

    return {
        "created_at": now_iso(),
        "workspace": str(ROOT),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "fixed_config": FIXED_CONFIG,
        "required_imports": required_imports,
        "torch_status": torch_status,
        "model_resources": model_status,
        "dataset_resources": dataset_status,
        "missing_blockers": missing,
        "network_note": (
            "ModelScope resume for Qwen/Qwen2.5-7B-Instruct was attempted. The sandboxed run failed "
            "with WinError 10013 socket permission denial; an escalated run timed out before all "
            "protocol weight shards completed. No fallback generator was used."
        ),
    }


def config_yaml(run: dict) -> str:
    lines = [
        f"run_id: {run['run_id']}",
        f"description: \"{run['description']}\"",
        f"pipeline: {run['pipeline']}",
        f"corpus: {run['corpus']}",
        f"gate: {run['gate']}",
        "status: not_run_resource_unavailable",
        "fixed_config:",
    ]
    for key, value in FIXED_CONFIG.items():
        if isinstance(value, dict):
            lines.append(f"  {key}:")
            for sub_key, sub_value in value.items():
                v = str(sub_value).lower() if isinstance(sub_value, bool) else sub_value
                lines.append(f"    {sub_key}: {v}")
        elif isinstance(value, str):
            lines.append(f"  {key}: \"{value}\"")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def unavailable_row(reason: str) -> dict:
    return {
        "status": "not_run",
        "reason": reason,
        "created_at": now_iso(),
    }


def build_run_dirs(preflight_data: dict) -> list[dict]:
    rows = []
    reason = "; ".join(preflight_data["missing_blockers"]) or "unknown resource preflight failure"
    for run in RUNS:
        run_dir = OUT / "runs" / run["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        write_text(run_dir / "config.yaml", config_yaml(run))
        write_json(
            run_dir / "dataset_manifest.json",
            {
                "dataset": FIXED_CONFIG["dataset"],
                "status": "not_available",
                "required_calibration_queries": FIXED_CONFIG["calibration_queries"],
                "required_eval_queries": FIXED_CONFIG["eval_queries"],
                "split_seed": FIXED_CONFIG["split_seed"],
                "local_resource_audit": preflight_data["dataset_resources"],
                "blocking_reason": "NQ data not found locally",
            },
        )
        write_json(
            run_dir / "corpus_manifest.json",
            {
                "chunking": FIXED_CONFIG["chunking"],
                "status": "not_built",
                "clean_corpus_has_poison_chunks": None,
                "poisoned_corpus_seed": 42 if run["corpus"] == "poisoned" else None,
                "model_resource_audit": {
                    "dense_retriever_usable": preflight_data["model_resources"]["dense_retriever"][
                        "usable_for_protocol"
                    ],
                    "reranker_usable": preflight_data["model_resources"]["reranker"][
                        "usable_for_protocol"
                    ],
                    "generator_usable": preflight_data["model_resources"]["generator_protocol"][
                        "usable_for_protocol"
                    ],
                },
                "blocking_reason": reason,
            },
        )
        write_json(
            run_dir / "poison_manifest.json",
            {
                "attack": FIXED_CONFIG["attack"] if run["corpus"] == "poisoned" else None,
                "attack_seed": 42 if run["corpus"] == "poisoned" else None,
                "poison_budget_chunks_per_query": (
                    FIXED_CONFIG["poison_budget_chunks_per_query"] if run["corpus"] == "poisoned" else 0
                ),
                "status": "not_generated",
                "note": "clean run has no poison" if run["corpus"] == "clean" else "blocked before poison generation",
            },
        )
        write_jsonl(run_dir / "retrieval_results.jsonl", [unavailable_row(reason)])
        write_jsonl(
            run_dir / "rerank_results.jsonl",
            [
                unavailable_row(
                    "simplified run has no reranker" if run["pipeline"] == "simplified" else reason
                )
            ],
        )
        write_jsonl(
            run_dir / "gate_decisions.jsonl",
            [unavailable_row("no-gate run" if run["gate"] == "none" else reason)],
        )
        write_jsonl(run_dir / "generation_outputs.jsonl", [unavailable_row(reason)])
        metrics = {
            "run_id": run["run_id"],
            "status": "not_run_resource_unavailable",
            "clean_em": None,
            "clean_f1": None,
            "asr": None,
            "asr_drop": None,
            "tpr": None,
            "fpr": None,
            "utility_drop_f1": None,
            "poison_presence_at_5": None,
            "poison_fraction_at_5": None,
            "latency_seconds": None,
            "blocking_reason": reason,
        }
        write_json(run_dir / "metrics.json", metrics)
        write_json(
            run_dir / "run_log.json",
            {
                "run_id": run["run_id"],
                "created_at": now_iso(),
                "status": "not_run_resource_unavailable",
                "events": [
                    {
                        "time": now_iso(),
                        "event": "preflight_blocked",
                        "reason": reason,
                    }
                ],
            },
        )
        rows.append(
            {
                "run_id": run["run_id"],
                "description": run["description"],
                "pipeline": run["pipeline"],
                "corpus": run["corpus"],
                "gate": run["gate"],
                "status": "not_run_resource_unavailable",
                "output_path": str(run_dir),
                "clean_em": "",
                "clean_f1": "",
                "asr": "",
                "asr_drop": "",
                "tpr": "",
                "fpr": "",
                "utility_drop_f1": "",
                "poison_presence_at_5": "",
                "poison_fraction_at_5": "",
                "latency_seconds": "",
                "blocking_reason": reason,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_reports(preflight_data: dict, rows: list[dict]) -> None:
    write_json(OUT / "preflight_resource_audit.json", preflight_data)
    write_csv(OUT / "aggregate_metrics.csv", rows, CSV_COLUMNS)

    reason = "; ".join(preflight_data["missing_blockers"])
    write_csv(
        OUT / "gate_position_summary.csv",
        [
            {
                "comparison": "P_ret_vs_P_gen",
                "status": "not_available",
                "p_ret_asr_drop": "",
                "p_gen_asr_drop": "",
                "delta_asr_drop": "",
                "p_ret_fpr": "",
                "p_gen_fpr": "",
                "delta_fpr": "",
                "p_ret_utility_drop_f1": "",
                "p_gen_utility_drop_f1": "",
                "delta_utility_drop_f1": "",
                "reason": reason,
            }
        ],
        [
            "comparison",
            "status",
            "p_ret_asr_drop",
            "p_gen_asr_drop",
            "delta_asr_drop",
            "p_ret_fpr",
            "p_gen_fpr",
            "delta_fpr",
            "p_ret_utility_drop_f1",
            "p_gen_utility_drop_f1",
            "delta_utility_drop_f1",
            "reason",
        ],
    )

    blockers = "\n".join(f"- {b}" for b in preflight_data["missing_blockers"]) or "- none"
    run_paths = "\n".join(f"- {row['run_id']}: `{row['output_path']}`" for row in rows)
    dense_ok = preflight_data["model_resources"]["dense_retriever"]["usable_for_protocol"]
    reranker_ok = preflight_data["model_resources"]["reranker"]["usable_for_protocol"]
    qwen_ok = preflight_data["model_resources"]["generator_protocol"]["usable_for_protocol"]
    fallback_exists = preflight_data["model_resources"]["generator_fallback_not_used"]["exists"]
    incomplete_files = preflight_data["model_resources"]["generator_modelscope_cache"].get(
        "incomplete_files", []
    )
    incomplete_lines = "\n".join(f"- `{p}`" for p in incomplete_files) or "- none"

    write_text(
        OUT / "pre_experiment_report.md",
        f"""# Phase 3 Minipilot Pre-Experiment Report

Date: {now_iso()}

## Status

The strict Phase 3 minipilot did not execute the 8 runs. Resource preflight blocked before data processing, retrieval, reranking, gate calibration, or generation.

## Fixed Protocol Checked

- Dataset: Natural Questions / NQ
- Calibration queries: 100
- Eval queries: 100
- Split seed: 20260905
- Attack seed: 42
- Chunking: C128-S32
- Retriever: BAAI/bge-base-en-v1.5
- Reranker: BAAI/bge-reranker-base
- Generator: Qwen/Qwen2.5-7B-Instruct
- Decoding: temperature=0, do_sample=false, top_p=1.0, max_new_tokens=64
- Attack: targeted_template_poison, 5 poisoned chunks per query
- Gate signal: query_overlap_anomaly, alpha=0.05 on clean calibration candidates

## Resource Audit

- Dense retriever usable locally: {dense_ok}
- Reranker usable locally: {reranker_ok}
- Protocol generator usable locally: {qwen_ok}
- Qwen3.5 fallback exists but was not used: {fallback_exists}
- NQ local candidate found: {preflight_data["dataset_resources"]["usable_local_candidate_found"]}

## Blocking Findings

{blockers}

## Incomplete Qwen2.5 Cache Files

{incomplete_lines}

## Run Outputs

{run_paths}

## Metrics

No ASR, ASR-Drop, Clean EM/F1, TPR, FPR, Utility Drop, PoisonPresence@5, PoisonFraction@5, or Latency values were produced because no protocol-compliant run reached retrieval or generation.
""",
    )

    write_text(
        OUT / "failure_cases.md",
        f"""# Failure Cases

No per-query failure cases exist because the run matrix was blocked during resource preflight.

## Resource Failures

{blockers}

## Engineering Interpretation

This is not yet a data-processing, corpus/chunk, retrieval, reranking, gate-threshold, generation, or metrics-calculation failure. Those stages were not reached. The immediate blockers are the missing local NQ data and incomplete protocol generator.
""",
    )

    write_text(
        OUT / "protocol_deviation.md",
        """none

No replacement dataset, retriever, reranker, generator, attack, gate signal, threshold rule, or expanded run matrix was used. The discovered Qwen3.5 fallback was not substituted for the fixed Qwen/Qwen2.5-7B-Instruct generator.
""",
    )

    write_text(
        OUT / "PHASE3_HANDOFF_TO_COMMANDER.md",
        f"""# Phase 3 Handoff To Commander

Date: {now_iso()}

## Completion

8 core runs completed: no.

All 8 run directories were created with config, manifests, stage logs, metrics, and run logs. Every run is marked `not_run_resource_unavailable`.

## Run Output Paths

{run_paths}

## Simplified vs Multi-Stage Clean QA And No-Gate ASR

No comparison is available. R1/R3 clean QA and R2/R4 no-gate ASR were not executed.

## P_ret vs P_gen ASR-Drop / FPR / Utility Drop

No comparison is available. R5/R6/R7/R8 were not executed.

## SafeGate Minimal Discrimination

Not evaluated. The `query_overlap_anomaly` threshold could not be calibrated because clean calibration candidates were not generated.

## Leakage / Input Consistency / Metrics / Protocol

- Data leakage: no evidence; no split, corpus, retrieval, gate, or generation stage executed.
- Input inconsistency: no evidence; no model inputs were produced.
- Metrics script error: no evidence; metrics are intentionally null for blocked runs.
- Protocol deviation: none. No fallback model or substitute dataset was used.

## Current Go Decision

No-Go for expanding experiments now. This is a resource-blocked No-Go, not a research-direction No-Go.

## Most Needed Fix

Resource provisioning is the first blocker:

1. Provide local Natural Questions / BEIR NQ data.
2. Complete local `Qwen/Qwen2.5-7B-Instruct`.

After those are available, the next implementation checks should be data processing, corpus/chunk construction, retrieval results, reranking output, gate threshold calibration, generation, and metric calculation.
""",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    preflight_data = preflight()
    rows = build_run_dirs(preflight_data)
    write_reports(preflight_data, rows)
    print(f"wrote {OUT}")
    if preflight_data["missing_blockers"]:
        print("status: not_run_resource_unavailable")
        for item in preflight_data["missing_blockers"]:
            print(f"blocker: {item}")
        return 2
    print("status: resources_available_but_full_execution_not_implemented")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
