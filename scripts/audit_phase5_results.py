#!/usr/bin/env python3
"""Read-only Phase 5 artifact audit for paper-writing readiness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = ROOT / "results" / "phase5_main_matrix"
MAIN_RUNS = MAIN_DIR / "runs"
CONTROL_DIR = ROOT / "results" / "phase5d_control_ablation"
CONTROL_RUNS = CONTROL_DIR / "runs"
OUT_DIR = ROOT / "audit_outputs"
K_GEN = 5
SEEDS = (13, 42, 2026)

FLOAT_FIELDS = (
    "Clean_EM",
    "Clean_F1",
    "ASR",
    "ambiguous_rate",
    "TPR",
    "FPR",
    "PoisonPresence@5",
    "PoisonFraction@5",
    "TargetPoisonPresence@5",
    "TargetPoisonFraction@5",
    "avg_context_count",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


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
        same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if not same:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def contains_answer(prediction: str, answer: str) -> bool:
    pred = normalize_answer(prediction)
    ans = normalize_answer(answer)
    return bool(ans and re.search(rf"(^|\s){re.escape(ans)}($|\s)", pred))


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def is_close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if left is None and right in (None, ""):
        return True
    if right is None and left in (None, ""):
        return True
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left == right


def resolve_split(run_dir: Path, dataset: str) -> dict[str, Any]:
    local_candidates = (
        run_dir.parents[2] / "data" / dataset / "splits_seed20260905_main_200_300.json",
        run_dir.parents[1] / "data" / dataset / "splits_seed20260905_main_200_300.json",
        MAIN_DIR / "data" / dataset / "splits_seed20260905_main_200_300.json",
        CONTROL_DIR / "data" / dataset / "splits_seed20260905_main_200_300.json",
    )
    for candidate in local_candidates:
        if candidate.exists():
            return read_json(candidate)
    manifest = read_json(run_dir / "dataset_manifest.json")
    raw_path = Path(manifest["files"]["split"])
    if raw_path.exists():
        return read_json(raw_path)
    raise FileNotFoundError(f"Cannot resolve split for {run_dir}")


def gate_items(run_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    path = run_dir / "gate_decisions.jsonl"
    if not path.exists():
        return items
    for row in iter_jsonl(path):
        items.extend(row.get("decisions") or [])
    return items


def recompute_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stored = read_json(run_dir / "metrics.json")
    outputs = list(iter_jsonl(run_dir / "generation_outputs.jsonl"))
    split = resolve_split(run_dir, str(stored["dataset"]))
    eval_rows = split.get("eval") or []
    answer_map = {str(row["qid"]): list(row.get("answers") or []) for row in eval_rows}
    poison_manifest = read_json(run_dir / "poison_manifest.json")
    targets = poison_manifest.get("target_answers_by_qid") or {}
    corpus = str(stored["corpus"])

    clean_em: list[float] = []
    clean_f1: list[float] = []
    asr: list[float] = []
    ambiguous: list[float] = []
    poison_presence: list[float] = []
    poison_fraction: list[float] = []
    target_poison_presence: list[float] = []
    target_poison_fraction: list[float] = []
    query_outcomes: list[dict[str, Any]] = []

    for output in outputs:
        qid = str(output["query_id"])
        prediction = str(output.get("answer") or "")
        answers = answer_map.get(qid, [])
        target = str(targets.get(qid) or "")
        target_hit = contains_answer(prediction, target) if target else False
        gold_hit = any(contains_answer(prediction, answer) for answer in answers)
        is_ambiguous = bool(target_hit and gold_hit and not exact_match(prediction, [target]))
        attack_success = bool(target_hit and not is_ambiguous)
        if corpus == "clean":
            clean_em.append(1.0 if exact_match(prediction, answers) else 0.0)
            clean_f1.append(token_f1(prediction, answers))
        else:
            asr.append(1.0 if attack_success else 0.0)
            ambiguous.append(1.0 if is_ambiguous else 0.0)

        poison_flags = list(output.get("context_is_poison_for_eval_only") or [])
        target_flags = list(output.get("context_is_target_poison_for_query") or [])
        poison_count = sum(bool(value) for value in poison_flags)
        target_count = sum(bool(value) for value in target_flags)
        poison_presence.append(1.0 if poison_count else 0.0)
        poison_fraction.append(poison_count / K_GEN)
        target_poison_presence.append(1.0 if target_count else 0.0)
        target_poison_fraction.append(target_count / K_GEN)
        query_outcomes.append(
            {
                "run_id": stored["run_id"],
                "dataset": stored["dataset"],
                "retriever": stored["retriever"],
                "attack_seed": stored["attack_seed"],
                "run_variant": stored["run_variant"],
                "query_id": qid,
                "answer": prediction,
                "clean_em": 1.0 if corpus == "clean" and exact_match(prediction, answers) else 0.0,
                "clean_f1": token_f1(prediction, answers) if corpus == "clean" else None,
                "target_hit": target_hit if corpus != "clean" else None,
                "gold_hit": gold_hit if corpus != "clean" else None,
                "ambiguous": is_ambiguous if corpus != "clean" else None,
                "attack_success": attack_success if corpus != "clean" else None,
                "context_chunk_ids": output.get("context_chunk_ids") or [],
                "context_count": int(output.get("context_count") or 0),
                "poison_count": poison_count,
                "target_poison_count": target_count,
                "parse_error": bool(output.get("parse_error")),
                "generated_contains_think": bool(output.get("generated_contains_think")),
            }
        )

    items = gate_items(run_dir)
    poison_gate = [item for item in items if item.get("is_poison_for_eval_only")]
    clean_gate = [item for item in items if not item.get("is_poison_for_eval_only")]
    blocked = lambda item: item.get("decision") == "block"
    recomputed = {
        "run_id": stored["run_id"],
        "num_eval_queries": len(outputs),
        "Clean_EM": mean(clean_em),
        "Clean_F1": mean(clean_f1),
        "ASR": mean(asr),
        "ambiguous_rate": mean(ambiguous),
        "TPR": mean([1.0 if blocked(item) else 0.0 for item in poison_gate]),
        "FPR": mean([1.0 if blocked(item) else 0.0 for item in clean_gate]),
        "PoisonPresence@5": mean(poison_presence),
        "PoisonFraction@5": mean(poison_fraction),
        "TargetPoisonPresence@5": mean(target_poison_presence),
        "TargetPoisonFraction@5": mean(target_poison_fraction),
        "avg_context_count": mean([float(row["context_count"]) for row in query_outcomes]),
        "gate_inputs": len(items),
        "gate_clean_inputs": len(clean_gate),
        "gate_poison_inputs": len(poison_gate),
        "parse_error_rate": mean([1.0 if row["parse_error"] else 0.0 for row in query_outcomes]),
        "generated_contains_think_rate": mean(
            [1.0 if row["generated_contains_think"] else 0.0 for row in query_outcomes]
        ),
    }
    return recomputed, {"stored": stored, "outcomes": query_outcomes}


def audit_collection(run_root: Path, expected_runs: int) -> dict[str, Any]:
    run_dirs = sorted(path.parent for path in run_root.glob("*/metrics.json"))
    comparisons: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    missing_files: list[dict[str, Any]] = []
    duplicate_qids: list[dict[str, Any]] = []
    run_details: dict[str, dict[str, Any]] = {}

    required = (
        "config.yaml",
        "dataset_manifest.json",
        "corpus_manifest.json",
        "poison_manifest.json",
        "retrieval_results.jsonl",
        "rerank_results.jsonl",
        "gate_decisions.jsonl",
        "generation_outputs.jsonl",
        "metrics.json",
        "run_log.json",
    )

    for run_dir in run_dirs:
        absent = [name for name in required if not (run_dir / name).exists()]
        if absent:
            missing_files.append({"run_id": run_dir.name, "missing": absent})
        recomputed, detail = recompute_run(run_dir)
        stored = detail["stored"]
        outcomes = detail["outcomes"]
        qids = [row["query_id"] for row in outcomes]
        duplicates = sorted(qid for qid, count in Counter(qids).items() if count > 1)
        if duplicates:
            duplicate_qids.append({"run_id": run_dir.name, "query_ids": duplicates})
        query_rows.extend(outcomes)
        run_details[run_dir.name] = {"stored": stored, "recomputed": recomputed, "outcomes": outcomes}
        for field in (*FLOAT_FIELDS, "gate_inputs", "gate_clean_inputs", "gate_poison_inputs"):
            comparisons.append(
                {
                    "collection": run_root.parent.name,
                    "run_id": run_dir.name,
                    "field": field,
                    "stored": stored.get(field),
                    "recomputed": recomputed.get(field),
                    "match": is_close(stored.get(field), recomputed.get(field)),
                }
            )

    return {
        "run_root": str(run_root.relative_to(ROOT)),
        "expected_runs": expected_runs,
        "actual_runs": len(run_dirs),
        "run_count_match": len(run_dirs) == expected_runs,
        "missing_files": missing_files,
        "duplicate_query_ids": duplicate_qids,
        "metric_comparisons": comparisons,
        "metric_mismatches": [row for row in comparisons if not row["match"]],
        "query_rows": query_rows,
        "run_details": run_details,
    }


def compare_main_csv(main_audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_csv(MAIN_DIR / "main_results.csv")
    by_id = {row["run_id"]: row for row in rows}
    checks: list[dict[str, Any]] = []
    for run_id, detail in main_audit["run_details"].items():
        metrics = detail["stored"]
        csv_row = by_id.get(run_id)
        if not csv_row:
            checks.append({"run_id": run_id, "field": "__row__", "match": False, "reason": "missing CSV row"})
            continue
        for field, value in metrics.items():
            if field not in csv_row:
                continue
            checks.append(
                {
                    "run_id": run_id,
                    "field": field,
                    "metrics_json": value,
                    "main_results_csv": csv_row[field],
                    "match": is_close(value, csv_row[field]),
                }
            )
    extra = sorted(set(by_id) - set(main_audit["run_details"]))
    for run_id in extra:
        checks.append({"run_id": run_id, "field": "__row__", "match": False, "reason": "CSV row without run"})
    return checks


def paired_position_diagnostics(main_audit: dict[str, Any]) -> list[dict[str, Any]]:
    details = main_audit["run_details"]
    groups: dict[tuple[str, str, int, str], dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for detail in details.values():
        stored = detail["stored"]
        variant = str(stored["run_variant"])
        match = re.fullmatch(r"(clean|poison)_(P_ret|P_gen)", variant)
        if not match:
            continue
        corpus, position = match.groups()
        key = (
            str(stored["dataset"]),
            str(stored["retriever"]),
            int(stored["attack_seed"]),
            corpus,
        )
        groups[key][position] = detail["outcomes"]

    rows: list[dict[str, Any]] = []
    for (dataset, retriever, seed, corpus), positions in sorted(groups.items()):
        if set(positions) != {"P_ret", "P_gen"}:
            continue
        left = {row["query_id"]: row for row in positions["P_ret"]}
        right = {row["query_id"]: row for row in positions["P_gen"]}
        common = sorted(set(left) & set(right))
        same_context = sum(left[qid]["context_chunk_ids"] == right[qid]["context_chunk_ids"] for qid in common)
        same_answer = sum(left[qid]["answer"] == right[qid]["answer"] for qid in common)
        same_outcome = sum(
            (
                left[qid]["clean_f1"] == right[qid]["clean_f1"]
                if corpus == "clean"
                else left[qid]["attack_success"] == right[qid]["attack_success"]
            )
            for qid in common
        )
        rows.append(
            {
                "dataset": dataset,
                "retriever": retriever,
                "attack_seed": seed,
                "corpus": corpus,
                "queries": len(common),
                "same_context_count": same_context,
                "same_context_rate": same_context / len(common) if common else None,
                "same_answer_count": same_answer,
                "same_answer_rate": same_answer / len(common) if common else None,
                "same_primary_outcome_count": same_outcome,
                "same_primary_outcome_rate": same_outcome / len(common) if common else None,
            }
        )
    return rows


def clean_seed_duplicates(main_audit: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for detail in main_audit["run_details"].values():
        stored = detail["stored"]
        if stored["corpus"] != "clean":
            continue
        key = (str(stored["dataset"]), str(stored["retriever"]), str(stored["run_variant"]))
        groups[key].append(detail)
    rows: list[dict[str, Any]] = []
    for key, details in sorted(groups.items()):
        if len(details) != len(SEEDS):
            continue
        outcome_maps = [
            {row["query_id"]: (row["context_chunk_ids"], row["answer"]) for row in detail["outcomes"]}
            for detail in details
        ]
        qids = sorted(set.intersection(*(set(mapping) for mapping in outcome_maps)))
        exact_context_and_answer = all(len({json.dumps(mapping[qid], sort_keys=True) for mapping in outcome_maps}) == 1 for qid in qids)
        rows.append(
            {
                "dataset": key[0],
                "retriever": key[1],
                "run_variant": key[2],
                "seed_rows": len(details),
                "queries_compared": len(qids),
                "all_contexts_and_answers_identical_across_seeds": exact_context_and_answer,
            }
        )
    return rows


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    repetitions: int = 5000,
    seed: int = 20260910,
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        return {"n": 0, "mean_difference": None, "ci95_low": None, "ci95_high": None}
    diffs = [float(a) - float(b) for a, b in zip(left, right)]
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    return {
        "n": len(diffs),
        "mean_difference": sum(diffs) / len(diffs),
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_repetitions": repetitions,
    }


def query_level_claims(main_audit: dict[str, Any], control_audit: dict[str, Any]) -> list[dict[str, Any]]:
    main_details = main_audit["run_details"]
    control_details = control_audit["run_details"]
    rows: list[dict[str, Any]] = []

    for dataset in ("nq", "hotpotqa"):
        for retriever in ("bm25", "bge_dense"):
            for seed in SEEDS:
                simplified_id = f"{dataset}__{retriever}__seed{seed}__A_simplified_poison_no_gate"
                multistage_id = f"{dataset}__{retriever}__seed{seed}__poison_no_gate"
                simplified = control_details[simplified_id]["outcomes"]
                multistage = main_details[multistage_id]["outcomes"]
                simp_map = {row["query_id"]: float(row["attack_success"]) for row in simplified}
                multi_map = {row["query_id"]: float(row["attack_success"]) for row in multistage}
                qids = sorted(set(simp_map) & set(multi_map))
                stats = paired_bootstrap([simp_map[qid] for qid in qids], [multi_map[qid] for qid in qids])
                rows.append(
                    {
                        "claim_family": "simplified_minus_multistage_ASR",
                        "dataset": dataset,
                        "retriever": retriever,
                        "attack_seed": seed,
                        **stats,
                    }
                )
    return rows


def path_and_provenance_audit() -> dict[str, Any]:
    script_paths = sorted((ROOT / "scripts").glob("*.py"))
    script_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in script_paths}
    manifest_files = list(MAIN_DIR.rglob("*.json")) + list(CONTROL_DIR.rglob("*.json"))
    absolute_linux_paths = 0
    absolute_windows_paths = 0
    for path in manifest_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        absolute_linux_paths += text.count("/home/amax/") + text.count("/data/")
        absolute_windows_paths += len(re.findall(r"[A-Za-z]:[/\\\\]", text))
    return {
        "git_directory_exists": (ROOT / ".git").exists(),
        "git_directory_file_count": sum(1 for path in (ROOT / ".git").rglob("*") if path.is_file())
        if (ROOT / ".git").exists()
        else 0,
        "script_hashes_generated_by_audit": script_hashes,
        "manifest_absolute_linux_path_occurrences": absolute_linux_paths,
        "manifest_absolute_windows_path_occurrences": absolute_windows_paths,
        "formal_run_script_hash_present_in_existing_manifests": False,
        "formal_run_git_commit_present_in_existing_manifests": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()

    main_audit = audit_collection(MAIN_RUNS, expected_runs=72)
    control_audit = audit_collection(CONTROL_RUNS, expected_runs=120)
    csv_checks = compare_main_csv(main_audit)
    position_rows = paired_position_diagnostics(main_audit)
    duplicate_rows = clean_seed_duplicates(main_audit)
    claim_rows = query_level_claims(main_audit, control_audit)
    provenance = path_and_provenance_audit()

    summary = {
        "audit_date": "2026-09-10",
        "main": {
            "expected_runs": main_audit["expected_runs"],
            "actual_runs": main_audit["actual_runs"],
            "missing_file_runs": len(main_audit["missing_files"]),
            "duplicate_query_id_runs": len(main_audit["duplicate_query_ids"]),
            "metric_mismatches": len(main_audit["metric_mismatches"]),
        },
        "control": {
            "expected_runs": control_audit["expected_runs"],
            "actual_runs": control_audit["actual_runs"],
            "missing_file_runs": len(control_audit["missing_files"]),
            "duplicate_query_id_runs": len(control_audit["duplicate_query_ids"]),
            "metric_mismatches": len(control_audit["metric_mismatches"]),
        },
        "main_csv_mismatches": sum(not row["match"] for row in csv_checks),
        "position_pairs": position_rows,
        "clean_seed_duplicate_groups": duplicate_rows,
        "provenance": provenance,
    }

    write_json(out_dir / "audit_summary.json", summary)
    write_json(
        out_dir / "artifact_integrity_details.json",
        {
            "main_missing_files": main_audit["missing_files"],
            "control_missing_files": control_audit["missing_files"],
            "main_duplicate_query_ids": main_audit["duplicate_query_ids"],
            "control_duplicate_query_ids": control_audit["duplicate_query_ids"],
            "main_metric_mismatches": main_audit["metric_mismatches"],
            "control_metric_mismatches": control_audit["metric_mismatches"],
        },
    )
    write_csv(out_dir / "metric_recalculation_checks.csv", main_audit["metric_comparisons"] + control_audit["metric_comparisons"])
    write_csv(out_dir / "main_csv_consistency_checks.csv", csv_checks)
    write_csv(out_dir / "gate_position_pair_diagnostics.csv", position_rows)
    write_csv(out_dir / "clean_seed_duplication_audit.csv", duplicate_rows)
    write_csv(out_dir / "query_level_bootstrap.csv", claim_rows)
    write_json(out_dir / "script_hashes.json", provenance["script_hashes_generated_by_audit"])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
