# PHASE5_HANDOFF_TO_COMMANDER

Current judgment: `Conditional Go`
READY_MAIN_MATRIX_COMPLETE = true

## Completion
- Formal main runs completed: `72 / 72`.
- Output directory: `/home/amax/anquan/results/phase5_main_matrix`.
- Warm-up directory preserved and not overwritten: `true`.
- Warm-up results used as formal main results: `false`.
- Local run / no server: `true / true`.
- GPU monitor available in runner: `True`.
- Warm-up residual process detected at start: `False`.
- Runner exit observed after final run: `true`.
- Post-run GPU snapshot: `/home/amax/anquan/results/phase5_main_matrix/GPU_RESOURCE_SNAPSHOT_20260908_154703.md`.
- Post-run `run_phase5_main_matrix.py` process active: `false`.

## Matrix
- datasets: `NQ`, `HotpotQA`
- retrievers: `BM25`, `BGE dense`
- seeds: `13`, `42`, `2026`
- reranker: `BAAI/bge-reranker-base` on
- generator: `Qwen/Qwen3-8B` only
- gate positions: `P_ret`, `P_gen`

## Run Paths
- nq__bm25__seed13__clean_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed13__clean_no_gate`
- nq__bm25__seed13__poison_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed13__poison_no_gate`
- nq__bm25__seed13__clean_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed13__clean_P_ret`
- nq__bm25__seed13__poison_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed13__poison_P_ret`
- nq__bm25__seed13__clean_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed13__clean_P_gen`
- nq__bm25__seed13__poison_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed13__poison_P_gen`
- nq__bm25__seed42__clean_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed42__clean_no_gate`
- nq__bm25__seed42__poison_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed42__poison_no_gate`
- nq__bm25__seed42__clean_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed42__clean_P_ret`
- nq__bm25__seed42__poison_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed42__poison_P_ret`
- nq__bm25__seed42__clean_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed42__clean_P_gen`
- nq__bm25__seed42__poison_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed42__poison_P_gen`
- nq__bm25__seed2026__clean_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed2026__clean_no_gate`
- nq__bm25__seed2026__poison_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed2026__poison_no_gate`
- nq__bm25__seed2026__clean_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed2026__clean_P_ret`
- nq__bm25__seed2026__poison_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed2026__poison_P_ret`
- nq__bm25__seed2026__clean_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed2026__clean_P_gen`
- nq__bm25__seed2026__poison_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bm25__seed2026__poison_P_gen`
- nq__bge_dense__seed13__clean_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed13__clean_no_gate`
- nq__bge_dense__seed13__poison_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed13__poison_no_gate`
- nq__bge_dense__seed13__clean_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed13__clean_P_ret`
- nq__bge_dense__seed13__poison_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed13__poison_P_ret`
- nq__bge_dense__seed13__clean_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed13__clean_P_gen`
- nq__bge_dense__seed13__poison_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed13__poison_P_gen`
- nq__bge_dense__seed42__clean_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed42__clean_no_gate`
- nq__bge_dense__seed42__poison_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed42__poison_no_gate`
- nq__bge_dense__seed42__clean_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed42__clean_P_ret`
- nq__bge_dense__seed42__poison_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed42__poison_P_ret`
- nq__bge_dense__seed42__clean_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed42__clean_P_gen`
- nq__bge_dense__seed42__poison_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed42__poison_P_gen`
- nq__bge_dense__seed2026__clean_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed2026__clean_no_gate`
- nq__bge_dense__seed2026__poison_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed2026__poison_no_gate`
- nq__bge_dense__seed2026__clean_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed2026__clean_P_ret`
- nq__bge_dense__seed2026__poison_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed2026__poison_P_ret`
- nq__bge_dense__seed2026__clean_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed2026__clean_P_gen`
- nq__bge_dense__seed2026__poison_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/nq__bge_dense__seed2026__poison_P_gen`
- hotpotqa__bm25__seed13__clean_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed13__clean_no_gate`
- hotpotqa__bm25__seed13__poison_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed13__poison_no_gate`
- hotpotqa__bm25__seed13__clean_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed13__clean_P_ret`
- hotpotqa__bm25__seed13__poison_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed13__poison_P_ret`
- hotpotqa__bm25__seed13__clean_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed13__clean_P_gen`
- hotpotqa__bm25__seed13__poison_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed13__poison_P_gen`
- hotpotqa__bm25__seed42__clean_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed42__clean_no_gate`
- hotpotqa__bm25__seed42__poison_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed42__poison_no_gate`
- hotpotqa__bm25__seed42__clean_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed42__clean_P_ret`
- hotpotqa__bm25__seed42__poison_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed42__poison_P_ret`
- hotpotqa__bm25__seed42__clean_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed42__clean_P_gen`
- hotpotqa__bm25__seed42__poison_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed42__poison_P_gen`
- hotpotqa__bm25__seed2026__clean_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed2026__clean_no_gate`
- hotpotqa__bm25__seed2026__poison_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed2026__poison_no_gate`
- hotpotqa__bm25__seed2026__clean_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed2026__clean_P_ret`
- hotpotqa__bm25__seed2026__poison_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed2026__poison_P_ret`
- hotpotqa__bm25__seed2026__clean_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed2026__clean_P_gen`
- hotpotqa__bm25__seed2026__poison_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bm25__seed2026__poison_P_gen`
- hotpotqa__bge_dense__seed13__clean_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed13__clean_no_gate`
- hotpotqa__bge_dense__seed13__poison_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed13__poison_no_gate`
- hotpotqa__bge_dense__seed13__clean_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed13__clean_P_ret`
- hotpotqa__bge_dense__seed13__poison_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed13__poison_P_ret`
- hotpotqa__bge_dense__seed13__clean_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed13__clean_P_gen`
- hotpotqa__bge_dense__seed13__poison_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed13__poison_P_gen`
- hotpotqa__bge_dense__seed42__clean_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed42__clean_no_gate`
- hotpotqa__bge_dense__seed42__poison_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed42__poison_no_gate`
- hotpotqa__bge_dense__seed42__clean_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed42__clean_P_ret`
- hotpotqa__bge_dense__seed42__poison_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed42__poison_P_ret`
- hotpotqa__bge_dense__seed42__clean_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed42__clean_P_gen`
- hotpotqa__bge_dense__seed42__poison_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed42__poison_P_gen`
- hotpotqa__bge_dense__seed2026__clean_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed2026__clean_no_gate`
- hotpotqa__bge_dense__seed2026__poison_no_gate: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed2026__poison_no_gate`
- hotpotqa__bge_dense__seed2026__clean_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed2026__clean_P_ret`
- hotpotqa__bge_dense__seed2026__poison_P_ret: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed2026__poison_P_ret`
- hotpotqa__bge_dense__seed2026__clean_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed2026__clean_P_gen`
- hotpotqa__bge_dense__seed2026__poison_P_gen: `/home/amax/anquan/results/phase5_main_matrix/runs/hotpotqa__bge_dense__seed2026__poison_P_gen`

## Required Answers
1. Only Qwen/Qwen3-8B was used: `true`.
2. No fallback to Qwen2.5: `true`.
3. Matrix expansion beyond requested main matrix: `false`.
4. Warm-up treated as main result: `false`.
5. RQ / contribution / threshold / SafeGate definition modified: `false`.
6. Config, metrics, and latency schema consistent across completed runs: `true`.
7. Leakage/input/protocol deviations: `BM25 uses an in-repo fixed BM25 implementation because pyserini/rank_bm25 are not installed locally; scoring parameters k1=0.9,b=0.4 are preserved and tokenizer is disclosed.`.

## Formal Comparisons
- simplified RAG vs multi-stage RAG formal comparison: `not yet in this fixed main matrix`; latest user-fixed matrix keeps BGE-reranker on for all formal runs. Warm-up contains a separate NQ-only comparison but is not counted as formal main result.
- gate position formal comparison: `started/completed` for P_ret vs P_gen across every completed dataset/retriever/seed block.
- SafeGate single-signal formal comparison: `not run in this main-matrix pass`; `ablation_results.csv` records this as pending control, because the latest execution rule says finish the main matrix before considering control matrices.

## Key Metrics By Dataset/Retriever/Seed
| dataset | retriever | seed | Clean F1 no-gate | ASR no-gate | P_ret ASR-Drop | P_ret FPR | P_ret Utility Drop | P_gen ASR-Drop | P_gen FPR | P_gen Utility Drop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nq | bm25 | 13 | 15.1% | 38.3% | 4.0% | 4.4% | 1.4% | 4.0% | 8.9% | 1.4% |
| nq | bm25 | 42 | 15.1% | 44.0% | 3.3% | 4.4% | 1.4% | 3.3% | 8.9% | 1.4% |
| nq | bm25 | 2026 | 15.1% | 43.3% | 5.0% | 4.4% | 1.4% | 5.0% | 8.9% | 1.4% |
| nq | bge_dense | 13 | 14.5% | 26.7% | 2.0% | 5.0% | 1.3% | 2.0% | 9.2% | 1.3% |
| nq | bge_dense | 42 | 14.5% | 31.7% | 3.7% | 5.0% | 1.3% | 3.7% | 9.2% | 1.3% |
| nq | bge_dense | 2026 | 14.5% | 30.3% | 4.3% | 5.0% | 1.3% | 4.3% | 9.2% | 1.3% |
| hotpotqa | bm25 | 13 | 14.6% | 46.7% | 46.0% | 4.8% | 5.2% | 46.0% | 19.7% | 5.2% |
| hotpotqa | bm25 | 42 | 14.6% | 46.3% | 46.0% | 4.8% | 5.2% | 46.0% | 19.7% | 5.2% |
| hotpotqa | bm25 | 2026 | 14.6% | 48.0% | 48.0% | 4.8% | 5.2% | 48.0% | 19.7% | 5.2% |
| hotpotqa | bge_dense | 13 | 14.0% | 45.7% | 44.3% | 5.1% | 6.0% | 44.3% | 22.9% | 6.0% |
| hotpotqa | bge_dense | 42 | 14.0% | 46.0% | 44.7% | 5.1% | 6.0% | 44.7% | 22.9% | 6.0% |
| hotpotqa | bge_dense | 2026 | 14.0% | 48.0% | 46.3% | 5.1% | 6.0% | 46.3% | 22.9% | 6.0% |

## SafeGate Discrimination
- Minimum discrimination observed in any gate row: `True`.
- Interpretation remains empirical; SafeGate is reported only as a movable baseline/probe.

## Continue / Control Matrix
- Main matrix status determines next action. If complete, the next commander decision is whether to run the pending single-signal/control matrix.
