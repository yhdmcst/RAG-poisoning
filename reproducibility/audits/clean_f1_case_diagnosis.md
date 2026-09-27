# Clean F1 Case Diagnosis

- Source: Phase 5 formal main-matrix `clean_no_gate` runs, seed `13`, all 4 dataset-retriever combinations.
- Sampling: first 20 normalized-F1 errors per combination after deterministic query-id ordering; total 80 cases.
- Evidence presence is judged by qrel document ids from the frozen eval split.
- Truncation is a heuristic assessment because generation artifacts do not store exact generated token counts.

## Counts

| dataset | retriever | cases | retrieval_miss | rerank_or_context_miss | generation_error_or_truncation | evaluation_normalization_issue | implementation_bug | uncertain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa | bge_dense | 20 | 1 | 0 | 2 | 17 | 0 | 0 |
| hotpotqa | bm25 | 20 | 0 | 1 | 2 | 17 | 0 | 0 |
| nq | bge_dense | 20 | 2 | 7 | 0 | 11 | 0 | 0 |
| nq | bm25 | 20 | 7 | 1 | 0 | 11 | 0 | 1 |

## Interpretation

- `evaluation_normalization_issue` means the prediction has non-zero normalized token F1 but misses exact match; it is an evaluation-interpretation issue, not evidence of a code bug.
- `generation_error_or_truncation` is intentionally conservative for likely max-token cutoffs or retained-evidence / zero-F1 cases.
- `implementation_bug` is assigned only when an internal artifact contradiction is directly observed; this audit does not silently relabel a low score as a bug.
- Full case-level evidence is in `clean_f1_case_samples.csv`.
