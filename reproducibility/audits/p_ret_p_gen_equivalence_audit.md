# P_ret / P_gen Equivalence Audit

- Representative blocks: NQ/BM25 seed13, NQ/BGE dense seed42, HotpotQA/BM25 seed13, HotpotQA/BGE dense seed42.
- All comparisons are query-id aligned and use the formal main-matrix outputs.

## Protocol / Code Findings

- Signal family and threshold rule are shared: the same four signals, ReLU-z normalization, and clean-calibration-only alpha_block=0.05 rule.
- Numeric thresholds are recalibrated per gate position on position-specific candidate populations; therefore exact tau equality is not required and is not observed as a protocol invariant.
- P_ret gates retrieval-order top-M candidates before reranking; P_gen gates the reranked top-M queue. These are not the same rerank buffer.
- P_ret reranks the surviving candidates and then takes top-5; P_gen scans the reranked queue and backfills until five pass. These are not the same implementation path.

## Representative Query-Level Agreement

| dataset | retriever | seed | n | clean context exact agreement | poisoned context exact agreement | Clean F1 outcome agreement | ASR outcome agreement | poison-presence agreement |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| nq | bm25 | 13 | 300 | 1.000 | 0.960 | 1.000 | 1.000 | 1.000 |
| nq | bge_dense | 42 | 300 | 1.000 | 0.990 | 1.000 | 1.000 | 1.000 |
| hotpotqa | bm25 | 13 | 300 | 1.000 | 0.993 | 1.000 | 1.000 | 1.000 |
| hotpotqa | bge_dense | 42 | 300 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Interpretation

- Equal answer-level ASR-Drop is not sufficient to conclude interface equivalence: gate inputs, tau values, candidate order, and backfill paths differ.
- If representative final contexts and answer outcomes agree, the defensible interpretation is implementation-conditional outcome convergence under the shared scoring and context budget.
- TPR/FPR are chunk-level rates. P_gen can inspect a different number and composition of candidates than P_ret, so FPR/TPR denominator differences do not imply different answer-level defense strength by themselves.
- No implementation bug is inferred solely from equal aggregate ASR-Drop.
