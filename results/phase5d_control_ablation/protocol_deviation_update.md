# protocol_deviation_update

- RQ / contribution / SafeGate definition / threshold rule changed: `false`.
- Generator changed or Qwen2.5 fallback used: `false`; generator fixed to `Qwen/Qwen3-8B`.
- Dataset / retriever / attack / gate position expansion beyond Phase 5D request: `false`.
- Phase 5 main matrix overwritten: `false`.
- Warm-up results treated as Phase 5D or main result: `false`.
- `P_rerank`, low-overlap poison, fluent poison, second generator: `not used`.
- Implementation notes: `BM25 uses the same in-repo fixed BM25 implementation disclosed in Phase 5 main matrix because pyserini/rank_bm25 are not installed locally.`.
- Latency: `latency_not_comparable=true`.
