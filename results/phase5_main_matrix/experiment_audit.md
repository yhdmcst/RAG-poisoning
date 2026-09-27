# experiment_audit

## Resource Snapshot
- Local running / no server: `True` / `True`
- nvidia-smi available: `True`
- GPU rows: `['0, NVIDIA GeForce RTX 3090, 0, 595, 24576, 33, 20.85, 580.119.02', '1, NVIDIA GeForce RTX 3090, 0, 595, 24576, 31, 22.57, 580.119.02', '2, NVIDIA GeForce RTX 3090, 0, 595, 24576, 32, 22.15, 580.119.02', '3, NVIDIA GeForce RTX 3090, 0, 595, 24576, 33, 16.83, 580.119.02']`
- Compute process rows: `['GPU-ec8048c0-ef36-bb41-edfc-99e3811b424e, 1812, 572 MiB, /usr/local/bin/ollama', 'GPU-9b21c6e1-a131-efa4-d427-52a0aa502fdc, 1812, 572 MiB, /usr/local/bin/ollama', 'GPU-6f925463-3a97-4024-dcba-6dd9fecd8382, 1812, 572 MiB, /usr/local/bin/ollama', 'GPU-bb9884f3-99b0-43af-dc4d-0a0ec908e8e5, 1812, 572 MiB, /usr/local/bin/ollama']`
- Warm-up residual process detected: `False`
- Post-run snapshot file: `/home/amax/anquan/results/phase5_main_matrix/GPU_RESOURCE_SNAPSHOT_20260908_154703.md`
- Post-run `run_phase5_main_matrix.py` process active: `false`
- Post-run GPU rows: `['0, NVIDIA GeForce RTX 3090, 0, 595, 24576, 33, 20.94', '1, NVIDIA GeForce RTX 3090, 0, 595, 24576, 32, 22.97', '2, NVIDIA GeForce RTX 3090, 0, 595, 24576, 33, 21.75', '3, NVIDIA GeForce RTX 3090, 0, 595, 24576, 33, 16.76']`

## Compliance
- Qwen/Qwen3-8B only: `true`
- Qwen2.5 fallback: `false`
- Extra generator/dataset/attack/baseline: `false`
- Warm-up as main result: `false`
- RQ/contribution/SafeGate/threshold changes: `false`
- QA-as-document fallback: `false`
- Formal main matrix run count: `72 / 72`
- Extra run ids outside requested dataset x retriever x seed x variant grid: `0`
- Simplified RAG comparison in this formal matrix: `not run`; latest fixed main-matrix instruction used `reranker on` for all formal runs.
- Single-signal control in this formal matrix: `not run`; recorded as pending after main matrix.
- Protocol deviations or implementation notes: `BM25 uses an in-repo fixed BM25 implementation because pyserini/rank_bm25 are not installed locally; scoring parameters k1=0.9,b=0.4 are preserved and tokenizer is disclosed.`

## Data Leakage Checks
- Clean NQ corpus from BEIR raw passage corpus: `true`
- Clean HotpotQA corpus from local raw context paragraphs, not answer-as-document: `true`
- Synthetic support passages: `false`
- Gold answers written into clean corpus: `false`
- Target answers written into clean corpus: `false`
- Poison label used by retriever/reranker/generator/SafeGate: `false`
- Calibration/eval split overlap: `false`
- SafeGate threshold source: `clean calibration candidates only`
- Clean runs include poison chunks: `false`
- Poisoned runs use only seed-specific poison chunks: `true`

## Latency
- `latency_not_comparable=true` is set because loading, embedding/indexing, retrieval, reranking, gate, and generation are reported as separate stages and shared artifacts may be reused.
- Top-level latency manifest: `/home/amax/anquan/results/phase5_main_matrix/latency_manifest.json`
