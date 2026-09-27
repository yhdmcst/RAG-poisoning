# experiment_audit

## Resource Snapshot
- local run / no server: `true / true`
- nvidia-smi available: `True`
- GPU rows: `['0, NVIDIA GeForce RTX 3090, 0, 595, 24576, 33, 21.49, 580.119.02', '1, NVIDIA GeForce RTX 3090, 0, 595, 24576, 32, 22.84, 580.119.02', '2, NVIDIA GeForce RTX 3090, 0, 595, 24576, 33, 21.78, 580.119.02', '3, NVIDIA GeForce RTX 3090, 0, 595, 24576, 34, 17.25, 580.119.02']`
- compute process rows: `['GPU-ec8048c0-ef36-bb41-edfc-99e3811b424e, 1812, 572 MiB, /usr/local/bin/ollama', 'GPU-9b21c6e1-a131-efa4-d427-52a0aa502fdc, 1812, 572 MiB, /usr/local/bin/ollama', 'GPU-6f925463-3a97-4024-dcba-6dd9fecd8382, 1812, 572 MiB, /usr/local/bin/ollama', 'GPU-bb9884f3-99b0-43af-dc4d-0a0ec908e8e5, 1812, 572 MiB, /usr/local/bin/ollama']`
- warm-up residual process detected: `False`
- run_phase5_main_matrix.py residual detected: `False`

## Completion
- A simplified no-gate completed: `24 / 24`
- B single-signal gate completed: `48 / 48`
- C random matched-block completed: `48 / 48`
- failed worker runs after retry: `[]`

## Parallelism
- Generation used independent per-run worker processes with `CUDA_VISIBLE_DEVICES` binding.
- Shared Phase 5 main matrix artifacts were reused only as read-only inputs or symlinked immutable `.npy` cache files.
- No two workers write the same run directory.

## Compliance
- Qwen/Qwen3-8B only: `true`
- Qwen2.5 fallback: `false`
- Extra dataset/generator/attack/gate position/P_rerank: `false`
- RQ/contribution/SafeGate/threshold changes: `false`
- QA-as-document fallback: `false`
- Clean runs include poison chunks: `false`
- Poisoned runs use only seed-specific targeted_template_poison chunks: `true`
- Poison label used by model inputs: `false`; labels are used only for metric audit.
- Protocol deviations or notes: `BM25 uses the same in-repo fixed BM25 implementation disclosed in Phase 5 main matrix because pyserini/rank_bm25 are not installed locally.`
