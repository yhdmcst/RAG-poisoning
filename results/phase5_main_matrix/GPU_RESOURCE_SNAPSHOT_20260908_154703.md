# GPU Resource Snapshot - 2026-09-08 15:47:03 CST

## Scope
- Local run: `true`
- Server / SSH: `not used`
- Monitor tool: `nvidia-smi`
- nvidia-smi available: `true`

## GPU Summary
| GPU | Model | Util % | Memory Used MiB | Memory Total MiB | Temp C | Power W |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | NVIDIA GeForce RTX 3090 | 0 | 595 | 24576 | 33 | 20.94 |
| 1 | NVIDIA GeForce RTX 3090 | 0 | 595 | 24576 | 32 | 22.97 |
| 2 | NVIDIA GeForce RTX 3090 | 0 | 595 | 24576 | 33 | 21.75 |
| 3 | NVIDIA GeForce RTX 3090 | 0 | 595 | 24576 | 33 | 16.76 |

## Process Check
- `run_phase5_main_matrix.py` process active: `false`
- Warm-up residual process detected: `false`
- Other GPU compute process visible: `/usr/local/bin/ollama` on GPUs 0-3, about `572 MiB` each from prior `nvidia-smi` snapshots.
- No process was terminated or migrated.
