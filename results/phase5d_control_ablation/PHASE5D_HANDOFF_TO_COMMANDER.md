# PHASE5D_HANDOFF_TO_COMMANDER

Current status: `Ready for Phase 6`
READY_FOR_PHASE6 = true

## Completion
- A simplified vs multi-stage control: `24 / 24` runs completed.
- B single-signal SafeGate ablation: `48 / 48` runs completed.
- C random matched-block gate baseline: `48 / 48` runs completed.
- failed runs after retry: `[]`
- Phase 5 main matrix overwritten: `false`.
- Warm-up directory overwritten or reused as formal result: `false`.

## Resource
- local run / no server: `true / true`.
- GPU snapshot rows: `['0, NVIDIA GeForce RTX 3090, 0, 595, 24576, 33, 21.49, 580.119.02', '1, NVIDIA GeForce RTX 3090, 0, 595, 24576, 32, 22.84, 580.119.02', '2, NVIDIA GeForce RTX 3090, 0, 595, 24576, 33, 21.78, 580.119.02', '3, NVIDIA GeForce RTX 3090, 0, 595, 24576, 34, 17.25, 580.119.02']`
- warm-up residual process at start: `False`

## Simplified vs Multi-Stage
| dataset | retriever | mean_simplified_minus_multistage_Clean_F1 | mean_simplified_minus_multistage_ASR |
| --- | --- | --- | --- |
| hotpotqa | bge_dense | -1.0% | 2.2% |
| hotpotqa | bm25 | -3.9% | 15.6% |
| nq | bge_dense | 2.2% | 11.1% |
| nq | bm25 | -3.6% | 18.1% |

## Multi-Signal vs Single-Signal
| gate_position | mean_control_ASR-Drop | mean_multi_signal_ASR-Drop | mean_control_FPR_clean | mean_multi_signal_FPR_clean | mean_control_Utility_Drop | mean_multi_signal_Utility_Drop |
| --- | --- | --- | --- | --- | --- | --- |
| P_gen | 41.2% | 24.8% | 16.8% | 15.2% | 2.9% | 3.5% |
| P_ret | 41.2% | 24.8% | 3.8% | 4.9% | 2.9% | 3.5% |

## Multi-Signal vs Random Matched-Block
| gate_position | mean_control_ASR-Drop | mean_multi_signal_ASR-Drop | mean_control_FPR_clean | mean_multi_signal_FPR_clean | mean_control_Utility_Drop | mean_multi_signal_Utility_Drop |
| --- | --- | --- | --- | --- | --- | --- |
| P_gen | 3.8% | 24.8% | 15.2% | 15.2% | 1.1% | 3.5% |
| P_ret | 1.7% | 24.8% | 4.9% | 4.9% | 0.4% | 3.5% |

## Required Answers
1. A/B/C completed runs: `A=24/24`, `B=48/48`, `C=48/48`.
2. simplified vs multi-stage Clean F1 and no-gate ASR differences are in `simplified_multistage_results.csv`; mean deltas are shown above.
3. Multi-signal SafeGate ASR-Drop >= single-signal in `4 / 24` comparable gate rows.
4. Multi-signal SafeGate ASR-Drop >= random matched-block in `24 / 24` comparable gate rows.
5. Leakage/input/metric-script abnormalities: no data leakage or input inconsistency detected; Clean EM likely underestimates long-form answer utility, so Clean F1 should be emphasized.
6. Recommendation: `Ready for Phase 6`.

## Paper Placement
- Main-table candidates: Phase 5 main matrix plus Phase 5D simplified/multi-stage and gate-position comparisons if commander accepts the BM25 implementation note.
- Supplement candidates: single-signal ablation, random matched-block baseline, parse/thinking and latency audits.
- Do not write: any claim that SafeGate is a standalone detector or universally superior defense.
