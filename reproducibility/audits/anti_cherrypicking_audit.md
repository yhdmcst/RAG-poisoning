# Anti-Cherry-Picking Audit

## Coverage Checks

- Dataset coverage: `NQ` and `HotpotQA` are both present in the formal main matrix and Phase 5D controls.
- Retriever coverage: `BM25` and `BGE dense` are both present.
- Safety and utility metrics required by protocol: `ASR, ASR-Drop, TPR, FPR, Clean_EM, Clean_F1, Utility Drop`. They are present in main/ablation CSV schemas.
- Gate positions: both `P_ret` and `P_gen` are reported.
- Seeds: `13`, `42`, and `2026` are retained; no best-seed-only presentation is permitted.

## Required Main-Text Coverage

1. A table or figure must include all four dataset-retriever combinations, not HotpotQA alone.
2. Gate-position reporting must pair ASR-Drop with TPR, FPR, Clean EM/F1, and Utility Drop.
3. NQ's weaker SafeGate ASR-Drop and HotpotQA's stronger ASR-Drop must both remain visible.
4. The higher P_gen TPR must be shown together with its higher FPR; it cannot be presented as an unconditional improvement.
5. The multi-signal versus single-signal negative result must be mentioned in the main discussion, even if the complete table is supplementary.
6. The matched random-block comparison must be retained as a control, with its scope stated narrowly.

## Placement Recommendation

- Main text: simplified/multi-stage no-gate comparison, P_ret/P_gen safety-utility profile, NQ/HotpotQA sensitivity, and matched random-block control summary.
- Supplement: seed-wise rows, full single-signal ablation, case samples, parse/thinking audit, and latency-not-comparable details.
- Not permitted: hiding adverse seeds, reporting only ASR-Drop, or presenting SafeGate as a standalone detector/defense.

## Claim Boundary

- Use `observed`, `under the tested setting`, `suggests`, and `baseline/probe` language.
- Do not write `statistically significant` from these descriptive/bootstrapped intervals alone.
- Do not claim universal superiority of P_gen, multi-signal fusion, or SafeGate.
