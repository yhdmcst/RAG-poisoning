# Statistical Uncertainty Report

- Bootstrap resamples: `2000`; deterministic analysis seed: `20260910`.
- Query-level paired bootstrap is used for ASR and Clean F1-derived outcomes.
- TPR/FPR are chunk-level rates. Their uncertainty is estimated with paired query-cluster bootstrap; P_ret and P_gen can have different gate-input denominators.
- CI crossing zero means the current data do not support a stable directional difference at the 95% interval level. No claim of statistical significance is made.

## Scope

- simplified vs multi-stage: all 2 datasets × 2 retrievers × 3 seeds.
- P_ret vs P_gen: TPR, FPR, ASR-Drop, Utility Drop, all formal main blocks.
- multi-signal vs single-signal and matched random-block: Phase 5D controls, paired by dataset/retriever/seed/query.

## Observed CI Coverage

- simplified minus multi-stage no-gate ASR: `9` blocks exclude zero; `3` cross zero.
- P_ret minus P_gen TPR: `6` blocks exclude zero; `6` cross zero.
- P_ret minus P_gen FPR: `12` blocks exclude zero; `0` cross zero; the observed differences are negative for P_ret minus P_gen.
- P_ret minus P_gen ASR-Drop and Utility Drop are exact zero paired differences in the formal blocks; these are outcome-level convergence results, not evidence of a position advantage.
- multi-signal minus single-signal ASR-Drop: `16` blocks exclude zero; `8` cross zero.
- multi-signal minus single-signal Utility Drop: `0` blocks exclude zero; `24` cross zero.
- multi-signal minus matched random-block ASR-Drop: `17` blocks exclude zero; `7` cross zero.
- multi-signal minus matched random-block Utility Drop: `20` blocks exclude zero; `4` cross zero.

## Interpretation

- The CSV contains per-block means, standard deviations, 95% bootstrap intervals, and zero-crossing flags.
- These intervals support statements such as `observed difference` or `uncertain direction`; they do not license `statistically significant` without a prespecified test and multiplicity policy.
- The main evidence remains heterogeneous across NQ vs HotpotQA and BM25 vs BGE dense. Aggregate prose must retain that heterogeneity.
