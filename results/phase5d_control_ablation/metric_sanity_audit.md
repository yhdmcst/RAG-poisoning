# metric_sanity_audit

## Parse / Thinking Output
- Phase 5D outputs: `36000`; parse_error_rate: `0.0%`; generated_contains_think_rate: `0.0%`.
- Phase 5 main outputs: `21600`; parse_error_rate: `0.0%`; generated_contains_think_rate: `0.0%`.

## Clean EM/F1 Format Audit
- Phase 5D clean outputs audited: `18000`; partial non-EM rate: `63.0%`; long-answer partial non-EM rate: `61.6%`.
- Phase 5 main clean outputs audited: `10800`; partial non-EM rate: `61.4%`; long-answer partial non-EM rate: `59.9%`.
- Interpretation: Clean EM is conservative for generated long-form answers; Clean F1 is the safer primary utility indicator for these smoke-scale QA outputs.
- Latency remains `latency_not_comparable=true` because Phase 5D reuses caches and uses multi-GPU generation.
