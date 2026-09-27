# Phase 5 中段总结

生成时间：2026-09-08

用途：给 Phase 6 风险审计和后续论文写作做复查基线。这里记录的是“能写什么、不能写什么、证据在哪里”，不是重新解释全部 CSV。

## 1. 当前范围

- Phase 5 主矩阵已完成：`72 / 72`
- Phase 5D 控制与消融已完成：`120 / 120`
- 合计：`192` 个正式 run
- 模型：只用 `Qwen/Qwen3-8B`
- 未回退 `Qwen2.5`
- 未新增 dataset / generator / attack / P_rerank / gate position
- `latency_not_comparable=true`，不把耗时当横向结论

## 2. 一句话结论

- 多阶段管线确实改变了 ASR，但不是所有设置都同样明显。
- SafeGate 作为 probe 成立，且明显强于随机删块基线。
- 但单信号在不少条件下反而更强，不能把 multi-signal 写成“必然更优”。
- `P_gen` 不是稳定优于 `P_ret`，它更像是更激进、FPR 更高的版本。

## 3. 主矩阵核心证据

| dataset | retriever | Clean F1 no-gate | ASR no-gate | P_ret ASR-Drop / FPR | P_gen ASR-Drop / FPR |
|---|---|---:|---:|---:|---:|
| NQ | BM25 | 15.14% | 41.89% | 4.11% / 4.43% | 4.11% / 8.93% |
| NQ | BGE dense | 14.50% | 29.56% | 3.33% / 5.05% | 3.33% / 9.20% |
| HotpotQA | BM25 | 14.60% | 47.00% | 46.67% / 4.78% | 46.67% / 19.70% |
| HotpotQA | BGE dense | 13.99% | 46.56% | 45.11% / 5.15% | 45.11% / 22.92% |

### 读法

- NQ 上门控收益有限，更多体现为误报成本差异。
- HotpotQA 上门控收益很强，且 `P_gen` 的误报明显更高。
- 主矩阵最稳的表述是：**gate position 改变了安全-效用权衡**。

## 4. Phase 5D 关键控制

### 4.1 simplified vs multi-stage

| dataset / retriever | Simplified - Multi-stage Clean F1 | Simplified - Multi-stage ASR |
|---|---:|---:|
| NQ / BM25 | -3.6 pp | +18.1 pp |
| NQ / BGE dense | +2.2 pp | +11.1 pp |
| HotpotQA / BM25 | -3.9 pp | +15.6 pp |
| HotpotQA / BGE dense | -1.0 pp | +2.2 pp |

结论：
- 多阶段并不总是更准，但在投毒场景下通常更稳。
- 这组结果足够支持“管线形态影响安全结果”，但不足以支持“多阶段永远更好”。

### 4.2 multi-signal vs single-signal

- 单信号在 ASR-Drop 上更强，multi-signal 只在少数行不差于单信号。
- 结论上应写成：**多信号 SafeGate 不是对单信号的稳定统治性提升**。
- 这意味着方法定位要保守，不能写成“更强的新 detector”。

### 4.3 multi-signal vs random matched-block

- multi-signal SafeGate 在 `24 / 24` 可比行上 ASR-Drop 不低于 random matched-block。
- 这条是最重要的“不是随机删块”证据。

## 5. 可以写进正文的结论

- 多阶段 RAG 管线与 simplified RAG 在投毒下表现不同。
- gate position 会显著改变 TPR / FPR 和 utility trade-off。
- SafeGate 作为可移动 probe / baseline 可复现。
- 与随机删块相比，SafeGate 不是纯随机效应。
- Clean F1 比 Clean EM 更适合本课题主叙述，尤其在长答案场景里。

## 6. 不能写重的话

- 不写“首个”“完全真实”“全面防御”
- 不写“新防线”“first-line defense”
- 不写“优于所有已有防御”
- 不写“P_gen 明显优于 P_ret”
- 不写“multi-signal 明显优于 single-signal”
- 不写“latency 可直接横比”
- 不写“SafeGate 可证明识别所有投毒”

## 7. 已知限制

- BM25 使用的是仓库内固定实现，不是 pyserini / rank_bm25。
- `latency_not_comparable=true`，只能当运行痕迹。
- Clean EM 对长答案偏保守，后续解释优先看 Clean F1。
- 结论是经验性的，不能推广成通用安全保证。

## 8. 论文写作文件入口

### 主结果

- `phase5_main_matrix/main_results.csv`
- `phase5_main_matrix/gate_position_results.csv`
- `phase5_main_matrix/ablation_results.csv`
- `phase5_main_matrix/experiment_audit.md`
- `phase5_main_matrix/failure_cases.md`
- `phase5_main_matrix/PHASE5_HANDOFF_TO_COMMANDER.md`

### 控制与消融

- `phase5d_control_ablation/control_results.csv`
- `phase5d_control_ablation/simplified_multistage_results.csv`
- `phase5d_control_ablation/single_signal_ablation_results.csv`
- `phase5d_control_ablation/random_gate_matched_results.csv`
- `phase5d_control_ablation/metric_sanity_audit.md`
- `phase5d_control_ablation/protocol_deviation_update.md`
- `phase5d_control_ablation/failure_cases.md`
- `phase5d_control_ablation/experiment_audit.md`
- `phase5d_control_ablation/PHASE5D_HANDOFF_TO_COMMANDER.md`

## 9. Phase 6 建议

- 先写 Results / Discussion / Limitations 的主框架，再回填图表。
- 主图优先级：
  1. gate position 对比
  2. simplified vs multi-stage
  3. SafeGate vs random matched-block
  4. single-signal ablation
- 补充材料优先放：
  - latency
  - parse / think 审计
  - 失败案例
  - BM25 实现说明

