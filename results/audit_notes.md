# Notes: 实验与论文写作准备审计

## 审计基准
- 审计日期：2026-09-10
- 工作区：`E:\1keyan\安全`
- 审计对象：实验数据、结果、代码与工作流、技术报告、计划、论文草稿及辅助材料。

## 证据分类
- **明确事实**：文件、字段、日志或代码直接支持。
- **复核结果**：通过脚本执行、重新计算或交叉比较得到。
- **推断**：基于现有证据的合理判断，需要在报告中说明不确定性。
- **缺口**：材料缺失、口径不明、无法复现或相互矛盾。

## 初步资产概览
- 项目按 Phase 0-6 组织，另有 `results`、`scripts`、论文草稿与分析资产目录。
- 根目录包含项目简报、研究大纲、课题边界、科研 proposal、技术工作流文档以及 PDF/DOCX 汇报材料。
- 排除 `results/phase5_qwen3_compatibility/pydeps` 后，研究区仍有约 3,231 个结果文件，结果目录约 64.6 GB。
- 正式实验主体：
  - Phase 5 主矩阵：72 个 `metrics.json`。
  - Phase 5D 控制与消融：120 个 `metrics.json`。
  - 合计：192 个正式 run。
- 主矩阵覆盖 2 个数据集、2 个 retriever、3 个 attack seed、6 个运行变体；Phase 5D 覆盖 simplified 对照、single-signal 消融和 random matched-block 基线。
- 每个正式 run 标称 300 个 eval query；主矩阵生成输出 21,600 条，Phase 5D 生成输出 36,000 条。

## 已确认的阶段事实
- 主实验 generator 固定为 `Qwen/Qwen3-8B`，解码为 `temperature=0`、`do_sample=false`。
- reranker 固定为 `BAAI/bge-reranker-base`；simplified 对照关闭 reranker。
- 数据集为 NQ 与 HotpotQA；retriever 为仓库内 fixed BM25 与 BGE dense。
- 主矩阵的 `ablation_results.csv` 是未运行占位记录，真正消融结果位于 Phase 5D。
- 已有 Phase 6 报告明确承认尚未重新执行脚本、逐个复算 `metrics.json` 或完成统计推断，本次审计将补齐这些检查。

## 初步风险线索
- Clean run 的 `attack_seed` 仅改变 run id；在确定性生成配置下，同一 dataset/retriever/variant 的 clean 结果跨 seed 完全相同。三行不能被当作三次独立 clean 重复。
- `split_seed` 固定，所有 attack seed 共享同一 300 个 eval query。若做统计推断，问题级配对/分层 bootstrap 通常比把 3 个 seed 当独立样本更合适。
- `P_ret` 与 `P_gen` 的 answer-level ASR 和 Clean F1 在聚合表中完全相同，需要核查原始 contexts/output 是否也相同，以及“位置效应”究竟由输入集合、评分候选范围还是仅门控统计口径产生。
- 当前主结果文档多次使用“显著改变”一词，但尚无统计检验，正式论文应避免统计含义上的“显著”。

## 全量复核结果
- 192/192 个正式 run 完成。
- 57,600 条 generation output。
- 标准产物缺失 run：0。
- 重复 query ID run：0。
- 主矩阵与控制矩阵核心指标复算 mismatch：0。
- P_ret/P_gen 24 个配对的 answer 与 primary outcome 全部一致。
- 共同候选上的门控 decision agreement 为 100%。
- P_gen 平均只检查 P_ret 候选量的 14.43%。
- 12 个 clean 配置跨 3 个 attack seed 的 context 和 answer 全部逐题一致。

## 统计结论
- Simplified - multi-stage ASR 的 query-clustered bootstrap：
  - NQ/BM25：+18.11 pp，[14.89, 21.56]。
  - NQ/BGE：+11.11 pp，[7.78, 14.56]。
  - HotpotQA/BM25：+15.56 pp，[12.33, 18.78]。
  - HotpotQA/BGE：+2.22 pp，[-0.78, 5.11]。
- Multi-signal ASR-Drop：
  - NQ：3.33-4.11 pp。
  - HotpotQA：45.11-46.67 pp。
- Clean F1 drop：
  - NQ：约 1.3 pp，约为基线的 8.9%。
  - HotpotQA：5.16-6.03 pp，约为基线的 35.3%-43.1%。

## 方法与复现问题
- Pointwise gate 与 pointwise reranker 当前基本可交换，导致 answer-level position null result。
- 两位置原始 TPR/FPR 的候选总体和分母不同。
- P_gen 阈值在 reranked top-50 上校准，但在 high-rank prefix 上部署，出现 FPR shift。
- 每题 5 个 poison chunk 文本完全相同，且只使用 T0。
- Phase 2 变量表要求 3 个 deterministic templates，实际偏离未登记。
- NQ actual strict chunk 与 strict/corpus manifest 一致；dataset manifest 残留旧 bytes/hash。
- Phase 3C readiness 环境与正式 Phase 5 环境不同，没有正式环境 package lock。
- `.git` 为空，正式 run 无 commit 或运行时脚本 hash。
- 现有 failure case 文件只包含 clean QA miss，不能支撑平衡案例分析。

## 交付物
- `paper_writing_readiness/analysis-report.md`
- `paper_writing_readiness/stats-appendix.md`
- `paper_writing_readiness/figure-catalog.md`
- `paper_writing_readiness/claim-evidence-register.csv`
- `paper_writing_readiness/2026-09-10--phase5--r01--论文写作准备综合审计报告.md`
- `audit_outputs/` 全量复算与诊断文件。
