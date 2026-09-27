**表3 P\_ret 与 P\_gen 的安全—效用指标**  

**Table 3 Security-utility metrics at P\_ret and P\_gen**

| 数据集 | 检索器 | 指标 Metric | P\_ret 均值±SD | P\_gen 均值±SD |
| --- | --- | --- | --- | --- |
| NQ | BM25 | ASR-Drop | 4.1±0.8 | 4.1±0.8 |
| NQ | BM25 | TPR | 11.7±1.0 | 16.0±1.7 |
| NQ | BM25 | FPR | 4.4±0.0 | 8.9±0.0 |
| NQ | BM25 | Utility Drop | 1.4±0.0 | 1.4±0.0 |
| NQ | BGE密集 | ASR-Drop | 3.3±1.2 | 3.3±1.2 |
| NQ | BGE密集 | TPR | 11.9±1.1 | 20.3±0.7 |
| NQ | BGE密集 | FPR | 5.0±0.0 | 9.2±0.0 |
| NQ | BGE密集 | Utility Drop | 1.3±0.0 | 1.3±0.0 |
| HotpotQA | BM25 | ASR-Drop | 46.7±1.2 | 46.7±1.2 |
| HotpotQA | BM25 | TPR | 69.7±1.1 | 96.9±0.4 |
| HotpotQA | BM25 | FPR | 4.8±0.0 | 19.7±0.0 |
| HotpotQA | BM25 | Utility Drop | 5.2±0.0 | 5.2±0.0 |
| HotpotQA | BGE密集 | ASR-Drop | 45.1±1.1 | 45.1±1.1 |
| HotpotQA | BGE密集 | TPR | 83.6±1.1 | 96.8±0.9 |
| HotpotQA | BGE密集 | FPR | 5.1±0.0 | 22.9±0.0 |
| HotpotQA | BGE密集 | Utility Drop | 6.0±0.0 | 6.0±0.0 |

注：数值为百分比。答案级 ASR-Drop 与 Utility Drop 在 4/4 组合于 P\_ret、P\_gen 收敛（实现条件下收敛/配对 CI 跨零）；P\_gen 在 4/4 组合同时给出更高 TPR 与更高 FPR。
