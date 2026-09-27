**表1 实验矩阵与变量配置**  

**Table 1 Experiment matrix and variable configuration**

| 变量 Variable | 取值 Value |
| --- | --- |
| 数据集 Dataset | NQ, HotpotQA |
| 检索器 Retriever | BM25($k_1$=0.9, $b$=0.4, 仓库内固定实现), BGE密集 |
| 重排器 Reranker | BAAI/bge-reranker-base |
| 生成器 Generator | Qwen/Qwen3-8B（确定性解码, $T$=0, max\_new\_tokens=64）|
| 分块 Chunking | C128-S32（块长128, 步长32）|
| 候选池/上下文 | top\_m=50, $K_{gen}$=5 |
| 攻击 Attack | targeted_template_poison, 每问 5 毒块 |
| 门控位置 Gate position | P\_ret（检索后重排前）, P\_gen（重排后生成前）|
| 门控信号 Gate signals | 查询相似度/查询重叠/语料距离/表面重复（unweighted_mean_relu_zscores）|
| 阈值 Thresholds | alpha\_block=0.05, alpha\_down=0.1 |
| 攻击种子 Seeds | 13, 42, 2026（split\_seed=20260905）|
| 校准/评测问题数 | 校准 200 / 评测 300 |
| 正式运行数 Runs | 主矩阵 72 + 控制/消融 120 = 192 |
| 延迟 Latency | latency\_not\_comparable=True（分阶段计时, 不可横向比较）|
