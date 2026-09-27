# 匿名统计复现包

本目录用于从论文实验的冻结 CSV 重新计算描述性统计、稳健性汇总、论文表格和结果 JSON。

## 可复现范围

- 主矩阵和控制/消融结果汇总。
- 简化管线与多阶段管线比较。
- P_ret 与 P_gen 安全—效用比较。
- 多信号、查询重叠单信号和匹配随机阻断比较。
- 80例低 Clean F1 诊断汇总。
- 查询级成对 bootstrap 区间跨零统计。
- 论文表格 Markdown 和分析 JSON。
- 34条结果一致性硬约束检查。
- 去除本地绝对路径的协议配置和公开数据清单。
- Clean F1、门控位置和统计不确定性审计材料。

## 不包含的内容

- 模型权重。
- NQ 1.5 GB 原始语料和模型缓存。
- 原始逐查询推理目录。
- 从模型推理阶段新增攻击种子。
- 完整复现第三方公开防御。

因此，本包是“冻结结果统计复现包”，不是端到端模型推理复现包。

## 运行

可以从任意目录执行。以下示例假设当前目录为本复现包：

```powershell
python -m pip install -r code/requirements.txt
python code/main.py
python code/audit_recheck.py
```

成功时，`code/main.py` 会重建 `figures/all_results.json` 和 `figures/TABLE_*.md`；`audit_recheck.py` 应输出 `0 失败`。

如需重绘统计图，再运行 `figures/gen_fig_*.py`。绘图依赖 Matplotlib；论文随稿已附带由原SVG源生成的高分辨率图，不要求为验证统计结果而重绘。

## 已验证结果

- 统计主程序完整运行成功。
- 34条硬约束通过，0失败。
- 主矩阵72行、控制与消融120个运行的论文口径保持一致。

## 数据来源

所有 CSV 均来自项目冻结的 Phase 5、Phase 5D、Phase CE-1 和 Phase 6.5 结果，不含模拟数据。原始相对路径和分析证据见论文逐条修改说明。
