**表4 Clean F1 低值代表性失败案例（80 例诊断）**  

**Table 4 Representative Clean-F1 failure cases (80 diagnosed samples)**

| 错误类型 Type | 占比 | 代表问题 Question | 标准答案 | 生成答案 | F1 |
| --- | --- | --- | --- | --- | --- |
| 评测/归一化口径 | 70.0% | how many episodes are in chicago fire season 4 | 23 | The answer is 23 episodes. | 0.40 |
| 检索未命中 | 12.5% | how many books are in the one piece series | 83 // more than 80 | I don't know. | 0.00 |
| 重排/上下文丢失 | 11.2% | chemicals produced in one part of a plant that affect the growth and response of other par | Plant hormones // hormones | The answer is: hormones. | 0.50 |
| 生成错误/截断 | 5.0% | Who managed the 5,000 acre estate of Thomas Jefferson? | Edmund Bacon | I don't know. | 0.00 |
| 不确定 | 1.2% | when does boomer find out she a cylon | Kobol's Last Gleaming | Boomer finds out she is a Cylon after the events of the first season, specifical | 0.00 |

注：80 例 Clean F1 诊断未发现实现 bug；主因为评测/归一化口径（56 例）、检索/重排未命中、生成截断。支撑'Clean F1 绝对值偏低仅作协议内相对效用比较'的限制。 案例文本取自真实诊断样本 clean\_f1\_case\_samples.csv。
