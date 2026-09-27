# 生成 Markdown 表格（表1/表2/表3/失败案例表）到 figures/TABLE_*.md
# 全部数据来自 figures/*.json（真实计算结果），中英文标题对应，供 paper-write 直接嵌入。
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json
import utils as u

FIG = u.FIG_DIR
DS_EN = {"nq": "NQ", "hotpotqa": "HotpotQA"}
RT_EN = {"bm25": "BM25", "bge_dense": "BGE密集"}


def _load(name):
    with open(os.path.join(FIG, name), encoding="utf-8") as f:
        return json.load(f)


def _w(name, text):
    with open(os.path.join(FIG, name), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [saved] figures/{name} ({os.path.getsize(os.path.join(FIG,name))} bytes)")


def table1_matrix():
    d = _load("descriptive_stats.json")
    m, p = d["experiment_matrix"], d["protocol_config"]
    t = []
    t.append("**表1 实验矩阵与变量配置**  \n")
    t.append("**Table 1 Experiment matrix and variable configuration**\n")
    t.append("| 变量 Variable | 取值 Value |")
    t.append("| --- | --- |")
    t.append(f"| 数据集 Dataset | NQ, HotpotQA |")
    t.append(f"| 检索器 Retriever | BM25($k_1$={p['bm25']['k1']}, $b$={p['bm25']['b']}, 仓库内固定实现), BGE密集 |")
    t.append(f"| 重排器 Reranker | {p['reranker']} |")
    t.append(f"| 生成器 Generator | {p['generator']}（确定性解码, $T$=0, max\\_new\\_tokens={p['decoding']['max_new_tokens']}）|")
    t.append(f"| 分块 Chunking | {p['chunking']}（块长{p['chunk_size']}, 步长{p['stride']}）|")
    t.append(f"| 候选池/上下文 | top\\_m={p['top_m']}, $K_{{gen}}$={p['K_gen']} |")
    t.append(f"| 攻击 Attack | {p['attack']}, 每问 {p['poison_budget']} 毒块 |")
    t.append(f"| 门控位置 Gate position | P\\_ret（检索后重排前）, P\\_gen（重排后生成前）|")
    t.append(f"| 门控信号 Gate signals | 查询相似度/查询重叠/语料距离/表面重复（{p['gate_fusion']}）|")
    t.append(f"| 阈值 Thresholds | alpha\\_block={p['alpha_block']}, alpha\\_down={p['alpha_down']} |")
    t.append(f"| 攻击种子 Seeds | {', '.join(str(s) for s in p['attack_seeds'])}（split\\_seed={p['split_seed']}）|")
    t.append(f"| 校准/评测问题数 | 校准 {p['calibration_queries']} / 评测 {p['eval_queries']} |")
    t.append(f"| 正式运行数 Runs | 主矩阵 {m['main_matrix_runs']} + 控制/消融 120 = {m['formal_runs_total_reported']} |")
    t.append(f"| 延迟 Latency | latency\\_not\\_comparable=True（分阶段计时, 不可横向比较）|")
    _w("TABLE_1_experiment_matrix.md", "\n".join(t) + "\n")


def table2_pipeline():
    d = _load("analysis_1_pipeline_asr_results.json")
    t = []
    t.append("**表2 简化管线与多阶段管线的无门控攻击成功率对比**  \n")
    t.append("**Table 2 No-gate attack success rate: simplified vs multi-stage pipeline**\n")
    t.append("| 数据集 Dataset | 检索器 Retriever | 简化管线 ASR (13/42/2026) | 多阶段管线 ASR (13/42/2026) | 差值均值±SD Diff | 3种子CI跨零 | Clean F1差值 |")
    t.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in d["by_combo"]:
        s = "/".join(u.pct(x) for x in r["simplified_ASR"])
        m = "/".join(u.pct(x) for x in r["multistage_ASR"])
        diff = f"{u.pct(r['diff_ASR_mean'])}±{u.pct(r['diff_ASR_std'])}"
        cz = r["n_seeds_ci_crosses_zero"]
        cz_s = "0/3" if cz == 0 else f"{cz}/3"
        f1d = ("+" if r["clean_f1_diff_mean"] > 0 else "") + u.pct(r["clean_f1_diff_mean"])
        t.append(f"| {DS_EN[r['dataset']]} | {RT_EN[r['retriever']]} | {s} | {m} | {diff} | {cz_s} | {f1d} |")
    t.append("\n注：ASR、Clean F1 均为百分比；差值=简化−多阶段。CI 为描述性 paired-bootstrap 区间（非统计显著性）。"
             "HotpotQA/BGE密集检索 3 种子区间均跨零，不作稳定差异判断。")
    _w("TABLE_2_pipeline_asr.md", "\n".join(t) + "\n")


def table3_gate():
    d = _load("analysis_2_gate_position_results.json")
    t = []
    t.append("**表3 P\\_ret 与 P\\_gen 的安全—效用指标**  \n")
    t.append("**Table 3 Security-utility metrics at P\\_ret and P\\_gen**\n")
    t.append("| 数据集 | 检索器 | 指标 Metric | P\\_ret 均值±SD | P\\_gen 均值±SD |")
    t.append("| --- | --- | --- | --- | --- |")
    # 从 by_combo_gate 取每个 combo 的 P_ret/P_gen
    rows = d["by_combo_gate"]
    idx = {}
    for r in rows:
        idx[(r["dataset"], r["retriever"], r["gate_position"])] = r
    metric_disp = [("ASR-Drop", "ASR-Drop"), ("TPR", "TPR"),
                   ("FPR_clean", "FPR"), ("Utility Drop", "Utility Drop")]
    for ds, rt in u.COMBOS:
        pr = idx.get((ds, rt, "P_ret"))
        pg = idx.get((ds, rt, "P_gen"))
        for col, disp in metric_disp:
            a = f"{u.pct(pr[col+'__mean'])}±{u.pct(pr[col+'__std'])}"
            b = f"{u.pct(pg[col+'__mean'])}±{u.pct(pg[col+'__std'])}"
            t.append(f"| {DS_EN[ds]} | {RT_EN[rt]} | {disp} | {a} | {b} |")
    t.append("\n注：数值为百分比。答案级 ASR-Drop 与 Utility Drop 在 4/4 组合于 P\\_ret、P\\_gen 收敛"
             "（实现条件下收敛/配对 CI 跨零）；P\\_gen 在 4/4 组合同时给出更高 TPR 与更高 FPR。")
    _w("TABLE_3_gate_position.md", "\n".join(t) + "\n")


def table_failure():
    d = _load("analysis_4_failure_cases_results.json")
    t = []
    t.append("**表4 Clean F1 低值代表性失败案例（80 例诊断）**  \n")
    t.append("**Table 4 Representative Clean-F1 failure cases (80 diagnosed samples)**\n")
    t.append("| 错误类型 Type | 占比 | 代表问题 Question | 标准答案 | 生成答案 | F1 |")
    t.append("| --- | --- | --- | --- | --- | --- |")
    dist = {x["error_type"]: x for x in d["error_type_distribution"]}
    for c in d["representative_cases"]:
        share = u.pct(dist[c["error_type"]]["share"]) + "%"
        q = c["question"].replace("|", "/")
        gold = c["gold_answer"].replace("|", "/")
        gen = c["generated_answer"].replace("|", "/")
        t.append(f"| {c['label']} | {share} | {q} | {gold} | {gen} | {c['normalized_f1']:.2f} |")
    t.append(f"\n注：{d['diagnosis']['finding']} 案例文本取自真实诊断样本 clean\\_f1\\_case\\_samples.csv。")
    _w("TABLE_failure_cases.md", "\n".join(t) + "\n")


def run():
    print("[gen_tables] 生成 Markdown 表格 ...")
    table1_matrix()
    table2_pipeline()
    table3_gate()
    table_failure()


if __name__ == "__main__":
    run()
