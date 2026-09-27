# 约束复核：从最终落盘的 figures/all_results.json 重新读取并硬校验（不信任进程内变量）。
# 通过则打印 AUDIT_OK 行的 n_constraints。
import json, os, sys

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
with open(os.path.join(FIG, "all_results.json"), encoding="utf-8") as f:
    R = json.load(f)

checks = []  # (描述, 通过?)


def add(desc, ok):
    checks.append((desc, bool(ok)))


d = R["descriptive_stats"]["experiment_matrix"]
add("主矩阵 run == 72", d["main_matrix_runs"] == 72)
add("正式 run 合计 == 192", d["formal_runs_total_reported"] == 192)
add("数据集 == {nq, hotpotqa}", set(d["datasets"]) == {"nq", "hotpotqa"})
add("检索器 == {bm25, bge_dense}", set(d["retrievers"]) == {"bm25", "bge_dense"})

cf = R["descriptive_stats"]["clean_f1_no_gate"]
add("Clean F1 无门控 ∈ (0,1)", 0 < cf["min"] <= cf["max"] < 1)
asr = R["descriptive_stats"]["poison_asr_no_gate"]
add("无门控 ASR ∈ [0,1]", 0 <= asr["min"] <= asr["max"] <= 1)

a1 = R["analysis_1_pipeline_asr"]
add("§5.1 简化>多阶段 4/4 组合", a1["summary"]["n_combos_simplified_higher_asr"] == 4)
for r in a1["by_combo"]:
    add(f"§5.1 {r['dataset']}/{r['retriever']} 简化 ASR∈[0,1]",
        all(0 <= x <= 1 for x in r["simplified_ASR"]))
    add(f"§5.1 {r['dataset']}/{r['retriever']} 多阶段 ASR∈[0,1]",
        all(0 <= x <= 1 for x in r["multistage_ASR"]))

a2 = R["analysis_2_gate_position"]
add("§5.2 ASR-Drop 收敛 4/4", a2["summary"]["n_asr_drop_converged"] == 4)
add("§5.2 P_gen 更高 FPR 4/4", a2["summary"]["n_pgen_higher_fpr"] == 4)
add("§5.2 P_gen 更高 TPR 4/4", a2["summary"]["n_pgen_higher_tpr"] == 4)
for r in a2["by_combo_gate"]:
    ok = all(0 <= r[f"{c}__mean"] <= 1.0001 for c in ["ASR-Drop", "TPR", "FPR_clean", "Utility Drop"])
    add(f"§5.2 {r['dataset']}/{r['retriever']}/{r['gate_position']} 指标∈[0,1]", ok)

a3 = R["analysis_3_control_baselines"]
add("§5.4 多信号≥随机 24/24",
    a3["safegate_vs_random"]["n_rows_multi_ge_random"] ==
    a3["safegate_vs_random"]["n_comparable_rows"] == 24)
add("§5.5 多信号≥单信号 < 全部（负结果）",
    a3["multi_vs_single"]["n_rows_multi_ge_single"] < a3["multi_vs_single"]["n_comparable_rows"])

a4 = R["analysis_4_failure_cases"]
add("§5.6 无实现 bug", a4["diagnosis"]["implementation_bug_cases"] == 0)
add("§5.6 案例分布求和 == 总数",
    sum(x["n"] for x in a4["error_type_distribution"]) == a4["total_cases"])

rob = R["robustness"]["key_checks"]
add("稳健性 simplified-multistage 9 不跨零/3 跨零",
    rob["simplified_minus_multistage_ASR"]["n_exclude_zero"] == 9 and
    rob["simplified_minus_multistage_ASR"]["n_cross_zero"] == 3)
add("稳健性 multi-single 16 不跨零",
    rob["multi_minus_single_asr_drop"]["n_exclude_zero"] == 16)
add("稳健性 multi-random 17 不跨零",
    rob["multi_minus_random_asr_drop"]["n_exclude_zero"] == 17)
add("稳健性 P_ret-P_gen asr_drop 12/12 跨零",
    rob["Pret_minus_Pgen_asr_drop"]["n_cross_zero"] == 12)

n_pass = sum(1 for _, ok in checks if ok)
n_fail = sum(1 for _, ok in checks if not ok)
for desc, ok in checks:
    if not ok:
        print(f"  ❌ {desc}")
print(f"约束复核：{n_pass} 通过 / {n_fail} 失败（共 {len(checks)} 条）")
if n_fail:
    sys.exit(1)
print(f"AUDIT_N_CONSTRAINTS={n_pass}")
