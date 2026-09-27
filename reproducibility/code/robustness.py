# §4.2 稳健性/统计报告口径：paired-bootstrap CI 跨零统计 + 跨种子一致性
# 数据源：ce_bootstrap_ci_summary.csv, ce_seedwise_summary.csv
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import utils as u


def run():
    print("[robustness] paired-bootstrap CI 跨零 + 跨种子一致性 ...")
    ci = u.load("ce_bootstrap_ci_summary.csv")
    ci["cross"] = ci["ci_crosses_zero"].astype(str).str.lower().eq("true")

    # 每个 comparison×metric 的跨零 / 不跨零计数
    by_comp = []
    for (comp, metric), g in ci.groupby(["comparison", "metric"]):
        n = int(len(g))
        cross = int(g["cross"].sum())
        by_comp.append({
            "comparison": comp, "metric": metric,
            "n_blocks": n, "n_cross_zero": cross, "n_exclude_zero": n - cross,
        })

    # 大纲关键口径核对
    def get(comp, metric):
        for r in by_comp:
            if r["comparison"] == comp and r["metric"] == metric:
                return r
        return None

    simp = get("simplified_minus_multistage_no_gate_ASR", "ASR_difference")
    ms_single = get("multi_signal_minus_single_signal", "asr_drop")
    ms_random = get("multi_signal_minus_random_matched_block", "asr_drop")
    pret_pgen_asr = get("P_ret_minus_P_gen", "asr_drop")

    # 跨种子一致性：simplified-multistage ASR 差值方向是否随种子一致
    seedwise = u.load("ce_seedwise_summary.csv")

    out = {
        "section": "4.2 稳健性/统计口径",
        "resamples": 2000,
        "ci_type": "descriptive paired-bootstrap CI (not statistical significance)",
        "by_comparison_metric": by_comp,
        "key_checks": {
            "simplified_minus_multistage_ASR": simp,     # 期望 9 不跨零 / 3 跨零
            "multi_minus_single_asr_drop": ms_single,    # 16 不跨零 / 8 跨零
            "multi_minus_random_asr_drop": ms_random,    # 17 不跨零 / 7 跨零
            "Pret_minus_Pgen_asr_drop": pret_pgen_asr,   # 全部跨零（收敛/功效不足）
        },
        "power_note": ("每个 dataset×retriever×seed 仅 3 攻击种子，配对 CI 功效有限；"
                       "P_ret−P_gen 的 ASR-Drop 区间跨零应解释为'实现条件下收敛或功效不足'，"
                       "而非'位置无差异'的强结论。"),
        "seedwise_rows": int(len(seedwise)),
    }
    u.save_json("robustness_results.json", out)

    # 断言：关键跨零口径与冻结文档一致
    assert simp is not None and simp["n_exclude_zero"] == 9 and simp["n_cross_zero"] == 3, \
        f"simplified-multistage CI 应 9 不跨零/3 跨零，实得 {simp}"
    assert ms_single is not None and ms_single["n_exclude_zero"] == 16, \
        f"multi-single asr_drop 应 16 不跨零，实得 {ms_single}"
    assert ms_random is not None and ms_random["n_exclude_zero"] == 17, \
        f"multi-random asr_drop 应 17 不跨零，实得 {ms_random}"
    print(f"  simplified-multistage: {simp['n_exclude_zero']} 不跨零/{simp['n_cross_zero']} 跨零")
    print(f"  multi-single asr_drop: {ms_single['n_exclude_zero']}/{ms_single['n_blocks']} 不跨零")
    print(f"  multi-random asr_drop: {ms_random['n_exclude_zero']}/{ms_random['n_blocks']} 不跨零")
    print(f"  Pret-Pgen asr_drop: {pret_pgen_asr['n_cross_zero']}/{pret_pgen_asr['n_blocks']} 跨零")
    return out


if __name__ == "__main__":
    run()
