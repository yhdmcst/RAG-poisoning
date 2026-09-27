# §5.1 简化管线 vs 多阶段管线 无门控 ASR（支撑 表2 / 图2 fig_pipeline_asr）
# 数据源：simplified_multistage_results.csv + ce_bootstrap_ci_summary.csv
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import utils as u


def run():
    print("[analysis_1] 简化 vs 多阶段无门控 ASR ...")
    df = u.load("simplified_multistage_results.csv")
    ci = u.load("ce_bootstrap_ci_summary.csv")

    # 每个 dataset×retriever 聚合 3 seed 的 simplified / multi-stage ASR 与差值
    rows = []
    for (ds, rt), g in df.groupby(["dataset", "retriever"]):
        g = g.sort_values("attack_seed")
        simp = pd.to_numeric(g["simplified_ASR_no_gate"], errors="coerce").values
        multi = pd.to_numeric(g["multi_stage_ASR_no_gate"], errors="coerce").values
        diff = pd.to_numeric(g["simplified_minus_multistage_ASR"], errors="coerce").values
        f1_simp = pd.to_numeric(g["simplified_Clean_F1"], errors="coerce").values
        f1_multi = pd.to_numeric(g["multi_stage_Clean_F1"], errors="coerce").values
        f1_diff = pd.to_numeric(g["simplified_minus_multistage_Clean_F1"], errors="coerce").values

        # 从该组合的 paired-bootstrap CI 取每个 seed 是否跨零
        sub = ci[(ci["comparison"] == "simplified_minus_multistage_no_gate_ASR") &
                 (ci["dataset"] == ds) & (ci["retriever"] == rt)]
        cross = {int(r["seed"]): (str(r["ci_crosses_zero"]).lower() == "true")
                 for _, r in sub.iterrows()}
        seeds = g["attack_seed"].tolist()
        cross_flags = [cross.get(int(s), None) for s in seeds]

        rows.append({
            "dataset": ds, "retriever": rt,
            "seeds": [int(s) for s in seeds],
            "simplified_ASR": [float(x) for x in simp],
            "multistage_ASR": [float(x) for x in multi],
            "diff_ASR": [float(x) for x in diff],
            "diff_ASR_mean": float(np.mean(diff)),
            "diff_ASR_std": float(np.std(diff, ddof=1)),  # 样本SD, 与冻结CE表一致
            "simplified_ASR_mean": float(np.mean(simp)),
            "multistage_ASR_mean": float(np.mean(multi)),
            "clean_f1_simplified_mean": float(np.mean(f1_simp)),
            "clean_f1_multistage_mean": float(np.mean(f1_multi)),
            "clean_f1_diff_mean": float(np.mean(f1_diff)),
            "ci_crosses_zero_per_seed": cross_flags,
            "n_seeds_ci_crosses_zero": int(sum(1 for c in cross_flags if c)),
        })

    # 全局结论校验
    n_combo_simplified_higher = sum(1 for r in rows if r["diff_ASR_mean"] > 0)
    n_combo_all_ci_exclude_zero = sum(1 for r in rows if r["n_seeds_ci_crosses_zero"] == 0)
    f1_signs = {f'{r["dataset"]}/{r["retriever"]}': ("+" if r["clean_f1_diff_mean"] > 0 else "-")
                for r in rows}

    out = {
        "section": "5.1",
        "figure": "fig_pipeline_asr (图2)",
        "table": "表2",
        "by_combo": rows,
        "summary": {
            "n_combos": len(rows),
            "n_combos_simplified_higher_asr": n_combo_simplified_higher,
            "n_combos_all3seed_ci_exclude_zero": n_combo_all_ci_exclude_zero,
            "diff_mean_range_pp": [round(100*min(r["diff_ASR_mean"] for r in rows), 1),
                                   round(100*max(r["diff_ASR_mean"] for r in rows), 1)],
            "clean_f1_diff_signs": f1_signs,
            "finding": ("简化管线无门控 ASR 在 4/4 组合高于多阶段管线；"
                        "3/4 组合 3 种子 CI 均不跨零，HotpotQA/BGE 密集检索区间跨零不作稳定差异判断；"
                        "Clean F1 差值非单向（3 负 1 正）。"),
        },
    }
    u.save_json("analysis_1_pipeline_asr_results.json", out)

    # --- 断言：结论方向必须与冻结数据一致 ---
    assert n_combo_simplified_higher == 4, "应为 4/4 组合简化>多阶段"
    for r in rows:
        # ASR 为比率，必须在 [0,1]
        assert all(0 <= x <= 1 for x in r["simplified_ASR"]), f"{r} simplified ASR 越界"
        assert all(0 <= x <= 1 for x in r["multistage_ASR"]), f"{r} multistage ASR 越界"
    print(f"  4/4 组合简化>多阶段; {n_combo_all_ci_exclude_zero}/4 组合 3 种子 CI 均不跨零; "
          f"差值均值范围 {out['summary']['diff_mean_range_pp']} pp")
    print(f"  Clean F1 差值符号: {f1_signs}")
    return out


if __name__ == "__main__":
    run()
