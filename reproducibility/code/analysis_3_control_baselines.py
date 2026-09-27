# §5.4 SafeGate(多信号) vs 匹配随机阻断（random_gate_matched_results.csv）
# §5.5 多信号 vs 单信号消融（single_signal_ablation_results.csv）—— 负结果，必须保留
# 图4 fig_safegate_control
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import utils as u


def _agg_control(df, label):
    """按 dataset×retriever×gate 聚合 3 seed：multi vs control 的 ASR-Drop / Utility Drop。
    并逐行(dataset×retriever×gate×seed)统计 multi >= control 的行数（大纲口径 24 可比行）。"""
    rows = []
    n_rows_total = 0
    n_rows_multi_ge = 0
    for (ds, rt, gp), g in df.groupby(["dataset", "retriever", "gate_position"]):
        m_drop = pd.to_numeric(g["multi_signal_ASR-Drop"], errors="coerce").values
        c_drop = pd.to_numeric(g["control_ASR-Drop"], errors="coerce").values
        m_util = pd.to_numeric(g["multi_signal_Utility_Drop"], errors="coerce").values
        c_util = pd.to_numeric(g["control_Utility_Drop"], errors="coerce").values
        # 逐 seed 行比较（含容差，避免浮点误差把相等判成小于）
        for md, cd in zip(m_drop, c_drop):
            n_rows_total += 1
            if md >= cd - 1e-9:
                n_rows_multi_ge += 1
        rows.append({
            "dataset": ds, "retriever": rt, "gate_position": gp, "n_seeds": int(len(g)),
            "multi_asr_drop_mean": float(np.mean(m_drop)),
            "control_asr_drop_mean": float(np.mean(c_drop)),
            "multi_util_drop_mean": float(np.mean(m_util)),
            "control_util_drop_mean": float(np.mean(c_util)),
            "multi_ge_control_all_seeds": bool(np.all(m_drop >= c_drop - 1e-9)),
            "multi_higher_util_drop": bool(np.mean(m_util) > np.mean(c_util)),
        })
    return rows, n_rows_total, n_rows_multi_ge


def run():
    print("[analysis_3] SafeGate vs 随机阻断 / 单信号 ...")
    rnd = u.load("random_gate_matched_results.csv")
    sgl = u.load("single_signal_ablation_results.csv")

    rnd_rows, rnd_n, rnd_ge = _agg_control(rnd, "random")
    sgl_rows, sgl_n, sgl_ge = _agg_control(sgl, "single")

    # §5.4：多信号在可比行 ASR-Drop 是否不低于随机；同时报告 Utility Drop
    n_rnd_util_higher = sum(1 for r in rnd_rows if r["multi_higher_util_drop"])
    # §5.5：多信号是否不低于单信号（负结果：应仅少数行成立）
    n_sgl_util_higher = sum(1 for r in sgl_rows if r["multi_higher_util_drop"])

    out = {
        "section": "5.4 + 5.5",
        "figure": "fig_safegate_control (图4)",
        "safegate_vs_random": {
            "by_group": rnd_rows,
            "n_comparable_rows": rnd_n,
            "n_rows_multi_ge_random": rnd_ge,
            "n_groups_multi_higher_utility_drop": n_rnd_util_higher,
            "finding": (f"多信号在 {rnd_ge}/{rnd_n} 可比行 ASR-Drop 不低于匹配随机阻断；"
                        "但 Utility Drop 更高，不上升为通用检测能力。"),
        },
        "multi_vs_single": {
            "by_group": sgl_rows,
            "n_comparable_rows": sgl_n,
            "n_rows_multi_ge_single": sgl_ge,
            "finding": (f"多信号仅在 {sgl_ge}/{sgl_n} 可比行 ASR-Drop 不低于单信号（查询重叠）；"
                        "未形成对单信号的一致优势（负结果，必须保留）。"),
        },
        "summary": {
            "safegate_beats_random_all": rnd_ge == rnd_n,
            "multi_not_consistently_better_than_single": sgl_ge < sgl_n,
        },
    }
    u.save_json("analysis_3_control_baselines_results.json", out)

    # --- 断言：负结果方向 ---
    assert rnd_ge == rnd_n, f"多信号应在全部 {rnd_n} 可比行不低于随机（实得 {rnd_ge}）"
    assert sgl_ge < sgl_n, f"多信号不应在全部行不低于单信号（负结果，实得 {sgl_ge}/{sgl_n}）"
    print(f"  §5.4 多信号≥随机: {rnd_ge}/{rnd_n} 行; 多信号 Utility Drop 更高组数: {n_rnd_util_higher}/{len(rnd_rows)}")
    print(f"  §5.5 多信号≥单信号: {sgl_ge}/{sgl_n} 行 (负结果)")
    return out


if __name__ == "__main__":
    run()
