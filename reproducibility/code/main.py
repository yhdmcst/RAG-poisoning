# 主程序：串联所有分析 → 合并 figures/all_results.json → 生成 Markdown 表格
# 全部基于 data/ 下真实冻结 CSV，无任何仿真数据。
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json
import utils as u
import data_preparation
import analysis_1_pipeline_asr
import analysis_2_gate_position
import analysis_3_control_baselines
import analysis_4_failure_cases
import robustness
import gen_tables


def main():
    print("=" * 60)
    print("多阶段RAG投毒评测与门控位置分析 — 真实冻结数据")
    print("=" * 60)

    desc = data_preparation.run()
    a1 = analysis_1_pipeline_asr.run()
    a2 = analysis_2_gate_position.run()
    a3 = analysis_3_control_baselines.run()
    a4 = analysis_4_failure_cases.run()
    rob = robustness.run()

    all_results = {
        "meta": {
            "title": "多阶段RAG投毒评测与门控位置分析",
            "data_source": "real_frozen_experiment (Phase5/5D/6.5)",
            "simulation_used": False,
            "safegate_role": "可移动轻量评测探针/基线（非防御方法/检测器）",
        },
        "descriptive_stats": desc,
        "analysis_1_pipeline_asr": a1,
        "analysis_2_gate_position": a2,
        "analysis_3_control_baselines": a3,
        "analysis_4_failure_cases": a4,
        "robustness": rob,
    }
    u.save_json("all_results.json", all_results)

    gen_tables.run()

    print("\n" + "=" * 60)
    print("完成。核心发现：")
    print(f"  §5.1 简化>多阶段 {a1['summary']['n_combos_simplified_higher_asr']}/4 组合; "
          f"差值均值 {a1['summary']['diff_mean_range_pp']} pp")
    print(f"  §5.2 ASR-Drop 收敛 {a2['summary']['n_asr_drop_converged']}/4; "
          f"P_gen 更高 FPR {a2['summary']['n_pgen_higher_fpr']}/4")
    print(f"  §5.3 NQ 弱 / HotpotQA 强 (ASR-Drop)")
    print(f"  §5.4 多信号≥随机 {a3['safegate_vs_random']['n_rows_multi_ge_random']}/"
          f"{a3['safegate_vs_random']['n_comparable_rows']} 行")
    print(f"  §5.5 多信号≥单信号 {a3['multi_vs_single']['n_rows_multi_ge_single']}/"
          f"{a3['multi_vs_single']['n_comparable_rows']} 行 (负结果)")
    print(f"  §5.6 失败案例 {a4['total_cases']} 例, 实现 bug {a4['diagnosis']['implementation_bug_cases']}")
    print("=" * 60)
    return all_results


if __name__ == "__main__":
    main()
