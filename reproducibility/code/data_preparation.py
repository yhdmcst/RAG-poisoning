# 数据准备：校验真实 CSV 完整性 + 输出描述性统计到 figures/descriptive_stats.json
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import utils as u


def run():
    print("[data_preparation] 校验真实冻结数据 ...")
    main = u.load("main_results.csv")
    gate = u.load("gate_position_results.csv")
    simp = u.load("simplified_multistage_results.csv")

    # --- 实验矩阵计数（主矩阵 72 + 控制/消融 120 = 192 正式 run）---
    matrix = {
        "main_matrix_runs": int(len(main)),                 # 72
        "datasets": sorted(main["dataset"].unique().tolist()),
        "retrievers": sorted(main["retriever"].unique().tolist()),
        "attack_seeds": sorted(main["attack_seed"].unique().tolist()),
        "run_variants": sorted(main["run_variant"].unique().tolist()),
        "gate_positions": sorted([g for g in main["gate"].unique().tolist() if g != "none"]),
        "generator": sorted(main["generator"].unique().tolist()),
        "reranker": sorted(main["reranker"].unique().tolist()),
        "num_eval_queries": int(main["num_eval_queries"].iloc[0]),
        "pipeline": sorted(main["pipeline"].unique().tolist()),
        "latency_not_comparable": bool(main["latency_not_comparable"].iloc[0]),
    }
    # 控制/消融 run 计数（来自 5D 三个控制文件的行数，均为 dataset×retriever×seed[×gate]）
    simp_ct = u.load("simplified_multistage_results.csv")
    single_ct = u.load("single_signal_ablation_results.csv")
    random_ct = u.load("random_gate_matched_results.csv")
    matrix["control_ablation_rows"] = {
        "simplified_multistage": int(len(simp_ct)),   # 12 (4 combo × 3 seed)
        "single_signal_ablation": int(len(single_ct)),  # 24 (含 gate 维度)
        "random_gate_matched": int(len(random_ct)),      # 24
    }
    matrix["formal_runs_total_reported"] = 192  # 主矩阵72 + 控制/消融120（大纲口径）

    # --- 无门控 Clean F1 描述性范围（协议内相对效用基线，绝对值偏低）---
    clean_no_gate = main[main["run_variant"] == "clean_no_gate"].copy()
    clean_no_gate["Clean_F1"] = pd.to_numeric(clean_no_gate["Clean_F1"], errors="coerce")
    clean_f1_by_combo = []
    for (ds, rt), g in clean_no_gate.groupby(["dataset", "retriever"]):
        v = g["Clean_F1"].dropna().values
        clean_f1_by_combo.append({
            "dataset": ds, "retriever": rt,
            "clean_f1_mean": float(np.mean(v)), "n": int(len(v)),
        })
    clean_f1_all = clean_no_gate["Clean_F1"].dropna().values
    clean_f1_summary = {
        "min": float(np.min(clean_f1_all)),
        "max": float(np.max(clean_f1_all)),
        "mean": float(np.mean(clean_f1_all)),
        "by_combo": clean_f1_by_combo,
        "note": "无门控 Clean F1 绝对值偏低，仅作协议内相对效用比较基线（大纲论点8/限制）。",
    }

    # --- 无门控 poison ASR 范围（poison_no_gate 变体）---
    poison_no_gate = main[main["run_variant"] == "poison_no_gate"].copy()
    poison_no_gate["ASR"] = pd.to_numeric(poison_no_gate["ASR"], errors="coerce")
    asr_all = poison_no_gate["ASR"].dropna().values
    asr_summary = {
        "min": float(np.min(asr_all)), "max": float(np.max(asr_all)),
        "mean": float(np.mean(asr_all)), "n": int(len(asr_all)),
    }

    # --- 数据规模/协议关键配置（来自 config.yaml）---
    protocol = {
        "chunking": "C128-S32", "chunk_size": 128, "stride": 32,
        "top_m": 50, "K_gen": 5,
        "reranker": "BAAI/bge-reranker-base",
        "generator": "Qwen/Qwen3-8B",
        "decoding": {"temperature": 0, "do_sample": False, "top_p": 1.0, "max_new_tokens": 64},
        "attack": "targeted_template_poison", "poison_budget": 5,
        "bm25": {"k1": 0.9, "b": 0.4, "note": "仓库内固定实现，非标准 BM25"},
        "gate_signals": ["query_similarity_anomaly", "query_overlap_anomaly",
                         "corpus_distance_anomaly", "surface_repetition_anomaly"],
        "gate_fusion": "unweighted_mean_relu_zscores",
        "alpha_block": 0.05, "alpha_down": 0.10,
        "gate_encoder": "BAAI/bge-base-en-v1.5",
        "calibration_queries": 200, "eval_queries": 300,
        "split_seed": 20260905, "attack_seeds": [13, 42, 2026],
    }

    out = {
        "data_source": "real_frozen_experiment (Phase5/5D/6.5)",
        "simulation_used": False,
        "experiment_matrix": matrix,
        "clean_f1_no_gate": clean_f1_summary,
        "poison_asr_no_gate": asr_summary,
        "protocol_config": protocol,
    }
    u.save_json("descriptive_stats.json", out)

    # --- 基础完整性断言 ---
    assert matrix["main_matrix_runs"] == 72, "主矩阵应为 72 run"
    assert set(matrix["datasets"]) == {"nq", "hotpotqa"}
    assert set(matrix["retrievers"]) == {"bm25", "bge_dense"}
    assert 0 < clean_f1_summary["max"] < 1
    print(f"  主矩阵 run={matrix['main_matrix_runs']}, "
          f"Clean F1 无门控范围=[{clean_f1_summary['min']:.4f}, {clean_f1_summary['max']:.4f}], "
          f"无门控 ASR 范围=[{asr_summary['min']:.4f}, {asr_summary['max']:.4f}]")
    return out


if __name__ == "__main__":
    run()
