# 公共工具：加载真实冻结 CSV、按 dataset×retriever(×gate) 聚合、落盘 JSON。
# 本模块不生成任何仿真数据；所有数值均来自 data/ 下的真实实验 CSV。
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(_HERE), "data")
FIG_DIR = os.path.join(os.path.dirname(_HERE), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [13, 42, 2026]
COMBOS = [("nq", "bm25"), ("nq", "bge_dense"),
          ("hotpotqa", "bm25"), ("hotpotqa", "bge_dense")]


def load(name):
    """读取 data/ 下的真实 CSV。"""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少真实数据文件: {path}")
    return pd.read_csv(path)


# 跨 3 攻击种子报告 mean±SD 时用样本标准差 ddof=1，与冻结 CE-1 表一致（论文引用同一口径）。
STD_DDOF = 1


def agg_mean_std(df, group_cols, value_cols):
    """按 group_cols 分组，对 value_cols 求跨种子 mean/std（样本标准差 ddof=1 与冻结 CE-1 表一致）。"""
    out = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        rec["n_seeds"] = int(len(g))
        for c in value_cols:
            vals = pd.to_numeric(g[c], errors="coerce").dropna().values
            rec[f"{c}__mean"] = float(np.mean(vals)) if len(vals) else None
            rec[f"{c}__std"] = float(np.std(vals, ddof=STD_DDOF)) if len(vals) > 1 else 0.0
        out.append(rec)
    return out


def round_rec(obj, nd=6):
    """递归四舍五入，避免浮点噪声写入 JSON。"""
    if isinstance(obj, dict):
        return {k: round_rec(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_rec(v, nd) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(obj, nd)
    return obj


def save_json(name, obj):
    path = os.path.join(FIG_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(round_rec(obj), f, ensure_ascii=False, indent=2)
    print(f"  [saved] figures/{name} ({os.path.getsize(path)} bytes)")
    return path


def pct(x, nd=1):
    """小数 → 百分比字符串。"""
    if x is None:
        return "—"
    return f"{100*x:.{nd}f}"
