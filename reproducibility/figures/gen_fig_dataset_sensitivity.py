# -*- coding: utf-8 -*-
"""图5 fig_dataset_sensitivity（§5.3）：SafeGate 的数据集敏感性。

分组柱状图 + 误差棒。x 轴为 2 个数据集（NQ / HotpotQA），每组两根柱：
BM25 与 BGE密集 两种检索器下的 ASR-Drop（位置无关，P_ret/P_gen 收敛同值）。
误差棒为 3 攻击种子的样本 SD（取自 by_combo_gate 的 ASR-Drop__std）。
突出对比：NQ 约 3.3%~4.1%，HotpotQA 约 45.1%~46.7% —— SafeGate 探针高度数据集敏感。
数据来源 figures/analysis_2_gate_position_results.json（真实冻结实验，禁止改数）。
黑白印刷：灰阶填充 + hatch 纹理 + 黑描边区分两种检索器。
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_os.path.dirname(_HERE), _HERE):  # 工作区根（含 _utils）+ figures（含 _figbase）
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import numpy as np
import matplotlib.pyplot as plt
from _figbase import init, load, style_bar, save_all, PALETTE, auto_legend, dynamic_limits, declutter_axes, set_paper_placement

init()

data = load('analysis_2_gate_position_results.json')
# ASR-Drop 位置无关（P_ret=P_gen 收敛），SD 取 by_combo_gate 的 ASR-Drop__std（两位置相同）。
g = {(r['dataset'], r['retriever']): r
     for r in data['by_combo_gate'] if r['gate_position'] == 'P_gen'}

DATASETS = ['nq', 'hotpotqa']
DLABEL = ['NQ', 'HotpotQA']

bm25_mean = [g[(d, 'bm25')]['ASR-Drop__mean'] * 100 for d in DATASETS]
bm25_sd = [g[(d, 'bm25')]['ASR-Drop__std'] * 100 for d in DATASETS]
bge_mean = [g[(d, 'bge_dense')]['ASR-Drop__mean'] * 100 for d in DATASETS]
bge_sd = [g[(d, 'bge_dense')]['ASR-Drop__std'] * 100 for d in DATASETS]

x = np.arange(len(DATASETS))
bw = 0.34

fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.7))
set_paper_placement(fig)

b1 = ax.bar(x - bw / 2, bm25_mean, bw, yerr=bm25_sd, capsize=3,
            error_kw={'linewidth': 0.8, 'ecolor': 'black'},
            color=PALETTE[0], label='BM25')
b2 = ax.bar(x + bw / 2, bge_mean, bw, yerr=bge_sd, capsize=3,
            error_kw={'linewidth': 0.8, 'ecolor': 'black'},
            color=PALETTE[2], label='BGE密集')
style_bar(b1, 0)
style_bar(b2, 2)

ax.set_ylabel('ASR-Drop (%)')
ax.set_xlabel('数据集')
ax.set_xticks(x)
ax.set_xticklabels(DLABEL)
dynamic_limits(ax, y=[np.array(bm25_mean) + np.array(bm25_sd),
                      np.array(bge_mean) + np.array(bge_sd),
                      np.array([0.0])], pad=0.14, include_zero=True)
declutter_axes(ax, grid=True, grid_axis='y')
auto_legend(ax)

save_all(fig, 'fig_dataset_sensitivity')
