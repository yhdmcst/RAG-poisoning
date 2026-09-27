# -*- coding: utf-8 -*-
"""图2 fig_pipeline_asr（§5.1）：管线形态对无门控 ASR 的影响。

分组柱状图 + 误差棒。x 轴为 4 个「数据集—检索器」组合，每组两根柱：
简化管线 vs 多阶段管线的无门控 ASR（均值，误差棒为 3 攻击种子样本 SD, ddof=1）。
数据来源 figures/analysis_1_pipeline_asr_results.json（真实冻结实验，禁止改数）。
黑白印刷：灰阶填充 + hatch 纹理 + 黑描边区分两条管线。
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

data = load('analysis_1_pipeline_asr_results.json')

# 组合显示顺序与中文标签
ORDER = [('nq', 'bm25'), ('nq', 'bge_dense'), ('hotpotqa', 'bm25'), ('hotpotqa', 'bge_dense')]
LABEL = {
    ('nq', 'bm25'): 'NQ\nBM25',
    ('nq', 'bge_dense'): 'NQ\nBGE密集',
    ('hotpotqa', 'bm25'): 'HotpotQA\nBM25',
    ('hotpotqa', 'bge_dense'): 'HotpotQA\nBGE密集',
}
rows = {(r['dataset'], r['retriever']): r for r in data['by_combo']}

simp_mean, simp_sd, multi_mean, multi_sd, cats = [], [], [], [], []
for key in ORDER:
    r = rows[key]
    s = np.asarray(r['simplified_ASR'], dtype=float) * 100.0
    m = np.asarray(r['multistage_ASR'], dtype=float) * 100.0
    simp_mean.append(s.mean()); simp_sd.append(s.std(ddof=1))
    multi_mean.append(m.mean()); multi_sd.append(m.std(ddof=1))
    cats.append(LABEL[key])

x = np.arange(len(ORDER))
bw = 0.36

fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.7))
set_paper_placement(fig)

b1 = ax.bar(x - bw / 2, simp_mean, bw, yerr=simp_sd, capsize=3,
            error_kw={'linewidth': 0.8, 'ecolor': 'black'},
            color=PALETTE[0], label='简化管线')
b2 = ax.bar(x + bw / 2, multi_mean, bw, yerr=multi_sd, capsize=3,
            error_kw={'linewidth': 0.8, 'ecolor': 'black'},
            color=PALETTE[1], label='多阶段管线')
style_bar(b1, 0)
style_bar(b2, 1)

ax.set_ylabel('无门控攻击成功率 ASR (%)')
ax.set_xlabel('数据集—检索器组合')
ax.set_xticks(x)
ax.set_xticklabels(cats)
dynamic_limits(ax, y=[np.array(simp_mean) + np.array(simp_sd),
                      np.array(multi_mean) + np.array(multi_sd),
                      np.array([0.0])], pad=0.12, include_zero=True)
declutter_axes(ax, grid=True, grid_axis='y')
auto_legend(ax)

save_all(fig, 'fig_pipeline_asr')
