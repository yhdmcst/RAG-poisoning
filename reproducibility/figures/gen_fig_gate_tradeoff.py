# -*- coding: utf-8 -*-
"""图3 fig_gate_tradeoff（§5.2）：P_ret 与 P_gen 的安全—效用权衡（2 面板）。

左面板「块级检出特征」：4 组合的 TPR 与干净块误报率 FPR 在 P_ret/P_gen 下的对比 ——
  P_gen 通常同时给出更高 TPR 与更高 FPR。
右面板「答案级安全—效用」：ASR-Drop 与 Utility Drop 在 P_ret/P_gen 下收敛（同高）——
  属实现条件下收敛/功效不足，不判某位置更优。
编码约定：浅灰=P_ret，深灰=P_gen（位置）；实心=前一指标，斜纹=后一指标（指标）。
数据来源 figures/analysis_2_gate_position_results.json（真实冻结实验，禁止改数）。
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_os.path.dirname(_HERE), _HERE):  # 工作区根（含 _utils）+ figures（含 _figbase）
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import numpy as np
import matplotlib.pyplot as plt
from _figbase import init, load, save_all, PALETTE, auto_legend, dynamic_limits, declutter_axes, set_paper_placement

init()

data = load('analysis_2_gate_position_results.json')
rows = {r['combo']: r for r in data['pret_vs_pgen']}
ORDER = ['nq/bm25', 'nq/bge_dense', 'hotpotqa/bm25', 'hotpotqa/bge_dense']
CATS = ['NQ\nBM25', 'NQ\nBGE密集', 'HotpotQA\nBM25', 'HotpotQA\nBGE密集']

x = np.arange(len(ORDER))
bw = 0.20
# 四柱相对组中心的偏移
off = [-1.5 * bw, -0.5 * bw, 0.5 * bw, 1.5 * bw]
# (shade_idx, hatch, label)：浅灰 PALETTE[3]=P_ret，深灰 PALETTE[0]=P_gen
LIGHT, DARK = PALETTE[3], PALETTE[0]


def draw_group(ax, series):
    """series: list of dict(vals, color, hatch, label)."""
    for j, s in enumerate(series):
        bars = ax.bar(x + off[j], s['vals'], bw, color=s['color'],
                      label=s['label'])
        for p in bars:
            p.set_edgecolor('black'); p.set_linewidth(0.8)
            if s['hatch']:
                p.set_hatch(s['hatch'])


fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))

# ---- 左面板：TPR / FPR ----
axL = axes[0]
tpr_pret = [rows[k]['tpr_Pret'] * 100 for k in ORDER]
tpr_pgen = [rows[k]['tpr_Pgen'] * 100 for k in ORDER]
fpr_pret = [rows[k]['fpr_Pret'] * 100 for k in ORDER]
fpr_pgen = [rows[k]['fpr_Pgen'] * 100 for k in ORDER]
draw_group(axL, [
    {'vals': tpr_pret, 'color': LIGHT, 'hatch': '', 'label': 'TPR·P_ret'},
    {'vals': tpr_pgen, 'color': DARK, 'hatch': '', 'label': 'TPR·P_gen'},
    {'vals': fpr_pret, 'color': LIGHT, 'hatch': '////', 'label': '干净FPR·P_ret'},
    {'vals': fpr_pgen, 'color': DARK, 'hatch': '////', 'label': '干净FPR·P_gen'},
])
axL.set_title('(a) 块级检出特征', loc='left', fontsize=11)
axL.set_ylabel('比率 (%)')
axL.set_xlabel('数据集—检索器组合')
axL.set_xticks(x); axL.set_xticklabels(CATS)
dynamic_limits(axL, y=[np.array(tpr_pgen), np.array([0.0])], pad=0.10, include_zero=True)
declutter_axes(axL, grid=True, grid_axis='y')
auto_legend(axL, ncol=2)

# ---- 右面板：ASR-Drop / Utility Drop（收敛）----
axR = axes[1]
asr_pret = [rows[k]['asr_drop_Pret'] * 100 for k in ORDER]
asr_pgen = [rows[k]['asr_drop_Pgen'] * 100 for k in ORDER]
uti_pret = [rows[k]['util_Pret'] * 100 for k in ORDER]
uti_pgen = [rows[k]['util_Pgen'] * 100 for k in ORDER]
draw_group(axR, [
    {'vals': asr_pret, 'color': LIGHT, 'hatch': '', 'label': 'ASR-Drop·P_ret'},
    {'vals': asr_pgen, 'color': DARK, 'hatch': '', 'label': 'ASR-Drop·P_gen'},
    {'vals': uti_pret, 'color': LIGHT, 'hatch': '....', 'label': 'Utility Drop·P_ret'},
    {'vals': uti_pgen, 'color': DARK, 'hatch': '....', 'label': 'Utility Drop·P_gen'},
])
axR.set_title('(b) 答案级安全—效用（收敛）', loc='left', fontsize=11)
axR.set_ylabel('下降幅度 (%)')
axR.set_xlabel('数据集—检索器组合')
axR.set_xticks(x); axR.set_xticklabels(CATS)
dynamic_limits(axR, y=[np.array(asr_pgen), np.array([0.0])], pad=0.10, include_zero=True)
declutter_axes(axR, grid=True, grid_axis='y')
auto_legend(axR, ncol=2)

save_all(fig, 'fig_gate_tradeoff')
