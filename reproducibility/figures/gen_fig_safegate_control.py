# -*- coding: utf-8 -*-
"""图4 fig_safegate_control（§5.4+§5.5）：SafeGate 探针的两个对照（2 面板）。

左面板「多信号 vs 匹配随机阻断」（§5.4）：4 组合，每组 3 柱 ——
  多信号（位置无关，P_ret/P_gen 收敛同值）、随机@P_ret、随机@P_gen；
  多信号 ASR-Drop 在 24/24 可比行不低于匹配随机（但其 Utility Drop 也更高，见正文）。
右面板「多信号 vs 单信号（查询重叠）」（§5.5，负结果）：4 组合，每组 2 柱 ——
  多信号仅在 4/24 可比行不低于单信号，未形成一致优势（负结果，必须如实保留）。
编码约定：深灰=多信号；随机/单信号用浅灰+斜纹、中灰+反斜纹/叉纹区分（黑白可辨）。
数据来源 figures/analysis_3_control_baselines_results.json（真实冻结实验，禁止改数）。
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

data = load('analysis_3_control_baselines_results.json')
rnd = {(r['dataset'], r['retriever'], r['gate_position']): r
       for r in data['safegate_vs_random']['by_group']}
sgl = {(r['dataset'], r['retriever'], r['gate_position']): r
       for r in data['multi_vs_single']['by_group']}

ORDER = [('nq', 'bm25'), ('nq', 'bge_dense'), ('hotpotqa', 'bm25'), ('hotpotqa', 'bge_dense')]
CATS = ['NQ\nBM25', 'NQ\nBGE密集', 'HotpotQA\nBM25', 'HotpotQA\nBGE密集']
x = np.arange(len(ORDER))

# 灰阶配色：深灰=多信号（主角），其余对照用浅/中灰 + 纹理。
DARK, MID, LIGHT = PALETTE[0], PALETTE[2], PALETTE[3]


def draw_group(ax, series, bw, offs):
    for j, s in enumerate(series):
        bars = ax.bar(x + offs[j], s['vals'], bw, color=s['color'], label=s['label'])
        for p in bars:
            p.set_edgecolor('black'); p.set_linewidth(0.8)
            if s['hatch']:
                p.set_hatch(s['hatch'])


fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))

# ---- 左面板（§5.4）：多信号 vs 匹配随机阻断（随机分 P_ret/P_gen）----
axL = axes[0]
multi = [rnd[(d, r, 'P_gen')]['multi_asr_drop_mean'] * 100 for d, r in ORDER]
rand_pret = [rnd[(d, r, 'P_ret')]['control_asr_drop_mean'] * 100 for d, r in ORDER]
rand_pgen = [rnd[(d, r, 'P_gen')]['control_asr_drop_mean'] * 100 for d, r in ORDER]
bwA = 0.26
draw_group(axL, [
    {'vals': multi, 'color': DARK, 'hatch': '', 'label': '多信号'},
    {'vals': rand_pret, 'color': LIGHT, 'hatch': '////', 'label': '随机阻断·P_ret'},
    {'vals': rand_pgen, 'color': MID, 'hatch': '\\\\\\\\', 'label': '随机阻断·P_gen'},
], bwA, [-bwA, 0.0, bwA])
axL.set_title('(a) 多信号 vs 匹配随机阻断', loc='left', fontsize=11)
axL.set_ylabel('ASR-Drop (%)')
axL.set_xlabel('数据集—检索器组合')
axL.set_xticks(x); axL.set_xticklabels(CATS, fontsize=8.5)
dynamic_limits(axL, y=[np.array(multi), np.array([0.0])], pad=0.14, include_zero=True)
declutter_axes(axL, grid=True, grid_axis='y')
auto_legend(axL, ncol=1)

# ---- 右面板（§5.5，负结果）：多信号 vs 单信号（查询重叠）----
axR = axes[1]
multi2 = [sgl[(d, r, 'P_gen')]['multi_asr_drop_mean'] * 100 for d, r in ORDER]
single = [sgl[(d, r, 'P_gen')]['control_asr_drop_mean'] * 100 for d, r in ORDER]
bwB = 0.34
draw_group(axR, [
    {'vals': multi2, 'color': DARK, 'hatch': '', 'label': '多信号'},
    {'vals': single, 'color': MID, 'hatch': 'xxxx', 'label': '单信号（查询重叠）'},
], bwB, [-bwB / 2, bwB / 2])
axR.set_title('(b) 多信号 vs 单信号（负结果）', loc='left', fontsize=11)
axR.set_ylabel('ASR-Drop (%)')
axR.set_xlabel('数据集—检索器组合')
axR.set_xticks(x); axR.set_xticklabels(CATS, fontsize=8.5)
dynamic_limits(axR, y=[np.array(multi2), np.array(single), np.array([0.0])], pad=0.14, include_zero=True)
declutter_axes(axR, grid=True, grid_axis='y')
auto_legend(axR, ncol=1)

save_all(fig, 'fig_safegate_control', pad=1.2)
