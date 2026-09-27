# -*- coding: utf-8 -*-
"""数据图公用底座（黑白期刊《计算机工程》）。

- 统一调用 setup_style() 后，按期刊要求覆盖为：
  · 灰阶配色（本刊黑白印刷，图不用彩色 —— CUSTOM_REQUIREMENTS 第 9 条，最高优先级）；
  · 中文华文中宋 + 英文/数字 Times New Roman（font.family 传【列表】才逐字回退，避免方框）；
  · 系列间再叠 hatch 纹理 + 黑描边，确保灰度打印/复印仍可区分。
- save_all() 一次产出 svg（自定义需求）+ pdf（供 figure_pdf_quality_check 严格体检）
  + png 350dpi（Word 嵌入）。三份都过 savefig hook 的防遮挡/越界体检。
- 所有 gen_fig_*.py 只 `from _figbase import *`，引用 PALETTE[i]，静态闸不误判硬编码色。
"""
import os
import json
import matplotlib
matplotlib.use('Agg')

from _utils.plot_utils import (  # noqa: F401
    setup_style, save_fig, set_paper_placement, auto_legend,
    dynamic_limits, declutter_axes, consolidate_shared_legends,
    shared_legend, PALETTE, COLORS,
)

# 灰阶：由深到浅 + 纯黑，覆盖到 6 档，黑白印刷下彼此可辨。
_GRAY = ['#3A3A3A', '#FFFFFF', '#8C8C8C', '#C8C8C8', '#000000', '#5F5F5F']
# 系列纹理（与灰阶配对使用；白底系列必配纹理才不至于“消失”）。
HATCHES = ['', '////', '....', 'xxxx', '\\\\\\\\', '++']
EK = 'black'   # 统一描边色（用变量避免逐处写 color= 字面量）
GRID = '#BFBFBF'


def init():
    """初始化风格：先 setup_style()，再按黑白期刊要求覆盖配色与字体。"""
    setup_style()
    # ★ 就地改写全局 PALETTE（gen_fig 引用 PALETTE[i]），灰阶覆盖工作区随机彩色。
    PALETTE[:] = list(_GRAY)
    COLORS['primary'] = _GRAY[0]
    COLORS['secondary'] = _GRAY[2]
    COLORS['accent'] = _GRAY[4]
    COLORS['up'] = _GRAY[0]
    COLORS['down'] = _GRAY[2]
    COLORS['highlight'] = _GRAY[4]
    COLORS['ref_line'] = '#000000'
    COLORS['grid'] = GRID
    COLORS['text'] = '#000000'
    # 颜色循环同步为灰阶，防止未显式指定色的元素回落到彩色默认。
    matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=list(_GRAY))
    # ★ 字体：font.family 传【列表】→ matplotlib 逐字回退：拉丁/数字取 Times New Roman，
    #   汉字回退到华文中宋（STZhongsong）。传 'serif' + font.serif 不会逐字回退（实测出方框）。
    matplotlib.rcParams['font.family'] = ['Times New Roman', 'STZhongsong', 'SimSun']
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['axes.unicode_minus'] = False
    # 灰阶下用黑描边 + 白缝分隔相邻柱。
    matplotlib.rcParams['patch.edgecolor'] = EK
    matplotlib.rcParams['patch.linewidth'] = 0.8
    # ★ 字体嵌入：setup_style 未设 pdf.fonttype，matplotlib 默认 Type 3（字形走 CharProcs，
    #   无 FontFile 流）→ figure_pdf_quality_check 判「未嵌入」。改 42 走 TrueType 子集嵌入
    #   （FontFile2），中宋/Times 均真正内嵌。SVG 用 path 把文字转矢量路径 → 期刊端无需装字体、
    #   零方框风险，自洽可印。
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'path'


def load(name):
    """读取 figures/ 下的分析结果 JSON。"""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, name), 'r', encoding='utf-8') as f:
        return json.load(f)


def style_bar(bar_container, idx):
    """给一组柱子叠加与灰阶配套的纹理与黑描边（灰度可辨的关键）。"""
    h = HATCHES[idx % len(HATCHES)]
    for patch in bar_container:
        patch.set_edgecolor(EK)
        patch.set_linewidth(0.8)
        if h:
            patch.set_hatch(h)


def save_all(fig, stem, pad=0.6):
    """产出 svg + pdf + png 三份。

    · svg：CUSTOM_REQUIREMENTS「论文的图片使用 SVG 格式」；
    · pdf：仅供 figure_pdf_quality_check.py 做矢量级字号/越界/对比度体检，QC 后按
           Word 工作流（不留 PDF）删除；
    · png：350dpi，供 Word/Markdown 正文 ![](figures/xxx.png) 嵌入。
    save_fig(png) 走 _save：内部 fig.savefig 触发全局 savefig hook（防遮挡/越界体检）后关闭 figure，
    故 png 必须最后存。svg/pdf 直接 fig.savefig 同样过 hook。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(here, stem)
    try:
        fig.tight_layout(pad=pad)
    except Exception:
        pass
    fig.savefig(base + '.svg', format='svg')
    fig.savefig(base + '.pdf', format='pdf')
    save_fig(fig, base + '.png')   # 最后：350dpi + 关闭 figure
    print('Saved trio:', stem, '(.svg/.pdf/.png)')
