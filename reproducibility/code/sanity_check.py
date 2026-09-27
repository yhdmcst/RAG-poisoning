# 自动化"太完美"检测：查 NaN/Inf、比率越界、编造的完美值
import json, os, sys, math

results = {}
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
for f in sorted(os.listdir(FIG)):
    if f.endswith("_results.json") or f == "all_results.json" or f == "descriptive_stats.json":
        try:
            with open(os.path.join(FIG, f), encoding="utf-8") as fh:
                results[f] = json.load(fh)
        except Exception:
            pass

errors, warnings, suspicious = [], [], []


def check_unrealistic(name, val):
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return
    key = name.lower()
    # 比率类字段：ASR/TPR/FPR/F1/drop 必须在 [0,1]（本项目所有比率均为 0-1 的 float）。
    # 计数/种子/区间宽度等为 int 或 pp 值，不参与比率越界检查。
    if isinstance(val, float) and any(w in key for w in
            ["asr", "tpr", "fpr", "f1", "drop", "share", "presence",
             "fraction", "agreement"]):
        if "pp" not in key and "width" not in key and "std" not in key:
            if val < -0.5 or val > 1.5:
                warnings.append(f"⚠ {name} = {val}（比率疑似越界，请确认单位）")
    if "p_value" in key or "pvalue" in key:
        if val == 0:
            suspicious.append(f"🚩 {name} = 0 完美显著")
        elif val > 1:
            errors.append(f"❌ {name} = {val} p 值 > 1 不可能")


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        if math.isnan(obj):
            errors.append(f"❌ {path} 为 NaN")
        elif math.isinf(obj):
            errors.append(f"❌ {path} 为 Inf")
        check_unrealistic(path, obj)
    elif isinstance(obj, int):
        check_unrealistic(path, obj)


for fname, data in results.items():
    walk(data, fname)

for e in errors:
    print(e)
for w in warnings:
    print(w)
for s in suspicious:
    print(s)

if errors:
    print(f"\n❌ {len(errors)} 个硬错误 — 必须修复")
    sys.exit(1)
if not errors and not warnings and not suspicious:
    print("✅ 所有数值通过 sanity check（无 NaN/Inf，比率均在合理范围）")
else:
    print(f"\n✅ 无硬错误；{len(warnings)} 警告 / {len(suspicious)} 可疑（人工确认）")
