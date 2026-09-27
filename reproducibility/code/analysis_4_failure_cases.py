# §5.6 失败案例与 Clean F1 诊断（clean_f1_case_samples.csv, 80 例）
# 输出：错误类型分布 + 每类一条真实代表案例（含真实 query 文本）→ 失败案例表
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from collections import Counter, OrderedDict
import pandas as pd
import utils as u

# 错误类型 → 中文标签 + 写作口径
TYPE_LABEL = OrderedDict([
    ("evaluation_normalization_issue", ("评测/归一化口径", "答案语义正确但归一化 F1 未满分，低估效用")),
    ("retrieval_miss", ("检索未命中", "top-50 无 qrel 证据，非门控引起")),
    ("rerank_or_context_miss", ("重排/上下文丢失", "检索到证据但重排 top-5 未保留")),
    ("generation_error_or_truncation", ("生成错误/截断", "证据在上下文但生成未给出正确答案")),
    ("uncertain", ("不确定", "非终止式结尾/中等长度，难以归因")),
])


def run():
    print("[analysis_4] 失败案例与 Clean F1 诊断 ...")
    df = u.load("clean_f1_case_samples.csv")

    counts = Counter(df["error_type"])
    total = int(len(df))
    dist = [{"error_type": k, "label": TYPE_LABEL.get(k, (k, ""))[0],
             "n": int(counts[k]), "share": round(counts[k] / total, 4),
             "writing_note": TYPE_LABEL.get(k, (k, ""))[1]}
            for k in sorted(counts, key=lambda x: -counts[x])]

    # 每个错误类型取一条真实代表案例（截断长文本，保留真实 query/gold/gen）
    cases = []
    for et in TYPE_LABEL:
        sub = df[df["error_type"] == et]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        cases.append({
            "error_type": et,
            "label": TYPE_LABEL[et][0],
            "dataset": r["dataset"], "retriever": r["retriever"],
            "query_id": str(r["query_id"]),
            "question": str(r["question"])[:90],
            "gold_answer": str(r["gold_answer"])[:60],
            "generated_answer": str(r["generated_answer"])[:80],
            "normalized_f1": float(r["normalized_f1"]),
            "error_reason": str(r["error_reason"])[:80],
        })

    # 关键诊断结论：主因非实现 bug，而是评测/归一化 + 检索/重排未命中 + 生成截断
    impl_bug = counts.get("implementation_bug", 0)
    out = {
        "section": "5.6",
        "table": "失败案例表（随文）",
        "total_cases": total,
        "error_type_distribution": dist,
        "representative_cases": cases,
        "diagnosis": {
            "implementation_bug_cases": int(impl_bug),
            "dominant_cause": dist[0]["error_type"] if dist else None,
            "dominant_share": dist[0]["share"] if dist else None,
            "finding": ("80 例 Clean F1 诊断未发现实现 bug；主因为评测/归一化口径"
                        f"（{counts.get('evaluation_normalization_issue',0)} 例）、检索/重排未命中、生成截断。"
                        "支撑'Clean F1 绝对值偏低仅作协议内相对效用比较'的限制。"),
        },
    }
    u.save_json("analysis_4_failure_cases_results.json", out)

    assert impl_bug == 0, "诊断不应出现 implementation_bug 案例"
    assert sum(d["n"] for d in dist) == total
    print(f"  {total} 例；主因={dist[0]['error_type']}({dist[0]['n']}); 实现 bug={impl_bug}")
    return out


if __name__ == "__main__":
    run()
