from collections import defaultdict
from typing import Dict, Any, List


def summarize(per_item: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate statistics per language and method,
    and compute EN vs UR deltas.
    """
    agg = defaultdict(lambda: defaultdict(lambda: {"bias": [], "utility": [], "score": []}))

    for row in per_item:
        lang = row["lang"]
        for method in ("baseline", "select", "sequential"):
            m = row[method]
            if not m or not m.get("word"):
                continue
            agg[lang][method]["bias"].append(m["bias"])
            agg[lang][method]["utility"].append(m["utility"])
            agg[lang][method]["score"].append(m["score"])

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    out: Dict[str, Any] = {}
    for lang, methods in agg.items():
        out[lang] = {}
        for method, vals in methods.items():
            out[lang][method] = {
                "n": len(vals["bias"]),
                "mean_bias": mean(vals["bias"]),
                "mean_utility": mean(vals["utility"]),
                "mean_score": mean(vals["score"]),
            }

    if "en" in out and "ur" in out:
        delta = {}
        for method in out["en"]:
            if method not in out["ur"]:
                continue
            delta[method] = {
                "bias_en_minus_ur": out["en"][method]["mean_bias"] - out["ur"][method]["mean_bias"],
                "utility_en_minus_ur": out["en"][method]["mean_utility"] - out["ur"][method]["mean_utility"],
                "score_en_minus_ur": out["en"][method]["mean_score"] - out["ur"][method]["mean_score"],
            }
        out["en_vs_ur"] = delta

    return out