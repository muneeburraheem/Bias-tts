from typing import Dict, Any, List, Set
from src.llm import LLMGenerator
from src.prm import GPT4oMiniJudge

def prm_sequential(
    template: str,
    lang: str,
    llm: LLMGenerator,
    prm: GPT4oMiniJudge,
    alpha: float,
    max_steps: int,
    bias_threshold: float,
) -> Dict[str, Any]:
    """
    - Start from baseline word
    - At each step, get critique and propose a new word
    - Keep track of trajectory; stop if:
       bias >= threshold, or
       composite score stops improving
    """
    traj: List[Dict[str, Any]] = []

    baseline_list = llm.generate_words(template, lang=lang, n=1, mode="fill")
    cur = baseline_list[0] if baseline_list else ""
    scores = prm.score_word(template, cur, lang) if cur else {"bias": 0.5, "utility": 0.5}
    comp = (1 - alpha) * scores["bias"] + alpha * scores["utility"]
    traj.append({"word": cur, "bias": scores["bias"], "utility": scores["utility"], "score": comp, "critique": ""})
    best_idx = 0

    if scores["bias"] >= bias_threshold:
        return {"chosen_word": cur, "trajectory": traj}

    seen: Set[str] = {cur.lower()} if cur else set()

    for step in range(1, max_steps + 1):
        prev = traj[-1]
        critique = prm.critique(template, prev["word"], lang)
        new_words = llm.generate_words(
            template,
            lang=lang,
            n=1,
            mode="revise",
            prev_word=prev["word"],
            critique=critique,
            forbid=list(seen),
        )
        if not new_words:
            break
        w = new_words[0]
        seen.add(w.lower())

        s = prm.score_word(template, w, lang)
        bias, util = s["bias"], s["utility"]
        score = (1 - alpha) * bias + alpha * util
        traj.append({"word": w, "bias": bias, "utility": util, "score": score, "critique": critique})

        if score < traj[best_idx]["score"]:
            break
        best_idx = len(traj) - 1
        if bias >= bias_threshold:
            break

    chosen = traj[best_idx]["word"]
    return {"chosen_word": chosen, "trajectory": traj}