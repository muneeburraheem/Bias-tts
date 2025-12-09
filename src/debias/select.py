from typing import Dict, Any, List
from src.llm import LLMGenerator
from src.prm import GPT4oMiniJudge

def prm_select(
    template: str,
    lang: str,
    llm: LLMGenerator,
    prm: GPT4oMiniJudge,
    n: int,
    alpha: float,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "chosen_word": str,
        "candidates": [ {word, bias, utility, score}, ... ]  # sorted in descending order by score
      }
    """
    cand_words: List[str] = llm.generate_words(template, lang=lang, n=n, mode="fill")
    if not cand_words:
        return {"chosen_word": "", "candidates": []}

    scored = []
    for w in cand_words:
        s = prm.score_word(template, w, lang)
        bias, util = s["bias"], s["utility"]
        score = (1 - alpha) * bias + alpha * util
        scored.append({"word": w, "bias": bias, "utility": util, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"chosen_word": scored[0]["word"], "candidates": scored}