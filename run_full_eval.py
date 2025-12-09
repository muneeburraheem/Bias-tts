#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Any, List
from src.utils.io import load_json, load_jsonl, save_json, timestamped_results_path
from src.llm import LLMGenerator
from src.prm import GPT4oMiniJudge
from src.debias import prm_select, prm_sequential
from src.eval.metrics import summarize

def run_full_eval(
    config_path: str,
    prompts_path: str,
    override_alpha: float | None = None,
    override_bestof: int | None = None,
    override_seqsteps: int | None = None,
):

    cfg = load_json(config_path)
    prompts = load_jsonl(prompts_path)

    cand_cfg = cfg["candidate_model"]
    local_cfg = cfg.get("local_model", {})
    provider = "openai" if cand_cfg.get("provider", "openai") == "openai" else "local"

    llm = LLMGenerator(
        provider=provider,
        model_name=cand_cfg["model_name"],
        max_tokens=cand_cfg.get("max_tokens", 3),
        temperature=cand_cfg.get("temperature", 0.9),
        top_p=cand_cfg.get("top_p", 1.0),
        presence_penalty=cand_cfg.get("presence_penalty", 0.8),
        frequency_penalty=cand_cfg.get("frequency_penalty", 0.4),
        local_device=local_cfg.get("device", "auto"),
    )

    prm_cfg = cfg["prm"]
    prm = GPT4oMiniJudge(model_name=prm_cfg["model_name"])

    alpha = override_alpha if override_alpha is not None else float(prm_cfg.get("alpha", 0.5))
    bias_threshold = float(prm_cfg.get("bias_threshold", 0.8))
    best_of_n = override_bestof if override_bestof is not None else int(cfg["debias"].get("best_of_n", 8))
    seq_max_steps = override_seqsteps if override_seqsteps is not None else int(cfg["debias"].get("seq_max_steps", 5))

    per_item: List[Dict[str, Any]] = []

    for row in prompts:
        pid = row["id"]
        lang = row["lang"]
        template = row["template"]

        print(f"\n[ID={pid} | lang={lang}] {template}")

        # Baseline
        base_words = llm.generate_words(template, lang=lang, n=1, mode="fill")
        base_word = base_words[0] if base_words else ""
        base_stats = prm.score_word(template, base_word, lang) if base_word else {"bias": 0.5, "utility": 0.5}
        base_score = (1 - alpha) * base_stats["bias"] + alpha * base_stats["utility"]

        print(f"  Baseline → {base_word} | bias={base_stats['bias']:.3f}, util={base_stats['utility']:.3f}")

        # PRM Select
        sel_res = prm_select(template, lang, llm, prm, n=best_of_n, alpha=alpha)
        if sel_res["candidates"]:
            top = sel_res["candidates"][0]
            sel_bias = top["bias"]
            sel_util = top["utility"]
            sel_score = top["score"]
            sel_word = top["word"]
        else:
            sel_word = ""
            sel_bias = sel_util = sel_score = 0.0

        print(f"  PRM-Select → {sel_word} | bias={sel_bias:.3f}, util={sel_util:.3f}")

        # PRM Sequential
        seq_res = prm_sequential(
            template, lang, llm, prm, alpha=alpha,
            max_steps=seq_max_steps, bias_threshold=bias_threshold
        )
        last = seq_res["trajectory"][-1]
        seq_word = seq_res["chosen_word"]
        seq_bias = last["bias"]
        seq_util = last["utility"]
        seq_score = last["score"]

        print(f"  PRM-Sequential → {seq_word} | bias={seq_bias:.3f}, util={seq_util:.3f}")

        per_item.append({
            "id": pid,
            "lang": lang,
            "template": template,
            "baseline": {
                "word": base_word,
                "bias": base_stats["bias"],
                "utility": base_stats["utility"],
                "score": base_score,
            },
            "select": {
                "word": sel_word,
                "bias": sel_bias,
                "utility": sel_util,
                "score": sel_score,
                "candidates": sel_res["candidates"],
            },
            "sequential": {
                "word": seq_word,
                "bias": seq_bias,
                "utility": seq_util,
                "score": seq_score,
                "trajectory": seq_res["trajectory"],
            },
        })

    # Summaries
    summary = summarize(per_item)

    results = {
        "config": cfg,
        "alpha_used": alpha,
        "best_of_n_used": best_of_n,
        "seq_steps_used": seq_max_steps,
        "prompt_file": prompts_path,
        "items": per_item,
        "summary": summary,
    }

    out_dir = cfg["run"]["results_dir"]
    prefix = cfg["run"].get("run_name_prefix", "run")

    out_path = timestamped_results_path(out_dir, prefix=prefix)
    save_json(out_path, results)

    print(f"\n[✓ DONE] Saved exhaustive results → {out_path}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config/config.json")
    parser.add_argument("--prompts", type=str, default="data/prompts_200.jsonl")

    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--bestof", type=int, default=None)
    parser.add_argument("--seqsteps", type=int, default=None)

    args = parser.parse_args()

    run_full_eval(
        config_path=args.config,
        prompts_path=args.prompts,
        override_alpha=args.alpha,
        override_bestof=args.bestof,
        override_seqsteps=args.seqsteps,
    )