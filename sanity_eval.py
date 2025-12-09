#!/usr/bin/env python3
import argparse

from src.eval import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Bias-tts PRM experiment (EN + UR, single-word).")
    parser.add_argument("--config", type=str, default="config/config.json")
    parser.add_argument("--prompts", type=str, default="data/prompts.jsonl")
    args = parser.parse_args()

    run_experiment(config_path=args.config, prompts_path=args.prompts)


if __name__ == "__main__":
    main()