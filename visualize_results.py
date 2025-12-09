#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_results(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_tables(summary: Dict[str, Any]):
    en = summary["en"]
    ur = summary["ur"]
    delta = summary["en_vs_ur"]

    table1_rows = []
    for lang_name, lang_data in (("English", en), ("Urdu", ur)):
        for method, vals in lang_data.items():
            table1_rows.append({
                "Language": lang_name,
                "Method": method.capitalize(),
                "Bias": vals["mean_bias"],
                "Utility": vals["mean_utility"],
                "Composite Score": vals["mean_score"],
            })
    table1 = pd.DataFrame(table1_rows)

    table2_rows = []
    for method, dvals in delta.items():
        table2_rows.append({
            "Method": method.capitalize(),
            "Bias Δ (EN–UR)": dvals["bias_en_minus_ur"],
            "Utility Δ (EN–UR)": dvals["utility_en_minus_ur"],
            "Score Δ (EN–UR)": dvals["score_en_minus_ur"]
        })
    table2 = pd.DataFrame(table2_rows)

    table3_rows = []
    for lang_name, lang_data in (("English", en), ("Urdu", ur)):
        base = lang_data["baseline"]
        sel = lang_data["select"]
        seq = lang_data["sequential"]

        table3_rows.append({
            "Language": lang_name,
            "From": "Baseline → Select",
            "Bias Δ": sel["mean_bias"] - base["mean_bias"],
            "Utility Δ": sel["mean_utility"] - base["mean_utility"],
            "Composite Δ": sel["mean_score"] - base["mean_score"]
        })
        table3_rows.append({
            "Language": lang_name,
            "From": "Select → Sequential",
            "Bias Δ": seq["mean_bias"] - sel["mean_bias"],
            "Utility Δ": seq["mean_utility"] - sel["mean_utility"],
            "Composite Δ": seq["mean_score"] - sel["mean_score"]
        })

    table3 = pd.DataFrame(table3_rows)

    return table1, table2, table3


def plot_metric_bar(table: pd.DataFrame, metric: str, output_dir: Path):
    # Bar chart comparing EN vs UR per method for a given metric.
    
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=table,
        x="Method",
        y=metric,
        hue="Language",
        palette="Set2"
    )
    plt.title(f"{metric} Across Methods (English vs Urdu)")
    plt.ylabel(metric)
    plt.xlabel("Method")
    plt.tight_layout()
    path = output_dir / f"bar_{metric.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved] {path}")


def plot_delta_heatmap(table2: pd.DataFrame, output_dir: Path):
    
    # Heatmap of EN–UR deltas for fairness, utility, score.
    
    plt.figure(figsize=(6, 4))
    df = table2.set_index("Method")
    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0
    )
    plt.title("English–Urdu Differences (Δ EN–UR)")
    plt.tight_layout()
    path = output_dir / "heatmap_en_vs_ur.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved] {path}")


def plot_improvement_lines(table3: pd.DataFrame, output_dir: Path):
    # Lineplot showing baseline→select→sequential trajectory.
    
    melted = table3.melt(
        id_vars=["Language", "From"],
        value_vars=["Bias Δ", "Utility Δ", "Composite Δ"],
        var_name="Metric",
        value_name="Delta"
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=melted,
        x="From",
        y="Delta",
        hue="Metric",
        palette="Set1"
    )
    plt.title("Improvement Across Debiasing Stages")
    plt.xticks(rotation=15)
    plt.tight_layout()
    path = output_dir / "improvement_stages.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved] {path}")


def generate_all_plots_and_tables(result_json: str, output_dir: str = "results/plots"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(result_json)
    summary = results["summary"]

    print("[INFO] Building tables…")
    table1, table2, table3 = build_tables(summary)

    # Save tables
    table1.to_csv(output_dir / "table1_mean_metrics.csv", index=False)
    table2.to_csv(output_dir / "table2_language_deltas.csv", index=False)
    table3.to_csv(output_dir / "table3_stage_improvements.csv", index=False)

    print("[INFO] Saved table1, table2, table3")

    # Build plots
    print("[INFO] Generating plots…")
    plot_metric_bar(table1, "Bias", output_dir)
    plot_metric_bar(table1, "Utility", output_dir)
    plot_metric_bar(table1, "Composite Score", output_dir)

    plot_delta_heatmap(table2, output_dir)
    plot_improvement_lines(table3, output_dir)

    print(f"[DONE] All tables and plots generated in: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to final run_*.json")
    parser.add_argument("--out", type=str, default="results/plots")

    args = parser.parse_args()
    generate_all_plots_and_tables(args.results, args.out)