from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_rel, wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


ANCHOR_MODEL = "HAD-CNN"
BASELINES = ["1D-CNN", "ResNet1D", "Concat-CNN"]
METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]


def holm_correction(p_values: List[float]) -> List[float]:
    """
    Holm-Bonferroni correction
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    prev = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = (m - rank + 1) * p
        adj = max(adj, prev)
        adj = min(adj, 1.0)
        adjusted[idx] = adj
        prev = adj

    return adjusted


def cohen_dz(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    diff_std = diff.std(ddof=1)
    if diff_std == 0:
        return 0.0
    return float(diff.mean() / diff_std)


def wins_ties_losses(diff: np.ndarray, eps: float = 1e-12) -> Tuple[int, int, int]:
    wins = int(np.sum(diff > eps))
    ties = int(np.sum(np.abs(diff) <= eps))
    losses = int(np.sum(diff < -eps))
    return wins, ties, losses


def paired_tests(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    result = {
        "t_pvalue": np.nan,
        "wilcoxon_pvalue": np.nan,
    }

    if not SCIPY_AVAILABLE:
        return result

    try:
        _, t_p = ttest_rel(x, y)
        result["t_pvalue"] = float(t_p)
    except Exception:
        pass

    try:
        diff = x - y
        if np.allclose(diff, 0):
            result["wilcoxon_pvalue"] = 1.0
        else:
            _, w_p = wilcoxon(x, y, zero_method="wilcox", correction=False, alternative="two-sided")
            result["wilcoxon_pvalue"] = float(w_p)
    except Exception:
        pass

    return result


def summarize_mean_sd(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("model", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_precision_mean=("macro_precision", "mean"),
            macro_precision_std=("macro_precision", "std"),
            macro_recall_mean=("macro_recall", "mean"),
            macro_recall_std=("macro_recall", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
    )

    preferred_order = ["1D-CNN", "ResNet1D", "Concat-CNN", "HAD-CNN"]
    summary["__order"] = summary["model"].apply(
        lambda x: preferred_order.index(x) if x in preferred_order else 999
    )
    summary = summary.sort_values(["__order", "model"]).drop(columns="__order").reset_index(drop=True)
    return summary


def build_pairwise_results(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    models_present = set(df["model"].unique().tolist())
    if ANCHOR_MODEL not in models_present:
        raise ValueError(f"输入 CSV 中未找到 anchor model: {ANCHOR_MODEL}")

    for metric in METRICS:
        pvals_for_holm: List[float] = []
        temp_rows: List[Dict] = []

        for baseline in BASELINES:
            if baseline not in models_present:
                continue

            anchor_df = df[df["model"] == ANCHOR_MODEL][["seed", metric]].rename(
                columns={metric: "anchor_value"}
            )
            base_df = df[df["model"] == baseline][["seed", metric]].rename(
                columns={metric: "baseline_value"}
            )

            merged = anchor_df.merge(base_df, on="seed", how="inner").sort_values("seed")
            if merged.empty:
                continue

            x = merged["anchor_value"].to_numpy(dtype=float)
            y = merged["baseline_value"].to_numpy(dtype=float)
            diff = x - y

            tests = paired_tests(x, y)
            dz = cohen_dz(x, y)
            wins, ties, losses = wins_ties_losses(diff)

            row = {
                "metric": metric,
                "anchor_model": ANCHOR_MODEL,
                "baseline_model": baseline,
                "n_pairs": len(merged),
                "anchor_mean": float(np.mean(x)),
                "anchor_std": float(np.std(x, ddof=1)),
                "baseline_mean": float(np.mean(y)),
                "baseline_std": float(np.std(y, ddof=1)),
                "paired_diff_mean": float(np.mean(diff)),
                "paired_diff_std": float(np.std(diff, ddof=1)),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "t_pvalue": tests["t_pvalue"],
                "wilcoxon_pvalue": tests["wilcoxon_pvalue"],
                "cohen_dz": dz,
            }
            temp_rows.append(row)
            pvals_for_holm.append(
                tests["wilcoxon_pvalue"] if not np.isnan(tests["wilcoxon_pvalue"]) else 1.0
            )

        if temp_rows:
            adjusted = holm_correction(pvals_for_holm)
            for row, adj_p in zip(temp_rows, adjusted):
                row["holm_adjusted_wilcoxon_pvalue"] = adj_p
                rows.append(row)

    out = pd.DataFrame(rows)
    metric_order = {m: i for i, m in enumerate(METRICS)}
    baseline_order = {m: i for i, m in enumerate(BASELINES)}
    out["__m"] = out["metric"].map(metric_order)
    out["__b"] = out["baseline_model"].map(baseline_order)
    out = out.sort_values(["__m", "__b"]).drop(columns=["__m", "__b"]).reset_index(drop=True)
    return out


def plot_macro_f1_seed_lines(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))

    for model, sub in df.groupby("model"):
        sub = sub.sort_values("seed")
        plt.plot(sub["seed"], sub["macro_f1"], marker="o", label=model)

    plt.xlabel("Random seed")
    plt.ylabel("Macro F1")
    plt.title("Experiment 5.12: Macro F1 across seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pairwise_macro_f1_differences(pairwise_df: pd.DataFrame, output_path: Path) -> None:
    sub = pairwise_df[pairwise_df["metric"] == "macro_f1"].copy()
    if sub.empty:
        return

    labels = sub["baseline_model"].tolist()
    means = sub["paired_diff_mean"].tolist()
    stds = sub["paired_diff_std"].tolist()

    plt.figure(figsize=(7, 4.8))
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.axhline(0.0, linestyle="--")
    plt.ylabel("Paired difference in Macro F1\n(HAD-CNN - baseline)")
    plt.title("Experiment 5.12: Pairwise Macro F1 differences")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 5.12 statistical significance analysis."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/exp512_statistics/exp512_per_seed_results.csv",
        help="由 exp512_collect_per_seed_results.py 生成的 per-seed CSV 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/exp512_statistics",
        help="输出目录，建议保持默认到 THZ_Humidity/results/exp512_statistics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required_cols = {"model", "seed", "accuracy", "macro_precision", "macro_recall", "macro_f1"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"输入 CSV 缺少必要列: {sorted(missing)}")

    df["model"] = df["model"].astype(str)
    df["seed"] = df["seed"].astype(int)

    summary_df = summarize_mean_sd(df)
    pairwise_df = build_pairwise_results(df)

    summary_csv = output_dir / "exp512_summary_mean_sd.csv"
    pairwise_csv = output_dir / "exp512_pairwise_significance_results.csv"
    paper_table_csv = output_dir / "table_5_12_statistical_significance.csv"
    fig1 = output_dir / "figure_5_12_macro_f1_seed_lines.png"
    fig2 = output_dir / "figure_5_12_pairwise_macro_f1_differences.png"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    pairwise_df.to_csv(pairwise_csv, index=False, encoding="utf-8-sig")

    table_5_12 = pairwise_df[
        [
            "metric",
            "anchor_model",
            "baseline_model",
            "n_pairs",
            "paired_diff_mean",
            "paired_diff_std",
            "wins",
            "ties",
            "losses",
            "wilcoxon_pvalue",
            "holm_adjusted_wilcoxon_pvalue",
            "cohen_dz",
        ]
    ].copy()
    table_5_12.to_csv(paper_table_csv, index=False, encoding="utf-8-sig")

    plot_macro_f1_seed_lines(df, fig1)
    plot_pairwise_macro_f1_differences(pairwise_df, fig2)

    print("=" * 80)
    print("Experiment 5.12 statistical significance analysis finished.")
    print(f"Input      : {input_csv}")
    print(f"Saved      : {summary_csv}")
    print(f"Saved      : {pairwise_csv}")
    print(f"Saved      : {paper_table_csv}")
    print(f"Saved      : {fig1}")
    print(f"Saved      : {fig2}")
    print("=" * 80)

    print("\n[Summary mean ± SD]")
    print(summary_df.to_string(index=False))

    print("\n[Pairwise significance results]")
    print(pairwise_df.to_string(index=False))


if __name__ == "__main__":
    main()