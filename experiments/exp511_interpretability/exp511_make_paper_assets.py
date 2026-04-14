from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BAND_ORDER_DEFAULT = ["water_1p4", "water_1p9", "mineral_diag", "other"]


def aggregate_band_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["model", "humidity_group", "band_name"], as_index=False)
        .agg(
            mean_mass_ratio=("mass_ratio", "mean"),
            std_mass_ratio=("mass_ratio", "std"),
            n_samples=("mass_ratio", "count"),
        )
    )
    grouped["std_mass_ratio"] = grouped["std_mass_ratio"].fillna(0.0)
    return grouped



def aggregate_curve_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["model", "class_name", "humidity_group", "wavelength_um"], as_index=False)
        .agg(
            mean_attr=("mean_attr", "mean"),
            std_attr=("mean_attr", "std"),
            n_samples=("n_samples", "sum"),
        )
    )
    grouped["std_attr"] = grouped["std_attr"].fillna(0.0)
    return grouped


def make_curve_figure(curve_df: pd.DataFrame, out_path: Path, models: List[str]) -> None:
    model_list = [m for m in models if m in curve_df["model"].unique()]
    if not model_list:
        return

    n = len(model_list)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, model_list):
        sub = curve_df[(curve_df["model"] == model) & (curve_df["class_name"] == "ALL")].copy()
        for humidity_group in ["low", "high"]:
            ss = sub[sub["humidity_group"] == humidity_group].sort_values("wavelength_um")
            if ss.empty:
                continue
            ax.plot(ss["wavelength_um"], ss["mean_attr"], label=humidity_group)
        for start, end, label in [(1.35, 1.45, "1.4 μm"), (1.85, 1.95, "1.9 μm"), (2.15, 2.35, "diag.")]:
            ax.axvspan(start, end, alpha=0.10)
        ax.set_title(f"{model}: attribution curves by humidity group")
        ax.set_ylabel("Normalized attribution")
        ax.legend(frameon=False)

    axes[-1].set_xlabel("Wavelength (μm)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def make_band_bar_figure(band_df: pd.DataFrame, out_path: Path, models: List[str]) -> None:
    model_list = [m for m in models if m in band_df["model"].unique()]
    if not model_list:
        return

    n = len(model_list)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 3.0 * n), sharex=True)
    if n == 1:
        axes = [axes]

    band_order = [b for b in BAND_ORDER_DEFAULT if b in band_df["band_name"].unique()]
    x = np.arange(len(band_order))
    width = 0.35

    for ax, model in zip(axes, model_list):
        sub = band_df[band_df["model"] == model].copy()
        low = sub[sub["humidity_group"] == "low"].set_index("band_name")
        high = sub[sub["humidity_group"] == "high"].set_index("band_name")

        low_means = [float(low.loc[b, "mean_mass_ratio"]) if b in low.index else 0.0 for b in band_order]
        low_stds = [float(low.loc[b, "std_mass_ratio"]) if b in low.index else 0.0 for b in band_order]
        high_means = [float(high.loc[b, "mean_mass_ratio"]) if b in high.index else 0.0 for b in band_order]
        high_stds = [float(high.loc[b, "std_mass_ratio"]) if b in high.index else 0.0 for b in band_order]

        ax.bar(x - width / 2, low_means, width=width, yerr=low_stds, capsize=4, label="low")
        ax.bar(x + width / 2, high_means, width=width, yerr=high_stds, capsize=4, label="high")
        ax.set_title(f"{model}: attribution mass ratio in interpretable bands")
        ax.set_ylabel("Mass ratio")
        ax.legend(frameon=False)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(band_order)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create paper-ready assets for Experiment 5.11"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="results/exp511_interpretability/raw",
        help="Directory containing raw outputs from exp511_interpretability.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/exp511_interpretability/paper_assets",
        help="Directory for paper-ready tables and figures",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["had", "concat"],
        help="Model keys to include in the figures (default: had concat)",
    )
    return parser



def main(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_df_raw = pd.read_csv(raw_dir / "exp511_attribution_curve_summary.csv")
    band_sample_df = pd.read_csv(raw_dir / "exp511_band_mass_sample_level.csv")
    peak_df = pd.read_csv(raw_dir / "exp511_top_peaks_summary.csv")

    curve_df = aggregate_curve_summary(curve_df_raw)
    band_df = aggregate_band_summary(band_sample_df)

    curve_csv = output_dir / "table_5_11_attribution_curve_summary.csv"
    band_csv = output_dir / "table_5_11_band_mass_summary.csv"
    peaks_csv = output_dir / "table_5_11_top_peaks_summary.csv"
    fig_curve = output_dir / "figure_5_11_attribution_curves.png"
    fig_bands = output_dir / "figure_5_11_band_mass_ratios.png"

    curve_df.to_csv(curve_csv, index=False, encoding="utf-8-sig")
    band_df.to_csv(band_csv, index=False, encoding="utf-8-sig")
    peak_df.to_csv(peaks_csv, index=False, encoding="utf-8-sig")

    make_curve_figure(curve_df, fig_curve, args.models)
    make_band_bar_figure(band_df, fig_bands, args.models)

    print("=" * 88)
    print("Experiment 5.11 paper assets generated.")
    print(f"Saved: {curve_csv}")
    print(f"Saved: {band_csv}")
    print(f"Saved: {peaks_csv}")
    print(f"Saved: {fig_curve}")
    print(f"Saved: {fig_bands}")
    print("=" * 88)


if __name__ == "__main__":
    main(build_argparser().parse_args())
