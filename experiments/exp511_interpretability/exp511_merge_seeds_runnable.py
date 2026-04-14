from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


SEED_PATTERN = re.compile(r"raw_seed(\d+)$")


REQUIRED_FILES = {
    "band": "exp511_band_mass_sample_level.csv",
    "curve": "exp511_attribution_curve_summary.csv",
    "peak": "exp511_top_peaks_summary.csv",
}


BAND_ORDER = ["water_1p4", "water_1p9", "mineral_diag", "other"]
HUMIDITY_ORDER = ["low", "high"]
MODEL_ORDER = ["had", "concat", "cnn", "resnet", "resnet1d"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Experiment 5.11 interpretability outputs across multiple seed folders."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory that contains raw_seedXX folders, e.g. .../results/exp511_interpretability",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for merged tables",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit seed list, e.g. --seeds 42 52 62 72 82",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=["had", "concat"],
        help="Models to keep in the main wide table. Default: had concat",
    )
    return parser.parse_args()


def discover_seed_dirs(root: Path, wanted_seeds: Optional[List[int]]) -> List[Tuple[int, Path]]:
    found: List[Tuple[int, Path]] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        m = SEED_PATTERN.match(p.name)
        if not m:
            continue
        seed = int(m.group(1))
        if wanted_seeds is not None and seed not in wanted_seeds:
            continue
        found.append((seed, p))

    if wanted_seeds is not None:
        found_seeds = {seed for seed, _ in found}
        missing = [s for s in wanted_seeds if s not in found_seeds]
        if missing:
            raise FileNotFoundError(f"Missing requested seed folders: {missing}")

    if not found:
        raise FileNotFoundError(f"No raw_seedXX folders found under: {root}")
    return found


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def concat_seed_tables(seed_dirs: List[Tuple[int, Path]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    band_parts = []
    curve_parts = []
    peak_parts = []

    for seed, seed_dir in seed_dirs:
        band_df = read_csv_checked(seed_dir / REQUIRED_FILES["band"]).copy()
        curve_df = read_csv_checked(seed_dir / REQUIRED_FILES["curve"]).copy()
        peak_df = read_csv_checked(seed_dir / REQUIRED_FILES["peak"]).copy()

        band_df["seed"] = seed
        curve_df["seed"] = seed
        peak_df["seed"] = seed

        band_parts.append(band_df)
        curve_parts.append(curve_df)
        peak_parts.append(peak_df)

    band_all = pd.concat(band_parts, ignore_index=True)
    curve_all = pd.concat(curve_parts, ignore_index=True)
    peak_all = pd.concat(peak_parts, ignore_index=True)
    return band_all, curve_all, peak_all


def make_band_seed_level_means(band_all: pd.DataFrame) -> pd.DataFrame:
    required = {"seed", "model", "humidity_group", "band_name", "mass_ratio"}
    missing = required - set(band_all.columns)
    if missing:
        raise ValueError(f"band_mass table missing columns: {sorted(missing)}")

    group_cols = ["seed", "model", "humidity_group", "band_name"]
    if "class_name" in band_all.columns:
        group_cols.insert(2, "class_name")

    band_seedmean = (
        band_all.groupby(group_cols, as_index=False)
        .agg(seed_mean_mass_ratio=("mass_ratio", "mean"), n_sample_rows=("mass_ratio", "count"))
    )
    return band_seedmean


def summarize_band_across_seeds(band_seedmean: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["model", "class_name", "humidity_group", "band_name"] if c in band_seedmean.columns]
    summary = (
        band_seedmean.groupby(group_cols, as_index=False)
        .agg(
            mean_mass_ratio=("seed_mean_mass_ratio", "mean"),
            std_mass_ratio=("seed_mean_mass_ratio", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    summary["std_mass_ratio"] = summary["std_mass_ratio"].fillna(0.0)
    summary["mean_pm_sd"] = summary.apply(
        lambda r: f"{r['mean_mass_ratio']:.4f} ± {r['std_mass_ratio']:.4f}", axis=1
    )
    return summary


def make_main_wide_table(df):
    df = df.copy()

    df["column_key"] = df["humidity_group"].astype(str) + "__" + df["band_name"].astype(str)
    df["mean_pm_sd"] = df.apply(
        lambda r: f"{r['mean_mass_ratio']:.4f} ± {r['std_mass_ratio']:.4f}",
        axis=1,
    )

    wide_text = df.pivot_table(
        index="model",
        columns="column_key",
        values="mean_pm_sd",
        aggfunc="first",
    )

    desired_cols = [
        "low__water_1p4",
        "low__water_1p9",
        "low__mineral_diag",
        "low__other",
        "high__water_1p4",
        "high__water_1p9",
        "high__mineral_diag",
        "high__other",
    ]

    existing_cols = [c for c in desired_cols if c in wide_text.columns]
    wide_text = wide_text[existing_cols].reset_index()

    return wide_text


def summarize_curve_across_seeds(curve_all: pd.DataFrame) -> pd.DataFrame:
    required = {"seed", "model", "humidity_group", "wavelength_um", "mean_attr"}
    missing = required - set(curve_all.columns)
    if missing:
        raise ValueError(f"curve table missing columns: {sorted(missing)}")

    group_cols = [c for c in ["model", "class_name", "humidity_group", "wavelength_um"] if c in curve_all.columns]
    out = (
        curve_all.groupby(group_cols, as_index=False)
        .agg(
            mean_attr_across_seeds=("mean_attr", "mean"),
            std_attr_across_seeds=("mean_attr", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    out["std_attr_across_seeds"] = out["std_attr_across_seeds"].fillna(0.0)
    return out


def summarize_peak_consensus(peak_all: pd.DataFrame) -> pd.DataFrame:
    required = {"seed", "model", "humidity_group", "wavelength_um"}
    missing = required - set(peak_all.columns)
    if missing:
        raise ValueError(f"peak table missing columns: {sorted(missing)}")

    group_cols = [c for c in ["model", "class_name", "humidity_group", "rank", "wavelength_um"] if c in peak_all.columns]
    out = (
        peak_all.groupby(group_cols, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_peak_attr=("mean_attr", "mean") if "mean_attr" in peak_all.columns else ("seed", "size"),
        )
    )
    if "mean_peak_attr" in out.columns:
        out = out.sort_values(["model", "humidity_group", "rank", "n_seeds", "mean_peak_attr"], ascending=[True, True, True, False, False])
    else:
        out = out.sort_values(["model", "humidity_group", "rank", "n_seeds"], ascending=[True, True, True, False])
    return out.reset_index(drop=True)


def write_outputs(
    out_dir: Path,
    band_all: pd.DataFrame,
    band_seedmean: pd.DataFrame,
    band_summary: pd.DataFrame,
    band_wide: pd.DataFrame,
    curve_seedmean: pd.DataFrame,
    peak_consensus: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "band_all": out_dir / "exp511_band_mass_all_rows_with_seed.csv",
        "band_seedmean": out_dir / "exp511_band_mass_seedmean.csv",
        "band_summary": out_dir / "exp511_band_mass_summary_long.csv",
        "band_wide": out_dir / "exp511_table_main_band_mass_wide.csv",
        "curve_seedmean": out_dir / "exp511_curve_summary_seedmean.csv",
        "peak_consensus": out_dir / "exp511_peak_consensus.csv",
    }

    band_all.to_csv(paths["band_all"], index=False, encoding="utf-8-sig")
    band_seedmean.to_csv(paths["band_seedmean"], index=False, encoding="utf-8-sig")
    band_summary.to_csv(paths["band_summary"], index=False, encoding="utf-8-sig")
    band_wide.to_csv(paths["band_wide"], index=False, encoding="utf-8-sig")
    curve_seedmean.to_csv(paths["curve_seedmean"], index=False, encoding="utf-8-sig")
    peak_consensus.to_csv(paths["peak_consensus"], index=False, encoding="utf-8-sig")

    try:
        import openpyxl  # noqa: F401

        xlsx_path = out_dir / "exp511_merged_tables.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            band_all.to_excel(writer, sheet_name="band_all", index=False)
            band_seedmean.to_excel(writer, sheet_name="band_seedmean", index=False)
            band_summary.to_excel(writer, sheet_name="band_summary", index=False)
            band_wide.to_excel(writer, sheet_name="band_wide", index=False)
            curve_seedmean.to_excel(writer, sheet_name="curve_seedmean", index=False)
            peak_consensus.to_excel(writer, sheet_name="peak_consensus", index=False)
        print(f"[OK] Excel written: {xlsx_path}")
    except ImportError:
        print("[WARN] openpyxl not installed, skipped Excel export. CSV files were still saved.")

    print("[OK] wrote merged Experiment 5.11 tables to:", out_dir)
    for name, p in paths.items():
        print(f"  - {name}: {p}")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    seed_dirs = discover_seed_dirs(root, args.seeds)
    print("[INFO] Found seed folders:", [f"raw_seed{seed}" for seed, _ in seed_dirs])

    band_all, curve_all, peak_all = concat_seed_tables(seed_dirs)
    band_seedmean = make_band_seed_level_means(band_all)
    band_summary = summarize_band_across_seeds(band_seedmean)
    band_wide = make_main_wide_table(band_summary)
    curve_seedmean = summarize_curve_across_seeds(curve_all)
    peak_consensus = summarize_peak_consensus(peak_all)

    write_outputs(out, band_all, band_seedmean, band_summary, band_wide, curve_seedmean, peak_consensus)


if __name__ == "__main__":
    main()
