#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import pandas as pd

MODEL_NAME_MAP = {
    "cnn": "1D-CNN",
    "resnet": "ResNet1D",
    "concat": "Concat-CNN",
    "hda": "HDA-CNN",
}

METRIC_KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]


def parse_split_seed(split_dir_name: str):
    m = re.match(r"split_(\d+)_seed_(\d+)", split_dir_name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def parse_run_seed(seed_dir_name: str):
    m = re.match(r"seed_?(\d+)", seed_dir_name)
    if not m:
        return None
    return int(m.group(1))


def safe_read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_per_run_rows(root: Path):
    rows = []
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    split_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("split_")])

    for split_dir in split_dirs:
        split_idx, split_seed = parse_split_seed(split_dir.name)

        for model_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            model_key = model_dir.name
            model_name = MODEL_NAME_MAP.get(model_key, model_key)

            for seed_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
                metrics_path = seed_dir / "metrics.json"
                if not metrics_path.exists():
                    continue

                run_seed = parse_run_seed(seed_dir.name)
                metrics = safe_read_json(metrics_path)

                row = {
                    "split_dir": split_dir.name,
                    "split_idx": split_idx,
                    "split_seed": split_seed,
                    "model_key": model_key,
                    "model": model_name,
                    "run_seed": run_seed,
                    "metrics_path": str(metrics_path),
                }

                for k in METRIC_KEYS:
                    row[k] = metrics.get(k)

                rows.append(row)

    return pd.DataFrame(rows)


def summarize_split_mean(df: pd.DataFrame):
    grouped = (
        df.groupby(["split_dir", "split_idx", "split_seed", "model_key", "model"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_precision_mean=("macro_precision", "mean"),
            macro_precision_std=("macro_precision", "std"),
            macro_recall_mean=("macro_recall", "mean"),
            macro_recall_std=("macro_recall", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            n_runs=("run_seed", "count"),
        )
        .sort_values(["split_idx", "model"])
        .reset_index(drop=True)
    )
    return grouped


def make_macro_f1_wide(split_mean_df: pd.DataFrame):
    wide = (
        split_mean_df.pivot_table(
            index=["split_dir", "split_idx", "split_seed"],
            columns="model",
            values="macro_f1_mean",
            aggfunc="first",
        )
        .reset_index()
        .sort_values("split_idx")
        .reset_index(drop=True)
    )
    return wide


def make_hda_pairwise_diff(macro_f1_wide: pd.DataFrame):
    if "HDA-CNN" not in macro_f1_wide.columns:
        raise ValueError("Could not find HDA-CNN column in wide Macro F1 table.")

    out = macro_f1_wide[["split_dir", "split_idx", "split_seed"]].copy()
    baselines = [c for c in macro_f1_wide.columns if c not in ["split_dir", "split_idx", "split_seed", "HDA-CNN"]]

    for b in baselines:
        out[f"HDA-CNN - {b}"] = macro_f1_wide["HDA-CNN"] - macro_f1_wide[b]

    return out


def make_hda_winrate(pairwise_df: pd.DataFrame):
    diff_cols = [c for c in pairwise_df.columns if c.startswith("HDA-CNN - ")]
    rows = []

    for c in diff_cols:
        baseline = c.replace("HDA-CNN - ", "")
        valid = pairwise_df[c].dropna()
        win_rate = (valid > 0).mean() if len(valid) else None
        rows.append({
            "baseline_model": baseline,
            "hda_macro_f1_win_rate": win_rate,
            "n_splits": int(valid.shape[0]),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="results/repeated_holdout_5seeds")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_df = collect_per_run_rows(root)
    if per_run_df.empty:
        raise RuntimeError(f"No metrics.json files found under {root}")

    split_mean_df = summarize_split_mean(per_run_df)
    macro_f1_wide_df = make_macro_f1_wide(split_mean_df)
    pairwise_df = make_hda_pairwise_diff(macro_f1_wide_df)
    winrate_df = make_hda_winrate(pairwise_df)

    per_run_path = out_dir / "repeated_holdout_per_run.csv"
    split_mean_path = out_dir / "repeated_holdout_split_mean.csv"
    macro_f1_wide_path = out_dir / "repeated_holdout_macro_f1_wide.csv"
    pairwise_path = out_dir / "repeated_holdout_hda_pairwise_diff.csv"
    winrate_path = out_dir / "repeated_holdout_hda_winrate.csv"

    per_run_df.to_csv(per_run_path, index=False)
    split_mean_df.to_csv(split_mean_path, index=False)
    macro_f1_wide_df.to_csv(macro_f1_wide_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    winrate_df.to_csv(winrate_path, index=False)

    print("Saved files:")
    print(per_run_path)
    print(split_mean_path)
    print(macro_f1_wide_path)
    print(pairwise_path)
    print(winrate_path)
    print()
    print(split_mean_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
