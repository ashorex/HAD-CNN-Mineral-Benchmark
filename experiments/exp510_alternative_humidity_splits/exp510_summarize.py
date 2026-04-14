import json
from pathlib import Path
import pandas as pd

ORDER = ["split_A_current", "split_B_shifted", "split_C_shifted"]

def main(results_root="results/exp510_non_overlapping"):
    root = Path(results_root)
    rows = []

    for preset_dir in root.iterdir():
        if not preset_dir.is_dir():
            continue
        preset = preset_dir.name
        for seed_dir in preset_dir.iterdir():
            metrics_path = seed_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)

            seed = int(seed_dir.name.replace("seed_", ""))
            rows.append({
                "preset": preset,
                "seed": seed,
                "accuracy": m["accuracy"],
                "macro_precision": m["macro_precision"],
                "macro_recall": m["macro_recall"],
                "macro_f1": m["macro_f1"],
            })

    if not rows:
        raise RuntimeError(f"No metrics found under {root}")

    df = pd.DataFrame(rows)
    df.to_csv(root / "exp510_per_seed_results.csv", index=False)

    summary = (
        df.groupby("preset", as_index=False)
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
    summary["preset"] = pd.Categorical(summary["preset"], categories=ORDER, ordered=True)
    summary = summary.sort_values("preset").reset_index(drop=True)
    summary.to_csv(root / "exp510_summary.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="results/exp510_non_overlapping")
    args = parser.parse_args()
    main(args.results_root)
