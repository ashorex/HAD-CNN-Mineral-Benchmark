import json
from pathlib import Path

import pandas as pd

ORDER = ["alpha", "delta", "sigma0", "eta", "kappa"]


def main(results_root="results/exp59_sensitivity"):
    root = Path(results_root)
    rows = []

    for param_dir in root.iterdir():
        if not param_dir.is_dir():
            continue
        param_name = param_dir.name

        for value_dir in param_dir.iterdir():
            if not value_dir.is_dir():
                continue
            value_tag = value_dir.name.replace("value_", "")
            try:
                value = float(value_tag)
            except Exception:
                continue

            for seed_dir in value_dir.iterdir():
                metrics_path = seed_dir / "metrics.json"
                if not metrics_path.exists():
                    continue

                with open(metrics_path, "r", encoding="utf-8") as f:
                    m = json.load(f)

                seed = int(seed_dir.name.replace("seed_", ""))
                rows.append({
                    "param": param_name,
                    "value": value,
                    "seed": seed,
                    "accuracy": m["accuracy"],
                    "macro_precision": m["macro_precision"],
                    "macro_recall": m["macro_recall"],
                    "macro_f1": m["macro_f1"],
                })

    if not rows:
        raise RuntimeError(f"No metrics found under {root}")

    df = pd.DataFrame(rows)
    df.to_csv(root / "exp59_per_seed_results.csv", index=False)

    summary = (
        df.groupby(["param", "value"], as_index=False)
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
    summary["param"] = pd.Categorical(summary["param"], categories=ORDER, ordered=True)
    summary = summary.sort_values(["param", "value"]).reset_index(drop=True)
    summary.to_csv(root / "exp59_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="results/exp59_sensitivity")
    args = parser.parse_args()
    main(args.results_root)
