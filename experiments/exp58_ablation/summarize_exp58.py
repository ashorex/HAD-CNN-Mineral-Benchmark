import json
from pathlib import Path
import pandas as pd

ORDER = ["full", "late_concat_only", "modulation_only", "no_residual"]

def main(results_root="results/exp58_ablation"):
    root = Path(results_root)
    rows = []
    for variant_dir in root.iterdir():
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        for seed_dir in variant_dir.iterdir():
            metrics_path = seed_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            seed = int(seed_dir.name.replace("seed_", ""))
            rows.append({
                "variant": variant,
                "seed": seed,
                "accuracy": m["accuracy"],
                "macro_precision": m["macro_precision"],
                "macro_recall": m["macro_recall"],
                "macro_f1": m["macro_f1"],
            })

    if not rows:
        raise RuntimeError(f"No metrics found under {root}")

    df = pd.DataFrame(rows)
    df.to_csv(root / "exp58_per_seed_results.csv", index=False)

    summary = (
        df.groupby("variant", as_index=False)
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
    summary["variant"] = pd.Categorical(summary["variant"], categories=ORDER, ordered=True)
    summary = summary.sort_values("variant").reset_index(drop=True)
    summary.to_csv(root / "exp58_summary.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="results/exp58_ablation")
    args = parser.parse_args()
    main(args.results_root)
