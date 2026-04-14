from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ORDER = ["alpha", "delta", "sigma0", "eta", "kappa"]
YLIMS = {
    "accuracy_mean": (0.65, 0.86),
    "macro_f1_mean": (0.65, 0.86),
}


def plot_metric(summary_csv: str, metric: str = "macro_f1_mean", std_metric: str = "macro_f1_std"):
    df = pd.read_csv(summary_csv)
    out_root = Path(summary_csv).parent / "figures"
    out_root.mkdir(parents=True, exist_ok=True)

    for param in ORDER:
        sub = df[df["param"] == param].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.errorbar(
            sub["value"],
            sub[metric],
            yerr=sub[std_metric],
            marker="o",
            capsize=4,
            linewidth=1.5,
        )
        ax.set_xlabel(param)
        ax.set_ylabel(metric.replace("_mean", "").replace("_", " ").title() + " (mean ± SD)")
        ax.set_title(f"Sensitivity of {metric.replace('_mean','')} to {param}")
        if metric in YLIMS:
            ax.set_ylim(*YLIMS[metric])
        plt.tight_layout()
        out_path = out_root / f"{param}_{metric}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, default="results/exp59_sensitivity/exp59_summary.csv")
    parser.add_argument("--metric", type=str, default="macro_f1_mean")
    parser.add_argument("--std_metric", type=str, default="macro_f1_std")
    args = parser.parse_args()
    plot_metric(args.summary_csv, args.metric, args.std_metric)
