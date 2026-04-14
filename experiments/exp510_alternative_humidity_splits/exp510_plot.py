from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main(summary_csv="results/exp510_non_overlapping/exp510_summary.csv"):
    df = pd.read_csv(summary_csv)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(df["preset"], df["macro_f1_mean"], yerr=df["macro_f1_std"], capsize=5)
    ax.set_ylabel("Macro F1 (mean ± SD)")
    ax.set_title("Generalization under alternative non_overlapping humidity splits")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out1 = Path(summary_csv).parent / "exp510_macro_f1.png"
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(df["preset"], df["accuracy_mean"], yerr=df["accuracy_std"], capsize=5)
    ax.set_ylabel("Accuracy (mean ± SD)")
    ax.set_title("Generalization under alternative non_overlapping humidity splits")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out2 = Path(summary_csv).parent / "exp510_accuracy.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(out1)
    print(out2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, default="results/exp510_non_overlapping/exp510_summary.csv")
    args = parser.parse_args()
    main(args.summary_csv)
