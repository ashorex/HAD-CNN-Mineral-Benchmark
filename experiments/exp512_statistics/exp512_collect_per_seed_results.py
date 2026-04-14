from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


TARGET_METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]


def normalize_model_name(name: str) -> str:
    raw = name.strip().lower()
    alias_map = {
        "hda": "HAD-CNN",
        "had": "HAD-CNN",
        "hadcnn": "HAD-CNN",
        "had-cnn": "HAD-CNN",
        "hda_cnn": "HAD-CNN",
        "hda-cnn": "HAD-CNN",

        "concat": "Concat-CNN",
        "concatcnn": "Concat-CNN",
        "concat-cnn": "Concat-CNN",

        "cnn": "1D-CNN",
        "cnn1d": "1D-CNN",
        "1dcnn": "1D-CNN",
        "1d-cnn": "1D-CNN",

        "resnet": "ResNet1D",
        "resnet1d": "ResNet1D",
    }
    return alias_map.get(raw, name)


def parse_seed_from_name(name: str) -> Optional[int]:
    m = re.search(r"seed[_\-]?(\d+)", name.lower())
    if m:
        return int(m.group(1))
    return None


def safe_load_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def recursive_find(obj, candidate_keys: List[str]) -> Optional[float]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in candidate_keys:
                try:
                    return float(v)
                except Exception:
                    pass
            found = recursive_find(v, candidate_keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = recursive_find(item, candidate_keys)
            if found is not None:
                return found
    return None


def extract_metric(metrics: Dict, canonical_name: str) -> Optional[float]:
    """
    兼容你当前 metrics.json 的结构，例如：
    {
      "test_accuracy": ...,
      "classification_report": {
        "macro avg": {
          "precision": ...,
          "recall": ...,
          "f1-score": ...
        }
      }
    }
    """
    # 1) 直接平铺字段优先
    flat_aliases = {
        "accuracy": ["accuracy", "acc", "test_accuracy", "overall_accuracy"],
        "macro_precision": ["macro_precision", "precision"],
        "macro_recall": ["macro_recall", "recall"],
        "macro_f1": ["macro_f1", "f1", "f1-score", "macro_f1_score"],
    }

    for key in flat_aliases[canonical_name]:
        if key in metrics:
            try:
                return float(metrics[key])
            except Exception:
                pass

    # 2) 兼容 classification_report["macro avg"]
    report = metrics.get("classification_report")
    if isinstance(report, dict):
        macro_avg = report.get("macro avg")
        if isinstance(macro_avg, dict):
            if canonical_name == "macro_precision" and "precision" in macro_avg:
                return float(macro_avg["precision"])
            if canonical_name == "macro_recall" and "recall" in macro_avg:
                return float(macro_avg["recall"])
            if canonical_name == "macro_f1" and "f1-score" in macro_avg:
                return float(macro_avg["f1-score"])

        # accuracy 有时也在 classification_report 里
        if canonical_name == "accuracy" and "accuracy" in report:
            try:
                return float(report["accuracy"])
            except Exception:
                pass

    # 3) 兜底递归搜索
    recursive_aliases = {
        "accuracy": ["accuracy", "acc", "test_accuracy"],
        "macro_precision": ["macro_precision", "precision"],
        "macro_recall": ["macro_recall", "recall"],
        "macro_f1": ["macro_f1", "f1", "f1-score"],
    }
    return recursive_find(metrics, recursive_aliases[canonical_name])


def infer_model_from_metrics_path(metrics_path: Path, results_root: Path) -> Optional[str]:
    """
    适配你的目录结构:
    results/nir_crosshum_v2/hda/seed10/metrics.json
    """
    try:
        rel = metrics_path.relative_to(results_root)
        parts = rel.parts
        if len(parts) >= 3:
            model_dir_name = parts[-3]
            return normalize_model_name(model_dir_name)
    except Exception:
        pass
    return None


def infer_seed_from_metrics_path(metrics_path: Path) -> Optional[int]:
    for part in metrics_path.parts[::-1]:
        seed = parse_seed_from_name(part)
        if seed is not None:
            return seed
    return None


def collect_recursively(results_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []

    metrics_files = sorted(results_root.rglob("metrics.json"))
    if not metrics_files:
        raise RuntimeError(f"在 {results_root} 下没有找到任何 metrics.json")

    for metrics_path in metrics_files:
        metrics = safe_load_json(metrics_path)
        if metrics is None:
            continue

        model = infer_model_from_metrics_path(metrics_path, results_root)
        seed = infer_seed_from_metrics_path(metrics_path)

        # 如果路径推不出来，允许从 json 里兜底
        if model is None and "model" in metrics:
            model = normalize_model_name(str(metrics["model"]))
        if seed is None and "seed" in metrics:
            try:
                seed = int(metrics["seed"])
            except Exception:
                seed = None

        if model is None or seed is None:
            continue

        row = {
            "model": model,
            "seed": seed,
        }

        ok = True
        for target_metric in TARGET_METRICS:
            value = extract_metric(metrics, target_metric)
            if value is None:
                ok = False
                break
            row[target_metric] = value

        if ok:
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"在 {results_root} 下找到了 metrics.json，但未识别出可用指标。\n"
            f"请检查 metrics.json 是否包含 test_accuracy 与 classification_report['macro avg']。"
        )

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["model", "seed"])
        .sort_values(["model", "seed"])
        .reset_index(drop=True)
    )
    return df


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect per-seed experiment metrics for Experiment 5.12."
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results/nir_crosshum_v2",
        help="实验结果根目录，例如 results/nir_crosshum_v2",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/exp512_statistics",
        help="输出目录，建议保持默认",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_root.exists():
        raise FileNotFoundError(f"结果目录不存在: {results_root}")

    df = collect_recursively(results_root)
    summary = summarize_mean_sd(df)

    per_seed_csv = output_dir / "exp512_per_seed_results.csv"
    summary_csv = output_dir / "exp512_summary_mean_sd.csv"

    df.to_csv(per_seed_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("5.12 per-seed results collected successfully.")
    print(f"Input root : {results_root}")
    print(f"Rows       : {len(df)}")
    print(f"Saved      : {per_seed_csv}")
    print(f"Saved      : {summary_csv}")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\n[Summary mean ± SD]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()