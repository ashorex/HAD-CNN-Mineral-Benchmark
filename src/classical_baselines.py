import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


DEFAULT_SEEDS = [42, 52, 62, 72, 82]

LABEL_CANDIDATES = [
    "label", "target", "class", "mineral", "mineral_class", "y"
]

HUMIDITY_CANDIDATES = [
    "humidity", "rh", "h", "humidity_label"
]

GROUP_CANDIDATES = [
    "source_id", "source_file", "source", "parent_id", "file_id", "group", "group_id"
]

MODEL_NAME_MAP = {
    "pls": "PLS-DA",
    "svm": "SVM-RBF",
    "rf": "Random Forest",
}


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    label_col: str
    humidity_col: Optional[str]
    group_col: Optional[str]
    feature_cols: List[str]
    X_train: np.ndarray
    y_train: np.ndarray
    groups_train: Optional[np.ndarray]
    X_test: np.ndarray
    y_test: np.ndarray


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.lb_ = LabelBinarizer()
        Y = self.lb_.fit_transform(y)

        # Ensure multiclass-style output even in binary case
        if Y.ndim == 1:
            Y = np.column_stack([1 - Y, Y])
        elif Y.shape[1] == 1:
            Y = np.column_stack([1 - Y[:, 0], Y[:, 0]])

        self.classes_ = np.array(self.lb_.classes_)

        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)

        safe_n_components = max(
            1,
            min(
                int(self.n_components),
                Xs.shape[1],
                Y.shape[1]
            )
        )

        self.model_ = PLSRegression(n_components=safe_n_components)
        self.model_.fit(Xs, Y)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(X)
        scores = self.model_.predict(Xs)
        if scores.ndim == 1:
            scores = np.column_stack([1 - scores, scores])
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(X)
        pred_idx = np.argmax(scores, axis=1)
        return self.classes_[pred_idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Section 5.7 classical baseline experiment"
    )
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "pls", "svm", "rf"],
        help="Run a single model or all models."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run a single seed. If omitted, DEFAULT_SEEDS will be used."
    )

    parser.add_argument(
        "--label_col",
        type=str,
        default=None,
        help="Optional manual override for label column."
    )
    parser.add_argument(
        "--humidity_col",
        type=str,
        default=None,
        help="Optional manual override for humidity column."
    )
    parser.add_argument(
        "--group_col",
        type=str,
        default=None,
        help="Optional manual override for source/group column."
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Maximum number of CV folds."
    )

    return parser.parse_args()


def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def detect_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: Optional[str],
    humidity_col: Optional[str],
    group_col: Optional[str]
) -> Tuple[str, Optional[str], Optional[str]]:
    if label_col is None:
        label_col = find_first_existing_column(train_df, LABEL_CANDIDATES)
    if label_col is None or label_col not in train_df.columns:
        raise ValueError(
            f"Could not detect label column. Please specify --label_col explicitly. "
            f"Available columns: {list(train_df.columns)}"
        )

    if humidity_col is None:
        humidity_col = find_first_existing_column(train_df, HUMIDITY_CANDIDATES)
    if humidity_col is not None and humidity_col not in train_df.columns:
        humidity_col = None

    if group_col is None:
        group_col = find_first_existing_column(train_df, GROUP_CANDIDATES)
    if group_col is not None and group_col not in train_df.columns:
        group_col = None

    if label_col not in test_df.columns:
        raise ValueError(f"Label column '{label_col}' is missing in test CSV.")

    return label_col, humidity_col, group_col


def infer_feature_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exclude_cols: List[str]
) -> List[str]:
    common_cols = [c for c in train_df.columns if c in test_df.columns and c not in exclude_cols]
    feature_cols: List[str] = []

    for col in common_cols:
        train_num = pd.to_numeric(train_df[col], errors="coerce")
        test_num = pd.to_numeric(test_df[col], errors="coerce")

        if train_num.notna().all() and test_num.notna().all():
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError(
            "No numeric spectral feature columns were found.\n"
            "Please check whether your feature columns are numeric and shared by train/test."
        )

    return feature_cols


def load_dataset(args: argparse.Namespace) -> DatasetBundle:
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    label_col, humidity_col, group_col = detect_columns(
        train_df=train_df,
        test_df=test_df,
        label_col=args.label_col,
        humidity_col=args.humidity_col,
        group_col=args.group_col
    )

    exclude_cols = [label_col]
    if humidity_col is not None:
        exclude_cols.append(humidity_col)
    if group_col is not None:
        exclude_cols.append(group_col)

    feature_cols = infer_feature_columns(train_df, test_df, exclude_cols)

    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y_train = train_df[label_col].astype(str).to_numpy()

    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y_test = test_df[label_col].astype(str).to_numpy()

    groups_train = None
    if group_col is not None:
        groups_train = train_df[group_col].astype(str).to_numpy()

    return DatasetBundle(
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
        humidity_col=humidity_col,
        group_col=group_col,
        feature_cols=feature_cols,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        X_test=X_test,
        y_test=y_test,
    )


def get_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    cv_folds: int,
    seed: int
):
    if groups is not None:
        n_groups = len(np.unique(groups))
        if n_groups >= 2:
            n_splits = min(cv_folds, n_groups)
            if n_splits >= 2:
                gkf = GroupKFold(n_splits=n_splits)
                return list(gkf.split(X, y, groups))

    # Fallback: stratified CV
    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min())
    n_splits = min(cv_folds, min_class_count)
    n_splits = max(2, n_splits)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, y))


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def build_estimator(model_name: str, params: Dict[str, Any], seed: int):
    if model_name == "PLS-DA":
        return PLSDAClassifier(
            n_components=int(params["n_components"])
        )

    if model_name == "SVM-RBF":
        steps = [
            ("scaler", StandardScaler())
        ]

        if params.get("use_pca", False):
            steps.append((
                "pca",
                PCA(
                    n_components=params["pca_n_components"],
                    svd_solver="full",
                    whiten=False
                )
            ))

        steps.append((
            "svc",
            SVC(
                kernel="rbf",
                C=float(params["C"]),
                gamma=params["gamma"],
                class_weight="balanced",
                shrinking=True,
                cache_size=2048
            )
        ))

        return Pipeline(steps)

    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            criterion="gini",
            max_depth=int(params["max_depth"]),
            min_samples_split=6,
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            bootstrap=True,
            max_samples=0.80,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed
        )

    raise ValueError(f"Unsupported model: {model_name}")


def get_param_grid(model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> List[Dict[str, Any]]:
    if model_name == "PLS-DA":
        return list(ParameterGrid({
            "n_components": [2, 3, 4, 5, 6]
        }))

    if model_name == "SVM-RBF":
        grid_no_pca = list(ParameterGrid({
            "use_pca": [False],
            "C": [1, 10, 50, 100, 200],
            "gamma": ["scale", 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
        }))

        grid_with_pca = list(ParameterGrid({
            "use_pca": [True],
            "pca_n_components": [0.95, 0.99],
            "C": [1, 10, 50, 100, 200],
            "gamma": ["scale", 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
        }))

        return grid_no_pca + grid_with_pca

    if model_name == "Random Forest":
        return list(ParameterGrid({
            "n_estimators": [200, 300],
            "max_depth": [6, 8],
            "min_samples_leaf": [2, 4],
            "max_features": ["sqrt", 0.20]
        }))

    raise ValueError(f"Unsupported model: {model_name}")


def grid_search_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: Optional[np.ndarray],
    cv_folds: int,
    seed: int
) -> Tuple[Dict[str, Any], float]:
    splits = get_cv_splits(X_train, y_train, groups_train, cv_folds, seed)
    param_grid = get_param_grid(model_name, X_train, y_train)

    best_params: Optional[Dict[str, Any]] = None
    best_score = -np.inf

    for params in param_grid:
        fold_scores: List[float] = []

        for tr_idx, va_idx in splits:
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            estimator = build_estimator(model_name, params, seed)
            estimator.fit(X_tr, y_tr)
            pred_va = estimator.predict(X_va)

            metrics = macro_metrics(y_va, pred_va)
            fold_scores.append(metrics["macro_f1"])

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()

    assert best_params is not None
    return best_params, best_score


def train_and_evaluate_one_run(
    model_name: str,
    seed: int,
    data: DatasetBundle,
    cv_folds: int
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    best_params, best_cv_score = grid_search_model(
        model_name=model_name,
        X_train=data.X_train,
        y_train=data.y_train,
        groups_train=data.groups_train,
        cv_folds=cv_folds,
        seed=seed
    )

    final_estimator = build_estimator(model_name, best_params, seed)
    final_estimator.fit(data.X_train, data.y_train)
    pred_test = final_estimator.predict(data.X_test)

    test_metrics = macro_metrics(data.y_test, pred_test)

    run_info = {
        "model": model_name,
        "seed": seed,
        "best_cv_macro_f1": best_cv_score,
        **best_params
    }

    return run_info, test_metrics


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df
        .groupby("model", as_index=False)
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

    summary = summary.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    return summary


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    data = load_dataset(args)

    print("=" * 80)
    print("Loaded dataset")
    print(f"Train CSV     : {args.train_csv}")
    print(f"Test CSV      : {args.test_csv}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Label col     : {data.label_col}")
    print(f"Humidity col  : {data.humidity_col}")
    print(f"Group col     : {data.group_col}")
    print(f"#Features     : {len(data.feature_cols)}")
    print(f"Train shape   : {data.X_train.shape}")
    print(f"Test shape    : {data.X_test.shape}")
    print("=" * 80)

    if args.model == "all":
        models = ["PLS-DA", "SVM-RBF", "Random Forest"]
    else:
        models = [MODEL_NAME_MAP[args.model]]

    if args.seed is None:
        seeds = DEFAULT_SEEDS
    else:
        seeds = [args.seed]

    print(f"Models to run : {models}")
    print(f"Seeds to run  : {seeds}")
    print("=" * 80)

    all_rows: List[Dict[str, Any]] = []
    best_params_store: Dict[str, Dict[str, Any]] = {}

    for model_name in models:
        for seed in seeds:
            print(f"[Running] model={model_name}, seed={seed}")

            run_info, test_metrics = train_and_evaluate_one_run(
                model_name=model_name,
                seed=seed,
                data=data,
                cv_folds=args.cv_folds
            )

            row = {
                **run_info,
                **test_metrics
            }
            all_rows.append(row)

            best_params_store[f"{model_name}__seed_{seed}"] = {
                "best_cv_macro_f1": run_info["best_cv_macro_f1"],
                "params": {
                    k: v for k, v in run_info.items()
                    if k not in ["model", "seed", "best_cv_macro_f1"]
                }
            }

            print(
                f"  -> test accuracy={test_metrics['accuracy']:.4f}, "
                f"macro_f1={test_metrics['macro_f1']:.4f}"
            )

    results_df = pd.DataFrame(all_rows)
    summary_df = summarize_results(results_df)

    per_seed_csv = os.path.join(args.output_dir, "classical_per_seed_results.csv")
    summary_csv = os.path.join(args.output_dir, "classical_summary.csv")
    params_json = os.path.join(args.output_dir, "best_params.json")

    results_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with open(params_json, "w", encoding="utf-8") as f:
        json.dump(best_params_store, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Finished.")
    print(f"Saved: {per_seed_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {params_json}")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()