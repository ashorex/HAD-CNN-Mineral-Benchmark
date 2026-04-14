import os
import random
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def load_svm_data(dataset_dir):
    train_csv = os.path.join(dataset_dir, "train.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if "label" not in train_df.columns or "label" not in test_df.columns:
        raise ValueError("Column 'label' not found.")

    # 只保留列名可解释为波长的列
    feature_cols = []
    for c in train_df.columns:
        if c == "label":
            continue
        try:
            float(c)
            feature_cols.append(c)
        except ValueError:
            continue

    if len(feature_cols) == 0:
        raise ValueError("No spectral columns found. Please check column names.")

    print("Feature columns used by SVM:", feature_cols[:10], "...", feature_cols[-10:])
    print("Number of features:", len(feature_cols))

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(int)

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(int)
    print(train_df.columns.tolist())
    return X_train, y_train, X_test, y_test, feature_cols

def build_svm_model(C=10.0, gamma="scale", kernel="rbf"):
    """
    单输入 SVM
    - 先标准化
    - 再做 SVC
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            C=C,
            gamma=gamma,
            kernel=kernel,
            class_weight="balanced"
        ))
    ])
    return model


def train_svm_nir(
    dataset_dir="datasets/nir",
    exp_name="default_exp",
    seed=42,
    C=10.0,
    gamma="scale",
    kernel="rbf"
):
    set_seed(seed)

    print("===== SVM NIR Training =====")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Experiment name: {exp_name}")
    print(f"Seed: {seed}")
    print(f"SVM params: kernel={kernel}, C={C}, gamma={gamma}")

    X_train, y_train, X_test, y_test, feature_cols = load_svm_data(dataset_dir)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Num spectral features: {len(feature_cols)}")

    model = build_svm_model(C=C, gamma=gamma, kernel=kernel)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== Test Results =====")
    print("Accuracy:", acc)
    print("Report:\n", report_str)
    print("Confusion Matrix:\n", cm)

    # 保存结果
    save_dir = os.path.join("checkpoints", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    result_path = os.path.join(save_dir, f"svm_nir_seed{seed}_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("===== SVM NIR Results =====\n")
        f.write(f"Dataset dir: {dataset_dir}\n")
        f.write(f"Experiment name: {exp_name}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"SVM params: kernel={kernel}, C={C}, gamma={gamma}\n\n")
        f.write(f"Accuracy: {acc}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    print(f"Results saved to: {result_path}")

    return acc, report_dict, cm