import os
import random
from collections import Counter
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Models.HDA_CNN import HDA_CNN
from Models.CNN_1D import CNN_1D
from Models.Concat_CNN import Concat_CNN
from Utils.dataset_loader import SpectralDataset
from Utils.metrics import evaluate_metrics
from torch.utils.data import DataLoader, Subset
from Models.ResNet1D import ResNet1D


#单次训练，无seed
# def split_by_source_file_stratified(csv_path, val_ratio=0.2, seed=42):
#     """
#     按 source_file 分组、按类别分层划分 train/val
#     尽量保证：
#     1. 每个类别训练集至少有 1 个 source_file
#     2. 当类别 source_file 数 >= 2 时，验证集尽量也有 1 个 source_file
#     """
#     df = pd.read_csv(csv_path)
#
#     if "source_file" not in df.columns:
#         raise ValueError("train.csv 中缺少 source_file 列。")
#
#     rng = random.Random(seed)
#
#     # 每个 source_file 对应一个 label
#     source_label_df = df[["source_file", "label"]].drop_duplicates()
#
#     train_sources = []
#     val_sources = []
#
#     print("===== 分层 source_file 划分 =====")
#     for label in sorted(source_label_df["label"].unique()):
#         class_sources = source_label_df[source_label_df["label"] == label]["source_file"].tolist()
#         rng.shuffle(class_sources)
#
#         n_total = len(class_sources)
#
#         if n_total == 1:
#             # 只有1个源文件，只能用于训练
#             n_val = 0
#         else:
#             # 至少给验证集 1 个，但训练集也至少保留 1 个
#             n_val = max(1, int(round(n_total * val_ratio)))
#             n_val = min(n_val, n_total - 1)
#
#         val_part = class_sources[:n_val]
#         train_part = class_sources[n_val:]
#
#         train_sources.extend(train_part)
#         val_sources.extend(val_part)
#
#         print(
#             f"class {label}: total={n_total}, "
#             f"train_sources={len(train_part)}, val_sources={len(val_part)}"
#         )
#
#     train_sources = set(train_sources)
#     val_sources = set(val_sources)
#
#     train_idx = df.index[df["source_file"].isin(train_sources)].tolist()
#     val_idx = df.index[df["source_file"].isin(val_sources)].tolist()
#
#     print("训练 source_file 数:", len(train_sources))
#     print("验证 source_file 数:", len(val_sources))
#     print("训练样本数:", len(train_idx))
#     print("验证样本数:", len(val_idx))
#     print("source_file 重叠数:", len(train_sources.intersection(val_sources)))
#
#     # 再额外打印 train/val 中每类样本数，方便检查
#     train_labels = df.loc[train_idx, "label"].value_counts().sort_index()
#     val_labels = df.loc[val_idx, "label"].value_counts().sort_index()
#
#     print("===== train 中各类样本数 =====")
#     for i in sorted(source_label_df["label"].unique()):
#         print(f"class {i}: {train_labels.get(i, 0)}")
#
#     print("===== val 中各类样本数 =====")
#     for i in sorted(source_label_df["label"].unique()):
#         print(f"class {i}: {val_labels.get(i, 0)}")
#
#     return train_idx, val_idx
#
#
# def build_class_weights(dataset, indices, num_classes=4, device="cpu"):
#     """
#     根据训练子集标签分布构造类别权重
#     若某些类别在训练集中缺失，则直接报错
#     """
#     labels = [int(dataset.labels[i]) for i in indices]
#     counter = Counter(labels)
#
#     print("===== 训练集类别统计 =====")
#     for i in range(num_classes):
#         print(f"class {i}: {counter.get(i, 0)} samples")
#
#     missing_classes = [i for i in range(num_classes) if counter.get(i, 0) == 0]
#     if len(missing_classes) > 0:
#         raise ValueError(
#             f"训练集中缺少类别: {missing_classes}。"
#             f"请重新划分 train/val，确保每类至少有一个 source_file 在训练集中。"
#         )
#
#     counts = np.array([counter[i] for i in range(num_classes)], dtype=np.float32)
#
#     weights = counts.sum() / (num_classes * counts)
#     weights = np.clip(weights, 0.5, 5.0)
#
#     print("===== 类别权重 =====")
#     for i in range(num_classes):
#         print(f"class {i}: weight={weights[i]:.4f}")
#
#     return torch.tensor(weights, dtype=torch.float32).to(device)
#
#
# def train_nir(model_type="hda"):
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#
#     # 使用NIR数据集进行训练：
#     # train_csv = "Dataset/NIR/train.csv"
#     # test_csv = "Dataset/NIR/test.csv"
#
#     #泛化实验数据进行训练：
#     train_csv = "Dataset/NIR_cross_humidity/train.csv"
#     test_csv = "Dataset/NIR_cross_humidity/test.csv"
#
#     dataset = SpectralDataset(train_csv)
#     test_set = SpectralDataset(test_csv)
#
#     spec, hum, label = dataset[0]
#     print("spec shape:", spec.shape)
#     print("hum shape:", hum.shape)
#     print("label:", label)
#
#     train_idx, val_idx = split_by_source_file_stratified(train_csv, val_ratio=0.2, seed=42)
#
#     train_set = Subset(dataset, train_idx)
#     val_set = Subset(dataset, val_idx)
#
#     train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
#     test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
#
#     num_c = len(np.unique(dataset.labels))
#
#     if model_type == "hda":
#         model = HDA_CNN(num_classes=num_c)
#     elif model_type == "concat":
#         model = Concat_CNN(num_classes=num_c)
#     else:
#         model = CNN_1D(num_classes=num_c)
#
#     model = model.to(device)
#
#     class_weights = build_class_weights(dataset, train_idx, num_classes=num_c, device=device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
#
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode="max",
#         factor=0.5,
#         patience=5,
#         verbose=True
#     )
#
#     os.makedirs("checkpoints", exist_ok=True)
#     #使用NIR数据集保存的模型路径：
#     # best_model_path = f"checkpoints/{model_type}_nir_best.pth"
#
#     #泛化实验保存路径：
#     best_model_path = f"checkpoints/{model_type}_nir_crosshum_best.pth"
#
#     epochs = 40
#     best_val_acc = -1.0
#
#     history = {
#         "train_loss": [],
#         "val_acc": []
#     }
#
#     for epoch in range(epochs):
#
#         model.train()
#         total_loss = 0.0
#         total_num = 0
#
#         for spec, hum, label in train_loader:
#             spec = spec.to(device)
#             hum = hum.to(device)
#             label = label.to(device)
#
#             optimizer.zero_grad()
#
#             if model_type in ["hda", "concat"]:
#                 output = model(spec, hum)
#             else:
#                 output = model(spec)
#
#             loss = criterion(output, label)
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item() * spec.size(0)
#             total_num += spec.size(0)
#
#         train_loss = total_loss / max(total_num, 1)
#
#         model.eval()
#         y_true = []
#         y_pred = []
#
#         with torch.no_grad():
#             for spec, hum, label in val_loader:
#                 spec = spec.to(device)
#                 hum = hum.to(device)
#
#                 if model_type in ["hda", "concat"]:
#                     output = model(spec, hum)
#                 else:
#                     output = model(spec)
#
#                 pred = torch.argmax(output, dim=1).cpu().numpy()
#                 y_true.extend(label.numpy())
#                 y_pred.extend(pred)
#
#         val_acc, report, cm = evaluate_metrics(y_true, y_pred)
#
#         history["train_loss"].append(train_loss)
#         history["val_acc"].append(val_acc)
#
#         print(
#             f"Epoch {epoch+1:03d} | "
#             f"Train Loss: {train_loss:.4f} | "
#             f"Val Acc: {val_acc:.4f} | "
#             f"LR: {optimizer.param_groups[0]['lr']:.6f}"
#         )
#
#         scheduler.step(val_acc)
#
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), best_model_path)
#             print(f"Best model saved. Val Acc = {best_val_acc:.4f}")
#
#     print(f"Loading best model from: {best_model_path}")
#     model.load_state_dict(torch.load(best_model_path, map_location=device))
#     model.eval()
#
#     y_true = []
#     y_pred = []
#
#     with torch.no_grad():
#         for spec, hum, label in test_loader:
#             spec = spec.to(device)
#             hum = hum.to(device)
#
#             if model_type in ["hda", "concat"]:
#                 output = model(spec, hum)
#             else:
#                 output = model(spec)
#
#             pred = torch.argmax(output, dim=1).cpu().numpy()
#             y_true.extend(label.numpy())
#             y_pred.extend(pred)
#
#     acc, report, cm = evaluate_metrics(y_true, y_pred)
#
#     print("===== Test Results =====")
#     print("Best Val Accuracy:", best_val_acc)
#     print("Test Accuracy:", acc)
#     print("Confusion Matrix:\n", cm)
#     print("Classification Report:\n", report)
#     #NIR数据：
#     # torch.save(model.state_dict(), f"checkpoints/{model_type}_nir_final.pth")
#
#     #泛化实验：
#     torch.save(model.state_dict(), f"checkpoints/{model_type}_nir_crosshum_final.pth")

#seed实验
# def split_by_source_file_stratified(csv_path, val_ratio=0.2, seed=42):
#     """
#     按 source_file 分组、按类别分层划分 train/val
#     尽量保证：
#     1. 每个类别训练集至少有 1 个 source_file
#     2. 当类别 source_file 数 >= 2 时，验证集尽量也有 1 个 source_file
#     """
#     df = pd.read_csv(csv_path)
#
#     if "source_file" not in df.columns:
#         raise ValueError("train.csv 中缺少 source_file 列。")
#
#     rng = random.Random(seed)
#
#     # 每个 source_file 对应一个 label
#     source_label_df = df[["source_file", "label"]].drop_duplicates()
#
#     train_sources = []
#     val_sources = []
#
#     print("===== 分层 source_file 划分 =====")
#     for label in sorted(source_label_df["label"].unique()):
#         class_sources = source_label_df[source_label_df["label"] == label]["source_file"].tolist()
#         rng.shuffle(class_sources)
#
#         n_total = len(class_sources)
#
#         if n_total == 1:
#             # 只有1个源文件，只能用于训练
#             n_val = 0
#         else:
#             # 至少给验证集 1 个，但训练集也至少保留 1 个
#             n_val = max(1, int(round(n_total * val_ratio)))
#             n_val = min(n_val, n_total - 1)
#
#         val_part = class_sources[:n_val]
#         train_part = class_sources[n_val:]
#
#         train_sources.extend(train_part)
#         val_sources.extend(val_part)
#
#         print(
#             f"class {label}: total={n_total}, "
#             f"train_sources={len(train_part)}, val_sources={len(val_part)}"
#         )
#
#     train_sources = set(train_sources)
#     val_sources = set(val_sources)
#
#     train_idx = df.index[df["source_file"].isin(train_sources)].tolist()
#     val_idx = df.index[df["source_file"].isin(val_sources)].tolist()
#
#     print("训练 source_file 数:", len(train_sources))
#     print("验证 source_file 数:", len(val_sources))
#     print("训练样本数:", len(train_idx))
#     print("验证样本数:", len(val_idx))
#     print("source_file 重叠数:", len(train_sources.intersection(val_sources)))
#
#     # 再额外打印 train/val 中每类样本数，方便检查
#     train_labels = df.loc[train_idx, "label"].value_counts().sort_index()
#     val_labels = df.loc[val_idx, "label"].value_counts().sort_index()
#
#     print("===== train 中各类样本数 =====")
#     for i in sorted(source_label_df["label"].unique()):
#         print(f"class {i}: {train_labels.get(i, 0)}")
#
#     print("===== val 中各类样本数 =====")
#     for i in sorted(source_label_df["label"].unique()):
#         print(f"class {i}: {val_labels.get(i, 0)}")
#
#     return train_idx, val_idx
#
#
# def build_class_weights(dataset, indices, num_classes=4, device="cpu"):
#     """
#     根据训练子集标签分布构造类别权重
#     若某些类别在训练集中缺失，则直接报错
#     """
#     labels = [int(dataset.labels[i]) for i in indices]
#     counter = Counter(labels)
#
#     print("===== 训练集类别统计 =====")
#     for i in range(num_classes):
#         print(f"class {i}: {counter.get(i, 0)} samples")
#
#     missing_classes = [i for i in range(num_classes) if counter.get(i, 0) == 0]
#     if len(missing_classes) > 0:
#         raise ValueError(
#             f"训练集中缺少类别: {missing_classes}。"
#             f"请重新划分 train/val，确保每类至少有一个 source_file 在训练集中。"
#         )
#
#     counts = np.array([counter[i] for i in range(num_classes)], dtype=np.float32)
#
#     weights = counts.sum() / (num_classes * counts)
#     weights = np.clip(weights, 0.5, 5.0)
#
#     print("===== 类别权重 =====")
#     for i in range(num_classes):
#         print(f"class {i}: weight={weights[i]:.4f}")
#
#     return torch.tensor(weights, dtype=torch.float32).to(device)
#
#
# def train_nir(
#     model_type="hda",
#     dataset_dir="Dataset/NIR_cross_humidity",
#     exp_name="nir_crosshum_no_val",
#     seed=42
# ):
#     # 设置随机种子
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#     print("Seed:", seed)
#     print("Dataset dir:", dataset_dir)
#
#     train_csv = os.path.join(dataset_dir, "train.csv")
#     test_csv = os.path.join(dataset_dir, "test.csv")
#
#     dataset = SpectralDataset(train_csv)
#     test_set = SpectralDataset(test_csv)
#
#     spec, hum, label = dataset[0]
#     print("spec shape:", spec.shape)
#     print("hum shape:", hum.shape)
#     print("label:", label)
#
#     # 无验证集：直接全量 train.csv 用于训练
#     train_set = dataset
#
#     train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
#
#     num_c = len(np.unique(dataset.labels))
#
#     if model_type == "hda":
#         model = HDA_CNN(num_classes=num_c)
#     elif model_type == "concat":
#         model = Concat_CNN(num_classes=num_c)
#     else:
#         model = CNN_1D(num_classes=num_c)
#
#     model = model.to(device)
#
#     # 直接用整个训练集构建类别权重
#     full_train_idx = list(range(len(dataset)))
#     class_weights = build_class_weights(dataset, full_train_idx, num_classes=num_c, device=device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
#
#     # 无验证集时，不再用 ReduceLROnPlateau，改成固定步长衰减
#     scheduler = torch.optim.lr_scheduler.StepLR(
#         optimizer,
#         step_size=20,
#         gamma=0.5
#     )
#
#     # 分 seed 保存，避免覆盖
#     ckpt_dir = os.path.join("checkpoints", exp_name, model_type, f"seed{seed}")
#     result_dir = os.path.join("results", exp_name, model_type, f"seed{seed}")
#     os.makedirs(ckpt_dir, exist_ok=True)
#     os.makedirs(result_dir, exist_ok=True)
#
#     final_model_path = os.path.join(ckpt_dir, "final.pth")
#     epochs = 60
#
#     history = {
#         "train_loss": []
#     }
#
#     for epoch in range(epochs):
#
#         model.train()
#         total_loss = 0.0
#         total_num = 0
#
#         for spec, hum, label in train_loader:
#             spec = spec.to(device)
#             hum = hum.to(device)
#             label = label.to(device)
#
#             optimizer.zero_grad()
#
#             if model_type in ["hda", "concat"]:
#                 output = model(spec, hum)
#             else:
#                 output = model(spec)
#
#             loss = criterion(output, label)
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item() * spec.size(0)
#             total_num += spec.size(0)
#
#         scheduler.step()
#
#         train_loss = total_loss / max(total_num, 1)
#         history["train_loss"].append(train_loss)
#
#         print(
#             f"Epoch {epoch+1:03d} | "
#             f"Train Loss: {train_loss:.4f} | "
#             f"LR: {optimizer.param_groups[0]['lr']:.6f}"
#         )
#
#     # ===== 最终测试 =====
#     model.eval()
#     y_true = []
#     y_pred = []
#     all_humidity = []
#
#     with torch.no_grad():
#         for spec, hum, label in test_loader:
#             spec = spec.to(device)
#             hum = hum.to(device)
#
#             if model_type in ["hda", "concat"]:
#                 output = model(spec, hum)
#             else:
#                 output = model(spec)
#
#             pred = torch.argmax(output, dim=1).cpu().numpy()
#             y_true.extend(label.numpy())
#             y_pred.extend(pred)
#             all_humidity.extend(hum.cpu().numpy().flatten().tolist())
#
#     acc, report, cm = evaluate_metrics(y_true, y_pred)
#
#     print("===== Test Results =====")
#     print("Test Accuracy:", acc)
#     print("Confusion Matrix:\n", cm)
#     print("Classification Report:\n", report)
#
#     torch.save(model.state_dict(), final_model_path)
#
#     # 保存训练过程
#     with open(os.path.join(result_dir, "history.json"), "w", encoding="utf-8") as f:
#         json.dump(history, f, ensure_ascii=False, indent=2)
#
#     # 保存整体指标
#     metrics_out = {
#         "seed": seed,
#         "model": model_type,
#         "dataset_dir": dataset_dir,
#         "test_accuracy": float(acc),
#         "classification_report": report,
#         "confusion_matrix": cm.tolist()
#     }
#     with open(os.path.join(result_dir, "metrics.json"), "w", encoding="utf-8") as f:
#         json.dump(metrics_out, f, ensure_ascii=False, indent=2)
#
#     # 保存逐样本预测
#     pred_df = pd.DataFrame({
#         "y_true": y_true,
#         "y_pred": y_pred,
#         "humidity": all_humidity
#     })
#     pred_df.to_csv(os.path.join(result_dir, "predictions.csv"), index=False)

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Models.HDA_CNN import HDA_CNN
from Models.CNN_1D import CNN_1D
from Models.ResNet1D import ResNet1D
from Utils.dataset_loader import SpectralDataset
from Utils.metrics import evaluate_metrics


import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Models.HDA_CNN import HDA_CNN
from Models.CNN_1D import CNN_1D
from Models.Concat_CNN import Concat_CNN
from Models.ResNet1D import ResNet1D

from Utils.dataset_loader import SpectralDataset
from Utils.metrics import evaluate_metrics


import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Models.HDA_CNN import HDA_CNN
from Models.CNN_1D import CNN_1D
from Models.Concat_CNN import Concat_CNN
from Models.ResNet1D import ResNet1D

from Utils.dataset_loader import SpectralDataset
from Utils.metrics import evaluate_metrics


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了尽量可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_type="hda", num_classes=4):
    if model_type == "hda":
        return HDA_CNN(num_classes=num_classes)
    elif model_type == "cnn":
        return CNN_1D(num_classes=num_classes)
    elif model_type == "concat":
        return Concat_CNN(num_classes=num_classes)
    elif model_type == "resnet":
        return ResNet1D(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def infer_num_classes(dataset):
    """
    尽量从 dataset 中自动推断类别数。
    如果你的 SpectralDataset 已经带 labels/y，就优先使用。
    """
    if hasattr(dataset, "labels"):
        return len(set(dataset.labels))
    elif hasattr(dataset, "y"):
        return len(set(dataset.y))
    else:
        label_set = set()
        for i in range(len(dataset)):
            _, _, label = dataset[i]
            if torch.is_tensor(label):
                label = label.item()
            label_set.add(int(label))
        return len(label_set)


def compute_class_weights(dataset, num_classes):
    """
    计算 weighted cross-entropy 所需类别权重
    """
    counts = np.zeros(num_classes, dtype=np.int64)

    for i in range(len(dataset)):
        _, _, label = dataset[i]
        if torch.is_tensor(label):
            label = label.item()
        counts[int(label)] += 1

    counts = np.maximum(counts, 1)
    weights = len(dataset) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def forward_by_model(model, model_type, spec, hum):
    """
    统一处理单模态 / 双模态前向传播：
    - cnn / resnet: 单模态，只输入 spec
    - concat / hda: 双模态，输入 spec + hum
    """
    if model_type in ["concat", "hda"]:
        return model(spec, hum)
    else:
        return model(spec)


@torch.no_grad()
def evaluate_model(model, loader, device, model_type="hda"):
    model.eval()
    y_true = []
    y_pred = []

    for spec, hum, label in loader:
        spec = spec.to(device)
        hum = hum.to(device)
        label = label.to(device)

        output = forward_by_model(model, model_type, spec, hum)
        pred = torch.argmax(output, dim=1)

        y_true.extend(label.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    acc, report, cm = evaluate_metrics(y_true, y_pred)
    return acc, report, cm


def train_nir(
    model_type="hda",
    dataset_dir="datasets/nir",
    exp_name="default_exp",
    seed=42,
    epochs=60
):
    """
    正式实验版：
    - 只使用 train.csv 和 test.csv
    - 不额外划分 validation set
    - 训练完直接在独立 test set 上评估
    """
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model type: {model_type}")
    print(f"Random seed: {seed}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Experiment name: {exp_name}")

    train_csv = os.path.join(dataset_dir, "train.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"test.csv not found: {test_csv}")

    # ===== 数据 =====
    train_set = SpectralDataset(train_csv)
    test_set = SpectralDataset(test_csv)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    num_classes = infer_num_classes(train_set)
    print(f"Detected num_classes = {num_classes}")

    # ===== 模型 =====
    model = build_model(model_type=model_type, num_classes=num_classes).to(device)

    # ===== 损失函数 =====
    class_weights = compute_class_weights(train_set, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ===== 优化器 =====
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )

    # ===== 训练 =====
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for spec, hum, label in train_loader:
            spec = spec.to(device)
            hum = hum.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = forward_by_model(model, model_type, spec, hum)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            batch_size = label.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        scheduler.step()

        avg_loss = running_loss / max(total_samples, 1)
        print(f"Epoch {epoch + 1:03d}/{epochs:03d} | Train Loss: {avg_loss:.4f}")

    # ===== 测试 =====
    test_acc, test_report, test_cm = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        model_type=model_type
    )

    print("\n===== Test Results =====")
    print("Accuracy:", test_acc)
    print("Report:\n", test_report)
    print("Confusion Matrix:\n", test_cm)

    # ===== 保存模型 =====
    save_dir = os.path.join("checkpoints", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_type}_nir_seed{seed}.pth")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

    return test_acc, test_report, test_cm