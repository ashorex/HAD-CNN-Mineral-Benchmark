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