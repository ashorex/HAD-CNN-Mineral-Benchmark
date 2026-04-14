import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=7, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, num_classes=4, in_channels=1, base_channels=32, dropout=0.1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            ResidualBlock1D(base_channels, base_channels, stride=1, kernel_size=7, dropout=dropout),
            ResidualBlock1D(base_channels, base_channels, stride=1, kernel_size=7, dropout=dropout)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock1D(base_channels, base_channels * 2, stride=2, kernel_size=5, dropout=dropout),
            ResidualBlock1D(base_channels * 2, base_channels * 2, stride=1, kernel_size=5, dropout=dropout)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock1D(base_channels * 2, base_channels * 4, stride=2, kernel_size=3, dropout=dropout),
            ResidualBlock1D(base_channels * 4, base_channels * 4, stride=1, kernel_size=3, dropout=dropout)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 兼容 [B, L] 或 [B, 1, L]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x