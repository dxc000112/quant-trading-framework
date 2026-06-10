"""
Time-series datasets and data loaders for PyTorch training.

面试问题：时间序列 DataLoader 和 CV 图像 DataLoader 有什么区别？
答：不能 shuffle（时序依赖）；需要滑窗切分；
    train/val/test 必须按时间切分，不能随机 split。
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for time-series prediction.

    Converts a (T, F) feature matrix and a (T,) label vector into
    overlapping windows of length `window_size`, where each window
    maps to the label at the end of the window.

    Args:
        features: np.ndarray of shape (T, num_features)
        labels: np.ndarray of shape (T,)
        window_size: int, number of time steps per sample

    面试辩护：
        Why sliding window instead of full-sequence?
        → Memory efficiency for long intraday series (>10k bars).
        → Consistent input shape for batch training.
        Failure mode: overlapping windows cause data leakage between
        train/val if not split by time first, then windowed.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, window_size: int = 20):
        if len(features) != len(labels):
            raise ValueError(
                f"features length ({len(features)}) != labels length ({len(labels)})"
            )
        if window_size >= len(features):
            raise ValueError(
                f"window_size ({window_size}) must be < data length ({len(features)})"
            )

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]   # (window_size, F)
        y = self.labels[idx + self.window_size]            # scalar
        return x, y


class CrossSectionalDataset(Dataset):
    """
    Dataset for cross-sectional (point-in-time) factor models.

    Each sample is a single row: (features, label).
    Used when replacing RandomForest with MLP for meta-labeling
    or factor selection, where temporal ordering is handled by
    the outer train/test split.

    Args:
        features: np.ndarray of shape (N, num_features)
        labels: np.ndarray of shape (N,)
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        if len(features) != len(labels):
            raise ValueError(
                f"features length ({len(features)}) != labels length ({len(labels)})"
            )
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def temporal_train_val_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    Splits data strictly by time order (no shuffling).

    面试辩护：
        Why not random split?
        → Random split in time series causes future data to leak into training.
        → Standard practice in quant finance: walk-forward or expanding window.

    Returns:
        tuple of (train_features, train_labels,
                  val_features, val_labels,
                  test_features, test_labels)
    """
    n = len(features)
    test_start = int(n * (1 - test_ratio))
    val_start = int(n * (1 - test_ratio - val_ratio))

    return (
        features[:val_start], labels[:val_start],
        features[val_start:test_start], labels[val_start:test_start],
        features[test_start:], labels[test_start:],
    )


def build_dataloaders(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int = 20,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    dataset_type: str = 'timeseries',
):
    """
    End-to-end builder: data → temporal split → DataLoaders.

    Args:
        dataset_type: 'timeseries' for sliding-window, 'cross_sectional' for point-in-time

    Returns:
        dict with keys 'train', 'val', 'test', each a DataLoader
    """
    (train_X, train_y,
     val_X, val_y,
     test_X, test_y) = temporal_train_val_test_split(
        features, labels, val_ratio=val_ratio, test_ratio=test_ratio
    )

    DatasetClass = TimeSeriesDataset if dataset_type == 'timeseries' else CrossSectionalDataset

    if dataset_type == 'timeseries':
        train_ds = DatasetClass(train_X, train_y, window_size=window_size)
        val_ds = DatasetClass(val_X, val_y, window_size=window_size)
        test_ds = DatasetClass(test_X, test_y, window_size=window_size)
    else:
        train_ds = DatasetClass(train_X, train_y)
        val_ds = DatasetClass(val_X, val_y)
        test_ds = DatasetClass(test_X, test_y)

    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }
