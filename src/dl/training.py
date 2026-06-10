"""
Training loop, evaluation, and checkpointing utilities for PyTorch models.

Provides a framework-agnostic Trainer class that handles:
  - Train/val/test loops with early stopping
  - Metric tracking (loss, accuracy, Rank IC)
  - Model checkpointing (best model by val loss)
  - TensorBoard logging integration
  - Device management (CPU/MPS/CUDA)
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Auto-detect the best available device.

    Priority: CUDA → MPS (Apple Silicon) → CPU

    面试问题：Apple Silicon 的 MPS 后端有什么限制？
    答：某些 op 不支持（如 complex tensor），某些精度行为不同。
        建议开发用 CPU，训练用 CUDA/Colab。
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@dataclass
class TrainConfig:
    """Training hyperparameters — centralizes all magic numbers."""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 10            # early stopping patience
    min_delta: float = 1e-4       # minimum improvement to reset patience
    grad_clip: float = 1.0        # max gradient norm
    log_dir: str = 'runs'         # TensorBoard log directory
    checkpoint_dir: str = 'checkpoints'
    device: Optional[str] = None  # 'cpu', 'cuda', 'mps', or None (auto)


class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    面试问题：Early stopping 和 L2 正则化的关系？
    答：都是防止过拟合。Early stopping 隐式控制模型复杂度
        （等价于限制权重从初始值移动的距离），
        类似于 L2 正则化但不需要调 lambda。
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    General-purpose PyTorch training loop.

    Usage:
        model = MetaLabelNet(input_dim=7)
        trainer = Trainer(model, config=TrainConfig(epochs=50))
        history = trainer.fit(train_loader, val_loader)
        metrics = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainConfig] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.config = config or TrainConfig()
        self.device = (
            torch.device(self.config.device)
            if self.config.device
            else get_device()
        )
        self.model = model.to(self.device)
        self.loss_fn = loss_fn or nn.BCELoss()
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = scheduler
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
        )
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
        }

        # TensorBoard (optional, only if installed)
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.config.log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=self.config.log_dir)
        except ImportError:
            logger.info("TensorBoard not installed. Skipping logging.")

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(batch_x).squeeze(-1)
            loss = self.loss_fn(pred, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            pred = self.model(batch_x).squeeze(-1)
            loss = self.loss_fn(pred, batch_y)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping and checkpointing.

        Returns:
            history dict with 'train_loss' and 'val_loss' per epoch
        """
        best_val_loss = float('inf')
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 'best_model.pt'
        )

        logger.info(
            f"Training on {self.device} | "
            f"epochs={self.config.epochs} | patience={self.config.patience}"
        )

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss = self._val_epoch(val_loader)
            elapsed = time.time() - t0

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            # TensorBoard logging
            if self._writer:
                self._writer.add_scalar('Loss/train', train_loss, epoch)
                self._writer.add_scalar('Loss/val', val_loss, epoch)
                if self.scheduler:
                    self._writer.add_scalar(
                        'LR', self.optimizer.param_groups[0]['lr'], epoch
                    )

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Logging
            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d}/{self.config.epochs} | "
                    f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                    f"{elapsed:.1f}s"
                )

            # Early stopping
            if self.early_stopping.step(val_loss):
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(best val_loss={best_val_loss:.6f})"
                )
                break

        # Load best model
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(
                f"Loaded best model from epoch {checkpoint['epoch']} "
                f"(val_loss={checkpoint['val_loss']:.6f})"
            )

        if self._writer:
            self._writer.close()

        return self.history

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Returns:
            dict with 'test_loss' and 'accuracy' (for classification)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        n_batches = 0

        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            pred = self.model(batch_x).squeeze(-1)
            loss = self.loss_fn(pred, batch_y)
            total_loss += loss.item()
            n_batches += 1

            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        metrics = {
            'test_loss': total_loss / max(n_batches, 1),
            'n_samples': len(preds),
        }

        # Binary classification accuracy
        if set(np.unique(labels)).issubset({0.0, 1.0}):
            binary_preds = (preds > 0.5).astype(float)
            metrics['accuracy'] = float(np.mean(binary_preds == labels))
            metrics['precision'] = float(
                np.sum((binary_preds == 1) & (labels == 1))
                / max(np.sum(binary_preds == 1), 1)
            )
            metrics['recall'] = float(
                np.sum((binary_preds == 1) & (labels == 1))
                / max(np.sum(labels == 1), 1)
            )

        logger.info(f"Test metrics: {metrics}")
        return metrics

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        """Run inference and return predictions as numpy array."""
        self.model.eval()
        all_preds = []
        for batch_x, _ in loader:
            batch_x = batch_x.to(self.device)
            pred = self.model(batch_x).squeeze(-1)
            all_preds.append(pred.cpu().numpy())
        return np.concatenate(all_preds)
