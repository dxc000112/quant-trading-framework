"""
Neural network architectures for quantitative trading.

Each model includes:
  - Docstring with intuition, expected benefit, and failure mode
  - Interview Q&A annotations for technical justification

Models:
  - MetaLabelNet: MLP for meta-labeling (replaces RF in meta_model.py)
  - PriceLSTM: LSTM for sequential price/feature modeling
  - FactorCNN: 1D-CNN for automatic factor extraction
  - TemporalTransformer: Transformer encoder for time-series forecasting
  - FactorAutoEncoder: AutoEncoder for 3000+ factor dimensionality reduction
"""

import math
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════
#  MLP — Meta-Label Classifier (replaces RandomForest)
# ══════════════════════════════════════════════════════════════════════

class MetaLabelNet(nn.Module):
    """
    Feed-forward MLP for meta-labeling (binary classification).

    Replaces: sklearn.ensemble.RandomForestClassifier in meta_model.py

    Intuition:
        Hidden layers capture non-linear interactions between factors
        (e.g., RSI × Bollinger %B × VWAP distance) that tree ensembles
        treat as axis-aligned splits.

    Expected benefit:
        Better probability calibration than RF's predict_proba,
        which is known to be poorly calibrated for imbalanced data.

    Failure mode:
        With < 1000 samples, the MLP will overfit badly.
        Mitigation: Dropout + BatchNorm + early stopping.

    面试问题：为什么用 NN 替代 RF 做 Meta-Labeling？
    答：RF 是 bagging，splits 是 axis-aligned，无法捕获因子间的乘积交互；
        NN 的隐层天然适合 factor interaction；
        此外 NN 输出概率校准优于 RF 的 predict_proba。
    反问：RF 的优势？→ 不需要标准化、特征重要性可解释、不容易过拟合小数据。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) — cross-sectional feature row
        Returns:
            (batch, 1) — probability of profitable trade
        """
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════
#  LSTM — Sequential Price/Feature Modeling
# ══════════════════════════════════════════════════════════════════════

class PriceLSTM(nn.Module):
    """
    LSTM encoder → FC head for sequence classification/regression.

    Replaces: rolling window features like serial_corr in meta_model.py

    Intuition:
        LSTM's gating mechanism learns which past bars to remember
        and which to forget, automatically discovering patterns like
        mean reversion after extended runs.

    Expected benefit:
        Captures longer-range dependencies than fixed-window rolling stats.
        Hidden state acts as a compressed "market memory".

    Failure mode:
        Financial time series have extremely low SNR (signal-to-noise ratio).
        LSTM may just learn noise patterns that don't generalize.
        Mitigation: Use direction/regime labels instead of raw price prediction.

    面试问题：为什么量化交易中 LSTM 实际效果往往不好？
    答：金融时序的信噪比极低（SNR < 1）；LSTM 擅长的是序列中有明确模式的数据
        （语言、语音）；但股价 next-step prediction 几乎是随机游走。
    优化路径：不预测价格本身，预测波动率 regime 或 direction label。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        scale = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * scale, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, output_dim) — uses last hidden state
        """
        out, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        # Take the last layer's hidden state
        if self.lstm.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]
        return self.fc(last_hidden)


# ══════════════════════════════════════════════════════════════════════
#  1D-CNN — Automatic Factor Extraction
# ══════════════════════════════════════════════════════════════════════

class FactorCNN(nn.Module):
    """
    Multi-scale 1D-CNN that replaces hand-crafted rolling window factors.

    Replaces: factor_library.py's enumerate-all-windows approach

    Intuition:
        Each Conv1d kernel learns the optimal window weighting.
        A kernel of size 3 might learn [1, -2, 1] ≈ acceleration,
        while a kernel of size 5 might learn mean reversion patterns.

    Expected benefit:
        End-to-end learned factors vs. brute-force enumeration of
        (2,3,5,...,252) windows × (zscore, range_pos, etc.) transforms.
        Reduces 3000+ hand-crafted factors to ~64 learned features.

    Failure mode:
        Needs large datasets to learn meaningful kernels.
        With < 500 samples, hand-crafted factors will outperform.

    面试问题：用 1D-CNN 代替手工 rolling 因子的好处？
    答：CNN kernel 自动学习最优窗口长度和权重组合，
        比手工枚举 (2,3,5,10,20,...,252) 更高效。
    缺点：需要大量数据防止过拟合；可解释性下降。
    """

    def __init__(self, in_channels: int, output_dim: int = 1):
        super().__init__()
        # Multi-scale convolutions (short, medium, long windows)
        self.conv_short = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.conv_long = nn.Conv1d(in_channels, 32, kernel_size=9, padding=4)

        self.bn = nn.BatchNorm1d(96)  # 32 * 3 channels
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len) — features × time
        Returns:
            (batch, output_dim)
        """
        h_short = torch.relu(self.conv_short(x))
        h_mid = torch.relu(self.conv_mid(x))
        h_long = torch.relu(self.conv_long(x))

        h = torch.cat([h_short, h_mid, h_long], dim=1)  # (B, 96, T)
        h = self.bn(h)
        h = self.pool(h).squeeze(-1)  # (B, 96)
        return self.fc(h)


# ══════════════════════════════════════════════════════════════════════
#  Transformer Encoder — Time-Series Forecasting
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for intraday time-series forecasting.

    Replaces: spx_anchor/model.py's RF + Quantile GBM

    Intuition:
        Self-attention directly relates any two time steps,
        e.g., comparing "30 min ago VWAP deviation" with "current RSI"
        without needing LSTM's sequential information flow.

    Expected benefit:
        Better at capturing non-local dependencies in intraday structure
        (e.g., opening drive → lunch drift → closing magnet).

    Failure mode:
        Positional encoding assumes uniform time spacing.
        Intraday data has gaps (lunch, halts).
        Mitigation: Use relative positional encoding or learnable PE.

    面试问题：Transformer 在量化中比 LSTM 好在哪里？
    答：Self-attention 可以直接关联任意两个时间步，
        不需要像 LSTM 一样逐步传递。对于需要比较
        "30分钟前的 VWAP 偏离" 和 "当前的 RSI" 的场景更自然。
    限制：Positional encoding 在不等间距时间步上需要特殊处理。
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        output_dim: int = 1,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, output_dim) — uses last token representation
        """
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x[:, -1, :]  # Take last time step
        return self.output_head(x)


# ══════════════════════════════════════════════════════════════════════
#  AutoEncoder — Factor Dimensionality Reduction
# ══════════════════════════════════════════════════════════════════════

class FactorAutoEncoder(nn.Module):
    """
    AutoEncoder for 3000+ factor compression.

    Replaces: corrwith → top_factor_count selection in factor_selection_model.py

    Intuition:
        The encoder learns a nonlinear mapping from 3000 raw factors
        to a compact latent space. Each latent dimension represents a
        "synthetic factor" — a learned combination of raw inputs.

    Expected benefit:
        - Eliminates the look-ahead bias in corrwith-based selection
        - Captures nonlinear factor interactions that linear PCA misses
        - Latent space is reusable across different downstream tasks

    Failure mode:
        Reconstruction loss doesn't guarantee predictive utility.
        Mitigation: Add supervised signal (target_rank) as auxiliary loss.

    面试问题：为什么用 AE 而不是 PCA？
    答：PCA 只能线性降维，AE 的 encoder 可以学到非线性因子组合。
        且可以加入稀疏性约束（SAE），让合成因子更可解释。
    """

    def __init__(self, input_dim: int = 3000, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, input_dim) — raw factor vector
        Returns:
            reconstructed: (batch, input_dim)
            latent: (batch, latent_dim) — compressed representation
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent factors without reconstruction."""
        return self.encoder(x)


# ══════════════════════════════════════════════════════════════════════
#  scikit-learn Wrapper for MetaLabelNet
# ══════════════════════════════════════════════════════════════════════

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

class PyTorchMetaLabelClassifier(BaseEstimator, ClassifierMixin):
    """
    scikit-learn compatible classifier wrapper around MetaLabelNet MLP.
    
    Expected benefit:
        Drop-in replacement for RandomForestClassifier in existing
        meta-labeling pipelines (e.g. train_short.py, train_sparse.py).
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 40,
        batch_size: int = 64,
        patience: int = 10,
        val_ratio: float = 0.15,
    ):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_ratio = val_ratio
        
        self.model = None
        self.scaler = None
        self.classes_ = np.array([0.0, 1.0])
        self.input_dim = None

    def fit(self, X, y):
        # Convert pandas DataFrames or Series to numpy
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        self.input_dim = X.shape[1]
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split train/val by time to avoid future-leakage
        from src.dl.datasets import CrossSectionalDataset
        from src.dl.training import Trainer, TrainConfig
        from torch.utils.data import DataLoader
        
        if len(X_scaled) >= 20:
            val_size = int(len(X_scaled) * self.val_ratio)
            train_X, val_X = X_scaled[:-val_size], X_scaled[-val_size:]
            train_y, val_y = y[:-val_size], y[-val_size:]
        else:
            train_X, val_X = X_scaled, X_scaled
            train_y, val_y = y, y
            
        train_ds = CrossSectionalDataset(train_X, train_y)
        val_ds = CrossSectionalDataset(val_X, val_y)
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        
        self.model = MetaLabelNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
        
        config = TrainConfig(
            lr=self.lr,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            patience=self.patience,
            log_dir='.cache/tb_runs',
            checkpoint_dir='.cache/meta_ckpts',
        )
        trainer = Trainer(self.model, config=config)
        trainer.fit(train_loader, val_loader)
        
        return self

    def predict_proba(self, X):
        if self.model is None or self.scaler is None:
            raise ValueError("Classifier is not fitted yet.")
            
        if hasattr(X, 'values'):
            X = X.values
        X = X.astype(np.float32)
        
        X_scaled = self.scaler.transform(X)
        
        from src.dl.training import get_device
        device = get_device()
        self.model.to(device)
        self.model.eval()
        
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model(x_tensor).cpu().squeeze(-1).numpy()
            
        if len(preds.shape) == 0:
            preds = preds.reshape(1)
            
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 1.0 - preds
        probs[:, 1] = preds
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(float)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is not None:
            state['model_state_dict'] = {k: v.cpu() for k, v in self.model.state_dict().items()}
            del state['model']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'model_state_dict' in state and state['model_state_dict'] is not None:
            self.model = MetaLabelNet(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            )
            self.model.load_state_dict(state['model_state_dict'])
            del self.model_state_dict
