"""
Tests for the deep learning module (src/dl/).

Verifies:
  - Dataset creation and windowing
  - Model forward passes (shape correctness)
  - Training loop convergence on synthetic data
  - Temporal split correctness (no data leakage)
"""

import unittest
import numpy as np
import torch

from src.dl.datasets import (
    TimeSeriesDataset,
    CrossSectionalDataset,
    temporal_train_val_test_split,
    build_dataloaders,
)
from src.dl.models import (
    MetaLabelNet,
    PriceLSTM,
    FactorCNN,
    TemporalTransformer,
    FactorAutoEncoder,
)
from src.dl.training import Trainer, TrainConfig, EarlyStopping, get_device


class TestDatasets(unittest.TestCase):

    def test_timeseries_dataset_shapes(self):
        """Sliding window produces correct shapes."""
        T, F = 100, 5
        features = np.random.randn(T, F).astype(np.float32)
        labels = np.random.randint(0, 2, size=T).astype(np.float32)
        window = 20

        ds = TimeSeriesDataset(features, labels, window_size=window)
        self.assertEqual(len(ds), T - window)

        x, y = ds[0]
        self.assertEqual(x.shape, (window, F))
        self.assertEqual(y.shape, ())

    def test_timeseries_dataset_rejects_bad_window(self):
        """Window size >= data length should raise ValueError."""
        features = np.random.randn(10, 3).astype(np.float32)
        labels = np.random.randn(10).astype(np.float32)
        with self.assertRaises(ValueError):
            TimeSeriesDataset(features, labels, window_size=10)

    def test_cross_sectional_dataset_shapes(self):
        """Cross-sectional dataset returns individual rows."""
        N, F = 50, 7
        features = np.random.randn(N, F).astype(np.float32)
        labels = np.random.randn(N).astype(np.float32)

        ds = CrossSectionalDataset(features, labels)
        self.assertEqual(len(ds), N)

        x, y = ds[0]
        self.assertEqual(x.shape, (F,))
        self.assertEqual(y.shape, ())

    def test_temporal_split_no_overlap(self):
        """Train/val/test splits should not overlap and cover all data."""
        features = np.arange(100).reshape(-1, 1).astype(np.float32)
        labels = np.arange(100).astype(np.float32)

        tr_X, tr_y, va_X, va_y, te_X, te_y = temporal_train_val_test_split(
            features, labels, val_ratio=0.15, test_ratio=0.15
        )

        total = len(tr_X) + len(va_X) + len(te_X)
        self.assertEqual(total, 100)

        # Verify temporal ordering (no leakage)
        self.assertTrue(tr_X[-1] < va_X[0])
        self.assertTrue(va_X[-1] < te_X[0])

    def test_build_dataloaders_returns_correct_keys(self):
        """build_dataloaders should return train/val/test DataLoaders."""
        features = np.random.randn(200, 5).astype(np.float32)
        labels = np.random.randn(200).astype(np.float32)

        loaders = build_dataloaders(
            features, labels, window_size=10, batch_size=16,
            dataset_type='timeseries'
        )
        self.assertIn('train', loaders)
        self.assertIn('val', loaders)
        self.assertIn('test', loaders)


class TestModels(unittest.TestCase):

    def test_meta_label_net_forward(self):
        """MLP forward pass produces correct output shape."""
        model = MetaLabelNet(input_dim=7, hidden_dim=32)
        x = torch.randn(16, 7)
        out = model(x)
        self.assertEqual(out.shape, (16, 1))
        # Output should be in [0, 1] (sigmoid)
        self.assertTrue(torch.all(out >= 0) and torch.all(out <= 1))

    def test_price_lstm_forward(self):
        """LSTM forward pass produces correct output shape."""
        model = PriceLSTM(input_dim=5, hidden_dim=32, num_layers=2, output_dim=1)
        x = torch.randn(8, 20, 5)  # batch=8, seq=20, features=5
        out = model(x)
        self.assertEqual(out.shape, (8, 1))

    def test_price_lstm_bidirectional(self):
        """BiLSTM should also work."""
        model = PriceLSTM(input_dim=5, hidden_dim=32, bidirectional=True)
        x = torch.randn(4, 15, 5)
        out = model(x)
        self.assertEqual(out.shape, (4, 1))

    def test_factor_cnn_forward(self):
        """1D-CNN forward pass with multi-scale kernels."""
        model = FactorCNN(in_channels=5, output_dim=1)
        x = torch.randn(8, 5, 30)  # batch=8, channels=5, seq=30
        out = model(x)
        self.assertEqual(out.shape, (8, 1))

    def test_temporal_transformer_forward(self):
        """Transformer encoder forward pass."""
        model = TemporalTransformer(input_dim=7, d_model=32, nhead=4, num_layers=2)
        x = torch.randn(4, 20, 7)
        out = model(x)
        self.assertEqual(out.shape, (4, 1))

    def test_factor_autoencoder_forward(self):
        """AutoEncoder produces reconstruction and latent."""
        model = FactorAutoEncoder(input_dim=100, latent_dim=16)
        x = torch.randn(8, 100)
        reconstructed, latent = model(x)
        self.assertEqual(reconstructed.shape, (8, 100))
        self.assertEqual(latent.shape, (8, 16))

    def test_factor_autoencoder_encode(self):
        """Encode method returns latent only."""
        model = FactorAutoEncoder(input_dim=100, latent_dim=16)
        x = torch.randn(8, 100)
        latent = model.encode(x)
        self.assertEqual(latent.shape, (8, 16))


class TestTraining(unittest.TestCase):

    def test_early_stopping_triggers(self):
        """Early stopping should fire after patience epochs without improvement."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        losses = [1.0, 0.9, 0.89, 0.89, 0.89, 0.89]  # improves then plateaus

        stopped_at = None
        for i, loss in enumerate(losses):
            if es.step(loss):
                stopped_at = i
                break

        self.assertIsNotNone(stopped_at)
        self.assertEqual(stopped_at, 4)  # patience=3 after last improvement at index 1

    def test_device_detection(self):
        """get_device should return a valid torch.device."""
        device = get_device()
        self.assertIsInstance(device, torch.device)

    def test_trainer_fit_on_synthetic_data(self):
        """Trainer should reduce loss on a trivially learnable task."""
        # Create a simple linear classification task
        np.random.seed(42)
        N, F = 200, 4
        X = np.random.randn(N, F).astype(np.float32)
        w = np.array([1, -1, 0.5, -0.5], dtype=np.float32)
        y = (X @ w > 0).astype(np.float32)

        loaders = build_dataloaders(
            X, y, batch_size=32,
            val_ratio=0.15, test_ratio=0.15,
            dataset_type='cross_sectional',
        )

        model = MetaLabelNet(input_dim=F, hidden_dim=16)
        config = TrainConfig(
            epochs=20,
            patience=10,
            lr=1e-2,
            log_dir='/tmp/test_runs',
            checkpoint_dir='/tmp/test_ckpts',
        )
        trainer = Trainer(model, config=config)
        history = trainer.fit(loaders['train'], loaders['val'])

        # Loss should have decreased
        self.assertGreater(len(history['train_loss']), 0)
        self.assertLess(
            history['train_loss'][-1],
            history['train_loss'][0],
            "Training loss did not decrease — model failed to learn."
        )

        # Evaluate
        metrics = trainer.evaluate(loaders['test'])
        self.assertIn('test_loss', metrics)
        self.assertIn('accuracy', metrics)


if __name__ == '__main__':
    unittest.main()
