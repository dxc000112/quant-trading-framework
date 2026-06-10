# Intelligent SPY Intraday Monitor (ML-Enhanced)

A high-frequency intraday trading monitor for **SPY** (S&P 500 ETF) that combines technical analysis (VWAP/RSI) with Machine Learning (Meta-Labeling) to filter signals and send real-time visual alerts to Discord.

## 🚀 Key Features

- **Intraday Strategy**: Monitors 1-minute candles for momentum setups (Price > VWAP + RSI > 50).
- **ML Meta-Labeling**: Uses the **Triple Barrier Method** to label historical trades and trains a **Random Forest** classifier to predict trade success.
- **Smart Filtering**: The Live Bot only alerts if the ML model predicts a high probability of success (>70%).
- **Rich Notifications**: Sends instant **Discord Alerts** containing:
  - Trade details (Price, Stop Loss, Target).
  - **Live Chart Visualization**: A "Trade Box" showing the setup context and projected barriers.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```

2. **Create a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   # venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

1. **Discord Webhook**:
   - Go to your Discord Channel Settings -> Integrations -> Webhooks -> New Webhook.
   - Copy the **Webhook URL**.

2. **Environment Variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` to include Discord Webhook and **Alpaca Credentials**:
     ```env
     DISCORD_WEBHOOK_URL=https://discord.com/...
      ALPACA_API_KEY=PK...
      ALPACA_SECRET_KEY=...
      ALPACA_BASE_URL=https://paper-api.alpaca.markets
     ```

3. **Unified Market Data Layer**:
   - The project now has an internal `market_data` package that routes symbols to different providers behind one interface.
   - **US stocks** prefer **Alpaca**, then fall back to `yfinance`.
   - Historical results are cached locally under `.cache/market_data` by default.
## 🖥️ Usage

### Mode 1: Machine Learning (Recommended)

This mode uses a trained Random Forest model to filter out false positives.

1.  **Train the Model**:
    Generates synthetic training data (or fetches history), applies Triple Barrier labeling, and saves the model to `rf_model.pkl`.
    ```bash
    python -m src.train
    ```

2.  **Start Live Inference Bot**:
    Loads the trained model and monitors SPY in real-time. Alerts only on high-confidence setups.
    ```bash
    python -m src.live_bot
    ```

### Mode 2: Standard Rule-Based
Runs the strategy based purely on VWAP and RSI logic without ML filtering.
```bash
python main.py
```

### Mode 3: 3000-Factor Stock Selection
Builds a separate cross-sectional stock-selection model for U.S. single-stock research. It is now tuned for a semiconductor watchlist and can highlight a focus name such as `TSM` while still ranking the broader U.S. semiconductor universe.

Train the model:
```bash
python -m src.train_factor_model
```

Score the latest cross-section:
```bash
python -m src.score_factor_model
```

The scorer expects a trained model artifact in the project root. On a fresh checkout, always run the training command once before scoring.

Run an out-of-sample backtest and compute Sharpe:
```bash
python -m src.backtests.backtest_factor_model
```

Configure the universe and the number of picks in `config.yaml`:
```yaml
factor_model:
  universe_preset: "semis"
  top_n: 1
  focus_symbol: "TSM"
```

## 🧠 Deep Learning Extension (Phase 3 Upgrades)

The system is upgraded with PyTorch deep learning models, offering non-linear feature projection and better probability calibration.

### 1. PyTorch MLP Classifier (`PyTorchMetaLabelClassifier`)
- **Location**: [src/dl/models.py](src/dl/models.py#L379)
- **Concept**: A scikit-learn compatible wrapper around a PyTorch Multi-Layer Perceptron (`MetaLabelNet`). It replaces the Random Forest meta-labeler, capturing high-order multiplicative factor interactions.
- **Usage**: Train via command-line by selecting `mlp` as the model type:
  ```bash
  python -m src.train_short --model-type mlp
  ```

### 2. AutoEncoder Factor Compressor (`FactorAutoEncoder`)
- **Location**: [src/dl/models.py](src/dl/models.py#L313)
- **Concept**: Compresses the high-dimensional 3000+ factor space into a 32-dimensional dense latent space. It trains self-supervised, eliminating look-ahead target leakages during feature selection.
- **Usage**: Automatically used by the stock selection training pipeline:
  ```bash
  python -m src.train_factor_model
  ```

### 3. Model Comparisons & Backtests
We have built comparative analysis scripts to benchmark deep learning models against the Random Forest baselines.

- **SPY Reversal Benchmarking** (RF Classifier vs. PyTorch MLP Classifier):
  ```bash
  python src/backtests/compare_reversal_models.py
  ```
  Generates comparative metrics and stores a report in `backtest_outputs/reversal_comparison.md`.

- **Semiconductor Stock Selection Benchmarking** (Correlation Filter vs. AutoEncoder):
  ```bash
  python src/backtests/compare_factor_selection.py
  ```
  Generates comparative metrics and stores a report in `backtest_outputs/factor_selection_comparison.md`.

## 📂 Project Structure

- **`src/`**:
  - `data_loader.py`: Thin compatibility wrapper around the unified market-data layer.
  - `market_data/`: Internal U.S.-focused market-data package with provider routing, symbol normalization, and caching.
  - `factor_library.py`: Generates a large multi-family factor library (3000+ factors).
  - `factor_selection_model.py`: Cross-sectional training and ranking pipeline for stock selection.
  - `strategy.py`: Technical indicator calculation (VWAP, RSI).
  - `labeling.py`: **Triple Barrier Method** implementation for data labeling.
  - `meta_model.py`: Feature engineering and Random Forest training logic.
  - `plotting.py`: Generates the "Trade Box" visualization charts.
  - `notifier.py`: Handles Discord Webhook integration.
  - `train.py`: Script to train and persist the ML model.
  - `live_bot.py`: The main ML-driven execution loop.
  - `train_factor_model.py`: Training entry point for the new 3000-factor stock selection model.
  - `score_factor_model.py`: Latest cross-section scoring entry point.
  - `backtests/backtest_factor_model.py`: OOS backtest entry point for Sharpe and drawdown analysis.
- **`main.py`**: Legacy rule-based monitor.
- **`rf_model.pkl`**: The saved Machine Learning model (generated after training).

## 📊 Strategy Details

- **Primary Signal**: 
  - Condition: `Close > VWAP` AND `RSI(14) > 50`.
- **Meta-Labeling (Triple Barrier)**:
  - **Profit Taking (PT)**: dynamic based on volatility.
  - **Stop Loss (SL)**: dynamic based on volatility.
  - **Time Barrier**: Discards trades that take too long to resolve.
- **Model Features**: Volatility, Log Returns, Serial Correlation, RSI.
