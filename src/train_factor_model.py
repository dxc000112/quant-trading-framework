from src.factor_selection_model import (
    fetch_universe_history,
    get_focus_symbol,
    build_focus_snapshot,
    load_factor_model_settings,
    load_stock_universe,
    pick_top_stocks,
    score_latest_cross_section,
    train_factor_selection_model,
)


MODEL_PATH = 'factor_model.pkl'


def main():
    settings = load_factor_model_settings()
    tickers = load_stock_universe()
    focus_symbol = get_focus_symbol()
    print("--- 3000-Factor Cross-Sectional Model Training ---")
    print(f"Universe: {tickers}")
    print(
        "Preset: "
        f"{settings['universe_preset']} | Lookback: {settings['lookback_period']} | "
        f"Interval: {settings['bar_interval']} | Horizon: {settings['forward_horizon']} bars | "
        f"Pick: Top {settings['top_n']} | Focus: {focus_symbol}"
    )

    price_map = fetch_universe_history(
        tickers,
        period=settings['lookback_period'],
        interval=settings['bar_interval'],
        source='auto',
        min_rows=settings['min_rows'],
    )

    if len(price_map) < 2:
        raise ValueError("Need at least two stocks with usable history to train the factor model.")

    model_bundle = train_factor_selection_model(
        price_map,
        horizon=settings['forward_horizon'],
        min_factor_count=settings['min_factor_count'],
        top_factor_count=settings['top_factor_count'],
        top_n=settings['top_n'],
        save_path=MODEL_PATH,
    )

    print("\nTraining Metrics:")
    for key, value in model_bundle['metrics'].items():
        print(f"  {key}: {value}")

    print("\nLatest Pick Snapshot:")
    latest_scores = score_latest_cross_section(model_bundle, price_map)
    print(pick_top_stocks(latest_scores, top_n=settings['top_n']).to_string())

    focus_snapshot = build_focus_snapshot(latest_scores, focus_symbol)
    if focus_snapshot is not None:
        print(f"\nFocus Symbol Snapshot ({focus_symbol}):")
        print(focus_snapshot.to_string())

    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
