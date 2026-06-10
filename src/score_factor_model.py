from pathlib import Path

from src.factor_selection_model import (
    build_focus_snapshot,
    fetch_universe_history,
    get_focus_symbol,
    load_factor_model,
    load_factor_model_settings,
    load_stock_universe,
    pick_top_stocks,
    score_latest_cross_section,
)


MODEL_PATH = 'factor_model.pkl'


def resolve_model_path(model_path=MODEL_PATH):
    path = Path(model_path)
    if path.exists():
        return str(path)

    raise SystemExit(
        "Missing trained model artifact 'factor_model.pkl'. "
        "Run `python -m src.train_factor_model` first, then rerun the scorer."
    )


def main():
    settings = load_factor_model_settings()
    model_bundle = load_factor_model(resolve_model_path(MODEL_PATH))
    tickers = load_stock_universe()
    focus_symbol = get_focus_symbol()

    print("--- Latest 3000-Factor Pick ---")
    print(f"Universe preset: {settings['universe_preset']}")
    print(f"Universe size: {len(tickers)} | Pick size: {settings['top_n']} | Focus: {focus_symbol}")

    price_map = fetch_universe_history(
        tickers,
        period=settings['lookback_period'],
        interval=settings['bar_interval'],
        source='auto',
        min_rows=settings['min_rows'],
    )

    scores = score_latest_cross_section(model_bundle, price_map)
    print(pick_top_stocks(scores, top_n=settings['top_n']).to_string())

    focus_snapshot = build_focus_snapshot(scores, focus_symbol)
    if focus_snapshot is not None:
        print(f"\nFocus Symbol Snapshot ({focus_symbol}):")
        print(focus_snapshot.to_string())


if __name__ == "__main__":
    main()
