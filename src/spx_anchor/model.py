from dataclasses import dataclass, field
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src.spx_anchor.types import AnchorForecast, AnchorStructure


LABEL_COLUMNS = ['label_target_price', 'label_lower_bound', 'label_upper_bound']
NON_FEATURE_COLUMNS = {'session_date', 'as_of', 'future_close', 'future_vwap'}


@dataclass
class AnchorModelBundle:
    point_model: object
    lower_model: object
    upper_model: object
    feature_columns: List[str]
    fill_values: Dict[str, float]
    metadata: Dict[str, object] = field(default_factory=dict)


def infer_feature_columns(snapshot_panel: pd.DataFrame) -> List[str]:
    numeric_columns = snapshot_panel.select_dtypes(include=[np.number]).columns.tolist()
    excluded = set(LABEL_COLUMNS) | NON_FEATURE_COLUMNS
    return [column for column in numeric_columns if column not in excluded]


def _prepare_features(bundle: AnchorModelBundle, feature_frame: pd.DataFrame) -> pd.DataFrame:
    X = feature_frame[bundle.feature_columns].copy()
    for column, value in bundle.fill_values.items():
        if column in X.columns:
            X[column] = pd.to_numeric(X[column], errors='coerce').fillna(value)
    return X


def train_anchor_model(
    snapshot_panel: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    lower_alpha: float = 0.15,
    upper_alpha: float = 0.85,
) -> AnchorModelBundle:
    if snapshot_panel is None or snapshot_panel.empty:
        raise ValueError("snapshot_panel must contain training rows.")

    feature_columns = feature_columns or infer_feature_columns(snapshot_panel)
    if not feature_columns:
        raise ValueError("No numeric feature columns were available for model training.")

    train_frame = snapshot_panel.dropna(subset=LABEL_COLUMNS).copy()
    if train_frame.empty:
        raise ValueError("snapshot_panel does not contain usable labels.")

    fill_values = (
        train_frame[feature_columns]
        .apply(pd.to_numeric, errors='coerce')
        .median()
        .fillna(0.0)
        .to_dict()
    )

    X_train = train_frame[feature_columns].copy()
    for column, value in fill_values.items():
        X_train[column] = pd.to_numeric(X_train[column], errors='coerce').fillna(value)

    point_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    lower_model = GradientBoostingRegressor(
        loss='quantile',
        alpha=lower_alpha,
        random_state=42,
    )
    upper_model = GradientBoostingRegressor(
        loss='quantile',
        alpha=upper_alpha,
        random_state=42,
    )

    point_model.fit(X_train, train_frame['label_target_price'])
    lower_model.fit(X_train, train_frame['label_lower_bound'])
    upper_model.fit(X_train, train_frame['label_upper_bound'])

    return AnchorModelBundle(
        point_model=point_model,
        lower_model=lower_model,
        upper_model=upper_model,
        feature_columns=feature_columns,
        fill_values=fill_values,
        metadata={
            'lower_alpha': lower_alpha,
            'upper_alpha': upper_alpha,
            'training_rows': int(len(train_frame)),
            'model_name': 'rf-point-plus-quantile-gbm',
        },
    )


def predict_anchor_levels(bundle: AnchorModelBundle, feature_frame: pd.DataFrame) -> pd.DataFrame:
    if feature_frame is None or feature_frame.empty:
        raise ValueError("feature_frame must contain at least one row.")

    X = _prepare_features(bundle, feature_frame)
    prediction = pd.DataFrame(index=feature_frame.index)
    prediction['pred_target_price'] = bundle.point_model.predict(X)
    prediction['pred_lower_bound'] = bundle.lower_model.predict(X)
    prediction['pred_upper_bound'] = bundle.upper_model.predict(X)

    lower = prediction[['pred_lower_bound', 'pred_upper_bound']].min(axis=1)
    upper = prediction[['pred_lower_bound', 'pred_upper_bound']].max(axis=1)
    prediction['pred_lower_bound'] = lower
    prediction['pred_upper_bound'] = upper
    prediction['pred_target_price'] = prediction[['pred_target_price', 'pred_lower_bound', 'pred_upper_bound']].apply(
        lambda row: min(max(row['pred_target_price'], row['pred_lower_bound']), row['pred_upper_bound']),
        axis=1,
    )
    return prediction


def _build_drivers(feature_row: pd.Series, structure: AnchorStructure):
    spot_price = float(feature_row['spot_price'])
    pain_delta = spot_price - structure.gex_pain
    side = 'above' if pain_delta >= 0 else 'below'

    return [
        f"Spot is {abs(pain_delta):.1f} pts {side} GEX pain",
        f"Walls: put {structure.put_wall:.1f} / call {structure.call_wall:.1f}",
        f"Opening drive {feature_row.get('opening_drive_pct', 0.0):.2%} and VWAP delta {feature_row.get('vwap_distance', 0.0):.1f} pts",
        f"Gamma regime: {structure.gamma_regime}",
    ]


def predict_anchor_forecast(
    bundle: AnchorModelBundle,
    feature_row: pd.Series,
    structure: AnchorStructure,
) -> AnchorForecast:
    feature_frame = pd.DataFrame([feature_row])
    prediction = predict_anchor_levels(bundle, feature_frame).iloc[0]

    width = float(prediction['pred_upper_bound'] - prediction['pred_lower_bound'])
    expected_move = max(float(feature_row.get('expected_move', 0.0)), 1.0)
    confidence = 1.0 - min(width / (4.0 * expected_move), 1.0)

    numeric_snapshot = {}
    for key, value in feature_row.items():
        if isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value):
            numeric_snapshot[key] = float(value)

    return AnchorForecast(
        as_of=pd.Timestamp(feature_row['as_of']),
        spot_price=float(feature_row['spot_price']),
        target_price=float(prediction['pred_target_price']),
        lower_bound=float(prediction['pred_lower_bound']),
        upper_bound=float(prediction['pred_upper_bound']),
        confidence=float(max(0.0, min(confidence, 1.0))),
        structure=structure,
        feature_snapshot=numeric_snapshot,
        model_name=str(bundle.metadata.get('model_name', 'spx-close-anchor')),
        drivers=_build_drivers(feature_row, structure),
    )


def save_anchor_model(bundle: AnchorModelBundle, path: str):
    payload = {
        'point_model': bundle.point_model,
        'lower_model': bundle.lower_model,
        'upper_model': bundle.upper_model,
        'feature_columns': bundle.feature_columns,
        'fill_values': bundle.fill_values,
        'metadata': bundle.metadata,
    }
    joblib.dump(payload, path)


def load_anchor_model(path: str) -> AnchorModelBundle:
    payload = joblib.load(path)
    return AnchorModelBundle(
        point_model=payload['point_model'],
        lower_model=payload['lower_model'],
        upper_model=payload['upper_model'],
        feature_columns=payload['feature_columns'],
        fill_values=payload['fill_values'],
        metadata=payload.get('metadata', {}),
    )
