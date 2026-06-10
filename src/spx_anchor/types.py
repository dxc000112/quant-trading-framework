from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class GammaRankLevel:
    rank: int
    strike: float
    score: float
    magnet_weight: float
    pin_score: float
    gross_gex: float
    net_gex: float
    call_gex: float
    put_gex: float
    open_interest: float
    size_proxy: float
    zero_dte_oi: float
    gamma_exposure_share: float
    gamma_density: float
    oi_concentration: float
    distance_decay: float
    time_to_close_decay: float
    pin_risk_score: float


@dataclass
class AnchorStructure:
    as_of: pd.Timestamp
    spot_price: float
    call_wall: float
    put_wall: float
    long_gamma: float
    gex_pain: float
    gamma_flip: Optional[float]
    expected_move: float
    total_gex: float
    total_gross_gex: float
    front_expiry_gex: float
    gamma_slope: float
    call_concentration: float
    put_concentration: float
    gamma_regime: str
    market_regime: str
    weighted_gamma_center: float
    weighted_gamma_magnet_price: float
    strongest_magnet_strike: float
    strongest_pin_strike: float
    strongest_pin_score: float
    max_pain: float
    pin_risk_level: float
    pin_risk_score: float
    time_to_close_decay: float
    expiry_bucket_gex: Dict[str, float] = field(default_factory=dict)
    gamma_rank_levels: List[GammaRankLevel] = field(default_factory=list)
    gamma_ranks: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnchorForecast:
    as_of: pd.Timestamp
    spot_price: float
    target_price: float
    lower_bound: float
    upper_bound: float
    confidence: float
    structure: AnchorStructure
    feature_snapshot: Dict[str, float] = field(default_factory=dict)
    model_name: str = 'spx-close-anchor'
    drivers: List[str] = field(default_factory=list)
    invalidation_conditions: List[str] = field(default_factory=list)
    baselines: Dict[str, Dict[str, float]] = field(default_factory=dict)
    market_regime: str = ''
    shock_reprice: bool = False

    @property
    def forecast_low(self) -> float:
        return self.lower_bound

    @property
    def forecast_high(self) -> float:
        return self.upper_bound

    @property
    def confidence_score(self) -> float:
        return self.confidence

    @property
    def long_gamma_level(self) -> float:
        return self.structure.long_gamma

    @property
    def call_wall(self) -> float:
        return self.structure.call_wall

    @property
    def put_wall(self) -> float:
        return self.structure.put_wall

    @property
    def gex_pain(self) -> float:
        return self.structure.gex_pain

    def to_record(self) -> Dict[str, object]:
        record = {
            'as_of': self.as_of,
            'target_price': self.target_price,
            'forecast_low': self.forecast_low,
            'forecast_high': self.forecast_high,
            'long_gamma_level': self.long_gamma_level,
            'call_wall': self.call_wall,
            'put_wall': self.put_wall,
            'gex_pain': self.gex_pain,
            'weighted_gamma_magnet_price': self.structure.weighted_gamma_magnet_price,
            'strongest_magnet_strike': self.structure.strongest_magnet_strike,
            'strongest_pin_strike': self.structure.strongest_pin_strike,
            'strongest_pin_score': self.structure.strongest_pin_score,
            'market_regime': self.market_regime or self.structure.market_regime,
            'confidence_score': self.confidence_score,
            'invalidation_conditions': self.invalidation_conditions,
        }
        for key in ['pin_trust_multiplier', 'shock_score', 'deanchor_score', 'vol_convergence_score']:
            if key in self.feature_snapshot:
                record[key] = self.feature_snapshot[key]
        for idx, level in enumerate(self.structure.gamma_rank_levels[:5], start=1):
            record[f'gamma_rank_{idx}'] = level.strike
            record[f'gamma_rank_{idx}_score'] = level.score
            record[f'gamma_rank_{idx}_magnet_weight'] = level.magnet_weight
            record[f'gamma_rank_{idx}_pin_score'] = level.pin_score
        return record
