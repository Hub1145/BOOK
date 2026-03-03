# features/extractors/futures_open_interest_funding_extractor.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from collections import deque
import logging


logger = logging.getLogger(__name__)


class FuturesOpenInterestFundingExtractor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.lookback_periods = self.config.get('lookback_periods', [8, 24, 72, 168])
        self.funding_history = deque(maxlen=1000)
        self.oi_history = deque(maxlen=1000)
        
    def extract(self, futures_data: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        
        current_funding = futures_data.get('funding_rate', 0)
        current_oi = futures_data.get('open_interest', 0)
        mark_price = futures_data.get('mark_price', 0)
        index_price = futures_data.get('index_price', 0)
        
        self.funding_history.append(current_funding)
        self.oi_history.append(current_oi)
        
        features.update(self._extract_funding_features(current_funding))
        features.update(self._extract_oi_features(current_oi))
        features.update(self._extract_basis_features(mark_price, index_price))
        features.update(self._extract_liquidation_features(futures_data))
        features.update(self._extract_momentum_features())
        
        return features
        
    def _extract_funding_features(self, current_funding: float) -> Dict[str, float]:
        features = {}
        
        features['funding_rate'] = current_funding
        features['funding_rate_annual'] = current_funding * 3 * 365
        
        if len(self.funding_history) > 0:
            funding_array = np.array(list(self.funding_history))
            
            features['funding_mean_24h'] = np.mean(funding_array[-8:]) if len(funding_array) >= 8 else current_funding
            features['funding_std_24h'] = np.std(funding_array[-8:]) if len(funding_array) >= 8 else 0
            
            for period in self.lookback_periods:
                if len(funding_array) >= period:
                    period_data = funding_array[-period:]
                    features[f'funding_mean_{period}h'] = np.mean(period_data)
                    features[f'funding_cumulative_{period}h'] = np.sum(period_data)
                    
                    if np.std(period_data) > 0:
                        features[f'funding_zscore_{period}h'] = (current_funding - np.mean(period_data)) / np.std(period_data)
                    else:
                        features[f'funding_zscore_{period}h'] = 0
                        
        features['funding_direction'] = 1 if current_funding > 0 else -1 if current_funding < 0 else 0
        features['funding_magnitude'] = abs(current_funding)
        
        return features
        
    def _extract_oi_features(self, current_oi: float) -> Dict[str, float]:
        features = {}
        
        features['open_interest'] = current_oi
        
        if len(self.oi_history) > 1:
            oi_array = np.array(list(self.oi_history))
            
            features['oi_change_1h'] = current_oi - oi_array[-2] if len(oi_array) >= 2 else 0
            features['oi_change_pct_1h'] = (current_oi - oi_array[-2]) / oi_array[-2] if len(oi_array) >= 2 and oi_array[-2] > 0 else 0
            
            for period in self.lookback_periods:
                if len(oi_array) >= period:
                    period_start = oi_array[-period]
                    if period_start > 0:
                        features[f'oi_change_{period}h'] = current_oi - period_start
                        features[f'oi_change_pct_{period}h'] = (current_oi - period_start) / period_start
                        
            if len(oi_array) >= 24:
                features['oi_volatility_24h'] = np.std(np.diff(oi_array[-24:]))
                
        features['oi_concentration'] = self._calculate_oi_concentration(current_oi)
        
        return features
        
    def _extract_basis_features(self, mark_price: float, index_price: float) -> Dict[str, float]:
        features = {}
        
        if index_price > 0:
            basis = mark_price - index_price
            basis_pct = basis / index_price
            
            features['basis'] = basis
            features['basis_pct'] = basis_pct
            features['basis_annual'] = basis_pct * 365
            
            features['contango'] = 1 if basis > 0 else 0
            features['backwardation'] = 1 if basis < 0 else 0
            
            features['basis_abs'] = abs(basis)
            features['basis_strength'] = abs(basis_pct)
            
        return features
        
    def _extract_liquidation_features(self, futures_data: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        
        long_liquidations = futures_data.get('long_liquidations_24h', 0)
        short_liquidations = futures_data.get('short_liquidations_24h', 0)
        total_liquidations = long_liquidations + short_liquidations
        
        features['long_liquidations_24h'] = long_liquidations
        features['short_liquidations_24h'] = short_liquidations
        features['total_liquidations_24h'] = total_liquidations
        
        if total_liquidations > 0:
            features['liquidation_ratio'] = long_liquidations / total_liquidations
            features['liquidation_imbalance'] = (long_liquidations - short_liquidations) / total_liquidations
        else:
            features['liquidation_ratio'] = 0.5
            features['liquidation_imbalance'] = 0
            
        estimated_liq_levels = futures_data.get('estimated_liquidation_levels', {})
        current_price = futures_data.get('mark_price', 0)
        
        if estimated_liq_levels and current_price > 0:
            long_liq_price = estimated_liq_levels.get('long_liquidation_price', 0)
            short_liq_price = estimated_liq_levels.get('short_liquidation_price', 0)
            
            if long_liq_price > 0:
                features['distance_to_long_liq'] = (current_price - long_liq_price) / current_price
                
            if short_liq_price > 0:
                features['distance_to_short_liq'] = (short_liq_price - current_price) / current_price
                
        features['liquidation_pressure'] = self._calculate_liquidation_pressure(futures_data)
        
        return features
        
    def _extract_momentum_features(self) -> Dict[str, float]:
        features = {}
        
        if len(self.funding_history) >= 24:
            funding_array = np.array(list(self.funding_history))[-24:]
            
            funding_changes = np.diff(funding_array)
            features['funding_momentum'] = np.mean(funding_changes)
            features['funding_acceleration'] = np.diff(funding_changes).mean() if len(funding_changes) > 1 else 0
            
            positive_funding = (funding_array > 0).sum()
            features['positive_funding_ratio_24h'] = positive_funding / len(funding_array)
            
            funding_flips = np.sum(np.diff(np.sign(funding_array)) != 0)
            features['funding_flips_24h'] = funding_flips
            
        if len(self.oi_history) >= 24:
            oi_array = np.array(list(self.oi_history))[-24:]
            
            oi_changes = np.diff(oi_array)
            features['oi_momentum'] = np.mean(oi_changes)
            features['oi_trend'] = np.polyfit(range(len(oi_array)), oi_array, 1)[0]
            
        return features
        
    def _calculate_oi_concentration(self, current_oi: float) -> float:
        if len(self.oi_history) < 100:
            return 0.5
            
        oi_array = np.array(list(self.oi_history))
        percentile = np.percentile(oi_array, [10, 90])
        
        if current_oi <= percentile[0]:
            return 0
        elif current_oi >= percentile[1]:
            return 1
        else:
            return (current_oi - percentile[0]) / (percentile[1] - percentile[0])
            
    def _calculate_liquidation_pressure(self, futures_data: Dict[str, Any]) -> float:
        liquidations = futures_data.get('recent_liquidations', [])
        
        if not liquidations:
            return 0
            
        pressure_score = 0
        current_time = pd.Timestamp.now()
        
        for liq in liquidations:
            time_diff = (current_time - pd.Timestamp(liq['timestamp'])).total_seconds() / 3600
            
            if time_diff < 24:
                weight = np.exp(-time_diff / 6)
                size_normalized = liq['size'] / futures_data.get('open_interest', 1)
                pressure_score += weight * size_normalized
                
        return min(pressure_score, 1.0)