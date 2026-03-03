# features/extractors/advanced_cryptofeed_extractor.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging


logger = logging.getLogger(__name__)


class AdvancedCryptofeedExtractor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.depth_levels = self.config.get('depth_levels', [1, 5, 10, 20, 50])
        self.volume_buckets = self.config.get('volume_buckets', [0.1, 0.5, 1.0, 5.0, 10.0])
        
    def extract(self, orderbook: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return self._empty_features()
            
        features.update(self._extract_depth_features(bids, asks))
        features.update(self._extract_imbalance_features(bids, asks))
        features.update(self._extract_shape_features(bids, asks))
        features.update(self._extract_liquidity_features(bids, asks))
        features.update(self._extract_microstructure_features(bids, asks))
        
        return features
        
    def _extract_depth_features(self, bids: List, asks: List) -> Dict[str, float]:
        features = {}
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        
        for level in self.depth_levels:
            bid_depth = min(len(bids), level)
            ask_depth = min(len(asks), level)
            
            bid_volume = sum(float(b[1]) for b in bids[:bid_depth])
            ask_volume = sum(float(a[1]) for a in asks[:ask_depth])
            
            features[f'bid_volume_{level}'] = bid_volume
            features[f'ask_volume_{level}'] = ask_volume
            features[f'total_volume_{level}'] = bid_volume + ask_volume
            
            if bid_depth > 0 and ask_depth > 0:
                bid_price_range = abs(float(bids[bid_depth-1][0]) - best_bid) / mid_price if mid_price else 0
                ask_price_range = abs(float(asks[ask_depth-1][0]) - best_ask) / mid_price if mid_price else 0
                features[f'bid_price_range_{level}'] = bid_price_range
                features[f'ask_price_range_{level}'] = ask_price_range
                
        return features
        
    def _extract_imbalance_features(self, bids: List, asks: List) -> Dict[str, float]:
        features = {}
        
        for level in self.depth_levels:
            bid_volume = sum(float(b[1]) for b in bids[:level])
            ask_volume = sum(float(a[1]) for a in asks[:level])
            total = bid_volume + ask_volume
            
            if total > 0:
                features[f'imbalance_{level}'] = (bid_volume - ask_volume) / total
                features[f'bid_ask_ratio_{level}'] = bid_volume / ask_volume if ask_volume > 0 else 10.0
            else:
                features[f'imbalance_{level}'] = 0
                features[f'bid_ask_ratio_{level}'] = 1.0
                
        return features
        
    def _extract_shape_features(self, bids: List, asks: List) -> Dict[str, float]:
        features = {}
        
        bid_volumes = [b[1] for b in bids[:20]]
        ask_volumes = [a[1] for a in asks[:20]]
        
        features['bid_volume_mean'] = np.mean(bid_volumes) if bid_volumes else 0
        features['bid_volume_std'] = np.std(bid_volumes) if bid_volumes else 0
        features['bid_volume_skew'] = self._calculate_skew(bid_volumes)
        features['bid_volume_kurtosis'] = self._calculate_kurtosis(bid_volumes)
        
        features['ask_volume_mean'] = np.mean(ask_volumes) if ask_volumes else 0
        features['ask_volume_std'] = np.std(ask_volumes) if ask_volumes else 0
        features['ask_volume_skew'] = self._calculate_skew(ask_volumes)
        features['ask_volume_kurtosis'] = self._calculate_kurtosis(ask_volumes)
        
        bid_gradients = np.gradient(bid_volumes) if len(bid_volumes) > 1 else [0]
        ask_gradients = np.gradient(ask_volumes) if len(ask_volumes) > 1 else [0]
        
        features['bid_gradient_mean'] = np.mean(bid_gradients)
        features['ask_gradient_mean'] = np.mean(ask_gradients)
        
        return features
        
    def _extract_liquidity_features(self, bids: List, asks: List) -> Dict[str, float]:
        features = {}
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        spread = best_ask - best_bid if best_bid and best_ask else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        
        features['spread'] = spread
        features['spread_bps'] = (spread / mid_price * 10000) if mid_price else 0
        
        for pct in [0.1, 0.25, 0.5, 1.0]:
            price_up = mid_price * (1 + pct/100) if mid_price else 0
            price_down = mid_price * (1 - pct/100) if mid_price else 0
            
            volume_to_move_up = sum(float(a[1]) for a in asks if float(a[0]) <= price_up)
            volume_to_move_down = sum(float(b[1]) for b in bids if float(b[0]) >= price_down)
            
            features[f'volume_to_move_{int(pct*100)}bps_up'] = volume_to_move_up
            features[f'volume_to_move_{int(pct*100)}bps_down'] = volume_to_move_down
            
        return features
        
    def _extract_microstructure_features(self, bids: List, asks: List) -> Dict[str, float]:
        features = {}
        
        bid_prices = [float(b[0]) for b in bids[:50]]
        ask_prices = [float(a[0]) for a in asks[:50]]
        
        if len(bid_prices) > 1:
            bid_price_diffs = np.diff(sorted(bid_prices, reverse=True))
            features['bid_tick_variance'] = np.var(bid_price_diffs) if len(bid_price_diffs) > 0 else 0
            features['bid_tick_mean'] = np.mean(bid_price_diffs) if len(bid_price_diffs) > 0 else 0
            
        if len(ask_prices) > 1:
            ask_price_diffs = np.diff(sorted(ask_prices))
            features['ask_tick_variance'] = np.var(ask_price_diffs) if len(ask_price_diffs) > 0 else 0
            features['ask_tick_mean'] = np.mean(ask_price_diffs) if len(ask_price_diffs) > 0 else 0
            
        bid_weighted_price = sum(float(b[0]) * float(b[1]) for b in bids[:20]) / sum(float(b[1]) for b in bids[:20]) if bids else 0
        ask_weighted_price = sum(float(a[0]) * float(a[1]) for a in asks[:20]) / sum(float(a[1]) for a in asks[:20]) if asks else 0
        
        features['bid_weighted_price'] = bid_weighted_price
        features['ask_weighted_price'] = ask_weighted_price
        features['weighted_mid_price'] = (bid_weighted_price + ask_weighted_price) / 2 if bid_weighted_price and ask_weighted_price else 0
        
        return features
        
    def _calculate_skew(self, values: List[float]) -> float:
        if len(values) < 3:
            return 0
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0
        return np.mean(((arr - mean) / std) ** 3)
        
    def _calculate_kurtosis(self, values: List[float]) -> float:
        if len(values) < 4:
            return 0
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0
        return np.mean(((arr - mean) / std) ** 4) - 3
        
    def _empty_features(self) -> Dict[str, float]:
        features = {}
        for level in self.depth_levels:
            features.update({
                f'bid_volume_{level}': 0,
                f'ask_volume_{level}': 0,
                f'total_volume_{level}': 0,
                f'bid_price_range_{level}': 0,
                f'ask_price_range_{level}': 0,
                f'imbalance_{level}': 0,
                f'bid_ask_ratio_{level}': 1.0
            })
        return features