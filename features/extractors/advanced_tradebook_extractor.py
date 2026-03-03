# features/extractors/advanced_tradebook_extractor.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from collections import defaultdict
from scipy import stats
import logging


logger = logging.getLogger(__name__)


class AdvancedTradebookExtractor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.windows = self.config.get('windows', [10, 50, 100, 500])
        self.volume_buckets = self.config.get('volume_buckets', [0.1, 1.0, 10.0, 100.0])
        
    def extract(self, trades: pd.DataFrame) -> Dict[str, float]:
        if trades.empty:
            return self._empty_features()

        # Ensure price and volume columns are numeric
        for col in ['price', 'volume']:
            if col in trades.columns:
                # Convert to numeric, coercing errors to NaN
                trades[col] = pd.to_numeric(trades[col], errors='coerce')
                # Fill NaN values with 0.0 to prevent arithmetic errors later
                trades[col] = trades[col].fillna(0.0)
        
        # After converting and filling, if price or volume are still somehow problematic,
        # or if other critical columns are missing, return empty features.
        # This also implicitly handles cases where trades.empty was true initially.
        if trades['price'].isnull().any() or trades['volume'].isnull().any() or trades.empty:
            return self._empty_features()
            
        features = {}
        features.update(self._extract_flow_features(trades))
        features.update(self._extract_entropy_features(trades))
        features.update(self._extract_clustering_features(trades))
        features.update(self._extract_signature_features(trades))
        features.update(self._extract_microstructure_flow(trades))
        
        return features
        
    def _extract_flow_features(self, trades: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        for window in self.windows:
            recent_trades = trades.tail(window)
            
            buy_volume = recent_trades[recent_trades['side'] == 'buy']['volume'].sum()
            sell_volume = recent_trades[recent_trades['side'] == 'sell']['volume'].sum()
            total_volume = buy_volume + sell_volume
            
            features[f'buy_volume_{window}'] = buy_volume
            features[f'sell_volume_{window}'] = sell_volume
            features[f'volume_imbalance_{window}'] = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
            features[f'trade_count_{window}'] = len(recent_trades)
            
            if len(recent_trades) > 0:
                features[f'avg_trade_size_{window}'] = recent_trades['volume'].mean()
                features[f'trade_size_std_{window}'] = recent_trades['volume'].std()
                features[f'vwap_{window}'] = (recent_trades['price'] * recent_trades['volume']).sum() / recent_trades['volume'].sum()
                
        return features
        
    def _extract_entropy_features(self, trades: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        for window in self.windows:
            recent_trades = trades.tail(window)
            
            if len(recent_trades) > 1:
                price_changes = recent_trades['price'].pct_change().dropna()
                volume_dist = recent_trades['volume'] / recent_trades['volume'].sum()
                
                features[f'price_entropy_{window}'] = stats.entropy(np.histogram(price_changes, bins=10)[0] + 1e-10)
                features[f'volume_entropy_{window}'] = stats.entropy(volume_dist + 1e-10)
                
                side_counts = recent_trades['side'].value_counts()
                side_probs = side_counts / len(recent_trades)
                features[f'side_entropy_{window}'] = stats.entropy(side_probs + 1e-10)
                
        return features
        
    def _extract_clustering_features(self, trades: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        if len(trades) < 10:
            return features
            
        time_diffs = trades['timestamp'].diff().dt.total_seconds().dropna()
        
        features['avg_time_between_trades'] = time_diffs.mean()
        features['std_time_between_trades'] = time_diffs.std()
        features['burst_ratio'] = (time_diffs < time_diffs.quantile(0.1)).sum() / len(time_diffs)
        
        for bucket in self.volume_buckets:
            large_trades = trades[trades['volume'] >= bucket]
            features[f'large_trade_ratio_{bucket}'] = len(large_trades) / len(trades)
            
            if len(large_trades) > 1:
                large_time_diffs = large_trades['timestamp'].diff().dt.total_seconds().dropna()
                features[f'large_trade_clustering_{bucket}'] = 1 / (large_time_diffs.mean() + 1) if len(large_time_diffs) > 0 else 0
                
        return features
        
    def _extract_signature_features(self, trades: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        if len(trades) < 20:
            return features
            
        buy_indicator = (trades['side'] == 'buy').astype(int) * 2 - 1
        cumulative_imbalance = buy_indicator.cumsum()
        
        x = np.arange(len(cumulative_imbalance))
        slope, intercept, r_value, _, _ = stats.linregress(x, cumulative_imbalance)
        
        features['signature_slope'] = slope
        features['signature_r_squared'] = r_value ** 2
        features['signature_final_imbalance'] = cumulative_imbalance.iloc[-1]
        
        features['signature_max_excursion'] = cumulative_imbalance.max()
        features['signature_min_excursion'] = cumulative_imbalance.min()
        features['signature_range'] = features['signature_max_excursion'] - features['signature_min_excursion']
        
        return features
        
    def _extract_microstructure_flow(self, trades: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        if len(trades) < 10:
            return features
            
        features['kyle_lambda'] = self._estimate_kyle_lambda(trades)
        features['price_impact'] = self._estimate_price_impact(trades)
        features['realized_spread'] = self._estimate_realized_spread(trades)
        
        tick_sizes = trades['price'].diff().abs().dropna()
        tick_sizes = tick_sizes[tick_sizes > 0]
        
        if len(tick_sizes) > 0:
            features['effective_tick_size'] = tick_sizes.min()
            features['tick_size_ratio'] = tick_sizes.std() / tick_sizes.mean() if tick_sizes.mean() > 0 else 0
            
        volume_weighted_prices = []
        for i in range(0, len(trades), 10):
            chunk = trades.iloc[i:i+10]
            if len(chunk) > 0 and chunk['volume'].sum() > 0:
                vwap = (chunk['price'] * chunk['volume']).sum() / chunk['volume'].sum()
                volume_weighted_prices.append(vwap)
                
        if len(volume_weighted_prices) > 1:
            features['vwap_volatility'] = np.std(volume_weighted_prices)
            
        return features
        
    def _estimate_kyle_lambda(self, trades: pd.DataFrame) -> float:
        if len(trades) < 20:
            return 0
            
        signed_volume = trades['volume'].values * ((trades['side'] == 'buy').astype(int) * 2 - 1)
        price_changes = trades['price'].diff().fillna(0).values
        
        if np.std(signed_volume) > 0:
            return np.cov(price_changes[1:], signed_volume[:-1])[0, 1] / np.var(signed_volume[:-1])
        return 0
        
    def _estimate_price_impact(self, trades: pd.DataFrame) -> float:
        if len(trades) < 10:
            return 0
            
        large_trades = trades[trades['volume'] > trades['volume'].quantile(0.9)]
        
        if len(large_trades) < 2:
            return 0
            
        impacts = []
        for idx in large_trades.index[:-1]:
            pos = trades.index.get_loc(idx)
            if pos < len(trades) - 5:
                pre_price = trades.iloc[max(0, pos-5):pos]['price'].mean()
                post_price = trades.iloc[pos+1:pos+6]['price'].mean()
                
                if pre_price > 0:
                    if large_trades.loc[idx, 'side'] == 'buy':
                        impact = (post_price - pre_price) / pre_price
                    else:
                        impact = (pre_price - post_price) / pre_price
                    impacts.append(impact)
                    
        return np.mean(impacts) if impacts else 0
        
    def _estimate_realized_spread(self, trades: pd.DataFrame) -> float:
        if len(trades) < 20:
            return 0
            
        spreads = []
        for i in range(10, len(trades) - 10):
            current_price = trades.iloc[i]['price']
            future_price = trades.iloc[i+10]['price']
            
            if trades.iloc[i]['side'] == 'buy':
                spread = 2 * (future_price - current_price) / current_price
            else:
                spread = 2 * (current_price - future_price) / current_price
                
            spreads.append(spread)
            
        return np.mean(spreads) if spreads else 0
        
    def _empty_features(self) -> Dict[str, float]:
        features = {}
        for window in self.windows:
            features.update({
                f'buy_volume_{window}': 0,
                f'sell_volume_{window}': 0,
                f'volume_imbalance_{window}': 0,
                f'trade_count_{window}': 0,
                f'avg_trade_size_{window}': 0,
                f'trade_size_std_{window}': 0,
                f'vwap_{window}': 0,
                f'price_entropy_{window}': 0,
                f'volume_entropy_{window}': 0,
                f'side_entropy_{window}': 0
            })
        return features