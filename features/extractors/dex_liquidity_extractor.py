# features/extractors/dex_liquidity_extractor.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from web3 import Web3
import logging


logger = logging.getLogger(__name__)


class DEXLiquidityExtractor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.slippage_amounts = self.config.get('slippage_amounts', [100, 1000, 10000, 100000])
        self.pool_fee = self.config.get('pool_fee', 0.003)
        
    def extract(self, pool_data: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        
        reserve0 = pool_data.get('reserve0', 0)
        reserve1 = pool_data.get('reserve1', 0)
        
        if reserve0 == 0 or reserve1 == 0:
            return self._empty_features()
            
        features.update(self._extract_liquidity_metrics(reserve0, reserve1))
        features.update(self._extract_slippage_features(reserve0, reserve1))
        features.update(self._extract_concentration_features(pool_data))
        features.update(self._extract_pool_health_metrics(pool_data))
        
        return features
        
    def _extract_liquidity_metrics(self, reserve0: float, reserve1: float) -> Dict[str, float]:
        features = {}
        
        total_liquidity = np.sqrt(reserve0 * reserve1)
        features['total_liquidity'] = total_liquidity
        features['reserve_ratio'] = reserve0 / reserve1
        features['liquidity_depth'] = min(reserve0, reserve1)
        
        k_constant = reserve0 * reserve1
        features['constant_product'] = k_constant
        
        price = reserve1 / reserve0
        features['pool_price'] = price
        
        liquidity_usd = 2 * np.sqrt(reserve0 * reserve1 * price)
        features['liquidity_usd_estimate'] = liquidity_usd
        
        return features
        
    def _extract_slippage_features(self, reserve0: float, reserve1: float) -> Dict[str, float]:
        features = {}
        
        for amount in self.slippage_amounts:
            buy_slippage = self._calculate_slippage(amount, reserve0, reserve1, True)
            sell_slippage = self._calculate_slippage(amount, reserve1, reserve0, False)
            
            features[f'buy_slippage_{amount}'] = buy_slippage
            features[f'sell_slippage_{amount}'] = sell_slippage
            features[f'avg_slippage_{amount}'] = (buy_slippage + sell_slippage) / 2
            
        price_impact_gradient = self._calculate_price_impact_gradient(reserve0, reserve1)
        features['price_impact_gradient'] = price_impact_gradient
        
        return features
        
    def _extract_concentration_features(self, pool_data: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        
        tick_data = pool_data.get('tick_data', {})
        if tick_data:
            liquidity_distribution = list(tick_data.values())
            
            if liquidity_distribution:
                total_liquidity = sum(liquidity_distribution)
                
                if total_liquidity > 0:
                    normalized_dist = [l / total_liquidity for l in liquidity_distribution]
                    features['liquidity_concentration'] = 1 - stats.entropy(normalized_dist + 1e-10)
                    features['liquidity_gini'] = self._calculate_gini(liquidity_distribution)
                    
                    top_10_pct = sum(sorted(liquidity_distribution, reverse=True)[:len(liquidity_distribution)//10])
                    features['top_10_concentration'] = top_10_pct / total_liquidity
                    
        features['active_liquidity_ratio'] = pool_data.get('active_liquidity', 0) / pool_data.get('total_liquidity', 1)
        
        return features
        
    def _extract_pool_health_metrics(self, pool_data: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        
        volume_24h = pool_data.get('volume_24h', 0)
        tvl = pool_data.get('tvl', np.sqrt(pool_data.get('reserve0', 0) * pool_data.get('reserve1', 0)) * 2)
        
        if tvl > 0:
            features['volume_tvl_ratio'] = volume_24h / tvl
            features['annualized_fee_apr'] = (volume_24h * self.pool_fee * 365) / tvl
        else:
            features['volume_tvl_ratio'] = 0
            features['annualized_fee_apr'] = 0
            
        features['pool_utilization'] = min(volume_24h / (tvl * 10), 1.0) if tvl > 0 else 0
        
        price_range = pool_data.get('price_range', {})
        if price_range:
            current_price = pool_data.get('current_price', 0)
            lower_price = price_range.get('lower', 0)
            upper_price = price_range.get('upper', float('inf'))
            
            if lower_price > 0 and upper_price < float('inf') and current_price > 0:
                price_position = (current_price - lower_price) / (upper_price - lower_price)
                features['price_range_position'] = max(0, min(1, price_position))
                features['price_range_width'] = (upper_price - lower_price) / current_price
                
        return features
        
    def _calculate_slippage(self, amount_in: float, reserve_in: float, reserve_out: float, is_buy: bool) -> float:
        if reserve_in == 0 or reserve_out == 0:
            return 0
            
        amount_in_with_fee = amount_in * (1 - self.pool_fee)
        amount_out = (amount_in_with_fee * reserve_out) / (reserve_in + amount_in_with_fee)
        
        ideal_price = reserve_out / reserve_in
        actual_price = amount_out / amount_in
        
        slippage = abs(ideal_price - actual_price) / ideal_price
        
        return slippage
        
    def _calculate_price_impact_gradient(self, reserve0: float, reserve1: float) -> float:
        if reserve0 == 0 or reserve1 == 0:
            return 0
            
        test_amounts = [100, 1000, 10000]
        impacts = []
        
        for amount in test_amounts:
            impact = self._calculate_slippage(amount, reserve0, reserve1, True)
            impacts.append(impact)
            
        if len(impacts) > 1:
            x = np.log(test_amounts)
            y = np.log(np.array(impacts) + 1e-10)
            
            if np.std(x) > 0:
                gradient = np.polyfit(x, y, 1)[0]
                return gradient
                
        return 0
        
    def _calculate_gini(self, values: List[float]) -> float:
        if not values:
            return 0
            
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
    def _empty_features(self) -> Dict[str, float]:
        features = {
            'total_liquidity': 0,
            'reserve_ratio': 1,
            'liquidity_depth': 0,
            'constant_product': 0,
            'pool_price': 0,
            'liquidity_usd_estimate': 0
        }
        
        for amount in self.slippage_amounts:
            features.update({
                f'buy_slippage_{amount}': 0,
                f'sell_slippage_{amount}': 0,
                f'avg_slippage_{amount}': 0
            })
            
        return features