# features/extractors/cross_exchange_discrepancy_extractor.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging


logger = logging.getLogger(__name__)


class CrossExchangeDiscrepancyExtractor:
    def __init__(self, config: Optional[Dict] = None, exchange_manager=None):
        self.config = config or {}
        self.exchange_manager = exchange_manager
        self.lookback_window = self.config.get('lookback_window', 300)
        
        # Initialize self.exchanges here, but it will be updated dynamically in extract()
        self.exchanges = self.config.get('exchanges', [])
        
    def extract(self, cross_exchange_data: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        
        # Dynamically get enabled exchanges from exchange_manager if available
        if self.exchange_manager:
            self.exchanges = list(self.exchange_manager.exchanges.keys())
        # If no exchange_manager, self.exchanges remains from config (or empty list)
            
        # Filter cross_exchange_data to only include active/enabled exchanges
        active_exchanges_data = {ex_id: data for ex_id, data in cross_exchange_data.items() if ex_id in self.exchanges}
        
        features.update(self._extract_price_discrepancies(active_exchanges_data))
        features.update(self._extract_volume_dominance(active_exchanges_data))
        features.update(self._extract_latency_features(active_exchanges_data))
        features.update(self._extract_arbitrage_opportunities(active_exchanges_data))
        features.update(self._extract_correlation_features(active_exchanges_data))
        
        return features
        
    def _extract_price_discrepancies(self, data: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        
        prices = {}
        for exchange, exchange_data in data.items():
            if 'price' in exchange_data:
                prices[exchange] = exchange_data['price']
                
        if len(prices) < 2:
            return features
            
        price_values = list(prices.values())
        
        features['price_dispersion'] = np.std(price_values) / np.mean(price_values) if np.mean(price_values) > 0 else 0
        features['price_range'] = max(price_values) - min(price_values)
        features['price_range_pct'] = features['price_range'] / np.mean(price_values) if np.mean(price_values) > 0 else 0
        
        max_exchange = max(prices, key=prices.get)
        min_exchange = min(prices, key=prices.get)
        
        features['max_price_exchange'] = self.exchanges.index(max_exchange) if max_exchange in self.exchanges else -1
        features['min_price_exchange'] = self.exchanges.index(min_exchange) if min_exchange in self.exchanges else -1
        features['price_spread_basis'] = (prices[max_exchange] - prices[min_exchange]) / prices[min_exchange] if prices[min_exchange] > 0 else 0
        
        median_price = np.median(price_values)
        for exchange, price in prices.items():
            features[f'{exchange}_price_deviation'] = (price - median_price) / median_price if median_price > 0 else 0
            
        return features
        
    def _extract_volume_dominance(self, data: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        
        volumes = {}
        for exchange, exchange_data in data.items():
            if 'volume_24h' in exchange_data:
                volumes[exchange] = exchange_data['volume_24h']
                
        if not volumes:
            return features
            
        total_volume = sum(volumes.values())
        
        if total_volume > 0:
            for exchange, volume in volumes.items():
                features[f'{exchange}_volume_share'] = volume / total_volume
                
            volume_values = list(volumes.values())
            features['volume_concentration'] = self._calculate_herfindahl_index(volume_values)
            
            dominant_exchange = max(volumes, key=volumes.get)
            features['dominant_exchange'] = self.exchanges.index(dominant_exchange) if dominant_exchange in self.exchanges else -1
            features['dominant_exchange_share'] = volumes[dominant_exchange] / total_volume
            
        features['volume_disparity'] = np.std(list(volumes.values())) / np.mean(list(volumes.values())) if volumes and np.mean(list(volumes.values())) > 0 else 0
        
        return features
        
    def _extract_latency_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        
        timestamps = {}
        for exchange, exchange_data in data.items():
            if 'timestamp' in exchange_data:
                timestamps[exchange] = pd.Timestamp(exchange_data['timestamp'])
                
        if len(timestamps) < 2:
            return features
            
        timestamp_values = list(timestamps.values())
        
        earliest = min(timestamp_values)
        latest = max(timestamp_values)
        
        features['max_timestamp_difference'] = (latest - earliest).total_seconds()
        features['timestamp_dispersion'] = np.std([(t - earliest).total_seconds() for t in timestamp_values])
        
        for exchange, timestamp in timestamps.items():
            features[f'{exchange}_latency'] = (timestamp - earliest).total_seconds()
            
        price_movements = []
        for exchange, exchange_data in data.items():
            if 'price_changes' in exchange_data:
                price_movements.append(exchange_data['price_changes'])
                
        if len(price_movements) > 1:
            features['price_movement_correlation'] = self._calculate_movement_correlation(price_movements)
            
        return features
        
    def _extract_arbitrage_opportunities(self, data: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        
        orderbooks = {}
        for exchange, exchange_data in data.items():
            if 'orderbook' in exchange_data:
                orderbooks[exchange] = exchange_data['orderbook']
                
        if len(orderbooks) < 2:
            return features
            
        arbitrage_opportunities = []
        
        for ex1 in orderbooks:
            for ex2 in orderbooks:
                if ex1 != ex2:
                    opportunity = self._calculate_arbitrage_opportunity(
                        orderbooks[ex1], orderbooks[ex2], ex1, ex2
                    )
                    arbitrage_opportunities.append(opportunity)
                    
        if arbitrage_opportunities:
            features['max_arbitrage_opportunity'] = max(arbitrage_opportunities)
            features['avg_arbitrage_opportunity'] = np.mean(arbitrage_opportunities)
            features['arbitrage_opportunities_count'] = sum(1 for opp in arbitrage_opportunities if opp > 0.001)
            
        features['cross_exchange_efficiency'] = 1 - features.get('avg_arbitrage_opportunity', 0)
        
        return features
        
    def _extract_correlation_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        
        price_series = {}
        for exchange, exchange_data in data.items():
            if 'price_history' in exchange_data:
                price_series[exchange] = exchange_data['price_history']
                
        if len(price_series) < 2:
            return features
            
        correlations = []
        exchanges_list = list(price_series.keys())
        
        for i in range(len(exchanges_list)):
            for j in range(i+1, len(exchanges_list)):
                series1 = np.array(price_series[exchanges_list[i]])
                series2 = np.array(price_series[exchanges_list[j]])
                
                if len(series1) == len(series2) and len(series1) > 10:
                    corr = np.corrcoef(series1, series2)[0, 1]
                    correlations.append(corr)
                    features[f'{exchanges_list[i]}_{exchanges_list[j]}_correlation'] = corr
                    
        if correlations:
            features['avg_price_correlation'] = np.mean(correlations)
            features['min_price_correlation'] = np.min(correlations)
            features['correlation_dispersion'] = np.std(correlations)
            
        return features
        
    def _calculate_herfindahl_index(self, values: List[float]) -> float:
        if not values or sum(values) == 0:
            return 0
            
        total = sum(values)
        shares = [v / total for v in values]
        return sum(s ** 2 for s in shares)
        
    def _calculate_movement_correlation(self, price_movements: List[List[float]]) -> float:
        if len(price_movements) < 2:
            return 0
            
        min_length = min(len(pm) for pm in price_movements)
        
        if min_length < 2:
            return 0
            
        truncated = [pm[:min_length] for pm in price_movements]
        correlations = []
        
        for i in range(len(truncated)):
            for j in range(i+1, len(truncated)):
                corr = np.corrcoef(truncated[i], truncated[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
                    
        return np.mean(correlations) if correlations else 0
        
    def _calculate_arbitrage_opportunity(self, ob1: Dict, ob2: Dict, ex1: str, ex2: str) -> float:
        if not ob1.get('bids') or not ob1.get('asks') or not ob2.get('bids') or not ob2.get('asks'):
            return 0
            
        best_bid_1 = ob1['bids'][0][0] if ob1['bids'] else 0
        best_ask_1 = ob1['asks'][0][0] if ob1['asks'] else 0
        best_bid_2 = ob2['bids'][0][0] if ob2['bids'] else 0
        best_ask_2 = ob2['asks'][0][0] if ob2['asks'] else 0
        
        if best_ask_1 == 0 or best_ask_2 == 0:
            return 0
            
        profit_1_to_2 = (best_bid_2 - best_ask_1) / best_ask_1 if best_bid_2 > best_ask_1 else 0
        profit_2_to_1 = (best_bid_1 - best_ask_2) / best_ask_2 if best_bid_1 > best_ask_2 else 0
        
        return max(profit_1_to_2, profit_2_to_1)