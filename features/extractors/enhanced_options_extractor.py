"""
Enhanced Options Feature Extractor with Full Statistical Analysis and Anomaly Detection
Integrates all pipeline features: microstructure, entropy, clustering, anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from scipy import stats
import logging

# Import all the options libraries
try:
    import mibian
except ImportError:
    mibian = None

try:
    from py_vollib.black_scholes import black_scholes as bs
    from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega, rho
    from py_vollib.black_scholes.implied_volatility import implied_volatility
    py_vollib_available = True
except ImportError:
    py_vollib_available = False

try:
    import QuantLib as ql
    quantlib_available = True
except ImportError:
    quantlib_available = False

# Import your existing feature extraction patterns
from features.extractors.advanced_cryptofeed_extractor import AdvancedCryptofeedExtractor
from features.extractors.advanced_tradebook_extractor import AdvancedTradebookExtractor

logger = logging.getLogger(__name__)


class EnhancedOptionsExtractor:
    """
    Complete options feature extractor with all statistical analysis,
    microstructure features, and anomaly detection preparation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.risk_free_rate = self.config.get('risk_free_rate', 0.05)
        self.dividend_yield = self.config.get('dividend_yield', 0.0)
        
        # Historical data buffers for time series analysis
        self.iv_history = defaultdict(lambda: deque(maxlen=1000))
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history = defaultdict(lambda: deque(maxlen=1000))
        self.greek_history = defaultdict(lambda: deque(maxlen=1000))
        self.orderbook_history = defaultdict(lambda: deque(maxlen=100))
        
        # Configuration for analysis windows
        self.windows = self.config.get('windows', [10, 50, 100, 500])
        self.depth_levels = self.config.get('depth_levels', [1, 5, 10, 20])
        self.volume_buckets = self.config.get('volume_buckets', [10, 100, 1000, 10000])
        
        # Initialize sub-extractors for orderbook analysis
        self.orderbook_extractor = AdvancedCryptofeedExtractor(config)
        
        # QuantLib setup if available
        if quantlib_available:
            self.calendar = ql.UnitedStates(ql.UnitedStates.Settlement)
            self.day_count = ql.Actual365Fixed()
            
    def extract(self, options_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract comprehensive options features including all stats and anomaly prep
        """
        features = {}
        
        # Extract basic options features
        features.update(self._extract_basic_features(options_data))
        
        # Extract Greeks and pricing features
        features.update(self._extract_greeks_and_pricing(options_data))
        
        # Extract orderbook microstructure features
        features.update(self._extract_orderbook_features(options_data))
        
        # Extract trade flow features
        features.update(self._extract_trade_flow_features(options_data))
        
        # Extract time series statistical features
        features.update(self._extract_time_series_features(options_data))
        
        # Extract entropy and complexity features
        features.update(self._extract_entropy_features(options_data))
        
        # Extract clustering and pattern features
        features.update(self._extract_clustering_features(options_data))
        
        # Extract cross-strike features
        features.update(self._extract_cross_strike_features(options_data))
        
        # Extract anomaly detection features
        features.update(self._extract_anomaly_features(options_data))
        
        # Update historical buffers
        self._update_history(options_data, features)
        
        return features
        
    def _extract_basic_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract basic option information and metrics"""
        features = {}
        
        # Parse option details
        symbol = data.get('symbol', '')
        option_details = self._parse_option_symbol(symbol)
        
        # Basic information
        underlying_price = data.get('underlying_price', 0)
        strike = data.get('strike', option_details.get('strike', 0))
        option_type = data.get('option_type', option_details.get('type', 'call'))
        expiry = data.get('expiry', option_details.get('expiry'))
        
        # Market prices
        bid = data.get('bid', 0)
        ask = data.get('ask', 0)
        last = data.get('last_price', 0)
        mid_price = (bid + ask) / 2 if bid and ask else last
        
        # Time calculations
        if isinstance(expiry, str):
            expiry = datetime.strptime(expiry, '%Y-%m-%d')
        tte_days = (expiry - datetime.now()).days
        tte_years = tte_days / 365.0
        
        features['underlying_price'] = underlying_price
        features['strike'] = strike
        features['time_to_expiry_days'] = tte_days
        features['time_to_expiry_years'] = tte_years
        features['option_price'] = mid_price
        features['bid'] = bid
        features['ask'] = ask
        features['last_price'] = last
        features['mid_price'] = mid_price
        
        # Spread metrics
        features['bid_ask_spread'] = ask - bid
        features['spread_bps'] = (ask - bid) / mid_price * 10000 if mid_price > 0 else 0
        features['spread_pct'] = (ask - bid) / mid_price if mid_price > 0 else 0
        
        # Moneyness
        if underlying_price > 0 and strike > 0:
            moneyness = underlying_price / strike
            features['moneyness'] = moneyness
            features['log_moneyness'] = np.log(moneyness)
            features['moneyness_pct'] = (underlying_price - strike) / strike * 100
            
            # ITM/OTM classification
            if option_type.lower() == 'call':
                features['in_the_money'] = 1 if underlying_price > strike else 0
                features['intrinsic_value'] = max(underlying_price - strike, 0)
                features['otm_distance'] = max(strike - underlying_price, 0) / underlying_price
            else:  # put
                features['in_the_money'] = 1 if underlying_price < strike else 0
                features['intrinsic_value'] = max(strike - underlying_price, 0)
                features['otm_distance'] = max(underlying_price - strike, 0) / underlying_price
                
            features['time_value'] = max(mid_price - features['intrinsic_value'], 0)
            features['time_value_ratio'] = features['time_value'] / mid_price if mid_price > 0 else 0
            
        # Volume and open interest
        features['volume'] = data.get('volume', 0)
        features['open_interest'] = data.get('open_interest', 0)
        features['volume_oi_ratio'] = features['volume'] / features['open_interest'] if features['open_interest'] > 0 else 0
        
        # Price momentum
        features['price_change_1h'] = data.get('price_change_1h', 0)
        features['price_change_24h'] = data.get('price_change_24h', 0)
        features['price_change_pct_1h'] = data.get('price_change_pct_1h', 0)
        features['price_change_pct_24h'] = data.get('price_change_pct_24h', 0)
        
        return features
        
    def _extract_greeks_and_pricing(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract Greeks and advanced pricing features"""
        features = {}
        
        S = data.get('underlying_price', 0)
        K = data.get('strike', 0)
        T = data.get('time_to_expiry_years', 0)
        r = self.risk_free_rate
        mid_price = data.get('mid_price', 0)
        option_type = data.get('option_type', 'call')
        
        if S > 0 and K > 0 and T > 0 and mid_price > 0:
            # Calculate implied volatility and Greeks
            if py_vollib_available:
                try:
                    flag = 'c' if option_type.lower() == 'call' else 'p'
                    
                    # Implied volatility
                    iv = implied_volatility(mid_price, S, K, T, r, flag)
                    features['implied_volatility'] = iv
                    features['implied_volatility_pct'] = iv * 100
                    
                    # Greeks
                    features['delta'] = delta(flag, S, K, T, r, iv)
                    features['gamma'] = gamma(flag, S, K, T, r, iv)
                    features['theta'] = theta(flag, S, K, T, r, iv) / 365  # Daily
                    features['vega'] = vega(flag, S, K, T, r, iv) / 100  # Per 1% IV
                    features['rho'] = rho(flag, S, K, T, r, iv) / 100  # Per 1% rate
                    
                    # Second-order Greeks
                    features['speed'] = self._calculate_speed(S, K, T, iv, flag)
                    features['charm'] = self._calculate_charm(S, K, T, iv, flag)
                    features['vanna'] = self._calculate_vanna(S, K, T, iv, flag)
                    features['volga'] = self._calculate_volga(S, K, T, iv, flag)
                    
                    # Dollar Greeks
                    features['dollar_delta'] = features['delta'] * S
                    features['dollar_gamma'] = features['gamma'] * S * S / 100
                    features['dollar_theta'] = features['theta']
                    features['dollar_vega'] = features['vega']
                    
                    # Greeks ratios
                    features['gamma_theta_ratio'] = abs(features['gamma'] / features['theta']) if features['theta'] != 0 else 0
                    features['delta_decay'] = features['charm'] / features['delta'] if features['delta'] != 0 else 0
                    
                except Exception as e:
                    logger.warning(f"Greeks calculation failed: {e}")
                    features['implied_volatility'] = 0
                    
        # Historical volatility comparison
        if 'historical_volatility' in data:
            hv = data['historical_volatility']
            features['historical_volatility'] = hv
            features['iv_hv_ratio'] = features.get('implied_volatility', 0) / hv if hv > 0 else 0
            features['iv_premium'] = features.get('implied_volatility', 0) - hv
            
        return features
        
    def _extract_orderbook_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract orderbook microstructure features"""
        features = {}
        
        orderbook = data.get('orderbook', {})
        if not orderbook:
            return features
            
        # Use the advanced orderbook extractor
        orderbook_features = self.orderbook_extractor.extract(orderbook)
        
        # Prefix with 'opt_ob_' for options orderbook
        for key, value in orderbook_features.items():
            features[f'opt_ob_{key}'] = value
            
        # Additional options-specific orderbook features
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if bids and asks:
            # Quote stability
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            
            # Tick analysis
            tick_size = self._estimate_tick_size(bids, asks)
            features['tick_size'] = tick_size
            features['spread_in_ticks'] = (best_ask - best_bid) / tick_size if tick_size > 0 else 0
            
            # Market maker presence
            features['quote_stuffing_indicator'] = self._detect_quote_stuffing(orderbook)
            features['market_maker_spread'] = self._estimate_market_maker_spread(bids, asks)
            
            # Options-specific imbalances
            features['put_call_bid_imbalance'] = self._calculate_put_call_imbalance(data, 'bid')
            features['put_call_ask_imbalance'] = self._calculate_put_call_imbalance(data, 'ask')
            
        return features
        
    def _extract_trade_flow_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract trade flow and execution features"""
        features = {}
        
        trades = data.get('recent_trades', [])
        if not trades:
            return features
            
        # Convert to DataFrame for easier analysis
        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            return features
            
        # Basic flow metrics
        for window in self.windows:
            recent = trades_df.tail(window)
            if len(recent) > 0:
                features[f'trade_count_{window}'] = len(recent)
                features[f'trade_volume_{window}'] = recent['volume'].sum()
                features[f'avg_trade_size_{window}'] = recent['volume'].mean()
                features[f'trade_size_std_{window}'] = recent['volume'].std()
                
                # Buy/sell imbalance
                if 'side' in recent.columns:
                    buy_volume = recent[recent['side'] == 'buy']['volume'].sum()
                    sell_volume = recent[recent['side'] == 'sell']['volume'].sum()
                    total_volume = buy_volume + sell_volume
                    
                    features[f'buy_volume_{window}'] = buy_volume
                    features[f'sell_volume_{window}'] = sell_volume
                    features[f'trade_imbalance_{window}'] = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
                    
                # VWAP
                if 'price' in recent.columns:
                    features[f'vwap_{window}'] = (recent['price'] * recent['volume']).sum() / recent['volume'].sum()
                    
        # Large trade detection
        for bucket in self.volume_buckets:
            large_trades = trades_df[trades_df['volume'] >= bucket]
            features[f'large_trade_count_{bucket}'] = len(large_trades)
            features[f'large_trade_ratio_{bucket}'] = len(large_trades) / len(trades_df) if len(trades_df) > 0 else 0
            
        # Trade clustering
        if 'timestamp' in trades_df.columns:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            time_diffs = trades_df['timestamp'].diff().dt.total_seconds()
            
            features['avg_time_between_trades'] = time_diffs.mean()
            features['trade_burst_ratio'] = (time_diffs < 1).sum() / len(time_diffs) if len(time_diffs) > 0 else 0
            features['trade_intensity'] = len(trades_df) / (trades_df['timestamp'].max() - trades_df['timestamp'].min()).total_seconds()
            
        return features
        
    def _extract_time_series_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract time series statistical features"""
        features = {}
        
        symbol = data.get('symbol', '')
        
        # IV time series analysis
        if symbol in self.iv_history and len(self.iv_history[symbol]) > 10:
            iv_series = np.array(list(self.iv_history[symbol]))
            
            # Basic statistics
            features['iv_mean'] = np.mean(iv_series)
            features['iv_std'] = np.std(iv_series)
            features['iv_skew'] = stats.skew(iv_series)
            features['iv_kurtosis'] = stats.kurtosis(iv_series)
            
            # Volatility of volatility
            iv_returns = np.diff(iv_series) / iv_series[:-1]
            features['vol_of_vol'] = np.std(iv_returns) * np.sqrt(252)
            
            # IV momentum
            if len(iv_series) >= 20:
                features['iv_momentum_5'] = iv_series[-1] - iv_series[-5]
                features['iv_momentum_20'] = iv_series[-1] - iv_series[-20]
                features['iv_trend'] = np.polyfit(range(20), iv_series[-20:], 1)[0]
                
            # IV percentile
            features['iv_percentile'] = stats.percentileofscore(iv_series, iv_series[-1])
            
            # Regime detection
            features['iv_regime'] = self._detect_volatility_regime(iv_series)
            
        # Price time series analysis
        if symbol in self.price_history and len(self.price_history[symbol]) > 10:
            price_series = np.array(list(self.price_history[symbol]))
            
            # Price momentum
            features['price_momentum_10'] = (price_series[-1] - price_series[-10]) / price_series[-10] if len(price_series) >= 10 else 0
            
            # Price volatility
            returns = np.diff(price_series) / price_series[:-1]
            features['price_volatility'] = np.std(returns) * np.sqrt(252)
            
            # Autocorrelation
            if len(returns) > 20:
                features['returns_autocorr'] = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                
        # Greeks evolution
        if symbol in self.greek_history and len(self.greek_history[symbol]) > 10:
            greeks_df = pd.DataFrame(list(self.greek_history[symbol]))
            
            # Delta evolution
            if 'delta' in greeks_df.columns:
                features['delta_change_10'] = greeks_df['delta'].iloc[-1] - greeks_df['delta'].iloc[-10]
                features['delta_stability'] = greeks_df['delta'].std()
                
            # Gamma evolution
            if 'gamma' in greeks_df.columns:
                features['gamma_mean'] = greeks_df['gamma'].mean()
                features['gamma_max'] = greeks_df['gamma'].max()
                
        return features
        
    def _extract_entropy_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract entropy and complexity measures"""
        features = {}
        
        # Price entropy
        if 'recent_trades' in data and len(data['recent_trades']) > 10:
            prices = [t['price'] for t in data['recent_trades']]
            price_changes = np.diff(prices)
            
            if len(price_changes) > 0:
                # Discretize price changes
                bins = np.histogram_bin_edges(price_changes, bins=10)
                hist, _ = np.histogram(price_changes, bins=bins)
                
                # Shannon entropy
                prob = hist / hist.sum()
                features['price_entropy'] = stats.entropy(prob + 1e-10)
                
        # Volume entropy
        if 'recent_trades' in data and len(data['recent_trades']) > 10:
            volumes = [t['volume'] for t in data['recent_trades']]
            
            # Volume distribution entropy
            vol_hist, _ = np.histogram(volumes, bins=10)
            vol_prob = vol_hist / vol_hist.sum()
            features['volume_entropy'] = stats.entropy(vol_prob + 1e-10)
            
        # Orderbook entropy
        orderbook = data.get('orderbook', {})
        if orderbook.get('bids') and orderbook.get('asks'):
            # Bid volume distribution
            bid_volumes = [b[1] for b in orderbook['bids'][:20]]
            bid_prob = np.array(bid_volumes) / np.sum(bid_volumes)
            features['bid_entropy'] = stats.entropy(bid_prob + 1e-10)
            
            # Ask volume distribution
            ask_volumes = [a[1] for a in orderbook['asks'][:20]]
            ask_prob = np.array(ask_volumes) / np.sum(ask_volumes)
            features['ask_entropy'] = stats.entropy(ask_prob + 1e-10)
            
        # Greeks entropy (distribution across strikes)
        if 'surface_data' in data:
            surface = data['surface_data']
            if surface and len(surface) > 3:
                # IV distribution entropy
                ivs = [point.get('implied_volatility', 0) for point in surface]
                if ivs:
                    iv_hist, _ = np.histogram(ivs, bins=10)
                    iv_prob = iv_hist / iv_hist.sum()
                    features['iv_surface_entropy'] = stats.entropy(iv_prob + 1e-10)
                    
        return features
        
    def _extract_clustering_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract clustering and pattern detection features"""
        features = {}
        
        # Trade clustering
        if 'recent_trades' in data and len(data['recent_trades']) > 20:
            trades = data['recent_trades']
            
            # Time-based clustering
            timestamps = [pd.Timestamp(t['timestamp']) for t in trades]
            time_diffs = np.diff([t.timestamp() for t in timestamps])
            
            if len(time_diffs) > 0:
                # Detect bursts (trades within 1 second)
                burst_threshold = 1.0
                bursts = time_diffs < burst_threshold
                
                features['burst_count'] = np.sum(bursts)
                features['burst_intensity'] = np.sum(bursts) / len(bursts)
                
                # Cluster statistics
                cluster_sizes = []
                current_cluster = 1
                
                for i, is_burst in enumerate(bursts):
                    if is_burst:
                        current_cluster += 1
                    else:
                        if current_cluster > 1:
                            cluster_sizes.append(current_cluster)
                        current_cluster = 1
                        
                if cluster_sizes:
                    features['avg_cluster_size'] = np.mean(cluster_sizes)
                    features['max_cluster_size'] = np.max(cluster_sizes)
                    
        # Price level clustering
        orderbook = data.get('orderbook', {})
        if orderbook.get('bids') and len(orderbook['bids']) > 10:
            bid_prices = [b[0] for b in orderbook['bids'][:50]]
            
            # Detect price levels with multiple orders
            price_counts = pd.Series(bid_prices).value_counts()
            features['bid_price_clustering'] = (price_counts > 1).sum() / len(price_counts)
            
        # Strike clustering (for surface analysis)
        if 'surface_data' in data and len(data['surface_data']) > 5:
            surface = data['surface_data']
            
            # Group by expiry
            by_expiry = defaultdict(list)
            for point in surface:
                by_expiry[point['expiry']].append(point)
                
            # Analyze strike distribution
            for expiry, points in by_expiry.items():
                if len(points) > 3:
                    strikes = [p['strike'] for p in points]
                    strike_diffs = np.diff(sorted(strikes))
                    
                    if len(strike_diffs) > 0:
                        features[f'strike_regularity_{expiry}'] = np.std(strike_diffs) / np.mean(strike_diffs)
                        
        return features
        
    def _extract_cross_strike_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features across different strikes and expiries"""
        features = {}
        
        if 'surface_data' not in data or not data['surface_data']:
            return features
            
        surface = data['surface_data']
        
        # Group by expiry
        by_expiry = defaultdict(list)
        for point in surface:
            by_expiry[point['expiry']].append(point)
            
        # Volatility smile/skew features
        for expiry, points in by_expiry.items():
            if len(points) >= 3:
                points_sorted = sorted(points, key=lambda x: x['strike'])
                
                strikes = [p['strike'] for p in points_sorted]
                ivs = [p.get('implied_volatility', 0) for p in points_sorted]
                
                if len(strikes) >= 3 and all(ivs):
                    # Find ATM
                    underlying = data.get('underlying_price', 0)
                    atm_idx = np.argmin(np.abs(np.array(strikes) - underlying))
                    
                    # Skew measures
                    if atm_idx > 0 and atm_idx < len(ivs) - 1:
                        features[f'skew_25d_{expiry}'] = ivs[0] - ivs[-1]  # OTM put - OTM call
                        features[f'butterfly_25d_{expiry}'] = (ivs[0] + ivs[-1]) / 2 - ivs[atm_idx]
                        
                    # Smile curvature
                    if len(strikes) >= 5:
                        # Fit polynomial to IV smile
                        poly_coeff = np.polyfit(strikes, ivs, 2)
                        features[f'smile_curvature_{expiry}'] = poly_coeff[0]  # Quadratic term
                        
                    # Risk reversal
                        features[f'risk_reversal_{expiry}'] = ivs[-1] - ivs[0]  # Call IV - Put IV
                    
        # Term structure features
        if len(by_expiry) >= 2:
            expiries_sorted = sorted(by_expiry.keys())
            
            # ATM term structure
            atm_ivs = []
            for expiry in expiries_sorted:
                points = by_expiry[expiry]
                underlying = data.get('underlying_price', 0)
                
                # Find ATM IV
                atm_point = min(points, key=lambda x: abs(x['strike'] - underlying))
                atm_ivs.append(atm_point.get('implied_volatility', 0))
                
            if len(atm_ivs) >= 2 and all(atm_ivs):
                # Term structure slope
                features['term_structure_slope'] = (atm_ivs[-1] - atm_ivs[0]) / len(atm_ivs)
                
                # Contango/backwardation
                features['vol_contango'] = 1 if atm_ivs[-1] > atm_ivs[0] else 0
                
                # Term structure volatility
                if len(atm_ivs) >= 3:
                    features['term_structure_volatility'] = np.std(atm_ivs)
                    
        # Cross-strike correlations
        features.update(self._calculate_cross_strike_correlations(by_expiry))
        
        return features
        
    def _extract_anomaly_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features specifically designed for anomaly detection"""
        features = {}
        
        symbol = data.get('symbol', '')
        
        # Price anomalies
        mid_price = data.get('mid_price', 0)
        intrinsic = data.get('intrinsic_value', 0)
        
        # Check for pricing anomalies
        if mid_price > 0:
            # Option trading below intrinsic value
            if mid_price < intrinsic:
                features['below_intrinsic'] = 1
                features['intrinsic_violation'] = (intrinsic - mid_price) / intrinsic
            else:
                features['below_intrinsic'] = 0
                features['intrinsic_violation'] = 0
                
        # Put-call parity check
        if 'put_data' in data:
            parity_features = self._check_put_call_parity(data, data['put_data'])
            features.update(parity_features)
            
        # IV anomalies
        if symbol in self.iv_history and len(self.iv_history[symbol]) > 50:
            iv_series = np.array(list(self.iv_history[symbol]))
            current_iv = features.get('implied_volatility', 0)
            
            if current_iv > 0:
                # Z-score
                iv_mean = np.mean(iv_series)
                iv_std = np.std(iv_series)
                features['iv_zscore'] = (current_iv - iv_mean) / iv_std if iv_std > 0 else 0
                
                # Extreme IV flag
                features['extreme_iv'] = 1 if abs(features['iv_zscore']) > 3 else 0
                
                # IV spike detection
                if len(iv_series) >= 10:
                    recent_mean = np.mean(iv_series[-10:-1])
                    features['iv_spike'] = (current_iv - recent_mean) / recent_mean if recent_mean > 0 else 0
                    
        # Greeks anomalies
        greeks = {
            'delta': features.get('delta', 0),
            'gamma': features.get('gamma', 0),
            'theta': features.get('theta', 0),
            'vega': features.get('vega', 0)
        }
        
        # Unusual Greeks relationships
        if greeks['gamma'] > 0 and greeks['theta'] != 0:
            # Gamma-theta ratio anomaly
            expected_ratio = abs(greeks['gamma'] / greeks['theta'])
            features['gamma_theta_anomaly'] = 1 if expected_ratio > 0.1 else 0
            
        # Orderbook anomalies
        orderbook = data.get('orderbook', {})
        if orderbook:
            # Wide spread anomaly
            spread_pct = features.get('spread_pct', 0)
            features['wide_spread_anomaly'] = 1 if spread_pct > 0.05 else 0  # 5% spread
            
            # Orderbook imbalance anomaly
            imbalance = features.get('opt_ob_imbalance_10', 0)
            features['orderbook_anomaly'] = 1 if abs(imbalance) > 0.8 else 0
            
        # Volume anomalies
        if symbol in self.volume_history and len(self.volume_history[symbol]) > 20:
            vol_series = np.array(list(self.volume_history[symbol]))
            current_vol = data.get('volume', 0)
            
            if len(vol_series) > 0 and current_vol > 0:
                vol_mean = np.mean(vol_series)
                vol_std = np.std(vol_series)
                
                features['volume_zscore'] = (current_vol - vol_mean) / vol_std if vol_std > 0 else 0
                features['unusual_volume'] = 1 if abs(features['volume_zscore']) > 3 else 0
                
        # Pattern anomalies
        features['anomaly_score'] = self._calculate_composite_anomaly_score(features)
        
        return features
        
    def _parse_option_symbol(self, symbol: str) -> Dict[str, Any]:
        """Parse option symbol like 'BTC-28MAR25-50000-C'"""
        parts = symbol.split('-')
        if len(parts) >= 4:
            return {
                'underlying': parts[0],
                'expiry': parts[1],
                'strike': float(parts[2]),
                'type': 'call' if parts[3].upper() == 'C' else 'put'
            }
        return {}
        
    def _calculate_speed(self, S: float, K: float, T: float, sigma: float, flag: str) -> float:
        """Calculate Speed (Gamma derivative)"""
        dS = S * 0.001
        gamma_up = gamma(flag, S + dS, K, T, self.risk_free_rate, sigma)
        gamma_down = gamma(flag, S - dS, K, T, self.risk_free_rate, sigma)
        return (gamma_up - gamma_down) / (2 * dS)
        
    def _calculate_charm(self, S: float, K: float, T: float, sigma: float, flag: str) -> float:
        """Calculate Charm (Delta time derivative)"""
        dT = 1 / 365
        if T > dT:
            delta_now = delta(flag, S, K, T, self.risk_free_rate, sigma)
            delta_tomorrow = delta(flag, S, K, T - dT, self.risk_free_rate, sigma)
            return (delta_tomorrow - delta_now) / dT
        return 0
        
    def _calculate_vanna(self, S: float, K: float, T: float, sigma: float, flag: str) -> float:
        """Calculate Vanna (Delta volatility derivative)"""
        dsigma = 0.01
        delta_up = delta(flag, S, K, T, self.risk_free_rate, sigma + dsigma)
        delta_down = delta(flag, S, K, T, self.risk_free_rate, sigma - dsigma)
        return (delta_up - delta_down) / (2 * dsigma)
        
    def _calculate_volga(self, S: float, K: float, T: float, sigma: float, flag: str) -> float:
        """Calculate Volga (Vega volatility derivative)"""
        dsigma = 0.01
        vega_up = vega(flag, S, K, T, self.risk_free_rate, sigma + dsigma)
        vega_down = vega(flag, S, K, T, self.risk_free_rate, sigma - dsigma)
        return (vega_up - vega_down) / (2 * dsigma * 100)
        
    def _estimate_tick_size(self, bids: List, asks: List) -> float:
        """Estimate minimum tick size from orderbook"""
        all_prices = [b[0] for b in bids[:20]] + [a[0] for a in asks[:20]]
        if len(all_prices) < 2:
            return 0.01
            
        price_diffs = []
        sorted_prices = sorted(all_prices)
        
        for i in range(1, len(sorted_prices)):
            diff = sorted_prices[i] - sorted_prices[i-1]
            if diff > 0:
                price_diffs.append(diff)
                
        return min(price_diffs) if price_diffs else 0.01
        
    def _detect_quote_stuffing(self, orderbook: Dict) -> float:
        """Detect potential quote stuffing activity"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if len(bids) < 10 or len(asks) < 10:
            return 0
            
        # Check for unusually uniform order sizes
        bid_sizes = [b[1] for b in bids[:20]]
        ask_sizes = [a[1] for a in asks[:20]]
        
        bid_unique_ratio = len(set(bid_sizes)) / len(bid_sizes)
        ask_unique_ratio = len(set(ask_sizes)) / len(ask_sizes)
        
        # Low unique ratio suggests potential stuffing
        stuffing_score = 1 - (bid_unique_ratio + ask_unique_ratio) / 2
        
        return stuffing_score
        
    def _estimate_market_maker_spread(self, bids: List, asks: List) -> float:
        """Estimate the effective market maker spread"""
        if not bids or not asks:
            return 0
            
        # Look for significant volume levels
        bid_volumes = [b[1] for b in bids[:10]]
        ask_volumes = [a[1] for a in asks[:10]]
        
        # Find levels with above-average volume (potential MM levels)
        bid_threshold = np.mean(bid_volumes) * 2
        ask_threshold = np.mean(ask_volumes) * 2
        
        mm_bid = next((b[0] for b in bids if b[1] > bid_threshold), bids[0][0])
        mm_ask = next((a[0] for a in asks if a[1] > ask_threshold), asks[0][0])
        
        return mm_ask - mm_bid
        
    def _calculate_put_call_imbalance(self, data: Dict, side: str) -> float:
        """Calculate put-call imbalance for bid or ask side"""
        if 'put_data' not in data:
            return 0
            
        call_value = data.get(side, 0)
        put_value = data['put_data'].get(side, 0)
        
        total = call_value + put_value
        if total > 0:
            return (call_value - put_value) / total
        return 0
        
    def _detect_volatility_regime(self, iv_series: np.ndarray) -> int:
        """Detect current volatility regime (0=low, 1=normal, 2=high)"""
        if len(iv_series) < 20:
            return 1
            
        current_iv = iv_series[-1]
        percentile = stats.percentileofscore(iv_series, current_iv)
        
        if percentile < 20:
            return 0  # Low vol regime
        elif percentile > 80:
            return 2  # High vol regime
        else:
            return 1  # Normal regime
            
    def _check_put_call_parity(self, call_data: Dict, put_data: Dict) -> Dict[str, float]:
        """Check put-call parity relationship"""
        features = {}
        
        S = call_data.get('underlying_price', 0)
        K = call_data.get('strike', 0)
        T = call_data.get('time_to_expiry_years', 0)
        r = self.risk_free_rate
        
        call_price = call_data.get('mid_price', 0)
        put_price = put_data.get('mid_price', 0)
        
        if S > 0 and K > 0 and T > 0:
            # C - P = S - K*e^(-rT)
            theoretical_diff = S - K * np.exp(-r * T)
            actual_diff = call_price - put_price
            
            parity_deviation = actual_diff - theoretical_diff
            
            features['put_call_parity_deviation'] = parity_deviation
            features['parity_deviation_pct'] = parity_deviation / S if S > 0 else 0
            features['parity_violation'] = 1 if abs(parity_deviation) > S * 0.01 else 0  # 1% threshold
            
            # Synthetic positions
            features['synthetic_long'] = call_price - put_price + K * np.exp(-r * T)
            features['synthetic_short'] = put_price - call_price + S
            
        return features
        
    def _calculate_cross_strike_correlations(self, by_expiry: Dict) -> Dict[str, float]:
        """Calculate correlations between different strikes"""
        features = {}
        
        # For each expiry, calculate correlations between strikes
        for expiry, points in by_expiry.items():
            if len(points) >= 5:  # Need enough strikes
                # Create price matrix
                strikes = sorted(set(p['strike'] for p in points))
                
                # Get historical prices for each strike
                price_matrix = []
                for strike in strikes:
                    strike_point = next((p for p in points if p['strike'] == strike), None)
                    if strike_point and 'price_history' in strike_point:
                        price_matrix.append(strike_point['price_history'])
                        
                if len(price_matrix) >= 2:
                    # Calculate correlation matrix
                    corr_matrix = np.corrcoef(price_matrix)
                    
                    # Average correlation
                    features[f'avg_strike_correlation_{expiry}'] = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                    
        return features
        
    def _calculate_composite_anomaly_score(self, features: Dict[str, float]) -> float:
        """Calculate composite anomaly score from multiple indicators"""
        anomaly_indicators = [
            features.get('below_intrinsic', 0),
            features.get('extreme_iv', 0),
            features.get('wide_spread_anomaly', 0),
            features.get('orderbook_anomaly', 0),
            features.get('unusual_volume', 0),
            features.get('parity_violation', 0),
            abs(features.get('iv_zscore', 0)) > 2,
            abs(features.get('volume_zscore', 0)) > 2,
            features.get('gamma_theta_anomaly', 0)
        ]
        
        # Weight and combine
        weights = [2.0, 1.5, 1.0, 1.0, 1.2, 2.0, 1.3, 1.1, 1.0]
        weighted_sum = sum(ind * w for ind, w in zip(anomaly_indicators, weights))
        
        return weighted_sum / sum(weights)
        
    def _update_history(self, data: Dict[str, Any], features: Dict[str, float]) -> None:
        """Update historical buffers"""
        symbol = data.get('symbol', '')
        
        if symbol:
            # Update IV history
            if 'implied_volatility' in features:
                self.iv_history[symbol].append(features['implied_volatility'])
                
            # Update price history
            if 'mid_price' in features:
                self.price_history[symbol].append(features['mid_price'])
                
            # Update volume history
            if 'volume' in features:
                self.volume_history[symbol].append(features['volume'])
                
            # Update Greeks history
            greek_snapshot = {
                'delta': features.get('delta', 0),
                'gamma': features.get('gamma', 0),
                'theta': features.get('theta', 0),
                'vega': features.get('vega', 0),
                'timestamp': datetime.now()
            }
            self.greek_history[symbol].append(greek_snapshot)
            
            # Update orderbook history
            if 'orderbook' in data:
                self.orderbook_history[symbol].append(data['orderbook'])
                
    def get_historical_stats(self, symbol: str) -> Dict[str, Any]:
        """Get historical statistics for a symbol"""
        stats = {}
        
        if symbol in self.iv_history:
            iv_series = np.array(list(self.iv_history[symbol]))
            stats['iv'] = {
                'mean': np.mean(iv_series),
                'std': np.std(iv_series),
                'min': np.min(iv_series),
                'max': np.max(iv_series),
                'current_percentile': stats.percentileofscore(iv_series, iv_series[-1]) if len(iv_series) > 0 else 50
            }
            
        if symbol in self.volume_history:
            vol_series = np.array(list(self.volume_history[symbol]))
            stats['volume'] = {
                'mean': np.mean(vol_series),
                'std': np.std(vol_series),
                'total': np.sum(vol_series)
            }
            
        return stats