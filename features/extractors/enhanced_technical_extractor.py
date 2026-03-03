# features/extractors/enhanced_technical_extractor.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import ta
import pandas_ta as pta
from finta import TA as fta
import logging


logger = logging.getLogger(__name__)


class EnhancedTechnicalExtractor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.periods = self.config.get('periods', [5, 10, 20, 50, 100])
        
    def extract(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        if ohlcv.empty or len(ohlcv) < 100:
            return self._empty_features()
            
        features = {}
        features.update(self._extract_trend_indicators(ohlcv))
        features.update(self._extract_momentum_indicators(ohlcv))
        features.update(self._extract_volatility_indicators(ohlcv))
        features.update(self._extract_volume_indicators(ohlcv))
        features.update(self._extract_pattern_recognition(ohlcv))
        
        return features
        
    def _extract_trend_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        for period in self.periods:
            if len(df) >= period:
                sma = ta.trend.sma_indicator(df['close'], window=period)
                ema = ta.trend.ema_indicator(df['close'], window=period)
                
                features[f'sma_{period}'] = sma.iloc[-1] if not sma.empty else 0
                features[f'ema_{period}'] = ema.iloc[-1] if not ema.empty else 0
                features[f'price_sma_ratio_{period}'] = df['close'].iloc[-1] / sma.iloc[-1] if not sma.empty and sma.iloc[-1] != 0 else 1
                
        adx = ta.trend.adx(df['high'], df['low'], df['close'])
        features['adx'] = adx.iloc[-1] if not adx.empty else 0
        
        macd = ta.trend.MACD(df['close'])
        features['macd'] = macd.macd().iloc[-1] if not macd.macd().empty else 0
        features['macd_signal'] = macd.macd_signal().iloc[-1] if not macd.macd_signal().empty else 0
        features['macd_diff'] = macd.macd_diff().iloc[-1] if not macd.macd_diff().empty else 0
        
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        features['ichimoku_a'] = ichimoku.ichimoku_a().iloc[-1] if not ichimoku.ichimoku_a().empty else 0
        features['ichimoku_b'] = ichimoku.ichimoku_b().iloc[-1] if not ichimoku.ichimoku_b().empty else 0
        
        return features
        
    def _extract_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        rsi = ta.momentum.RSIIndicator(df['close'])
        features['rsi_14'] = rsi.rsi().iloc[-1] if not rsi.rsi().empty else 50
        
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        features['stoch_k'] = stoch.stoch().iloc[-1] if not stoch.stoch().empty else 50
        features['stoch_d'] = stoch.stoch_signal().iloc[-1] if not stoch.stoch_signal().empty else 50
        
        williams = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'])
        features['williams_r'] = williams.williams_r().iloc[-1] if not williams.williams_r().empty else -50
        
        roc = ta.momentum.ROCIndicator(df['close'])
        features['roc'] = roc.roc().iloc[-1] if not roc.roc().empty else 0
        
        ultimate = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close'])
        features['ultimate_oscillator'] = ultimate.ultimate_oscillator().iloc[-1] if not ultimate.ultimate_oscillator().empty else 50
        
        return features
        
    def _extract_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        bb = ta.volatility.BollingerBands(df['close'])
        features['bb_high'] = bb.bollinger_hband().iloc[-1] if not bb.bollinger_hband().empty else 0
        features['bb_low'] = bb.bollinger_lband().iloc[-1] if not bb.bollinger_lband().empty else 0
        features['bb_width'] = bb.bollinger_wband().iloc[-1] if not bb.bollinger_wband().empty else 0
        features['bb_percent'] = bb.bollinger_pband().iloc[-1] if not bb.bollinger_pband().empty else 0.5
        
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'])
        features['atr'] = atr.average_true_range().iloc[-1] if not atr.average_true_range().empty else 0
        
        keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        features['keltner_high'] = keltner.keltner_channel_hband().iloc[-1] if not keltner.keltner_channel_hband().empty else 0
        features['keltner_low'] = keltner.keltner_channel_lband().iloc[-1] if not keltner.keltner_channel_lband().empty else 0
        
        donchian = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
        features['donchian_high'] = donchian.donchian_channel_hband().iloc[-1] if not donchian.donchian_channel_hband().empty else 0
        features['donchian_low'] = donchian.donchian_channel_lband().iloc[-1] if not donchian.donchian_channel_lband().empty else 0
        
        features['historical_volatility'] = df['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) if len(df) >= 20 else 0
        
        return features
        
    def _extract_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
        features['obv'] = obv.on_balance_volume().iloc[-1] if not obv.on_balance_volume().empty else 0
        
        cmf = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'])
        features['cmf'] = cmf.chaikin_money_flow().iloc[-1] if not cmf.chaikin_money_flow().empty else 0
        
        adi = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume'])
        features['adi'] = adi.acc_dist_index().iloc[-1] if not adi.acc_dist_index().empty else 0
        
        vpt = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume'])
        features['vpt'] = vpt.volume_price_trend().iloc[-1] if not vpt.volume_price_trend().empty else 0
        
        features['volume_sma_ratio'] = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 and df['volume'].rolling(20).mean().iloc[-1] != 0 else 1
        
        return features
        
    def _extract_pattern_recognition(self, df: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        if len(df) < 50:
            return features
            
        close_prices = df['close'].values
        
        support_levels = []
        resistance_levels = []
        
        for i in range(10, len(close_prices) - 10):
            if close_prices[i] == min(close_prices[i-10:i+11]):
                support_levels.append(close_prices[i])
            if close_prices[i] == max(close_prices[i-10:i+11]):
                resistance_levels.append(close_prices[i])
                
        current_price = close_prices[-1]
        
        nearest_support = min(support_levels, key=lambda x: abs(x - current_price)) if support_levels else current_price
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price)) if resistance_levels else current_price
        
        features['distance_to_support'] = (current_price - nearest_support) / current_price
        features['distance_to_resistance'] = (nearest_resistance - current_price) / current_price
        
        price_changes = df['close'].pct_change().dropna()
        features['trend_strength'] = price_changes.rolling(20).mean().iloc[-1] / price_changes.rolling(20).std().iloc[-1] if len(price_changes) >= 20 and price_changes.rolling(20).std().iloc[-1] != 0 else 0
        
        features['fractal_dimension'] = self._calculate_fractal_dimension(close_prices)
        
        return features
        
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        if len(prices) < 50:
            return 1.5
            
        n = len(prices)
        max_price = np.max(prices)
        min_price = np.min(prices)
        
        if max_price == min_price:
            return 1.5
            
        normalized = (prices - min_price) / (max_price - min_price)
        
        cumsum = np.cumsum(normalized - normalized[0])
        R = np.max(cumsum) - np.min(cumsum)
        S = np.std(prices)
        
        if S == 0:
            return 1.5
            
        return 2 - np.log(R/S) / np.log(n/2)
        
    def _empty_features(self) -> Dict[str, float]:
        features = {}
        for period in self.periods:
            features.update({
                f'sma_{period}': 0,
                f'ema_{period}': 0,
                f'price_sma_ratio_{period}': 1
            })
        features.update({
            'adx': 0, 'macd': 0, 'macd_signal': 0, 'macd_diff': 0,
            'rsi_14': 50, 'stoch_k': 50, 'stoch_d': 50,
            'bb_high': 0, 'bb_low': 0, 'bb_width': 0,
            'atr': 0, 'obv': 0, 'cmf': 0
        })
        return features