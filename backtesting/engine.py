"""Backtesting engine for the trade & orderbook pipeline"""

import asyncio
import logging
import datetime as dt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import json
import pickle
from dataclasses import dataclass, field
from enum import Enum
import re
import io
from dateutil.parser import isoparse
from scipy.stats import skew, kurtosis
import gzip
import math # Added for advanced diagnostics

from core.exchange_interface import (
    NormalizedOrderBook, NormalizedTrade, NormalizedTicker, OrderBookLevel
)
from features.extractors.advanced_cryptofeed_extractor import AdvancedCryptofeedExtractor
from features.extractors.advanced_tradebook_extractor import AdvancedTradebookExtractor
from features.extractors.enhanced_technical_extractor import EnhancedTechnicalExtractor
from features.extractors.dex_liquidity_extractor import DEXLiquidityExtractor
from features.extractors.futures_open_interest_funding_extractor import FuturesOpenInterestFundingExtractor
from features.extractors.cross_exchange_discrepancy_extractor import CrossExchangeDiscrepancyExtractor
from features.extractors.enhanced_options_extractor import EnhancedOptionsExtractor # NEW

from detection.system import DetectionSystem
from core.gdrive_utils import GoogleDriveAPI

from core.logging_config import get_logger

logger = get_logger(__name__)
backtest_data_logger = get_logger("backtest_data")

# Utility functions for parsing CSV content from bytes
def _read_csv_from_bytes(content: bytes) -> pd.DataFrame:
    """Reads CSV content from bytes, handling gzip compression."""
    try:
        # Try to decompress as gzip first
        with gzip.open(io.BytesIO(content), 'rt') as f:
            return pd.read_csv(f)
    except Exception:
        # If not gzip, try reading directly as plain text
        return pd.read_csv(io.BytesIO(content))

def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names and data types of a DataFrame."""
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl in {"timestamp", "time", "ts", "date"}: ren[c] = "timestamp"
        elif cl in {"qty", "quantity", "size", "volume", "amount"}: ren[c] = "volume"
        elif cl in {"px", "prc"}: ren[c] = "price"
    if ren: df = df.rename(columns=ren)

    if "timestamp" not in df.columns:
        # If no timestamp column was found or renamed, add one with NaT
        df["timestamp"] = pd.NaT
        logger.warning("No timestamp column found during normalization. Adding a 'timestamp' column with NaT values.")
    
    # Ensure timestamp column is always datetime, coercing errors
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True, errors="coerce")
    
    for k in ("price", "volume"):
        if k in df.columns: df[k] = pd.to_numeric(df[k], errors="coerce")
    
    # Attempt to parse 'bids' and 'asks' columns if they exist and are strings
    for col in ['bids', 'asks']:
        if col in df.columns and df[col].apply(lambda x: isinstance(x, str)).any():
            try:
                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from column '{col}' during normalization.")
    return df


class DataSource(Enum):
    """Supported data sources for backtesting"""
    CSV = "csv"
    PARQUET = "parquet"
    PICKLE = "pickle"
    JSON = "json"
    DATABASE = "database"
    LIVE_RECORDING = "live_recording"
    LIVE_API = "live_api"
    GDRIVE = "gdrive"


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    pump_start_date: datetime
    pump_end_date: datetime
    control_start_date: datetime
    control_end_date: datetime

    pump_start_time_str: str = "00:00:00"
    pump_end_time_str: str = "23:59:59"
    control_start_time_str: str = "00:00:00"
    control_end_time_str: str = "23:59:59"
    
    pump_event_definitions: List[Tuple[str, datetime]] = field(default_factory=list)
    control_event_definitions: List[Tuple[str, datetime]] = field(default_factory=list)
    exchanges: List[str] = field(default_factory=list)
    data_source: DataSource = DataSource.GDRIVE
    data_path: str = "./historical_data"
    gdrive_api_key: Optional[str] = None
    gdrive_orderbook_root_id: Optional[str] = None
    gdrive_orderbook_binance_root_id: Optional[str] = None
    gdrive_tradebook_root_id: Optional[str] = None
    data_type: str = "all"
    
    # Replay settings
    replay_speed: float = 1.0
    orderbook_depth: int = 50
    trade_history_size: int = 1000
    
    # Feature extraction settings
    enable_features: bool = True
    feature_config: Dict[str, Any] = field(default_factory=dict)
    
    # Anomaly detection settings
    enable_anomaly_detection: bool = True
    detection_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    track_performance: bool = True
    save_results: bool = True
    results_path: str = "./backtest_results"
    output: Dict[str, Any] = field(default_factory=dict)

    # Event-level aggregation settings
    enable_event_aggregation: bool = False
    event_windows: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "pre_event": (-30, 0),  # -30 minutes to 0 minutes before event
        "post_event": (0, 15)   # 0 minutes to +15 minutes after event
    })
    features_to_aggregate: List[str] = field(default_factory=list)
    detectors_to_aggregate: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    config: BacktestConfig
    start_time: datetime
    end_time: datetime
    
    # Event counts
    total_orderbooks: int = 0
    total_trades: int = 0
    total_anomalies: int = 0
    
    # Performance metrics
    processing_time_ms: List[float] = field(default_factory=list)
    feature_extraction_time_ms: List[float] = field(default_factory=list)
    anomaly_detection_time_ms: List[float] = field(default_factory=list)
    
    # Detected patterns
    anomaly_timeline: List[Dict[str, Any]] = field(default_factory=list)
    feature_statistics: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    
    # Custom metrics from strategies
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    event_aggregated_stats: List[Dict[str, Any]] = field(default_factory=list)


# --- Advanced Diagnostics Constants and Helper Functions ---

# Features to aggregate for moments, quantiles, and tail ratios
FEATURES_ALL = [
    # depth volumes
    "cf_bid_volume_1", "cf_ask_volume_1", "cf_total_volume_1",
    "cf_bid_volume_5", "cf_ask_volume_5", "cf_total_volume_5",
    "cf_bid_volume_10", "cf_ask_volume_10", "cf_total_volume_10",
    "cf_bid_volume_20", "cf_ask_volume_20", "cf_total_volume_20",
    "cf_bid_volume_50", "cf_ask_volume_50", "cf_total_volume_50",

    # price ranges
    "cf_bid_price_range_1", "cf_ask_price_range_1",
    "cf_bid_price_range_5", "cf_ask_price_range_5",
    "cf_bid_price_range_10", "cf_ask_price_range_10",
    "cf_bid_price_range_20", "cf_ask_price_range_20",
    "cf_bid_price_range_50", "cf_ask_price_range_50",

    # imbalance & ratios
    "cf_imbalance_1", "cf_imbalance_5", "cf_imbalance_10",
    "cf_imbalance_20", "cf_imbalance_50",
    "cf_bid_ask_ratio_1", "cf_bid_ask_ratio_5",
    "cf_bid_ask_ratio_10", "cf_bid_ask_ratio_20", "cf_bid_ask_ratio_50",
    # Add other feature prefixes as needed, e.g., 'tb_', 'opt_', 'dex_', etc.
    "tb_buy_volume_10", "tb_sell_volume_10", "tb_volume_imbalance_10",
    "tb_avg_trade_size_10", "tb_vwap_10",
    "cx_price_dispersion", "cx_price_spread_basis",
    "opt_implied_volatility", "opt_delta", "opt_gamma",
    "fut_funding_rate", "fut_open_interest", "fut_basis",
]

# Quantiles to compute
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

# Features for cross-correlation matrix
CORR_FEATURES = [
    "cf_bid_volume_1", "cf_ask_volume_1", "cf_total_volume_1",
    "cf_imbalance_1",
    "cf_bid_volume_5",
    "tb_buy_volume_10", "tb_sell_volume_10",
]

# Features for in-window time-structure metrics
TIME_FEATURES = [
    "cf_bid_volume_1", "cf_ask_volume_1",
    "cf_total_volume_1", "cf_imbalance_1",
    "tb_vwap_10",
]

# Anomaly detectors to aggregate statistics for
DETECTORS = [
    "pyod_iforest",
    "order_skew",
    "changepoint",
    "hmm_detect",
    "matrix_profile",
    "level_shift",
]


def moment_stats(df_window: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    out = {}
    for col in feature_cols:
        if col not in df_window.columns:
            continue
        # Convert to numeric, coercing errors, then drop NaNs
        x = pd.to_numeric(df_window[col], errors='coerce').dropna()
        if x.empty:
            continue

        out[f"{col}_mean"] = float(x.mean())
        out[f"{col}_std"]  = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        out[f"{col}_min"]  = float(x.min())
        out[f"{col}_max"]  = float(x.max())

        # skew & kurtosis (Fisher definition, unbiased-ish)
        xm = x - x.mean()
        denom = (xm**2).sum()
        if len(x) > 3 and denom > 0:
            m2 = denom / len(x)
            m3 = (xm**3).sum() / len(x)
            m4 = (xm**4).sum() / len(x)
            skew_val = m3 / (m2 ** 1.5)
            kurt_val = m4 / (m2 ** 2) - 3.0
        else:
            skew_val, kurt_val = 0.0, 0.0

        out[f"{col}_skew"]    = float(skew_val)
        out[f"{col}_kurtosis"] = float(kurt_val)
    return out


def quantile_stats(df_window: pd.DataFrame, feature_cols: List[str], quantiles: List[float]=QUANTILES) -> Dict[str, float]:
    out = {}
    for col in feature_cols:
        if col not in df_window.columns:
            continue
        x = pd.to_numeric(df_window[col], errors='coerce').dropna()
        if x.empty:
            continue

        qs = x.quantile(quantiles)
        for q in quantiles:
            out[f"{col}_q{int(q*100)}"] = float(qs.loc[q])
    return out


def tail_ratio_stats(df_window: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    out = {}
    for col in feature_cols:
        if col not in df_window.columns:
            continue
        x = pd.to_numeric(df_window[col], errors='coerce').dropna()
        if len(x) < 5: # Need enough data for meaningful quantiles
            continue

        q10 = float(x.quantile(0.10))
        q50 = float(x.quantile(0.50))
        q90 = float(x.quantile(0.90))
        
        upper_ratio = np.nan
        lower_ratio = np.nan

        if q50 != 0:
            upper_ratio = q90 / q50
            lower_ratio = q10 / q50
        elif q90 > 0: # If median is 0 but 90th percentile is positive
            upper_ratio = np.inf
        elif q10 < 0: # If median is 0 but 10th percentile is negative
            lower_ratio = -np.inf
        else: # All values are likely zero
            upper_ratio = 1.0
            lower_ratio = 1.0

        out[f"{col}_upper_tail_ratio"] = float(upper_ratio)
        out[f"{col}_lower_tail_ratio"] = float(lower_ratio)
        out[f"{col}_tail_spread"]      = float(q90 - q10)
    return out


def correlation_stats(df_window: pd.DataFrame, corr_features: List[str]=CORR_FEATURES) -> Dict[str, float]:
    out = {}
    cols = [c for c in corr_features if c in df_window.columns]
    if len(cols) < 2:
        return out
    
    df_sub = df_window[cols].astype(float)
    df_sub = df_sub.dropna() # Drop rows with NaNs for correlation calculation
    if len(df_sub) < 2: # Need at least 2 non-NaN data points for correlation
        return out
 
    corr = df_sub.corr()
 
    # flatten upper triangle
    for i, col_i in enumerate(cols):
        for j in range(i+1, len(cols)):
            col_j = cols[j]
            val = corr.loc[col_i, col_j]
            out[f"corr_{col_i}_vs_{col_j}"] = float(val) if not pd.isna(val) else np.nan
    return out
 
 
def time_index(df_window: pd.DataFrame) -> pd.Series:
    # convert timestamps to numeric seconds since window start
    t = pd.to_datetime(df_window["timestamp"])
    t0 = t.iloc[0]
    return (t - t0).dt.total_seconds().astype(float)
 
 
def time_structure_stats(df_window: pd.DataFrame, feature_cols: List[str]=TIME_FEATURES) -> Dict[str, float]:
    out = {}
    if len(df_window) < 3: # Need at least 3 points for slope and R2, 2 for diff
        return out
 
    t = time_index(df_window)
    t_centered = t - t.mean()
    t2_sum = (t_centered**2).sum()
 
    for col in feature_cols:
        if col not in df_window.columns:
            continue
        x = pd.to_numeric(df_window[col], errors='coerce').dropna()
        
        # Align x and t, dropping NaNs from x
        temp_df = pd.DataFrame({'t': t, 'x': x}).dropna()
        if len(temp_df) < 3: # Not enough valid data points
            continue
        
        t_aligned = temp_df['t']
        x_aligned = temp_df['x']
        
        x_centered = x_aligned - x_aligned.mean()
        t_centered_aligned = t_aligned - t_aligned.mean()
        t2_sum_aligned = (t_centered_aligned**2).sum()
 
        # slope (OLS): cov(t, x) / var(t)
        cov_tx = (t_centered_aligned * x_centered).sum()
        if t2_sum_aligned > 0:
            slope = cov_tx / t2_sum_aligned
        else:
            slope = 0.0
 
        # R^2 = (cov^2) / (var(t)*var(x))
        var_t = t2_sum_aligned / len(t_aligned)
        var_x = (x_centered**2).sum() / len(x_aligned)
        if var_t > 0 and var_x > 0:
            r2 = (cov_tx / len(t_aligned))**2 / (var_t * var_x)
        else:
            r2 = 0.0
 
        out[f"{col}_trend_slope"] = float(slope)
        out[f"{col}_trend_r2"]    = float(r2)
 
        # lag-1 autocorrelation
        if len(x_aligned) > 2:
            x1 = x_centered.iloc[1:]
            x0 = x_centered.iloc[:-1]
            num = (x1 * x0).sum()
            den = (x_centered**2).sum()
            acf1 = num / den if den > 0 else 0.0
        else:
            acf1 = 0.0
        out[f"{col}_acf1"] = float(acf1)
 
        # max normalized jump
        if len(x_aligned) > 1:
            dx = x_aligned.diff().iloc[1:]
            std_x = x_aligned.std(ddof=1)
            if std_x > 0:
                max_jump = float(dx.abs().max() / std_x)
            else:
                max_jump = float(dx.abs().max()) # If std_x is 0, max_jump is just max abs diff
        else:
            max_jump = 0.0
        out[f"{col}_max_norm_jump"] = float(max_jump)
 
    return out
 
 
def detector_window_stats(df_window: pd.DataFrame, detectors: List[str]=DETECTORS) -> Dict[str, float]:
    out = {}
    n = len(df_window)
    if n == 0:
        return out
 
    t = time_index(df_window) # Need time_index for lead times
 
    for det in detectors:
        score_col = f"anomalies.detectors.{det}.score_raw"
        flag_col  = f"anomalies.detectors.{det}.is_anomaly"
        
        # Check for flattened column names
        if score_col not in df_window.columns or flag_col not in df_window.columns:
            score_col = f"det_{det}_score" # Fallback to user-suggested flattened name
            flag_col = f"det_{det}_is_anom"
            if score_col not in df_window.columns or flag_col not in df_window.columns:
                continue # Skip if neither naming convention works
 
        s = pd.to_numeric(df_window[score_col], errors='coerce').dropna()
        # Ensure flag column is numeric (0 or 1) for aggregation
        f = pd.to_numeric(df_window[flag_col], errors='coerce').fillna(0).astype(int)
 
        if s.empty: # No valid scores for this detector
            continue
 
        out[f"{det}_score_mean"] = float(s.mean())
        out[f"{det}_score_std"]  = float(s.std(ddof=1)) if n > 1 else 0.0
        out[f"{det}_score_max"]  = float(s.max())
        out[f"{det}_frac_flagged"] = float(f.sum() / n) if n > 0 else 0.0
 
        # first / last anomaly lead time relative to window start
        if f.sum() > 0: # If any anomalies were flagged
            # Find indices where f is 1 (anomaly detected)
            anomaly_indices = f[f == 1].index
            
            # Get the time index for these anomalies
            anomaly_times = t.loc[anomaly_indices]
            
            if not anomaly_times.empty:
                first_time_from_window_start = anomaly_times.min()
                last_time_from_window_start = anomaly_times.max()
                
                out[f"{det}_first_flag_sec_from_window_start"] = float(first_time_from_window_start)
                out[f"{det}_last_flag_sec_from_window_start"]  = float(last_time_from_window_start)
            else:
                out[f"{det}_first_flag_sec_from_window_start"] = np.nan
                out[f"{det}_last_flag_sec_from_window_start"]  = np.nan
        else:
            out[f"{det}_first_flag_sec_from_window_start"] = np.nan
            out[f"{det}_last_flag_sec_from_window_start"]  = np.nan
 
    return out
 
 
def aggregate_single_window(df_window: pd.DataFrame,
                            symbol: str,
                            event_time: datetime,
                            event_type: str,
                            window_name: str) -> Dict[str, Any]:
    df_window = df_window.copy()
    if not df_window.empty and "timestamp" in df_window.columns:
        # The timestamp column should already be datetime from processed_df
        df_window = df_window.sort_values("timestamp")
    else:
        # Handle empty df_window or missing timestamp column gracefully
        logger.warning(f"df_window is empty or missing 'timestamp' for {symbol} {event_type} {window_name}. Skipping detailed aggregation.")
        return {
            "symbol": symbol,
            "event_time": event_time.isoformat(),
            "event_type": event_type,
            "window_name": window_name,
            "window_start": np.nan,
            "window_end": np.nan,
            "num_snapshots": 0,
        }
 
    result = {
        "symbol": symbol,
        "event_time": event_time.isoformat(),
        "event_type": event_type,   # "pump" or "control"
        "window_name": window_name, # "pre_event" / "post_event" / etc.
        "window_start": df_window["timestamp"].iloc[0].isoformat(),
        "window_end": df_window["timestamp"].iloc[-1].isoformat(),
        "num_snapshots": len(df_window),
    }
 
    # 1. basic moments
    result.update(moment_stats(df_window, FEATURES_ALL))
 
    # 2. quantiles & tail ratios
    result.update(quantile_stats(df_window, FEATURES_ALL))
    result.update(tail_ratio_stats(df_window, FEATURES_ALL))
 
    # 3. correlations
    result.update(correlation_stats(df_window))
 
    # 4. time structure
    result.update(time_structure_stats(df_window, TIME_FEATURES))
 
    # 5. detector aggregates
    result.update(detector_window_stats(df_window))
 
    return result
 
 
class HistoricalDataLoader:
    """Loads historical data from various sources"""
    
    def __init__(self, config: BacktestConfig, exchange_manager: Optional[Any] = None, data_loader_instance: Optional[Any] = None):
        self.config = config
        self.exchange_manager = exchange_manager
        self.data_loader_instance = data_loader_instance
        self.data_cache = {}
        if self.config.gdrive_api_key:
            self.gdrive_api = GoogleDriveAPI(self.config.gdrive_api_key)
        else:
            self.gdrive_api = None
            logger.warning("Google Drive API key not provided in config. GDrive data loading will not work.")
 
 
    async def load_data(self, source_id: str, symbol: str, start_datetime: datetime, end_datetime: datetime, start_time_str: str, end_time_str: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load orderbook and trade data for a symbol
        
        Returns:
            Tuple of (orderbook_df, trades_df)
        """
        cache_key = f"{source_id}:{symbol}:{start_datetime.isoformat()}:{end_datetime.isoformat()}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
            
        if self.config.data_source == DataSource.CSV:
            data = await self._load_csv_data(source_id, symbol, start_datetime, end_datetime)
        elif self.config.data_source == DataSource.PARQUET:
            data = await self._load_parquet_data(source_id, symbol, start_datetime, end_datetime)
        elif self.config.data_source == DataSource.PICKLE:
            data = await self._load_pickle_data(source_id, symbol, start_datetime, end_datetime)
        elif self.config.data_source == DataSource.JSON:
            data = await self._load_json_data(source_id, symbol, start_datetime, end_datetime)
        elif self.config.data_source == DataSource.DATABASE:
            data = await self._load_database_data(source_id, symbol, start_datetime, end_datetime) # Assumed method
        elif self.config.data_source == DataSource.LIVE_RECORDING:
            data = await self._load_live_recording_data(source_id, symbol, start_datetime, end_datetime) # Assumed method
        elif self.config.data_source == DataSource.LIVE_API:
            data = await self._load_live_api_data(source_id, symbol, start_datetime, end_datetime)
        elif self.config.data_source == DataSource.GDRIVE:
            data = await self._load_gdrive_data(source_id, symbol, start_datetime, end_datetime, start_time_str, end_time_str)
        else:
            raise ValueError(f"Unsupported data source: {self.config.data_source}")
            
        self.data_cache[cache_key] = data
        return data
        
    async def _load_csv_data(self, source_id: str, symbol: str, start_datetime: datetime, end_datetime: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from CSV files"""
        base_path = Path(self.config.data_path) / source_id / symbol.replace('/', '_')
        
        orderbook_path = base_path / "orderbooks.csv"
        trades_path = base_path / "trades.csv"
        
        orderbook_df = pd.read_csv(orderbook_path, parse_dates=['timestamp'])
        trades_df = pd.read_csv(trades_path, parse_dates=['timestamp'])
        
        # Filter by date range
        orderbook_df = orderbook_df[
            (orderbook_df['timestamp'] >= start_datetime) &
            (orderbook_df['timestamp'] <= end_datetime)
        ]
        trades_df = trades_df[
            (trades_df['timestamp'] >= start_datetime) &
            (trades_df['timestamp'] <= end_datetime)
        ]
        
        return orderbook_df, trades_df
        
    async def _load_parquet_data(self, source_id: str, symbol: str, start_datetime: datetime, end_datetime: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from Parquet files"""
        base_path = Path(self.config.data_path) / source_id / symbol.replace('/', '_')
        
        orderbook_df = pd.read_parquet(base_path / "orderbooks.parquet")
        trades_df = pd.read_parquet(base_path / "trades.parquet")
        
        # Filter by date range
        orderbook_df = orderbook_df[
            (orderbook_df['timestamp'] >= start_datetime) &
            (orderbook_df['timestamp'] <= end_datetime)
        ]
        trades_df = trades_df[
            (trades_df['timestamp'] >= start_datetime) &
            (trades_df['timestamp'] <= end_datetime)
        ]
        
        return orderbook_df, trades_df
        
    async def _load_pickle_data(self, source_id: str, symbol: str, start_datetime: datetime, end_datetime: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from Pickle files"""
        base_path = Path(self.config.data_path) / source_id / symbol.replace('/', '_')
        
        with open(base_path / "orderbooks.pkl", 'rb') as f:
            orderbook_df = pickle.load(f)
        with open(base_path / "trades.pkl", 'rb') as f:
            trades_df = pickle.load(f)

        # Filter by date range
        orderbook_df = orderbook_df[
            (orderbook_df['timestamp'] >= start_datetime) &
            (orderbook_df['timestamp'] <= end_datetime)
        ]
        trades_df = trades_df[
            (trades_df['timestamp'] >= start_datetime) &
            (trades_df['timestamp'] <= end_datetime)
        ]
            
        return orderbook_df, trades_df
        
    async def _load_json_data(self, source_id: str, symbol: str, start_datetime: datetime, end_datetime: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from JSON files"""
        base_path = Path(self.config.data_path) / source_id / symbol.replace('/', '_')
        
        # Support compressed JSON
        orderbook_file = base_path / "orderbooks.json.gz"
        if orderbook_file.exists():
            with gzip.open(orderbook_file, 'rt') as f: # Corrected: Pass file path, not content var
                orderbook_data = json.load(f)
        else:
            with open(base_path / "orderbooks.json", 'r') as f:
                orderbook_data = json.load(f)
                
        trades_file = base_path / "trades.json.gz"
        if trades_file.exists():
            with gzip.open(trades_file, 'rt') as f: # Corrected: Pass file path, not content var
                trades_data = json.load(f)
        else:
            with open(base_path / "trades.json", 'r') as f:
                trades_data = json.load(f)
                
        orderbook_df = pd.DataFrame(orderbook_data)
        trades_df = pd.DataFrame(trades_data)

        # Filter by date range
        orderbook_df = orderbook_df[
            (orderbook_df['timestamp'] >= start_datetime) &
            (orderbook_df['timestamp'] <= end_datetime)
        ]
        trades_df = trades_df[
            (trades_df['timestamp'] >= start_datetime) &
            (trades_df['timestamp'] <= end_datetime)
        ]
        
        return orderbook_df, trades_df

    async def _load_database_data(self, source_id: str, symbol: str, start_datetime: datetime, end_datetime: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from database (placeholder)"""
        logger.warning(f"Database data loading not implemented for {source_id}, {symbol}.")
        return pd.DataFrame(), pd.DataFrame()

    async def _load_live_recording_data(self, source_id: str, symbol: str, start_datetime: datetime, end_datetime: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from live recording (placeholder)"""
        logger.warning(f"Live recording data loading not implemented for {source_id}, {symbol}.")
        return pd.DataFrame(), pd.DataFrame()
        
    async def _load_live_api_data(self, source_id: str, symbol: str, start_datetime: datetime, end_datetime: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load historical OHLCV data from live exchange API."""
        if not self.exchange_manager:
            raise ValueError("ExchangeManager not initialized. Cannot load live API data.")
        logger.info(f"Fetching historical OHLCV for {symbol} from {source_id} API...")
        exchange_adapter = self.exchange_manager.get_exchange(source_id)
        if not exchange_adapter:
            raise ValueError(f"Exchange {source_id} not found in ExchangeManager.")
        
        # Fetch OHLCV data (e.g., 1-minute candles)
        # Default to 1m timeframe, can be made configurable in BacktestConfig if needed
        ohlcv = await exchange_adapter.fetch_ohlcv(
            symbol=symbol,
            timeframe='1m',
            since=int(start_datetime.timestamp() * 1000),
            limit=None
        )
        
        if not ohlcv:
            logger.warning(f"No OHLCV data found for {symbol} on {source_id} for the specified period.")
            return pd.DataFrame(), pd.DataFrame()
        
        # Convert OHLCV to DataFrame
        ohlcv_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
        
        # Filter by end date
        ohlcv_df = ohlcv_df[ohlcv_df['timestamp'] <= end_datetime]
        
        return pd.DataFrame(), ohlcv_df
 
    async def _load_gdrive_data(self, source_id: str, symbol: str, start_datetime: datetime, end_datetime: datetime, start_time_str: str, end_time_str: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data for a symbol from Google Drive."""
        if not self.gdrive_api:
            raise ValueError("Google Drive API not initialized. Please provide gdrive_api_key in config.")
 
        logger.info(f"Loading data for {symbol} from Google Drive (source: {source_id})...")
        
        orderbook_df = pd.DataFrame()
        trades_df = pd.DataFrame()
 
        current_date = start_datetime.date()
        end_date_limit = end_datetime.date()
 
        # Define time filters
        start_time_obj = datetime.strptime(start_time_str, "%H:%M:%S").time()
        end_time_obj = datetime.strptime(end_time_str, "%H:%M:%S").time()
 
        while current_date <= end_date_limit:
            year = current_date.year
            month = current_date.month
            
            # Orderbook data loading
            if self.config.data_type == "orderbook" or self.config.data_type == "all":
                if self.config.gdrive_orderbook_root_id:
                    ob_symbol_folder_name_general = symbol.replace('/', '').lower()
                    ob_year_folder_name_general = str(year)
                    ob_month_folder_name_general = f"{month:02d}"
                    
                    try:
                        ob_content_general = await self.gdrive_api.get_file_from_drive(
                            self.config.gdrive_orderbook_root_id,
                            [ob_symbol_folder_name_general, ob_year_folder_name_general, ob_month_folder_name_general],
                            f"{current_date.strftime('%Y-%m-%d')}.csv.gz"
                        )
                        logger.debug(f"get_file_from_drive for general orderbook for {symbol} ({current_date}) returned content: {bool(ob_content_general)}")
                        if ob_content_general:
                            monthly_ob_df_general = _read_csv_from_bytes(ob_content_general)
                            monthly_ob_df_general = _normalize_dataframe(monthly_ob_df_general)
                            logger.debug(f"General orderbook DataFrame for {symbol} ({current_date}) empty after normalization: {monthly_ob_df_general.empty}")
                            if not monthly_ob_df_general.empty:
                                # Daily time filtering removed as it conflicts with pre-event window loading
                                # The overall date range filtering at the end of _load_gdrive_data will handle the boundaries.
                                # Daily time filtering is now handled by the overall date range filter at the end of the function.
                                logger.debug(f"Daily time filtering for general orderbook data for {current_date} is now handled by overall date range filter.")
                                orderbook_df = pd.concat([orderbook_df, monthly_ob_df_general], ignore_index=True)
                                logger.info(f"Loaded {len(monthly_ob_df_general)} general orderbook rows for {symbol} from Google Drive for {current_date}.")
                                logger.debug(f"Sample of general orderbook data for {current_date}:\n{monthly_ob_df_general.head()}")
                            else:
                                logger.warning(f"General orderbook DataFrame was empty after normalization for {symbol} from Google Drive for {current_date}.")
                    except Exception as e:
                        logger.warning(f"Error loading general orderbook data for {symbol} from Google Drive for {current_date}: {e}")
 
                # Attempt to load from Binance-specific Orderbook Root (only if source_id is 'binance' for backward compatibility or specific configuration)
                if source_id.lower() == 'binance' and self.config.gdrive_orderbook_binance_root_id:
                    ob_symbol_folder_name_binance = symbol.replace('/', '').upper()
                    ob_year_month_folder_name_binance = f"{year}_{month:02d}"
 
                    try:
                        ob_content_binance = await self.gdrive_api.get_file_from_drive(
                            self.config.gdrive_orderbook_binance_root_id,
                            [ob_symbol_folder_name_binance, ob_year_month_folder_name_binance],
                            f"{current_date.strftime('%Y-%m-%d')}.csv.gz"
                        )
                        logger.debug(f"get_file_from_drive for Binance-specific orderbook for {symbol} ({current_date}) returned content: {bool(ob_content_binance)}")
                        if ob_content_binance:
                            logger.debug(f"Attempting to read and normalize Binance-specific orderbook content for {symbol} from Google Drive for {current_date}.")
                            monthly_ob_df_binance = _read_csv_from_bytes(ob_content_binance)
                            monthly_ob_df_binance = _normalize_dataframe(monthly_ob_df_binance)
                            logger.debug(f"Read {len(monthly_ob_df_binance)} rows from Binance-specific orderbook CSV for {symbol} from Google Drive for {current_date}.")
                            logger.debug(f"Normalized Binance-specific orderbook DataFrame has {len(monthly_ob_df_binance)} rows for {symbol} from Google Drive for {current_date}.")
                            logger.debug(f"Binance-specific orderbook DataFrame for {symbol} ({current_date}) empty after normalization: {monthly_ob_df_binance.empty}")
                            
                            if not monthly_ob_df_binance.empty:
                                # Daily time filtering removed as it conflicts with pre-event window loading
                                # The overall date range filtering at the end of _load_gdrive_data will handle the boundaries.
                                # Daily time filtering is now handled by the overall date range filter at the end of the function.
                                logger.debug(f"Daily time filtering for Binance-specific orderbook data for {current_date} is now handled by overall date range filter.")
                                orderbook_df = pd.concat([orderbook_df, monthly_ob_df_binance], ignore_index=True)
                                logger.info(f"Loaded {len(monthly_ob_df_binance)} Binance-specific orderbook rows for {symbol} from Google Drive for {current_date}.")
                                logger.debug(f"Sample of Binance-specific orderbook data for {current_date}:\n{monthly_ob_df_binance.head()}")
                            else:
                                logger.warning(f"Binance-specific orderbook DataFrame was empty after normalization for {symbol} from Google Drive for {current_date}.")
                    except Exception as e:
                        logger.exception(f"Error loading Binance-specific orderbook data for {symbol} from Google Drive for {current_date}.")
            
            # Tradebook data loading
            if self.config.data_type == "tradebook" or self.config.data_type == "all":
                tr_root_id = self.config.gdrive_tradebook_root_id
                tr_symbol_folder_name = symbol.replace('/', '').upper()
                tr_year_month_folder_name = f"{year}_{month:02d}"
                
                if tr_root_id:
                    try:
                        trade_content = await self.gdrive_api.get_file_from_drive(
                            tr_root_id,
                            [tr_symbol_folder_name, tr_year_month_folder_name],
                            f"{current_date.strftime('%Y_%m_%d')}.csv.gz"
                        )
                        logger.debug(f"get_file_from_drive for tradebook for {symbol} ({current_date}) returned content: {bool(trade_content)}")
                        if trade_content:
                            monthly_trades_df = _read_csv_from_bytes(trade_content)
                            monthly_trades_df = _normalize_dataframe(monthly_trades_df)
                            logger.debug(f"Tradebook DataFrame for {symbol} ({current_date}) empty after normalization: {monthly_trades_df.empty}")
                            # Daily time filtering removed as it conflicts with pre-event window loading
                            # The overall date range filtering at the end of _load_gdrive_data will handle the boundaries.
                            # Daily time filtering is now handled by the overall date range filter at the end of the function.
                            logger.debug(f"Daily time filtering for tradebook data for {current_date} is now handled by overall date range filter.")
                            logger.debug(f"Sample of tradebook data for {current_date}:\n{monthly_trades_df.head()}")
                            trades_df = pd.concat([trades_df, monthly_trades_df], ignore_index=True)
                            logger.info(f"Loaded {len(monthly_trades_df)} tradebook rows for {symbol} from Google Drive for {current_date}.")
                            logger.debug(f"Sample of tradebook data for {current_date}:\n{monthly_trades_df.head()}")
                    except Exception as e:
                        logger.warning(f"Error loading tradebook data for {symbol} from Google Drive for {current_date}: {e}")
            
            current_date += timedelta(days=1)
            
        # Filter by the exact date range after concatenation
        orderbook_df = orderbook_df[
            (orderbook_df['timestamp'] >= start_datetime) &
            (orderbook_df['timestamp'] <= end_datetime)
        ] if not orderbook_df.empty else pd.DataFrame()
        
        trades_df = trades_df[
            (trades_df['timestamp'] >= start_datetime) &
            (trades_df['timestamp'] <= end_datetime)
        ] if not trades_df.empty else pd.DataFrame()
 
        logger.debug(f"Final orderbook_df for {symbol} after date range filtering. Empty: {orderbook_df.empty}, Rows: {len(orderbook_df)}")
        logger.debug(f"Final trades_df for {symbol} after date range filtering. Empty: {trades_df.empty}, Rows: {len(trades_df)}")
 
        # --- Pre-flight Non-Zero Gate ---
        # Check if the loaded DataFrames are empty or contain only zeros
        if orderbook_df.empty and trades_df.empty:
            logger.warning(f"No data loaded for {symbol} from Google Drive.")
            return pd.DataFrame(), pd.DataFrame()
 
        if not orderbook_df.empty and (orderbook_df['price'].sum() == 0 or orderbook_df['volume'].sum() == 0):
            logger.warning(f"Orderbook data for {symbol} from Google Drive contains all zero prices or volumes. Returning empty DataFrame.")
            orderbook_df = pd.DataFrame()
 
        if not trades_df.empty and (trades_df['price'].sum() == 0 or trades_df['volume'].sum() == 0):
            logger.warning(f"Trade data for {symbol} from Google Drive contains all zero prices or volumes. Returning empty DataFrame.")
            trades_df = pd.DataFrame()
        
        return orderbook_df, trades_df
        
        
class BacktestingEngine:
    """Main backtesting engine that replays historical data through the pipeline"""
    
    def __init__(self, config: BacktestConfig, exchange_manager: Optional[Any] = None, data_loader_instance: Optional[Any] = None):
        self.config = config
        self.exchange_manager = exchange_manager
        self.data_loader_instance = data_loader_instance
        
        # Conditionally initialize HistoricalDataLoader based on data_source
        if config.data_source == DataSource.LIVE_API:
            if not exchange_manager or not data_loader_instance:
                raise ValueError("ExchangeManager and DataLoader instance are required for LIVE_API data source.")
            self.data_loader = HistoricalDataLoader(config, exchange_manager, data_loader_instance)
        else:
            self.data_loader = HistoricalDataLoader(config)
        
        # Initialize feature extractors if enabled
        if config.enable_features:
            self.feature_extractors = self._initialize_feature_extractors()
        else:
            self.feature_extractors = {}
            
        # Initialize anomaly detection if enabled
        if config.enable_anomaly_detection:
            self.detection_system = DetectionSystem(config.detection_config)
        else:
            self.detection_system = None
            
        # Event callbacks
        self.orderbook_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        self.feature_callbacks: List[Callable] = []
        self.anomaly_callbacks: List[Callable] = []
        
        # Internal state
        self.current_time = config.pump_start_date # Changed to pump_start_date
        self.results = BacktestResult(config=config, start_time=datetime.utcnow(), end_time=datetime.utcnow())
        self.orderbook_buffers = defaultdict(lambda: deque(maxlen=100))
        self.trade_buffers = defaultdict(lambda: deque(maxlen=config.trade_history_size))
        self.all_processed_data: List[Dict[str, Any]] = [] # New buffer for all processed data
        
    def _initialize_feature_extractors(self) -> Dict[str, Any]:
        """Initialize feature extractors with config"""
        extractors = {}
        fe_config = self.config.feature_config
        
        extractors['cryptofeed'] = AdvancedCryptofeedExtractor(fe_config.get('cryptofeed', {}))
        extractors['tradebook'] = AdvancedTradebookExtractor(fe_config.get('tradebook', {}))
        extractors['technical'] = EnhancedTechnicalExtractor(fe_config.get('technical', {}))
        extractors['dex'] = DEXLiquidityExtractor(fe_config.get('dex', {}))
        extractors['futures'] = FuturesOpenInterestFundingExtractor(fe_config.get('futures', {}))
        extractors['options'] = EnhancedOptionsExtractor(fe_config.get('options', {})) # Added EnhancedOptionsExtractor
        
        # CrossExchangeDiscrepancyExtractor needs exchange_manager if data_source is LIVE_API
        if self.config.data_source == DataSource.LIVE_API and self.exchange_manager:
            extractors['cross_exchange'] = CrossExchangeDiscrepancyExtractor(fe_config.get('cross_exchange', {}), self.exchange_manager)
        else:
            # For file-based data sources, pass None or a mock if the extractor can handle it
            extractors['cross_exchange'] = CrossExchangeDiscrepancyExtractor(fe_config.get('cross_exchange', {}), None)
        
        return extractors
        
    def add_orderbook_callback(self, callback: Callable) -> None:
        """Add callback for orderbook updates"""
        self.orderbook_callbacks.append(callback)
        
    def add_trade_callback(self, callback: Callable) -> None:
        """Add callback for trade updates"""
        self.trade_callbacks.append(callback)
        
    def add_feature_callback(self, callback: Callable) -> None:
        """Add callback for extracted features"""
        self.feature_callbacks.append(callback)
        
    def add_anomaly_callback(self, callback: Callable) -> None:
        """Add callback for detected anomalies"""
        self.anomaly_callbacks.append(callback)
        
    async def run(self) -> BacktestResult:
        """Run the backtest"""
        logger.info(f"Starting backtest from {self.config.pump_start_date} to {self.config.pump_end_date} using data from Google Drive.")
        
        logger.debug("Preparing event timeline...")
        all_events = await self._prepare_event_timeline()
        logger.info(f"Event timeline prepared with {len(all_events)} events.")
        
        # Process events in chronological order
        try:
            # --- First Pass: Process all events to populate self.all_processed_data ---
            logger.debug("Starting first pass: processing all events for feature extraction and anomaly detection.")
            for i, event in enumerate(all_events):
                self.current_time = event['timestamp']
                
                # Always process orderbook and trade events to populate buffers
                if event['type'] == 'orderbook':
                    await self._process_orderbook_event(event)
                elif event['type'] == 'trade':
                    await self._process_trade_event(event)
                
                features = {}
                feature_stats = {}
                anomalies_result = {}
                if self.config.enable_features:
                    features, feature_stats, anomalies_result = await self._extract_and_process_features(event['source_id'], event['symbol'])
                
                # Store all processed data for later aggregation (including pre-event windows)
                processed_data_point = {
                    "timestamp": self.current_time,
                    "source_id": event['source_id'],
                    "symbol": event['symbol'],
                    "symbol_group": event['symbol_group'],
                    "event_time": event['event_time'],
                    "features": features,
                    "feature_stats": feature_stats,
                    "anomalies": anomalies_result
                }
                self.all_processed_data.append(processed_data_point)
            logger.debug("First pass complete. self.all_processed_data is populated.")
 
            # --- Second Pass: Simulate backtest and generate unified output for events within the main timeframe ---
            logger.debug("Starting second pass: simulating backtest and generating unified output.")
            last_event_timestamp = None
            for processed_data_point in self.all_processed_data:
                self.current_time = processed_data_point['timestamp']
                
                # Only simulate and output for events within the main backtest time range
                if self.config.pump_start_date <= self.current_time <= self.config.pump_end_date: # Changed to pump_start/end_date
                    start_process = datetime.utcnow() # Reset start_process for this pass
 
                    if self.config.output.get("mode") == "full":
                        from core.unified_output import write_unified_record
                        write_unified_record(
                            config=self.config,
                            exchange=processed_data_point['source_id'],
                            symbol=processed_data_point['symbol'],
                            features=processed_data_point['features'],
                            feature_stats=processed_data_point['feature_stats'],
                            detectors=processed_data_point['anomalies'].get("detectors", {}),
                            meta_stats=processed_data_point['anomalies'].get("meta_statistics", {}),
                            composite=processed_data_point['anomalies'].get("composite", {})
                        )
                    
                    # Simulate replay speed
                    if self.config.replay_speed < float('inf') and last_event_timestamp is not None:
                        time_diff_real = (self.current_time - last_event_timestamp).total_seconds()
                        simulated_delay = time_diff_real / self.config.replay_speed
                        if simulated_delay > 0.001:
                            await asyncio.sleep(simulated_delay)
                    last_event_timestamp = self.current_time
                    
                    process_time = (datetime.utcnow() - start_process).total_seconds() * 1000
                    self.results.processing_time_ms.append(process_time)
            
            logger.info("Second pass complete. All events processed. Finalizing backtest results.")
            self.results.end_time = datetime.utcnow()
 
            if self.config.enable_event_aggregation:
                logger.info("Aggregating event-level statistics...")
                await self._aggregate_event_statistics()
                await self._save_aggregated_results() # Save aggregated results after aggregation
        except Exception as e:
            logger.exception(f"An error occurred during event processing: {e}")
            self.results.end_time = datetime.utcnow()
            logger.warning("Backtest terminated due to an error. Attempting to save partial results.")
        
        if self.config.save_results:
            await self._save_results()
            
        logger.info("Backtest run completed.")
        return self.results
        
    async def _prepare_event_timeline(self) -> List[Dict[str, Any]]:
        """Prepare chronologically sorted timeline of all events"""
        all_events = []
 
        # Determine the source_id for backtesting. Assuming a single source_id for GDrive backtesting.
        # If config.exchanges is not empty, it implies specific exchanges are configured,
        # otherwise, we use a generic "gdrive_data_source" to represent the source of historical data.
        backtest_source_id = self.config.exchanges[0] if self.config.exchanges else "gdrive_data_source"
        logger.debug(f"Backtest source ID: {backtest_source_id}")
 
        # Process pump events
        for symbol, event_time in self.config.pump_event_definitions:
            await self._load_and_add_events(
                backtest_source_id, symbol, "pump", event_time, all_events,
                self.config.pump_start_date, self.config.pump_end_date,
                self.config.pump_start_time_str, self.config.pump_end_time_str
            )
 
        # Process control events
        for symbol, event_time in self.config.control_event_definitions:
            await self._load_and_add_events(
                backtest_source_id, symbol, "control", event_time, all_events,
                self.config.control_start_date, self.config.control_end_date,
                self.config.control_start_time_str, self.config.control_end_time_str
            )
                    
        # Sort by timestamp
        all_events.sort(key=lambda x: x['timestamp'])
        
        logger.info(f"Prepared {len(all_events)} events for replay.")
        return all_events
 
    async def _load_and_add_events(
        self,
        source_id: str,
        symbol: str,
        symbol_group: str,
        event_time: datetime,
        all_events: List[Dict[str, Any]],
        data_start_date: datetime,
        data_end_date: datetime,
        data_start_time_str: str,
        data_end_time_str: str
    ):
        """Helper to load data and add events for a given symbol and group, relative to an event_time.
        Uses data_start_date and data_end_date for filtering specific to the event group.
        """
        try:
            # Adjust start and end dates for data loading to cover event windows
            min_start_date = event_time + timedelta(minutes=min(w[0] for w in self.config.event_windows.values()))
            max_end_date = event_time + timedelta(minutes=max(w[1] for w in self.config.event_windows.values()))
 
            # Use the provided data_start_date and data_end_date for filtering
            effective_start_date = min(data_start_date, min_start_date)
            effective_end_date = max(data_end_date, max_end_date)
 
            logger.debug(f"Adjusted data loading range for {symbol} around event {event_time} ({symbol_group}): {effective_start_date} to {effective_end_date}")
            
            # Now pass the effective date/time range to the data_loader's load_data method
            orderbook_df, trades_df = await self.data_loader.load_data(source_id, symbol, effective_start_date, effective_end_date, data_start_time_str, data_end_time_str)
 
            if self.config.data_source == DataSource.LIVE_API:
                for _, row in trades_df.iterrows():
                    all_events.append({
                        'timestamp': row['timestamp'],
                        'type': 'trade',
                        'source_id': source_id,
                        'symbol': symbol,
                        'symbol_group': symbol_group,
                        'event_time': event_time, # Add event_time to the event
                        'data': {
                            'price': row['close'],
                            'volume': row['volume'],
                            'side': 'buy' if row['open'] <= row['close'] else 'sell'
                        }
                    })
            else:
                if self.config.data_type in ["orderbook", "all"]:
                    # Group by timestamp to reconstruct order book snapshots
                    grouped_orderbooks = orderbook_df.groupby('timestamp')
                    for timestamp, group in grouped_orderbooks:
                        bids_for_ts = []
                        asks_for_ts = []
                        for _, ob_row in group.iterrows():
                            # Ensure 'bids' and 'asks' are lists of lists before extending
                            if 'bids' in ob_row and isinstance(ob_row['bids'], list):
                                bids_for_ts.extend([OrderBookLevel(price=float(b[0]), volume=float(b[1])) for b in ob_row['bids']])
                            elif ob_row['type'] == 'b':
                                bids_for_ts.append(OrderBookLevel(price=float(ob_row['price']), volume=float(ob_row['volume'])))
                            
                            if 'asks' in ob_row and isinstance(ob_row['asks'], list):
                                asks_for_ts.extend([OrderBookLevel(price=float(a[0]), volume=float(a[1])) for a in ob_row['asks']])
                            elif ob_row['type'] == 'a':
                                asks_for_ts.append(OrderBookLevel(price=float(ob_row['price']), volume=float(ob_row['volume'])))
 
                        all_events.append({
                            'timestamp': timestamp,
                            'type': 'orderbook',
                            'source_id': source_id,
                            'symbol': symbol,
                            'symbol_group': symbol_group,
                            'event_time': event_time, # Add event_time to the event
                            'data': {
                                'bids': bids_for_ts,
                                'asks': asks_for_ts,
                                'nonce': None
                            }
                        })
                        
                if self.config.data_type in ["tradebook", "all"]:
                    for _, row in trades_df.iterrows():
                        all_events.append({
                            'timestamp': row['timestamp'],
                            'type': 'trade',
                            'source_id': source_id,
                            'symbol': symbol,
                            'symbol_group': symbol_group,
                            'event_time': event_time, # Add event_time to the event
                            'data': row
                        })
                
        except Exception as e:
            logger.error(f"Failed to load data for {symbol} from {source_id} - {e}")
     
            
    async def _process_orderbook_event(self, event: Dict[str, Any]) -> None:
        """Process a single orderbook event"""
        data = event['data']
        
        # Convert to normalized format
        orderbook = self._create_normalized_orderbook(event)
        
        # Store in buffer
        key = f"{event['source_id']}:{event['symbol']}"
        self.orderbook_buffers[key].append(orderbook)
        
        # Call callbacks
        for callback in self.orderbook_callbacks:
            await callback(orderbook)
            
        # Feature extraction and anomaly detection will be handled in the main loop
        # after both orderbook and trade buffers are potentially updated.
            
        self.results.total_orderbooks += 1
        
    async def _process_trade_event(self, event: Dict[str, Any]) -> None:
        """Process a single trade event"""
        trade = self._create_normalized_trade(event)
        
        # Store in buffer
        key = f"{event['source_id']}:{event['symbol']}"
        self.trade_buffers[key].append(trade)
        
        # Call callbacks
        for callback in self.trade_callbacks:
            await callback(trade)
            
        self.results.total_trades += 1
        
    async def _extract_and_process_features(self, source_id: str, symbol: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Extract features and run anomaly detection"""
        logger.debug(f"Attempting to extract features for {symbol} from data source: {source_id}")
        start_time = datetime.utcnow()
        
        # Get current state
        key = f"{source_id}:{symbol}"
        orderbooks = list(self.orderbook_buffers[key])
        trades = list(self.trade_buffers[key])
        
        if not orderbooks and not trades and self.config.data_source != DataSource.LIVE_API:
            logger.debug(f"Skipping feature extraction for {symbol} from data source: {source_id}: No orderbooks or trades available and not LIVE_API source.")
            return {}, {}, {}
        
        # Extract features
        features = {}
        
        # Cryptofeed features from orderbook (only if orderbook data is available)
        if orderbooks and 'cryptofeed' in self.feature_extractors:
            latest_ob = orderbooks[-1]
            ob_dict = {
                'bids': [[float(level.price), float(level.volume)] for level in latest_ob.bids],
                'asks': [[float(level.price), float(level.volume)] for level in latest_ob.asks]
            }
            cf_features = self.feature_extractors['cryptofeed'].extract(ob_dict)
            features.update({f'cf_{k}': v for k, v in cf_features.items()})
            logger.debug(f"Extracted cryptofeed features for {symbol} from data source: {source_id}. Count: {len(cf_features)}")
            
        # Tradebook features
        if trades and 'tradebook' in self.feature_extractors:
            trades_df = pd.DataFrame([{
                'timestamp': t.timestamp,
                'price': t.price,
                'volume': t.volume,
                'side': t.side
            } for t in trades])
            
            tb_features = self.feature_extractors['tradebook'].extract(trades_df)
            features.update({f'tb_{k}': v for k, v in tb_features.items()})
            logger.debug(f"Extracted tradebook features for {symbol} from data source: {source_id}. Count: {len(tb_features)}")
            
        # Cross-exchange features (requires exchange_manager for live API)
        if 'cross_exchange' in self.feature_extractors:
            # For backtesting with file data, cross-exchange features might need a different approach
            # or could be skipped if not relevant for historical data without a live manager.
            # Assuming for now it needs some form of data, potentially from loaded dataframes
            pass # Placeholder for actual cross-exchange feature extraction logic if needed
            
        # Other extractors would go here similarly
        
        # Track feature extraction time
        feature_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.results.feature_extraction_time_ms.append(feature_time)
        logger.debug(f"Feature extraction for {symbol} from data source: {source_id} completed in {feature_time:.2f} ms.")
        
        # Call feature callbacks (pass features without metadata)
        for callback in self.feature_callbacks:
            await callback(features)
            
        # Initialize feature_stats to an empty dictionary
        feature_stats = {}
        
        # Calculate and store feature statistics
        numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float)) and not pd.isna(v)}
        if numeric_features:
            df = pd.DataFrame([numeric_features])
            if len(df) > 1: # This condition is likely incorrect for a single row DataFrame, should be for multiple data points over time
                desc_stats = {
                    "mean": df.mean(axis=0).to_dict(),
                    "variance": df.var(axis=0).to_dict(),
                    "skewness": df.apply(lambda x: skew(x, nan_policy='omit')).to_dict(),
                    "kurtosis": df.apply(lambda x: kurtosis(x, nan_policy='omit')).to_dict()
                }
            else: # For a single data point, variance, skewness, kurtosis are not meaningful
                desc_stats = {
                    "mean": df.mean(axis=0).to_dict(),
                    "variance": {k: 0.0 for k in numeric_features.keys()},
                    "skewness": {k: 0.0 for k in numeric_features.keys()},
                    "kurtosis": {k: 0.0 for k in numeric_features.keys()}
                }
            
            if source_id not in self.results.feature_statistics:
                self.results.feature_statistics[source_id] = {}
            self.results.feature_statistics[source_id][symbol] = desc_stats
            feature_stats = desc_stats
        
        # Prepare features for logging and anomaly detection by adding metadata
        # This dictionary will be used for logging and passed to the detection system.
        # It should NOT be returned as the primary 'features' output to avoid column conflicts.
        features_for_logging_and_detection = features.copy()
        features_for_logging_and_detection['_timestamp'] = self.current_time.isoformat()
        features_for_logging_and_detection['_source_id'] = source_id
        features_for_logging_and_detection['_symbol'] = symbol
        
        # Log extracted features and statistics to backtest_data_logger
        backtest_data_logger.info(json.dumps({
            "event_type": "features_extracted",
            "timestamp": self.current_time.isoformat(),
            "source_id": source_id,
            "symbol": symbol,
            "features": features_for_logging_and_detection, # Use the version with metadata for logging
            "feature_statistics": feature_stats
        }, default=str))
        
        # Run anomaly detection if enabled
        if self.detection_system and self.config.enable_anomaly_detection:
            logger.debug(f"Features for anomaly detection for {symbol} from data source {source_id}: {features_for_logging_and_detection}")
            logger.debug(f"Feature statistics for anomaly detection for {symbol} from data source {source_id}: {feature_stats}")
            logger.debug(f"Calling anomaly detection for {symbol} from data source: {source_id}.")
            
            # Pass features_for_logging_and_detection to detection system
            anomalies_result = await self._run_anomaly_detection(features_for_logging_and_detection, source_id, symbol)
        else:
            logger.debug(f"Anomaly detection skipped for {symbol} from data source: {source_id}. Enabled: {self.config.enable_anomaly_detection}, System: {bool(self.detection_system)}")
            anomalies_result = {}
        
        # Log anomaly detection results to backtest_data_logger
        backtest_data_logger.info(json.dumps({
            "event_type": "anomaly_detection_results",
            "timestamp": self.current_time.isoformat(),
            "source_id": source_id,
            "symbol": symbol,
            "anomalies_result": anomalies_result
        }, default=str))
        
        return features, feature_stats, anomalies_result # Return features WITHOUT metadata
    
    async def _run_anomaly_detection(self, features_for_detection: Dict[str, Any], source_id: str, symbol: str) -> Dict[str, Any]:
        """Run anomaly detection on features"""
        logger.debug(f"Attempting to run anomaly detection for {symbol} from data source: {source_id}")
        start_time = datetime.utcnow()
        
        if not self.detection_system or not self.config.enable_anomaly_detection:
            return {}
        
        anomalies = await self.detection_system.detect_anomalies(features_for_detection)
        
        # Track detection time
        detection_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.results.anomaly_detection_time_ms.append(detection_time)
        logger.debug(f"Anomaly detection for {symbol} from data source: {source_id} completed in {detection_time:.2f} ms. Severity: {anomalies['composite']['severity']}")
 
        # Record anomalies
        if anomalies['composite']['severity'] != 'normal':
            self.results.total_anomalies += 1
            self.results.anomaly_timeline.append({
                'timestamp': self.current_time,
                'source_id': source_id,
                'symbol': symbol,
                'anomaly': anomalies
            })
            logger.info(f"Anomaly detected for {symbol} from data source: {source_id}: {anomalies['composite']['severity']} - Detectors Flagged: {anomalies['composite']['num_detectors_flagged']}")
            
        # Call anomaly callbacks
        for callback in self.anomaly_callbacks:
            await callback(anomalies)
        return anomalies
            
    def _create_normalized_orderbook(self, event: Dict[str, Any]) -> NormalizedOrderBook:
        """Create normalized orderbook from event data"""
        data = event['data']
        
        bids = []
        asks = []
 
        # Assuming 'data' now contains 'bids' and 'asks' lists directly from _prepare_event_timeline
        if 'bids' in data and isinstance(data['bids'], list):
            bids = data['bids'][:self.config.orderbook_depth]
        
        if 'asks' in data and isinstance(data['asks'], list):
            asks = data['asks'][:self.config.orderbook_depth]
            
        return NormalizedOrderBook(
            exchange=event['source_id'], # Use source_id here
            symbol=event['symbol'],
            timestamp=event['timestamp'],
            bids=bids,
            asks=asks,
            sequence=data.get('nonce', 0)
        )
        
    def _create_normalized_trade(self, event: Dict[str, Any]) -> NormalizedTrade:
        """Create normalized trade from event data"""
        data = event['data']
        
        return NormalizedTrade(
            exchange=event['source_id'], # Use source_id here
            symbol=event['symbol'],
            timestamp=data.get('timestamp', event['timestamp']),
            id=str(data.get('id', '')),
            price=float(data['price']),
            volume=float(data.get('volume', data.get('amount'))),
            side=data.get('side', 'unknown'),
            taker_side=data.get('taker_side', data.get('side', 'unknown'))
        )
        
    async def _save_results(self) -> None:
        """Save backtest results"""
        results_dir = Path(self.config.results_path)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_results_{timestamp}.json"
        
        results_dict = {
            'config': {
                'pump_start_date': self.config.pump_start_date.isoformat(),
                'pump_end_date': self.config.pump_end_date.isoformat(),
                'pump_start_time_str': self.config.pump_start_time_str,
                'pump_end_time_str': self.config.pump_end_time_str,
                'control_start_date': self.config.control_start_date.isoformat(),
                'control_end_date': self.config.control_end_date.isoformat(),
                'control_start_time_str': self.config.control_start_time_str,
                'control_end_time_str': self.config.control_end_time_str,
                'pump_event_definitions': [(s, et.isoformat()) for s, et in self.config.pump_event_definitions],
                'control_event_definitions': [(s, et.isoformat()) for s, et in self.config.control_event_definitions],
                'exchanges': self.config.exchanges, # This will be empty for GDRIVE
                'replay_speed': self.config.replay_speed,
                'enable_features': self.config.enable_features,
                'enable_anomaly_detection': self.config.enable_anomaly_detection,
                'enable_event_aggregation': self.config.enable_event_aggregation,
                'event_windows': {k: (v[0], v[1]) for k, v in self.config.event_windows.items()},
                'features_to_aggregate': self.config.features_to_aggregate,
                'detectors_to_aggregate': self.config.detectors_to_aggregate
            },
            'summary': {
                'start_time': self.results.start_time.isoformat(),
                'end_time': self.results.end_time.isoformat(),
                'duration_seconds': (self.results.end_time - self.results.start_time).total_seconds(),
                'total_orderbooks': self.results.total_orderbooks,
                'total_trades': self.results.total_trades,
                'total_anomalies': self.results.total_anomalies
            },
            'performance': {
                'avg_processing_time_ms': np.mean(self.results.processing_time_ms) if self.results.processing_time_ms else 0,
                'avg_feature_time_ms': np.mean(self.results.feature_extraction_time_ms) if self.results.feature_extraction_time_ms else 0,
                'avg_detection_time_ms': np.mean(self.results.anomaly_detection_time_ms) if self.results.anomaly_detection_time_ms else 0
            },
            'feature_statistics': self.results.feature_statistics,
            'anomaly_timeline': self.results.anomaly_timeline,
            'custom_metrics': self.results.custom_metrics
        }
        
        with open(results_dir / filename, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
            
        logger.info(f"Saved backtest results to {results_dir / filename}")
 
    async def _aggregate_event_statistics(self) -> None:
        """
        Aggregates features and detector outputs over defined time windows
        for pump and control events.
        """
        if not self.config.enable_event_aggregation:
            logger.info("Event aggregation is disabled. Skipping aggregation.")
            return
 
        for event_type, event_definitions in [
            ("pump", self.config.pump_event_definitions),
            ("control", self.config.control_event_definitions),
        ]:
            for symbol, event_time in event_definitions:
                logger.info(f"Aggregating for {event_type} event: {symbol} at {event_time}")
                
                # Filter self.all_processed_data for the current event's symbol
                event_processed_data = [
                    d for d in self.all_processed_data 
                    if d['symbol'] == symbol and d['event_time'] == event_time
                ]
                
                if not event_processed_data:
                    logger.warning(f"No processed data found for {event_type} event: {symbol} at {event_time}. Skipping aggregation for this event.")
                    continue
 
                # Create a DataFrame from the filtered processed data
                df_processed = pd.json_normalize(event_processed_data, sep='_')
                
                # Ensure 'timestamp' is datetime for windowing operations
                df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
 
                # Rename columns to strip "features_" and "anomalies_detectors_" prefixes
                # This ensures that statistical helper functions can find the correct columns
                renamed_columns = {}
                for col in df_processed.columns:
                    if col.startswith('features_'):
                        renamed_columns[col] = col[len('features_'):]
                    elif col.startswith('anomalies_detectors_'):
                        # Flatten anomalies_detectors_pyod_iforest_score_raw to det_pyod_iforest_score
                        parts = col.split('_')
                        if len(parts) >= 4 and parts[0] == 'anomalies' and parts[1] == 'detectors':
                            detector_name = parts[2]
                            metric_type = '_'.join(parts[3:]) # score_raw or is_anomaly
                            if metric_type == 'score_raw':
                                renamed_columns[col] = f'det_{detector_name}_score'
                            elif metric_type == 'is_anomaly':
                                renamed_columns[col] = f'det_{detector_name}_is_anom'
                            else:
                                renamed_columns[col] = col # Keep original if not score_raw or is_anomaly
                        else:
                            renamed_columns[col] = col # Keep original if not matching pattern
                    else:
                        renamed_columns[col] = col
                df_processed = df_processed.rename(columns=renamed_columns)
                logger.debug(f"Columns after renaming: {df_processed.columns.tolist()}")
 
                # Now, df_window will be created from df_processed with renamed columns
 
                for window_name, (start_offset_min, end_offset_min) in self.config.event_windows.items():
                    window_start_time = event_time + timedelta(minutes=start_offset_min)
                    window_end_time = event_time + timedelta(minutes=end_offset_min)
 
                    df_window = df_processed[
                        (df_processed['timestamp'] >= window_start_time) &
                        (df_processed['timestamp'] < window_end_time)
                    ].copy() # Use .copy() to avoid SettingWithCopyWarning
                    
                    if df_window.empty:
                        logger.warning(f"No data for window '{window_name}' for {symbol} at {event_time}. Skipping aggregation for this window.")
                        continue
 
                    # Perform aggregation for the current window
                    aggregated_stats = aggregate_single_window(
                        df_window, symbol, event_time, event_type, window_name
                    )
                    
                    self.results.custom_metrics.setdefault("event_aggregated_stats", []).append(aggregated_stats)
                    logger.debug(f"Aggregated statistics for {window_name} of {event_type} event {symbol} at {event_time}.")
 
        logger.info("Finished aggregating all event statistics.")
 
    async def _save_aggregated_results(self) -> None:
        """Saves the aggregated event statistics to a JSONL file."""
        if not self.config.enable_event_aggregation or not self.results.custom_metrics.get("event_aggregated_stats"):
            return
 
        results_dir = Path(self.config.results_path)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"stats_{timestamp}.jsonl"
        filepath = results_dir / filename
 
        with open(filepath, 'w') as f:
            for record in self.results.custom_metrics["event_aggregated_stats"]:
                f.write(json.dumps(record, default=str) + "\n")
        
        logger.info(f"Saved aggregated event statistics to {filepath}")
