# trade_orderbook_pipeline.py
import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone, timedelta, time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import dataclasses
import numpy as np # Import numpy for nan check
 
import ccxt # Import ccxt to catch specific exceptions
import yaml
from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn
from prometheus_client import make_asgi_app, Info
 
# Helper to replace np.nan with None for JSON serialization
def replace_nan_with_none(obj):
    if isinstance(obj, list):
        return [replace_nan_with_none(elem) for elem in obj]
    elif isinstance(obj, dict):
        return {key: replace_nan_with_none(value) for key, value in obj.items()}
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj

import pandas as pd


from core.exchange_manager_updated import ExchangeManager
from core.data_loader import DataLoader
from core.exchange_interface import (
    NormalizedOrderBook, NormalizedTrade, NormalizedTicker
)


from features.extractors.advanced_cryptofeed_extractor import AdvancedCryptofeedExtractor
from features.extractors.advanced_tradebook_extractor import AdvancedTradebookExtractor
from features.extractors.enhanced_technical_extractor import EnhancedTechnicalExtractor
from features.extractors.dex_liquidity_extractor import DEXLiquidityExtractor
from features.extractors.futures_open_interest_funding_extractor import FuturesOpenInterestFundingExtractor
from features.extractors.cross_exchange_discrepancy_extractor import CrossExchangeDiscrepancyExtractor
from features.extractors.enhanced_options_extractor import EnhancedOptionsExtractor
from features.extractors.pre_pump_pipeline import build_feature_matrix as build_pre_pump_features_matrix

from detection.system import DetectionSystem
from core.unified_output import write_unified_record

from core.logging_config import init_logging, get_logger

# Global logger for initial setup messages, will be re-initialized within pipeline class
_global_logger = get_logger(__name__)

# NEW IMPORTS FOR ALL EXCHANGES AND BACKTESTING
try:
    from adapters.cex_adapters import CEX_ADAPTERS, get_cex_adapter
    from adapters.dex_adapters import DEX_ADAPTERS, get_dex_adapter
    from backtesting.engine import BacktestConfig, BacktestingEngine, DataSource
    ALL_EXCHANGES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import exchange adapters or backtesting engine: {e}")
    ALL_EXCHANGES_AVAILABLE = False
    CEX_ADAPTERS = {}
    DEX_ADAPTERS = {}

sys.path.insert(0, str(Path(__file__).parent / "adapters"))

info_metric = Info('trade_orderbook_pipeline', 'Trade and orderbook pipeline information')
info_metric.info({
    'version': '2.0.0',
    'start_time': datetime.now(timezone.utc).isoformat()
})

class TradeOrderbookPipeline:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        
        # Initialize logging with settings from the loaded config
        init_logging(pipeline_config=self.config)
        self.logger = get_logger(__name__)
        self.logger.info("Initializing Trade & Orderbook Pipeline...")

        # Load exchanges config from JSON (assuming config_path is the JSON config)
        exchanges_config_from_json = self.config.get("exchanges", {})

        # Ensure all exchanges are enabled as requested
        for ex_id, ex_conf in exchanges_config_from_json.items():
            if isinstance(ex_conf, dict):
                ex_conf['enabled'] = True

        self.exchange_manager = ExchangeManager(exchanges_config=exchanges_config_from_json, config_path=config_path)
        self.data_loader = DataLoader(self.exchange_manager, self.config.get('data_loader', {}))
        
        self.feature_extractors = self._initialize_feature_extractors()
        self.detection_system = DetectionSystem(pipeline_config=self.config)
        
        self.backtest_engine = None
        
        self.app = FastAPI(title="Trade & Orderbook Pipeline v2")
        self.websocket_clients: List[WebSocket] = []
        self._running = False
        self._setup_routes()
        self._setup_callbacks()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported config file format. Please use .json, .yaml, or .yml.")
            
    def _initialize_feature_extractors(self) -> Dict[str, Any]:
        # Only pass exchange_manager to CrossExchangeDiscrepancyExtractor if needed for live data
        cross_exchange_manager = self.exchange_manager if self.config.get('backtesting', {}).get('data_source') != 'gdrive' else None
        return {
            'cryptofeed': AdvancedCryptofeedExtractor(self.config.get('extractors', {}).get('cryptofeed', {})),
            'tradebook': AdvancedTradebookExtractor(self.config.get('extractors', {}).get('tradebook', {})),
            'technical': EnhancedTechnicalExtractor(self.config.get('extractors', {}).get('technical', {})),
            'dex': DEXLiquidityExtractor(self.config.get('extractors', {}).get('dex', {})),
            'futures': FuturesOpenInterestFundingExtractor(self.config.get('extractors', {}).get('futures', {})),
            'cross_exchange': CrossExchangeDiscrepancyExtractor(self.config.get('extractors', {}).get('cross_exchange', {}), cross_exchange_manager),
            'options': EnhancedOptionsExtractor(self.config.get('extractors', {}).get('options', {})),
            'pre_pump': None # Placeholder for pre_pump_pipeline, as it's a function not a class
        }
        
    def _normalize_symbol_for_ccxt(self, symbol: str) -> str:
        """Converts a symbol to CCXT's BASE/QUOTE format if not already."""
        # This normalization is primarily for live exchange interaction.
        # For GDrive backtesting, the symbol from the config is used directly.
        if '/' in symbol:
            return symbol.upper()
        if symbol.endswith('USDT'):
            return f"{symbol[:-4].upper()}/USDT"
        elif symbol.endswith('USD'):
            return f"{symbol[:-3].upper()}/USD"
        elif symbol.endswith('BTC'):
            return f"{symbol[:-3].upper()}/BTC"
        elif symbol.endswith('ETH'):
            return f"{symbol[:-3].upper()}/ETH"
        return symbol.upper()

    def _setup_routes(self) -> None:
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
            
        @self.app.get("/exchanges")
        async def get_exchanges():
            configured_exchanges = []
            for ex_id, ex_config in self.config.get('exchanges', {}).items():
                is_enabled = ex_config.get('enabled', False)
                is_connected = False
                if is_enabled and ex_id in self.exchange_manager.exchanges:
                    try:
                        is_connected = await self.exchange_manager.exchanges[ex_id].is_connected()
                    except Exception:
                        is_connected = False

                configured_exchanges.append({
                    "id": ex_id,
                    "enabled_in_config": is_enabled,
                    "is_active": is_connected
                })

            return JSONResponse(content={
                "configured_exchanges": configured_exchanges,
                "active_exchanges_count": len(self.exchange_manager.exchanges),
                "active_exchanges_list": list(self.exchange_manager.exchanges.keys())
            })
            
        @self.app.get("/instruments")
        async def get_instruments():
            return await self.exchange_manager.get_all_instruments()
            
        @self.app.get("/orderbook")
        async def get_orderbook_for_multiple_symbols(symbols: List[str] = Query(..., description="Comma-separated list of symbols")):
            """Get orderbooks for multiple symbols across all enabled exchanges concurrently."""
            all_orderbooks = {}

            async def fetch_ob(ex_id, ex_adapter, sym_raw):
                sym = self._normalize_symbol_for_ccxt(sym_raw)
                try:
                    res = await ex_adapter.fetch_order_book(sym)
                    return ex_id, sym, res
                except Exception as e:
                    self.logger.warning(f"Error fetching orderbook for {sym} on {ex_id}: {e}")
                    return ex_id, sym, None

            tasks = []
            for ex_id, ex_adapter in self.exchange_manager.exchanges.items():
                for sym_raw in symbols:
                    tasks.append(fetch_ob(ex_id, ex_adapter, sym_raw))

            results = await asyncio.gather(*tasks)
            for ex_id, sym, res in results:
                if res is not None:
                    if ex_id not in all_orderbooks: all_orderbooks[ex_id] = {}
                    all_orderbooks[ex_id][sym] = res.dict() if hasattr(res, "dict") else res

            if False: # Fixed 404
                raise HTTPException(status_code=404, detail=f"No orderbooks found for any symbols.")
            return all_orderbooks
            
        @self.app.get("/trades")
        async def get_trades_for_multiple_symbols(symbols: List[str] = Query(..., description="Comma-separated list of symbols"), limit: int = 100):
            """Get recent trades for multiple symbols across all enabled exchanges concurrently."""
            all_trades = {}

            async def fetch_tr(ex_id, ex_adapter, sym_raw):
                sym = self._normalize_symbol_for_ccxt(sym_raw)
                try:
                    res = await ex_adapter.fetch_trades(sym, limit=limit)
                    return ex_id, sym, res
                except Exception as e:
                    self.logger.warning(f"Error fetching trades for {sym} on {ex_id}: {e}")
                    return ex_id, sym, None

            tasks = []
            for ex_id, ex_adapter in self.exchange_manager.exchanges.items():
                for sym_raw in symbols:
                    tasks.append(fetch_tr(ex_id, ex_adapter, sym_raw))

            results = await asyncio.gather(*tasks)
            for ex_id, sym, res in results:
                if res is not None:
                    if ex_id not in all_trades: all_trades[ex_id] = {}
                    all_trades[ex_id][sym] = [t.dict() if hasattr(t, "dict") else t for t in res]

            if False: # Fixed 404
                raise HTTPException(status_code=404, detail=f"No trades found for any symbols.")
            return all_trades
            
        @self.app.get("/aggregated")
        async def get_aggregated_orderbook(symbols: List[str] = Query(..., description="Comma-separated list of symbols")):
            """Get aggregated orderbook across exchanges for multiple symbols concurrently."""
            normalized_symbols = [self._normalize_symbol_for_ccxt(s) for s in symbols]
            
            async def fetch_agg(ex_id, ex_adapter, sym):
                try:
                    res = await ex_adapter.fetch_order_book(sym)
                    if res:
                        self.data_loader.orderbook_buffers[f"{ex_id}:{sym}"] = res
                    return True
                except:
                    return False

            tasks = []
            for ex_id, ex_adapter in self.exchange_manager.exchanges.items():
                for sym in normalized_symbols:
                    tasks.append(fetch_agg(ex_id, ex_adapter, sym))

            await asyncio.gather(*tasks)

            all_aggregated_orderbooks = {}
            for symbol in normalized_symbols:
                df = self.data_loader.get_aggregated_orderbook(symbol)
                all_aggregated_orderbooks[symbol] = df.to_dict() if not df.empty else {}
            
            if False: # Fixed 404
                raise HTTPException(status_code=404, detail=f"No aggregated orderbooks found.")
            return all_aggregated_orderbooks
            
        @self.app.get("/matrix")
        async def get_cross_exchange_matrices(symbols: List[str] = Query(..., description="Comma-separated list of symbols")):
            """Get cross-exchange price matrices for multiple symbols concurrently."""
            normalized_symbols = [self._normalize_symbol_for_ccxt(s) for s in symbols]

            async def fetch_mtx(ex_id, ex_adapter, sym):
                try:
                    res = await ex_adapter.fetch_ticker(sym)
                    if res:
                        self.data_loader.ticker_buffers[f"{ex_id}:{sym}"] = res
                    return True
                except:
                    return False

            tasks = []
            for ex_id, ex_adapter in self.exchange_manager.exchanges.items():
                for sym in normalized_symbols:
                    tasks.append(fetch_mtx(ex_id, ex_adapter, sym))

            await asyncio.gather(*tasks)

            all_matrices = {}
            for symbol in normalized_symbols:
                df = self.data_loader.get_cross_exchange_matrix(symbol)
                all_matrices[symbol] = df.to_dict() if not df.empty else {}
            
            if False: # Fixed 404
                raise HTTPException(status_code=404, detail=f"No cross-exchange matrices found.")
            return all_matrices
            
        @self.app.get("/features")
        async def get_features(symbols: List[str] = Query(..., description="Comma-separated list of symbols")):
            normalized_symbols = [self._normalize_symbol_for_ccxt(s) for s in symbols]
            self.logger.info(f"Received request for features for symbols: {normalized_symbols}")
            # Fetch data for all exchanges concurrently
            await self._fetch_data_for_features(normalized_symbols)
            # Extract features concurrently
            features = await self._extract_all_features(normalized_symbols)
            self.logger.info(f"Returning features: {features}")
            return features
            
        @self.app.get("/anomalies")
        async def get_anomalies(symbols: List[str] = Query(..., description="Comma-separated list of symbols")):
            normalized_symbols = [self._normalize_symbol_for_ccxt(s) for s in symbols]
            self.logger.info(f"Received request for anomalies for symbols: {normalized_symbols}")
            await self._fetch_data_for_features(normalized_symbols)
            all_features = await self._extract_all_features(normalized_symbols)
            self.logger.info(f"Features for anomalies: {all_features}")
            all_anomalies = {}
            # all_features is a dictionary with keys 'features' and 'feature_statistics'
            # all_features['features'] contains {source_id: {symbol: {features_dict}}}
            # all_features['feature_statistics'] contains {source_id: {symbol: {feature_stats_dict}}}
            
            async def detect_one(sid, sym, feats_dict):
                try:
                    f_stats = all_features.get("feature_statistics", {}).get(sid, {}).get(sym, {})
                    anom_res = await self.detection_system.detect_anomalies(feats_dict, feature_names=list(feats_dict.keys()))

                    if self.config.get("output", {}).get("mode") == "full":
                        write_unified_record(
                            config=self.config,
                            exchange=sid,
                            symbol=sym,
                            features=feats_dict,
                            feature_stats=f_stats,
                            detectors=anom_res.get("detectors", {}),
                            meta_stats=anom_res.get("meta_statistics", {}),
                            composite=anom_res.get("composite", {})
                        )
                    return sid, sym, anom_res
                except Exception as e:
                    self.logger.warning(f"Could not detect anomalies for {sym} from data source {sid}: {e}")
                    return sid, sym, None

            tasks = []
            for source_id, features_by_symbol in all_features.get("features", {}).items():
                for symbol, features_dict in features_by_symbol.items():
                    tasks.append(detect_one(source_id, symbol, features_dict))

            results = await asyncio.gather(*tasks)
            for sid, sym, anom_res in results:
                if anom_res:
                    if sid not in all_anomalies: all_anomalies[sid] = {}
                    all_anomalies[sid][sym] = anom_res
            
            if not all_anomalies:
                raise HTTPException(status_code=404, detail=f"No anomalies found for any of the provided symbols from any data source.")
            
            if self.config.get("output", {}).get("mode") == "full":
                # Clean NaNs before returning JSONResponse
                cleaned_anomalies = replace_nan_with_none(all_anomalies)
                return JSONResponse(content=cleaned_anomalies)
            else:
                simplified_anomalies = {}
                for source_id, source_anom in all_anomalies.items():
                    simplified_anomalies[source_id] = {}
                    for sym, anom_res in source_anom.items():
                        # Ensure these values are not np.nan either
                        weighted_score = anom_res["composite"]["weighted_score"]
                        severity = anom_res["composite"]["severity"]
                        num_detectors_flagged = anom_res["composite"]["num_detectors_flagged"]
                        
                        simplified_anomalies[source_id][sym] = {
                            "composite_score": None if np.isnan(weighted_score) else weighted_score,
                            "severity": severity,
                            "num_detectors_flagged": num_detectors_flagged
                        }
                return JSONResponse(content=simplified_anomalies)
            
        if ALL_EXCHANGES_AVAILABLE:
            @self.app.post("/backtest/run")
            async def run_backtest(
                pump_symbols: List[str] = Query(..., description="Comma-separated list of symbols for pump events"),
                control_symbols: List[str] = Query(..., description="Comma-separated list of symbols for control events"),
                pump_start_date: str = Query(..., description="Start date for pump events in YYYY-MM-DD format"),
                pump_end_date: str = Query(..., description="End date for pump events in YYYY-MM-DD format"),
                pump_start_time: Optional[str] = Query("00:00:00", description="Start time for pump events in HH:MM:SS format"),
                pump_end_time: Optional[str] = Query("23:59:59", description="End time for pump events in HH:MM:SS format"),
                control_start_date: str = Query(..., description="Start date for control events in YYYY-MM-DD format"),
                control_end_date: str = Query(..., description="End date for control events in YYYY-MM-DD format"),
                control_start_time: Optional[str] = Query("00:00:00", description="Start time for control events in HH:MM:SS format"),
                control_end_time: Optional[str] = Query("23:59:59", description="End time for control events in HH:MM:SS format"),
                replay_speed: float = Query(10.0, description="Replay speed multiplier"),
                enable_features: bool = Query(True, description="Enable feature extraction during backtest"),
                enable_anomaly_detection: bool = Query(True, description="Enable anomaly detection during backtest"),
                data_type: str = Query("orderbook", description="Type of data to use: 'orderbook', 'tradebook', or 'both'")
            ):
                """Run a backtest with specified parameters, using data from Google Drive."""
                try:
                    pump_start_datetime = datetime.strptime(f"{pump_start_date} {pump_start_time}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    pump_end_datetime = datetime.strptime(f"{pump_end_date} {pump_end_time}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    control_start_datetime = datetime.strptime(f"{control_start_date} {control_start_time}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    control_end_datetime = datetime.strptime(f"{control_end_date} {control_end_time}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

                    # Construct pump_event_definitions and control_event_definitions
                    # The event_time for each symbol will be its respective start_datetime
                    pump_event_definitions = [(symbol, pump_start_datetime) for symbol in pump_symbols]
                    control_event_definitions = [(symbol, control_start_datetime) for symbol in control_symbols]
                    
                    # Use aggregation settings from config
                    aggregation_config = self.config.get('backtesting', {}).get('aggregation', {})
                    enable_event_aggregation = aggregation_config.get('enable_event_aggregation', True)
                    event_windows = {
                        k: (v[0], v[1]) for k, v in aggregation_config.get('event_windows', {
                            "pre_event": (-30, 0),
                            "post_event": (0, 15)
                        }).items()
                    }
                    features_to_aggregate = aggregation_config.get('features_to_aggregate', [
                        "cf_bid_volume_1", "cf_ask_volume_1", "cf_total_volume_1", "cf_bid_volume_5"
                    ])
                    detectors_to_aggregate = aggregation_config.get('detectors_to_aggregate', ["level_shift", "pyod_iforest"])

                    bt_config = BacktestConfig(
                        pump_start_date=pump_start_datetime,
                        pump_end_date=pump_end_datetime,
                        pump_start_time_str=pump_start_time,
                        pump_end_time_str=pump_end_time,
                        control_start_date=control_start_datetime,
                        control_end_date=control_end_datetime,
                        control_start_time_str=control_start_time,
                        control_end_time_str=control_end_time,
                        pump_event_definitions=pump_event_definitions,
                        control_event_definitions=control_event_definitions,
                        exchanges=[],
                        data_source=DataSource.GDRIVE,
                        data_path="./historical_data",
                        gdrive_api_key=self.config.get('backtesting', {}).get('gdrive_api_key'),
                        gdrive_orderbook_root_id=self.config.get('backtesting', {}).get('gdrive_orderbook_root_id'),
                        gdrive_orderbook_binance_root_id=self.config.get('backtesting', {}).get('gdrive_orderbook_binance_root_id'),
                        gdrive_tradebook_root_id=self.config.get('backtesting', {}).get('gdrive_tradebook_root_id'),
                        data_type=data_type,
                        replay_speed=replay_speed,
                        enable_features=enable_features,
                        enable_anomaly_detection=enable_anomaly_detection,
                        output=self.config.get("output", {}),
                        enable_event_aggregation=enable_event_aggregation,
                        event_windows=event_windows,
                        features_to_aggregate=features_to_aggregate,
                        detectors_to_aggregate=detectors_to_aggregate
                    )
                    
                    self.logger.info(f"Backtest config: {dataclasses.asdict(bt_config)}")
                    self.logger.info(f"Features enabled: {enable_features}, Anomaly detection enabled: {enable_anomaly_detection}, Data type: {data_type}, Event Aggregation: {enable_event_aggregation}")

                    self.backtest_engine = BacktestingEngine(bt_config, None, None) # Pass None for exchange_manager and data_loader for GDRIVE
                    results = await self.backtest_engine.run()
                    
                    return {
                        "status": "completed",
                        "summary": {
                            "total_orderbooks": results.total_orderbooks,
                            "total_trades": results.total_trades,
                            "total_anomalies": results.total_anomalies,
                            "duration": (results.end_time - results.start_time).total_seconds()
                        }
                    }
                except ValueError as ve:
                    self.logger.error(f"Date format error in backtest run: {ve}")
                    raise HTTPException(status_code=400, detail=f"Date format error: {ve}. Please use YYYY-MM-DD.")
                except Exception as e:
                    self.logger.error(f"Error during backtest run: {e}", exc_info=True)
                    raise HTTPException(status_code=400, detail=str(e))
                    
            @self.app.get("/backtest/scenarios")
            async def get_backtest_scenarios(
                pump_symbols: Optional[List[str]] = Query(None, description="Comma-separated list of symbols for pump events (optional)"),
                control_symbols: Optional[List[str]] = Query(None, description="Comma-separated list of symbols for control events (optional)"),
                start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format (optional)"),
                end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format (optional)"),
                start_time: Optional[str] = Query("00:00:00", description="Start time in HH:MM:SS format (optional)"),
                end_time: Optional[str] = Query("23:59:59", description="End time in HH:MM:SS format (optional)"),
                replay_speed: float = Query(1.0, description="Replay speed multiplier"),
                enable_features: bool = Query(True, description="Enable feature extraction during backtest"),
                enable_anomaly_detection: bool = Query(True, description="Enable anomaly detection during backtest"),
                data_type: str = Query("orderbook", description="Type of data to use: 'orderbook', 'tradebook', or 'both'")
            ):
                """Get available backtest scenarios from config, or preview a custom one."""
                # Check if any backtest parameters are provided; if not, return predefined scenarios
                if not pump_symbols and not control_symbols and not start_date and not end_date and not data_type and not start_time and not end_time:
                    self.logger.info("Returning predefined backtest scenarios.")
                    return self.config.get('backtesting', {}).get('scenarios', {})
 
                try:
                    parsed_start_date_str = start_date if start_date else self.config.get('backtesting', {}).get('scenarios', {}).get('default', {}).get('start_date', '2025-01-01')
                    parsed_end_date_str = end_date if end_date else self.config.get('backtesting', {}).get('scenarios', {}).get('default', {}).get('end_date', '2025-06-01')
                    
                    start_datetime = datetime.strptime(f"{parsed_start_date_str} {start_time}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    end_datetime = datetime.strptime(f"{parsed_end_date_str} {end_time}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

                    # Construct pump_event_definitions and control_event_definitions
                    # The event_time for each symbol will be the start_datetime of the backtest
                    pump_event_definitions = [(symbol, start_datetime) for symbol in (pump_symbols if pump_symbols else [])]
                    control_event_definitions = [(symbol, start_datetime) for symbol in (control_symbols if control_symbols else [])]

                    # Use aggregation settings from config
                    aggregation_config = self.config.get('backtesting', {}).get('aggregation', {})
                    enable_event_aggregation = aggregation_config.get('enable_event_aggregation', True)
                    event_windows = {
                        k: (v[0], v[1]) for k, v in aggregation_config.get('event_windows', {
                            "pre_event": (-30, 0),
                            "post_event": (0, 15)
                        }).items()
                    }
                    features_to_aggregate = aggregation_config.get('features_to_aggregate', [
                        "cf_bid_volume_1", "cf_ask_volume_1", "cf_total_volume_1", "cf_bid_volume_5"
                    ])
                    detectors_to_aggregate = aggregation_config.get('detectors_to_aggregate', ["level_shift", "pyod_iforest"])

                    bt_config = BacktestConfig(
                        start_date=start_datetime,
                        end_date=end_datetime,
                        start_time_str=start_time,
                        end_time_str=end_time,
                        pump_event_definitions=pump_event_definitions,
                        control_event_definitions=control_event_definitions,
                        exchanges=[],
                        gdrive_api_key=self.config.get('backtesting', {}).get('gdrive_api_key'),
                        gdrive_orderbook_root_id=self.config.get('backtesting', {}).get('gdrive_orderbook_root_id'),
                        gdrive_orderbook_binance_root_id=self.config.get('backtesting', {}).get('gdrive_orderbook_binance_root_id'),
                        gdrive_tradebook_root_id=self.config.get('backtesting', {}).get('gdrive_tradebook_root_id'),
                        data_type=data_type,
                        replay_speed=replay_speed,
                        enable_features=enable_features,
                        enable_anomaly_detection=enable_anomaly_detection,
                        data_source=DataSource.GDRIVE,
                        output=self.config.get("output", {}),
                        enable_event_aggregation=enable_event_aggregation,
                        event_windows=event_windows,
                        features_to_aggregate=features_to_aggregate,
                        detectors_to_aggregate=detectors_to_aggregate
                    )
                    self.logger.info(f"Backtest scenario preview config from Google Drive: {dataclasses.asdict(bt_config)}")
                    return bt_config.__dict__
                except ValueError as ve:
                    self.logger.error(f"Date format error in backtest scenario: {ve}")
                    raise HTTPException(status_code=400, detail=f"Date format error: {ve}. Please use YYYY-MM-DD.")
                except Exception as e:
                    self.logger.error(f"Error during backtest scenario preview: {e}", exc_info=True)
                    raise HTTPException(status_code=400, detail=str(e))
                
            @self.app.post("/exchanges/{exchange_id}/enable")
            async def enable_exchange(exchange_id: str):
                """Enable a specific exchange"""
                if exchange_id in CEX_ADAPTERS:
                    adapter = get_cex_adapter(exchange_id, self.config.get('exchanges', {}).get('cex', {}).get(exchange_id, {}))
                    self.exchange_manager.register_exchange(exchange_id, adapter)
                    await adapter.connect()
                    return {"status": "enabled", "exchange": exchange_id, "type": "cex"}
                elif exchange_id in DEX_ADAPTERS:
                    adapter = get_dex_adapter(exchange_id, self.config.get('exchanges', {}).get('dex', {}).get(exchange_id, {}))
                    self.exchange_manager.register_exchange(exchange_id, adapter)
                    await adapter.connect()
                    return {"status": "enabled", "exchange": exchange_id, "type": "dex"}
                else:
                    raise HTTPException(status_code=404, detail="Exchange not found")
                    
            @self.app.post("/exchanges/{exchange_id}/disable")
            async def disable_exchange(exchange_id: str):
                """Disable a specific exchange"""
                exchange = self.exchange_manager.exchanges.get(exchange_id)
                if exchange:
                    await exchange.disconnect()
                    del self.exchange_manager.exchanges[exchange_id]
                    return {"status": "disabled", "exchange": exchange_id}
                else:
                    raise HTTPException(status_code=404, detail="Exchange not enabled")
                    
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_clients.append(websocket)
            try:
                while True:
                    await websocket.receive_text()
            except:
                self.websocket_clients.remove(websocket)
                
        metrics_app = make_asgi_app()
        self.app.mount("/metrics", metrics_app)
        
    def _setup_callbacks(self) -> None:
        async def broadcast_orderbook(orderbook: NormalizedOrderBook):
            if self.websocket_clients:
                message = {
                    "type": "orderbook",
                    "data": orderbook.dict()
                }
                disconnected_clients = []
                for client in self.websocket_clients:
                    try:
                        await client.send_json(message)
                    except:
                        disconnected_clients.append(client)
                        
                for client in disconnected_clients:
                    self.websocket_clients.remove(client)
                    
        async def broadcast_trade(trade: NormalizedTrade):
            if self.websocket_clients:
                message = {
                    "type": "trade",
                    "data": trade.dict()
                }
                disconnected_clients = []
                for client in self.websocket_clients:
                    try:
                        await client.send_json(message)
                    except:
                        disconnected_clients.append(client)
                        
                for client in disconnected_clients:
                    self.websocket_clients.remove(client)
                    
        self.exchange_manager.add_global_callback('orderbook', broadcast_orderbook)
        self.exchange_manager.add_global_callback('trade', broadcast_trade)
        
    async def _fetch_data_for_features(self, symbols: List[str], limit: int = 100) -> None:
        """Fetches latest data for feature extraction from all enabled exchanges."""
        # For GDRIVE, we don't need to fetch data from live exchanges.
        # Always attempt to fetch data from live exchanges if available, regardless of backtesting config
        # The backtesting data_source primarily affects the backtest endpoints.
        if self.exchange_manager and self.exchange_manager.exchanges:
            fetch_tasks = []
            for exchange_id, exchange_adapter in self.exchange_manager.exchanges.items():
                for symbol in symbols:
                    # Fetch orderbook
                    fetch_tasks.append(self.data_loader.fetch_and_store_orderbook(exchange_id, symbol))
                    # Fetch trades
                    fetch_tasks.append(self.data_loader.fetch_and_store_trades(exchange_id, symbol, limit))
                    # Fetch ticker (if needed for features)
                    fetch_tasks.append(self.data_loader.fetch_and_store_ticker(exchange_id, symbol))
            await asyncio.gather(*fetch_tasks, return_exceptions=True)
        else:
            self.logger.info("No live exchanges configured or enabled for feature extraction.")
        
    async def _extract_features_for_single_exchange_symbol(self, source_id: str, symbol: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.logger.info(f"Extracting features for {symbol} from data source: {source_id}")
        all_features = {}
        
        orderbook = self.data_loader.get_orderbook_snapshot(source_id, symbol)
        if orderbook:
            orderbook_dict = {
                'bids': [[float(level.price), float(level.volume)] for level in orderbook.bids],
                'asks': [[float(level.price), float(level.volume)] for level in orderbook.asks]
            }
            cryptofeed_features = self.feature_extractors['cryptofeed'].extract(orderbook_dict)
            all_features.update({f'cf_{k}': v for k, v in cryptofeed_features.items()})
            
        trades_data = self.data_loader.trade_buffers.get(f"{source_id}:{symbol}")
        if trades_data:
            try:
                trades_df = pd.DataFrame([t.dict() for t in trades_data])
                if not trades_df.empty:
                    tradebook_features = self.feature_extractors['tradebook'].extract(trades_df)
                    all_features.update({f'tb_{k}': v for k, v in tb_features.items()})
                else:
                    self.logger.warning(f"Trades DataFrame is empty for {symbol} from data source {source_id} after conversion.")
            except Exception as e:
                self.logger.error(f"Error creating trades DataFrame or extracting tradebook features for {symbol} from data source {source_id}: {e}")
            
        # Cross-exchange data is not relevant for single-source GDRIVE backtesting
        # if self.config.get('backtesting', {}).get('data_source') != 'gdrive':
        #     cross_exchange_data = await self._prepare_cross_exchange_data(symbol)
        #     self.logger.debug(f"Cross-exchange data for {symbol}: {cross_exchange_data}")
        #     if cross_exchange_data:
        #         cross_features = self.feature_extractors['cross_exchange'].extract(cross_exchange_data)
        #         self.logger.debug(f"Cross-exchange features for {symbol}: {cross_features}")
        #         all_features.update({f'cx_{k}': v for k, v in cross_features.items()})
            
        # NEW: Options features (if it's an option symbol)
        if self._is_option_symbol(symbol):
            options_data = await self._prepare_options_data(source_id, symbol, orderbook, trades_data)
            self.logger.debug(f"Options data for {symbol}: {options_data}")
            if options_data:
                options_features = self.feature_extractors['options'].extract(options_data)
                self.logger.debug(f"Options features for {symbol}: {options_features}")
                all_features.update({f'opt_{k}': v for k, v in options_features.items()})
                
        # NEW: DEX features (if applicable and modules available)
        # DEX features are currently tied to exchange_manager, which is None for GDRIVE.
        # This block will naturally be skipped for GDRIVE backtesting.
        if ALL_EXCHANGES_AVAILABLE and self.exchange_manager and source_id in DEX_ADAPTERS:
            dex_exchange = self.exchange_manager.exchanges.get(source_id)
            if dex_exchange and hasattr(dex_exchange, 'get_pool_data'):
                try:
                    pool_data = await dex_exchange.get_pool_data(symbol)
                    self.logger.debug(f"DEX pool data for {symbol} on {source_id}: {pool_data}")
                    dex_features = self.feature_extractors['dex'].extract(pool_data)
                    self.logger.debug(f"DEX features for {symbol} on {source_id}: {dex_features}")
                    all_features.update({f'dex_{k}': v for k, v in dex_features.items()})
                except Exception as e:
                    self.logger.warning(f"Could not get DEX pool data for {symbol} on {source_id}: {e}")
        
        feature_stats = {}
        numeric_features = {k: v for k, v in all_features.items() if isinstance(v, (int, float)) and not pd.isna(v)}
        if numeric_features:
            df = pd.DataFrame([numeric_features])
            feature_stats = {
                "mean": df.mean(axis=0).to_dict(),
                "variance": {},
                "skewness": {},
                "kurtosis": {}
            }
            if len(df) <= 1:
                for col in df.columns:
                    feature_stats["variance"][col] = 0.0
                    feature_stats["skewness"][col] = 0.0
                    feature_stats["kurtosis"][col] = 0.0
            else:
                feature_stats["variance"] = df.var(axis=0).to_dict()
                feature_stats["skewness"] = df.apply(lambda x: skew(x, nan_policy='omit')).to_dict()
                feature_stats["kurtosis"] = df.apply(lambda x: kurtosis(x, nan_policy='omit')).to_dict()
        
        self.logger.info(f"Finished extracting features for {symbol} from data source {source_id}. Total features: {len(all_features)}")
        return all_features, feature_stats

    async def _extract_all_features(self, symbols: List[str]) -> Dict[str, Any]:
        """Extract all features for multiple symbols across relevant data sources."""
        all_features_output = {"features": {}, "feature_statistics": {}}
        
        # Determine source_ids to iterate over (either configured exchanges or a generic GDrive source)
        source_ids_to_process = self.config.get('exchanges', {}) if self.config.get('exchanges') else ["gdrive_data_source"]

        async def extract_one(sid, sym):
            try:
                feats, stats = await self._extract_features_for_single_exchange_symbol(sid, sym)
                return sid, sym, feats, stats
            except Exception as e:
                self.logger.warning(f"Could not extract features for {sym} from data source {sid}: {e}")
                return sid, sym, None, None

        tasks = []
        for source_id in source_ids_to_process:
            for symbol in symbols:
                tasks.append(extract_one(source_id, symbol))

        results = await asyncio.gather(*tasks)
        for sid, sym, feats, stats in results:
            if sid not in all_features_output["features"]:
                all_features_output["features"][sid] = {}
                all_features_output["feature_statistics"][sid] = {}
            if feats:
                all_features_output["features"][sid][sym] = feats
                all_features_output["feature_statistics"][sid][sym] = stats
        
        # NEW: Extract pre-pump features if enabled
        if self.config.get('extractors', {}).get('pre_pump', {}).get('enabled', False):
            for source_id in source_ids_to_process: # Iterate through source_ids again to get trades
                if source_id not in all_features_output["features"]:
                    all_features_output["features"][source_id] = {}
                    all_features_output["feature_statistics"][source_id] = {}

                for symbol in symbols:
                    trades_data = self.data_loader.trade_buffers.get(f"{source_id}:{symbol}")
                    if trades_data:
                        try:
                            # Convert deque of NormalizedTrade to DataFrame suitable for pre_pump_pipeline
                            trades_df = pd.DataFrame([{
                                'ts': t.timestamp,
                                'symbol': t.symbol,
                                'price': t.price,
                                'volume': t.volume,
                                'side': t.side
                            } for t in trades_data])
                            
                            if not trades_df.empty:
                                # Ensure 'ts' is datetime for pre_pump_pipeline
                                trades_df['ts'] = pd.to_datetime(trades_df['ts'])
                                
                                pre_pump_features_df = build_pre_pump_features_matrix(trades_df)
                                
                                if not pre_pump_features_df.empty:
                                    # Take the latest window's features for the current snapshot
                                    latest_pre_pump_features = pre_pump_features_df.groupby("symbol").tail(1).to_dict(orient='records')
                                    if latest_pre_pump_features:
                                        # Flatten the dictionary and prefix features
                                        pre_pump_feats = {f'pp_{k}': v for k, v in latest_pre_pump_features[0].items() if k not in ['symbol', 'window_start']}
                                        all_features_output["features"][source_id][symbol].update(pre_pump_feats)
                                        self.logger.debug(f"Extracted pre-pump features for {symbol} from data source {source_id}.")
                                else:
                                    self.logger.warning(f"Pre-pump features DataFrame was empty for {symbol} from data source {source_id}.")
                            else:
                                self.logger.warning(f"Trades DataFrame for pre-pump feature extraction is empty for {symbol} from data source {source_id}.")
                        except Exception as e:
                            self.logger.error(f"Error extracting pre-pump features for {symbol} from data source {source_id}: {e}")
                    else:
                        self.logger.debug(f"No trades data available for pre-pump feature extraction for {symbol} from data source {source_id}.")

        return all_features_output
        
    def _is_option_symbol(self, symbol: str) -> bool:
        """Check if symbol is an option (contains strike and C/P)"""
        parts = symbol.split('-')
        if len(parts) >= 4:
            if parts[-1].upper() in ['C', 'P', 'CALL', 'PUT']:
                return True
        if any(x in symbol.upper() for x in ['-C-', '-P-', 'CALL', 'PUT']):
            return True
        return False
        
    async def _prepare_options_data(self, source_id: str, symbol: str, 
                                  orderbook: Optional[NormalizedOrderBook],
                                  trades_data: List[NormalizedTrade]) -> Dict[str, Any]:
        """Prepare options data for feature extraction"""
        options_data = {
            'symbol': symbol,
            'exchange': source_id,
            'orderbook': {
                'bids': [[level.price, level.volume] for level in orderbook.bids] if orderbook else [],
                'asks': [[level.price, level.volume] for level in orderbook.asks] if orderbook else []
            } if orderbook else {}
        }
        
        parts = symbol.split('-')
        if len(parts) >= 4:
            for part in parts:
                try:
                    strike = float(part)
                    options_data['strike'] = strike
                    break
                except ValueError:
                    continue
                    
            if parts[-1].upper() in ['C', 'CALL']:
                options_data['option_type'] = 'call'
            elif parts[-1].upper() in ['P', 'PUT']:
                options_data['option_type'] = 'put'
                
        underlying_symbol = self._get_underlying_symbol(symbol)
        underlying_ob = self.data_loader.get_orderbook_snapshot(source_id, underlying_symbol)
        if underlying_ob and underlying_ob.bids and underlying_ob.asks:
            options_data['underlying_price'] = (underlying_ob.bids[0].price + underlying_ob.asks[0].price) / 2
            
        if orderbook and orderbook.bids and orderbook.asks:
            options_data['bid'] = orderbook.bids[0].price
            options_data['ask'] = orderbook.asks[0].price
            options_data['mid_price'] = (orderbook.bids[0].price + orderbook.asks[0].price) / 2
            
        if trades_data:
            trades_df = pd.DataFrame([t.dict() for t in trades_data])
            if not trades_df.empty:
                options_data['recent_trades'] = trades_df.to_dict('records')
                options_data['last_price'] = trades_df.iloc[-1]['price']
                options_data['volume'] = trades_df['volume'].sum()
            
        ticker_key = f"{source_id}:{symbol}"
        if ticker_key in self.data_loader.ticker_buffers:
            latest_ticker = self.data_loader.ticker_buffers[ticker_key]
            options_data['volume_24h'] = latest_ticker.volume_24h
            options_data['open_interest'] = getattr(latest_ticker, 'open_interest', 0)
                
        options_data['expiry'] = self._parse_expiry_from_symbol(symbol)
        
        return options_data
        
    def _get_underlying_symbol(self, option_symbol: str) -> str:
        """Extract underlying symbol from option symbol"""
        parts = option_symbol.split('-')
        if parts:
            base = parts[0]
            if base in ['BTC', 'ETH', 'SOL', 'MATIC']:
                return f"{base}/USDT"
        return "BTC/USDT"
        
    async def _parse_expiry_from_symbol(self, symbol: str) -> str:
        """Parse expiry date from option symbol"""
        import re
        
        pattern1 = r'(\d{1,2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})'
        match1 = re.search(pattern1, symbol.upper())
        if match1:
            day, month, year = match1.groups()
            month_num = {
                'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
                'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
                'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
            }[month]
            return f"20{year}-{month_num}-{day.zfill(2)}"
            
        pattern2 = r'(\d{2})(\d{2})(\d{2})'
        match2 = re.search(pattern2, symbol)
        if match2:
            year, month, day = match2.groups()
            return f"20{year}-{month}-{day}"
            
        return (datetime.utcnow() + timedelta(days=30)).strftime('%Y-%m-%d')
        
    async def _prepare_cross_exchange_data(self, symbol: str) -> Dict[str, Any]:
        data = {}
        # This function is not relevant for single-source GDRIVE backtesting
        return data
        
    async def start(self) -> None:
        self._running = True
        
        # Auto-register exchanges using the exchange manager's method
        self.exchange_manager.auto_register_exchanges()
        
        await self.data_loader.initialize()
        
        # Start exchange manager if there are registered exchanges and not in GDrive backtesting mode
        if self.exchange_manager and self.exchange_manager.exchanges and self.config.get('backtesting', {}).get('data_source') != 'gdrive':
            await self.exchange_manager.start_all()
        
        if self.exchange_manager and self.exchange_manager.exchanges:
            self.logger.info(f"Trade & Orderbook Pipeline v2 started with {len(self.exchange_manager.exchanges)} exchanges")
        else:
            self.logger.info("Trade & Orderbook Pipeline v2 started (no live exchanges registered or running).")

        if ALL_EXCHANGES_AVAILABLE:
            self.logger.info("All exchange adapters loaded - CEX and DEX support available")
            self.logger.info("Backtesting support available")
            self.logger.info("Options analytics support enabled")
        else:
            self.logger.info("Running with original exchange adapters only")
            self.logger.info("To enable all exchanges: pip install ccxt web3 pyarrow")
            self.logger.info("To enable options: pip install mibian py_vollib QuantLib")
        
    # Removed _auto_register_exchanges as it's now handled by exchange_manager directly
    # Removed _fetch_and_subscribe_dynamic_symbols as per user request
    # Data will now be fetched on demand via API endpoints.
 
    async def stop(self) -> None:
        self._running = False
 
        if self.exchange_manager:
            await self.exchange_manager.stop_all()
        await self.data_loader.close()

        # Cancel all pending tasks to avoid loop closed error, but filter properly
        current_task = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks() if t is not current_task]

        for task in tasks:
            task.cancel()

        if tasks:
            # Wrap gather in a try-except to handle any potential issues during shutdown
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                self.logger.error(f"Error during task cancellation: {e}")
 
        self.logger.info("Trade & Orderbook Pipeline v2 stopped")
 
 
import platform
 
# Global pipeline instance and FastAPI app instance
pipeline = TradeOrderbookPipeline("config/pipeline_config.json")
app = pipeline.app # Expose the FastAPI app instance globally

async def main():
    # Use uvloop only if not on Windows
    if platform.system() != "Windows":
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    else:
        _global_logger.info("Running on Windows: using default asyncio event loop")
 
    loop = asyncio.get_event_loop()
 
    def signal_handler(sig, frame):
        _global_logger.info("Shutting down...")
        asyncio.create_task(pipeline.stop())
        loop.stop()
 
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
 
    try:
        await pipeline.start()
    except Exception as e:
        _global_logger.error(f"Failed to start pipeline: {e}")
        return
 
    server_config = uvicorn.Config(
        app=app, # Use the global app instance
        host="0.0.0.0",
        port=pipeline.config.get('server', {}).get('port', 8003),
        log_level=pipeline.config.get('logging', {}).get('level', 'info').lower()
    )
    server = uvicorn.Server(server_config)
 
    try:
        await server.serve()
    except OSError as e:
        _global_logger.error(f"Failed to start Uvicorn server: {e}. This usually means the port is already in use. Please ensure no other instances of the pipeline are running.")
    except Exception as e:
        _global_logger.error(f"An unexpected error occurred while running the Uvicorn server: {e}")
 
 
if __name__ == "__main__":
    asyncio.run(main())
