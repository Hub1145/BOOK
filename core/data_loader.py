# core/data_loader.py
"""Data loader for buffering and managing market data"""


import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle

import ccxt # Import ccxt to catch specific exceptions

from core.exchange_interface import NormalizedOrderBook, NormalizedTrade, NormalizedTicker
from core.exchange_manager_updated import ExchangeManager


logger = logging.getLogger(__name__)


class CircularBuffer:
    """Circular buffer with time-based expiration"""

    def __init__(self, maxlen: int = 1000, max_age_seconds: int = 3600):
        self.data = deque(maxlen=maxlen)
        self.max_age = timedelta(seconds=max_age_seconds)

    def append(self, item: Tuple[datetime, Any]) -> None:
        """Add item with timestamp"""
        self.data.append(item)
        self._cleanup()

    def _cleanup(self) -> None:
        """Remove old items"""
        cutoff = datetime.utcnow() - self.max_age
        while self.data and self.data[0][0] < cutoff:
            self.data.popleft()

    def get_recent(self, seconds: int) -> List[Any]:
        """Get items from last N seconds"""
        cutoff = datetime.utcnow() - timedelta(seconds=seconds)
        return [item[1] for item in self.data if item[0] > cutoff]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        if not self.data:
            return pd.DataFrame()
        items = [item[1] for item in self.data]
        if hasattr(items[0], '__dict__'):
            return pd.DataFrame([item.__dict__ for item in items])
        return pd.DataFrame(items)


class DataLoader:
    """Manages data buffering and aggregation for on-demand fetching"""

    def __init__(self, exchange_manager: ExchangeManager, config: Dict[str, Any] = None):
        self.exchange_manager = exchange_manager
        self.config = config or {}

        # Configuration
        self.buffer_size = self.config.get('buffer_size', 1000)
        self.orderbook_snapshot_interval = self.config.get('orderbook_snapshot_interval', 60)
        self.trade_buffer_minutes = self.config.get('trade_buffer_minutes', 30)

        # Data buffers (will be populated on demand)
        self.orderbook_buffers: Dict[str, NormalizedOrderBook] = {} # Stores latest snapshot
        self.trade_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.buffer_size)) # Stores recent trades
        self.ticker_buffers: Dict[str, NormalizedTicker] = {} # Stores latest ticker

        # Persistence (retained but usage changes with on-demand fetching)
        self.enable_persistence = self.config.get('enable_persistence', False)
        self.persistence_path = Path(self.config.get('persistence_path', './data/buffers'))

    async def initialize(self) -> None:
        """Initialize data loader (no subscriptions, only loads persisted data if enabled)"""
        if self.enable_persistence:
            await self._load_persisted_data()

        logger.info("Data loader initialized")

    async def fetch_and_store_orderbook(self, exchange_id: str, symbol: str) -> Optional[NormalizedOrderBook]:
        """Fetches and stores the latest orderbook for a given symbol."""
        exchange_adapter = self.exchange_manager.get_exchange(exchange_id)
        if not exchange_adapter:
            logger.warning(f"Exchange adapter for {exchange_id} not found.")
            return None

        try:
            normalized_ob = await exchange_adapter.fetch_order_book(symbol)
            if normalized_ob:
                self.orderbook_buffers[f"{exchange_id}:{symbol}"] = normalized_ob
                logger.debug(f"Fetched and stored orderbook for {symbol} on {exchange_id}. Levels: bids={len(normalized_ob.bids)}, asks={len(normalized_ob.asks)}")
                return normalized_ob
            else:
                logger.debug(f"No orderbook data returned for {symbol} on {exchange_id}.")
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            logger.warning(f"Error fetching orderbook for {symbol} on {exchange_id}: {e}")
        return None

    async def fetch_and_store_trades(self, exchange_id: str, symbol: str, limit: int = 100) -> List[NormalizedTrade]:
        """Fetches and stores recent trades for a given symbol."""
        exchange_adapter = self.exchange_manager.get_exchange(exchange_id)
        if not exchange_adapter:
            logger.warning(f"Exchange adapter for {exchange_id} not found.")
            return []

        try:
            normalized_trades = await exchange_adapter.fetch_trades(symbol, limit=limit)
            if normalized_trades:
                key = f"{exchange_id}:{symbol}"
                # Append new trades to the deque, maintaining maxlen
                for trade in normalized_trades:
                    self.trade_buffers[key].append(trade)
                logger.debug(f"Fetched and stored {len(normalized_trades)} trades for {symbol} on {exchange_id}. Current buffer size: {len(self.trade_buffers[key])}")
                return normalized_trades
            else:
                logger.debug(f"No trade data returned for {symbol} on {exchange_id}.")
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            logger.warning(f"Error fetching trades for {symbol} on {exchange_id}: {e}")
        return []

    async def fetch_and_store_ticker(self, exchange_id: str, symbol: str) -> Optional[NormalizedTicker]:
        """Fetches and stores the latest ticker for a given symbol."""
        exchange_adapter = self.exchange_manager.get_exchange(exchange_id)
        if not exchange_adapter:
            logger.warning(f"Exchange adapter for {exchange_id} not found.")
            return None

        try:
            normalized_ticker = await exchange_adapter.fetch_ticker(symbol)
            if normalized_ticker:
                self.ticker_buffers[f"{exchange_id}:{symbol}"] = normalized_ticker
                logger.debug(f"Fetched and stored ticker for {symbol} on {exchange_id}.")
                return normalized_ticker
            else:
                logger.debug(f"No ticker data returned for {symbol} on {exchange_id}.")
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            logger.warning(f"Error fetching ticker for {symbol} on {exchange_id}: {e}")
        return None

    def get_orderbook_snapshot(self, exchange: str, symbol: str) -> Optional[NormalizedOrderBook]:
        """Retrieves latest orderbook snapshot from buffer."""
        key = f"{exchange}:{symbol}"
        return self.orderbook_buffers.get(key)

    def get_recent_trades(self, exchange: str, symbol: str, count: int = 100) -> List[NormalizedTrade]:
        """Retrieves recent trades from buffer."""
        key = f"{exchange}:{symbol}"
        # Return last 'count' trades from the deque
        return list(self.trade_buffers[key])[-count:]

    def get_aggregated_orderbook(self, symbol: str, depth: int = 20) -> pd.DataFrame:
        """Get aggregated orderbook across exchanges from currently buffered data"""
        data = []

        for key, orderbook in self.orderbook_buffers.items():
            if symbol in key: # Make sure this matches the symbol format used for keys
                exchange = key.split(':')[0]

                # Add bids
                for i, level in enumerate(orderbook.bids[:depth]):
                    data.append({
                        'exchange': exchange,
                        'side': 'bid',
                        'price': level.price,
                        'volume': level.volume,
                        'level': i
                    })

                # Add asks
                for i, level in enumerate(orderbook.asks[:depth]):
                    data.append({
                        'exchange': exchange,
                        'side': 'ask',
                        'price': level.price,
                        'volume': level.volume,
                        'level': i
                    })

        return pd.DataFrame(data)

    def get_cross_exchange_matrix(self, symbol: str) -> pd.DataFrame:
        """Get cross-exchange price matrix from currently buffered data"""
        data = {}

        for key, ticker in self.ticker_buffers.items():
            if symbol in key: # Make sure this matches the symbol format used for keys
                exchange = key.split(':')[0]
                data[exchange] = {
                    'bid': ticker.bid,
                    'ask': ticker.ask,
                    'last': ticker.last,
                    'spread': ticker.ask - ticker.bid,
                    'mid': (ticker.bid + ticker.ask) / 2
                }

        return pd.DataFrame(data).T

    async def persist_buffers(self) -> None:
        """Save buffers to disk"""
        if not self.enable_persistence:
            return

        self.persistence_path.mkdir(parents=True, exist_ok=True)

        # Save orderbooks
        orderbook_data = {
            key: ob.dict()
            for key, ob in self.orderbook_buffers.items()
        }

        with open(self.persistence_path / 'orderbooks.json', 'w') as f:
            json.dump(orderbook_data, f)

        # Save trades (need to convert deque to list for JSON serialization)
        trade_data = {
            key: [trade.dict() for trade in buffer]
            for key, buffer in self.trade_buffers.items()
        }

        with open(self.persistence_path / 'trades.json', 'w') as f:
            json.dump(trade_data, f)

        # Save tickers
        ticker_data = {
            key: ticker.dict()
            for key, ticker in self.ticker_buffers.items()
        }

        with open(self.persistence_path / 'tickers.json', 'w') as f:
            json.dump(ticker_data, f)

        logger.info("Persisted buffers to disk")

    async def _load_persisted_data(self) -> None:
        """Load persisted data"""
        # Load orderbooks
        orderbook_path = self.persistence_path / 'orderbooks.json'
        if orderbook_path.exists():
            with open(orderbook_path, 'r') as f:
                loaded_data = json.load(f)
                for key, data in loaded_data.items():
                    self.orderbook_buffers[key] = NormalizedOrderBook(**data)
            logger.info(f"Loaded {len(self.orderbook_buffers)} persisted orderbooks.")

        # Load trades
        trades_path = self.persistence_path / 'trades.json'
        if trades_path.exists():
            with open(trades_path, 'r') as f:
                loaded_data = json.load(f)
                for key, data_list in loaded_data.items():
                    for data in data_list:
                        self.trade_buffers[key].append(NormalizedTrade(**data))
            logger.info(f"Loaded {sum(len(b) for b in self.trade_buffers.values())} persisted trades.")

        # Load tickers
        tickers_path = self.persistence_path / 'tickers.json'
        if tickers_path.exists():
            with open(tickers_path, 'r') as f:
                loaded_data = json.load(f)
                for key, data in loaded_data.items():
                    self.ticker_buffers[key] = NormalizedTicker(**data)
            logger.info(f"Loaded {len(self.ticker_buffers)} persisted tickers.")

    async def close(self) -> None:
        """Close data loader"""
        if self.enable_persistence:
            await self.persist_buffers()