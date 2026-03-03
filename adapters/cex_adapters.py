"""Adapters for research-identified centralized exchanges"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import aiohttp

try:
    import ccxt.async_support as ccxt_async
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt_async = None
    ccxt = None

from core.exchange_interface import (
    ExchangeInterface, NormalizedOrderBook, NormalizedTrade,
    NormalizedTicker, OrderBookLevel
)

logger = logging.getLogger(__name__)

# --- Base Adapters ---

class CCXTAdapter(ExchangeInterface):
    """Base adapter for CCXT-compatible exchanges"""

    def __init__(self, exchange_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(exchange_id, config)
        if not CCXT_AVAILABLE:
            raise ImportError("CCXT is not available")

        self.exchange_class = getattr(ccxt_async, exchange_id)

        params = {
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            **(config or {})
        }

        self.exchange = self.exchange_class(params)

    async def connect(self) -> None:
        try:
            await self.exchange.load_markets()
            self.connected = True
            logger.info(f"Connected to CCXT exchange: {self.exchange_id}")
        except Exception as e:
            logger.error(f"Error connecting to CCXT {self.exchange_id}: {e}")
            raise

    async def disconnect(self) -> None:
        if hasattr(self.exchange, 'close'):
            await self.exchange.close()
        self.connected = False

    async def fetch_order_book(self, symbol: str) -> Dict:
        ob = await self.exchange.fetch_order_book(symbol)
        return self._normalize_orderbook(ob, symbol)

    async def fetch_trades(self, symbol: str, limit: Optional[int] = None) -> List[Dict]:
        trades = await self.exchange.fetch_trades(symbol, limit=limit)
        return [self._normalize_trade(t, symbol) for t in trades]

    async def fetch_ticker(self, symbol: str) -> Dict:
        ticker = await self.exchange.fetch_ticker(symbol)
        return self._normalize_ticker(ticker, symbol)

    def _normalize_orderbook(self, orderbook: Dict, symbol: str) -> NormalizedOrderBook:
        normalized_bids = [OrderBookLevel(price=float(b[0]), volume=float(b[1])) for b in orderbook.get('bids', [])[:50]]
        normalized_asks = [OrderBookLevel(price=float(a[0]), volume=float(a[1])) for a in orderbook.get('asks', [])[:50]]
        return NormalizedOrderBook(
            exchange=self.exchange_id,
            symbol=symbol,
            timestamp=datetime.fromtimestamp(orderbook['timestamp'] / 1000) if orderbook.get('timestamp') else datetime.utcnow(),
            bids=normalized_bids,
            asks=normalized_asks,
            sequence=orderbook.get('nonce', 0)
        )

    def _normalize_trade(self, trade: Dict, symbol: str) -> NormalizedTrade:
        return NormalizedTrade(
            exchange=self.exchange_id,
            symbol=symbol,
            timestamp=datetime.fromtimestamp(trade['timestamp'] / 1000) if trade.get('timestamp') else datetime.utcnow(),
            id=str(trade.get('id', '')),
            price=float(trade.get('price', 0.0)),
            volume=float(trade.get('amount', 0.0)),
            side=trade.get('side', 'unknown'),
            taker_side=trade.get('takerOrMaker', trade.get('side', 'unknown'))
        )

    def _normalize_ticker(self, ticker: Dict, symbol: str) -> NormalizedTicker:
        return NormalizedTicker(
            exchange=self.exchange_id,
            symbol=symbol,
            timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000) if ticker.get('timestamp') else datetime.utcnow(),
            bid=ticker.get('bid', 0),
            ask=ticker.get('ask', 0),
            last=ticker.get('last', 0),
            volume_24h=ticker.get('baseVolume', ticker.get('quoteVolume', 0)),
            high_24h=ticker.get('high', 0),
            low_24h=ticker.get('low', 0)
        )


class CustomCEXAdapter(ExchangeInterface):
    """Base for custom CEX adapters using public endpoints and returning normalized data"""
    def __init__(self, exchange_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(exchange_id, config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 0.5

    async def connect(self) -> None:
        if not self.session:
            self.session = aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0'})
        self.connected = True

    async def disconnect(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False

    async def _get(self, url: str, params: Optional[Dict] = None) -> Dict:
        if not self.session: await self.connect()
        try:
            async with self.session.get(url, params=params) as response:
                await asyncio.sleep(self.rate_limit_delay)
                if response.status == 200: return await response.json()
                return {}
        except: return {}

    def _normalize_ticker(self, raw: Dict, symbol: str) -> NormalizedTicker:
        return NormalizedTicker(
            exchange=self.exchange_id, symbol=symbol, timestamp=datetime.utcnow(),
            bid=float(raw.get('bid', 0)), ask=float(raw.get('ask', 0)),
            last=float(raw.get('last', 0)), volume_24h=float(raw.get('volume', 0)),
            high_24h=0, low_24h=0
        )

# --- Custom Implementations ---

class BitrueAdapter(CustomCEXAdapter):
    def __init__(self, config=None): super().__init__('bitrue', config)
    async def fetch_ticker(self, symbol: str) -> NormalizedTicker:
        s = symbol.replace('/', '').upper()
        raw = await self._get(f"https://openapi.bitrue.com/api/v1/ticker/24hr", {"symbol": s})
        return NormalizedTicker(
            exchange=self.exchange_id, symbol=symbol, timestamp=datetime.utcnow(),
            bid=float(raw.get('bidPrice', 0)), ask=float(raw.get('askPrice', 0)),
            last=float(raw.get('lastPrice', 0)), volume_24h=float(raw.get('volume', 0)),
            high_24h=float(raw.get('highPrice', 0)), low_24h=float(raw.get('lowPrice', 0))
        )
    async def fetch_order_book(self, symbol: str) -> NormalizedOrderBook:
        s = symbol.replace('/', '').upper()
        raw = await self._get(f"https://openapi.bitrue.com/api/v1/depth", {"symbol": s, "limit": 100})
        bids = [OrderBookLevel(price=float(b[0]), volume=float(b[1])) for b in raw.get('bids', [])]
        asks = [OrderBookLevel(price=float(a[0]), volume=float(a[1])) for a in raw.get('asks', [])]
        return NormalizedOrderBook(self.exchange_id, symbol, datetime.utcnow(), bids, asks)
    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[NormalizedTrade]:
        s = symbol.replace('/', '').upper()
        raw = await self._get(f"https://openapi.bitrue.com/api/v1/trades", {"symbol": s, "limit": limit})
        return [NormalizedTrade(self.exchange_id, symbol, datetime.utcnow(), str(t.get('id')), float(t.get('price')), float(t.get('qty')), 'buy', 'buy') for t in raw]

class CoinWAdapter(CustomCEXAdapter):
    def __init__(self, config=None): super().__init__('coinw', config)
    async def fetch_ticker(self, symbol: str) -> NormalizedTicker:
        s = symbol.replace('/', '').upper()
        data = await self._get(f"https://api.coinw.com/api/v1/public?command=returnTicker")
        raw = data.get(s, {})
        return NormalizedTicker(
            exchange=self.exchange_id, symbol=symbol, timestamp=datetime.utcnow(),
            bid=float(raw.get('highestBid', 0)), ask=float(raw.get('lowestAsk', 0)),
            last=float(raw.get('last', 0)), volume_24h=float(raw.get('baseVolume', 0)),
            high_24h=float(raw.get('high24hr', 0)), low_24h=float(raw.get('low24hr', 0))
        )
    async def fetch_order_book(self, symbol: str) -> NormalizedOrderBook:
        s = symbol.replace('/', '').upper()
        raw = await self._get(f"https://api.coinw.com/api/v1/public?command=returnOrderBook&symbol={s}")
        bids = [OrderBookLevel(price=float(b[0]), volume=float(b[1])) for b in raw.get('bids', [])]
        asks = [OrderBookLevel(price=float(a[0]), volume=float(a[1])) for a in raw.get('asks', [])]
        return NormalizedOrderBook(self.exchange_id, symbol, datetime.utcnow(), bids, asks)
    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[NormalizedTrade]:
        s = symbol.replace('/', '').upper()
        raw = await self._get(f"https://api.coinw.com/api/v1/public?command=returnTradeHistory&symbol={s}")
        return [NormalizedTrade(self.exchange_id, symbol, datetime.utcnow(), str(t.get('trade_id')), float(t.get('rate')), float(t.get('amount')), 'buy', 'buy') for t in raw]

class FameEXAdapter(CustomCEXAdapter):
    def __init__(self, config=None): super().__init__('fameex', config)
    async def fetch_ticker(self, symbol: str) -> NormalizedTicker:
        s = symbol.replace('/', '_').upper()
        raw = await self._get(f"https://openapi.fameex.com/v2/public/ticker?base={s.split('_')[0]}&quote={s.split('_')[1]}")
        return NormalizedTicker(self.exchange_id, symbol, datetime.utcnow(), float(raw.get('bid', 0)), float(raw.get('ask', 0)), float(raw.get('last', 0)), float(raw.get('volume', 0)), 0, 0)

class BitunixAdapter(CustomCEXAdapter):
    def __init__(self, config=None): super().__init__('bitunix', config)
    async def fetch_ticker(self, symbol: str) -> NormalizedTicker:
        s = symbol.replace('/', '').upper()
        raw = await self._get(f"https://api.bitunix.com/api/spot/v1/market/last_price?symbol={s}")
        price = float(raw.get('data', {}).get('last_price', 0))
        return NormalizedTicker(self.exchange_id, symbol, datetime.utcnow(), price, price, price, 0, 0, 0)

class LMAXAdapter(CustomCEXAdapter):
    def __init__(self, config=None): super().__init__('lmax', config)
    async def fetch_ticker(self, symbol: str) -> NormalizedTicker:
        s = symbol.replace('/', '-').upper()
        raw = await self._get(f"https://api.lmaxdigital.com/v1/ticker/{s}")
        return NormalizedTicker(self.exchange_id, symbol, datetime.utcnow(), float(raw.get('bid', 0)), float(raw.get('ask', 0)), float(raw.get('lastPrice', 0)), 0, 0, 0)

# Register all research exchanges, prefer CCXT
RESEARCH_CEX = [
    'mexc', 'bybit', 'kucoin', 'poloniex', 'bitmart', 'lbank', 'xt', 'htx', 'binance', 'okx',
    'bitget', 'upbit', 'whitebit', 'bithumb', 'bullish', 'bitrue', 'ascendex', 'digifinex',
    'coinw', 'p2b', 'bingx', 'toobit', 'coinex', 'bitvavo', 'hitbtc', 'gateio', 'mercado',
    'bitopro', 'paymium', 'phemex', 'bitflyer', 'coincheck', 'gemini', 'cryptocom', 'bitstamp',
    'kraken', 'coinbase', 'blofin', 'bydfi', 'coinmate', 'wazirx', 'pionex', 'probit',
    'tidex', 'korbit', 'paribu', 'bitcastle', 'hibt', 'btcc', 'azbit', 'coinstore', 'bitunix',
    'lmax', 'inx', 'buyucoin', 'ueex', 'bika', 'kcex', 'bkex', 'fameex', 'weex', 'zengo', 'uphold', 'egemoney'
]

CEX_ADAPTERS = {}

for ex_id in RESEARCH_CEX:
    if CCXT_AVAILABLE and ex_id in ccxt.exchanges:
        class ResearchCCXTAdapter(CCXTAdapter):
            def __init__(self, config=None, eid=ex_id):
                super().__init__(eid, config)
        CEX_ADAPTERS[ex_id] = ResearchCCXTAdapter
    else:
        # Custom or generic fallback
        if ex_id == 'bitrue': CEX_ADAPTERS[ex_id] = BitrueAdapter
        elif ex_id == 'coinw': CEX_ADAPTERS[ex_id] = CoinWAdapter
        elif ex_id == 'fameex': CEX_ADAPTERS[ex_id] = FameEXAdapter
        elif ex_id == 'bitunix': CEX_ADAPTERS[ex_id] = BitunixAdapter
        elif ex_id == 'lmax': CEX_ADAPTERS[ex_id] = LMAXAdapter
        else:
            class GenericCustomAdapter(CustomCEXAdapter):
                def __init__(self, config=None, eid=ex_id): super().__init__(eid, config)
                async def fetch_ticker(self, symbol: str) -> NormalizedTicker:
                    return NormalizedTicker(self.exchange_id, symbol, datetime.utcnow(), 0, 0, 0, 0, 0, 0)
                async def fetch_order_book(self, symbol: str) -> NormalizedOrderBook:
                    return NormalizedOrderBook(self.exchange_id, symbol, datetime.utcnow(), [], [])
                async def fetch_trades(self, symbol: str, limit: int = 100) -> List[NormalizedTrade]:
                    return []
            CEX_ADAPTERS[ex_id] = GenericCustomAdapter

def get_cex_adapter(exchange_id: str, config: Optional[Dict[str, Any]] = None) -> ExchangeInterface:
    adapter_class = CEX_ADAPTERS.get(exchange_id)
    if not adapter_class:
        raise ValueError(f"Unsupported exchange: {exchange_id}")
    return adapter_class(config)
