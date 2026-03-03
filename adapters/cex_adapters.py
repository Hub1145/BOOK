"""Adapters for all major centralized exchanges"""

from typing import Dict, List, Optional, Any, Callable
import asyncio
from datetime import datetime
import logging
from abc import abstractmethod

try:
    import ccxt.async_support as ccxt_async
    CCXT_ASYNC_AVAILABLE = True
except ImportError:
    import ccxt
    ccxt_async = ccxt # Fallback to sync if async not available
    CCXT_ASYNC_AVAILABLE = False
    logger.warning("ccxt.async_support not found, falling back to synchronous ccxt. "
                   "Install with 'pip install ccxt[async]' for better performance.")

from core.exchange_interface import (
    ExchangeInterface, NormalizedOrderBook, NormalizedTrade, 
    NormalizedTicker, OrderBookLevel
)

logger = logging.getLogger(__name__)


class CCXTAdapter(ExchangeInterface):
    """Base adapter for CCXT-compatible exchanges"""
    
    def __init__(self, exchange_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(exchange_id, config)
        self.exchange_class = getattr(ccxt_async, exchange_id) # Use ccxt_async
        
        # Base parameters for CCXT client
        params = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot', # Default to spot, can be overridden by config
                'watchOrderBook': {'limit': 100},
                'watchTrades': {'limit': 100}
            },
            **(config or {}) # Merge any additional config provided
        }
        
        # If testnet is enabled in config, use CCXT's set_sandbox_mode method
        # Instantiate the exchange using ccxt_async
        self.exchange = self.exchange_class(params)
        if config and config.get('testnet', False):
            if hasattr(self.exchange, 'set_sandbox_mode'):
                self.exchange.set_sandbox_mode(True)
                logger.info(f"Configuring {exchange_id} for testnet using set_sandbox_mode.")
            else:
                logger.warning(f"Testnet requested for {exchange_id}, but set_sandbox_mode is not available. "
                               "Falling back to default URLs. Please check CCXT documentation for this exchange's testnet setup.")
        
    async def connect(self) -> None:
        """Initialize connection to exchange"""
        try:
            # Load markets to initialize the exchange instance and verify connection
            await self.exchange.load_markets()
            self.connected = True
            logger.info(f"Successfully connected to {self.exchange_id} (Testnet: {self.config.get('testnet', False)})")
        except ccxt.NetworkError as e:
            logger.error(f"Network error connecting to {self.exchange_id}: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error connecting to {self.exchange_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while connecting to {self.exchange_id}: {e}")
            raise
            
    async def disconnect(self) -> None:
        """Close connection to exchange"""
        if hasattr(self.exchange, 'close'):
            await self.exchange.close()
            await asyncio.sleep(0.1) # Give event loop a chance to process aiohttp cleanup
        self.connected = False
        logger.info(f"Disconnected from {self.exchange_id}")
        
    async def fetch_order_book(self, symbol: str) -> Dict:
        """Fetch orderbook data for a symbol using REST."""
        return await self.exchange.fetch_order_book(symbol)

    async def fetch_trades(self, symbol: str, limit: Optional[int] = None) -> List[Dict]:
        """Fetch trade data for a symbol using REST."""
        return await self.exchange.fetch_trades(symbol, limit=limit)

    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch ticker data for a symbol using REST."""
        return await self.exchange.fetch_ticker(symbol)

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', since: Optional[int] = None, limit: Optional[int] = None, params: Optional[Dict] = None) -> List[List[float]]:
        """Fetch OHLCV data."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since, limit, params)
            return ohlcv
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol} on {self.exchange_id}: {e}")
            return []

    async def get_instruments(self) -> List[str]:
        """Get list of available instruments from the exchange."""
        try:
            await self.exchange.load_markets() # Ensure markets are loaded
            return list(self.exchange.symbols)
        except Exception as e:
            logger.error(f"Failed to get instruments from {self.exchange_id}: {e}")
            return []

    def _normalize_orderbook(self, orderbook: Dict, symbol: str) -> NormalizedOrderBook:
        """Convert exchange orderbook to normalized format"""
        
        normalized_bids = []
        for i, bid in enumerate(orderbook['bids'][:50]):
            try:
                price = float(bid[0])
                volume = float(bid[1])
                normalized_bids.append(OrderBookLevel(price=price, volume=volume))
            except ValueError as e:
                # Log error, but don't re-raise to keep pipeline running
                continue
                
        normalized_asks = []
        for i, ask in enumerate(orderbook['asks'][:50]):
            try:
                price = float(ask[0])
                volume = float(ask[1])
                normalized_asks.append(OrderBookLevel(price=price, volume=volume))
            except ValueError as e:
                # Log error, but don't re-raise to keep pipeline running
                continue
                
        return NormalizedOrderBook(
            exchange=self.exchange_id,
            symbol=symbol,
            timestamp=datetime.fromtimestamp(orderbook['timestamp'] / 1000) if 'timestamp' in orderbook and orderbook['timestamp'] is not None else datetime.utcnow(),
            bids=normalized_bids,
            asks=normalized_asks,
            sequence=orderbook.get('nonce', 0)
        )
        
    def _normalize_trade(self, trade: Dict, symbol: str) -> NormalizedTrade:
        """Convert exchange trade to normalized format"""
        return NormalizedTrade(
            exchange=self.exchange_id,
            symbol=symbol,
            timestamp=datetime.fromtimestamp(trade['timestamp'] / 1000) if 'timestamp' in trade and trade['timestamp'] is not None else datetime.utcnow(),
            id=str(trade['id']),
            price=float(trade.get('price', 0.0)),
            volume=float(trade.get('amount', 0.0)),
            side=trade['side'],
            taker_side=trade.get('takerOrMaker', trade['side']) # Use takerOrMaker if available
        )

    def _normalize_ticker(self, ticker: Dict, symbol: str) -> NormalizedTicker:
        """Convert exchange ticker to normalized format"""
        return NormalizedTicker(
            exchange=self.exchange_id,
            symbol=symbol,
            timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000) if 'timestamp' in ticker and ticker['timestamp'] is not None else datetime.utcnow(),
            bid=ticker.get('bid', 0),
            ask=ticker.get('ask', 0),
            last=ticker.get('last', 0),
            volume_24h=ticker.get('baseVolume', 0),
            high_24h=ticker.get('high', 0),
            low_24h=ticker.get('low', 0),
            vwap_24h=ticker.get('vwap'),
            open_interest=ticker.get('openInterest'),
            funding_rate=ticker.get('fundingRate')
        )


# Create specific adapters for each major exchange
class BinanceAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('binance', config)


class BinanceUSAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('binanceus', config)


class CoinbaseAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('coinbase', config)


class KrakenAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('kraken', config)


class OKXAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('okx', config)


class BybitAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('bybit', config)


class GateioAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('gateio', config)


class HuobiAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('huobi', config)


class KucoinAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('kucoin', config)


class BitfinexAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('bitfinex', config)


class BitstampAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('bitstamp', config)


class GeminiAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('gemini', config)


class BitMEXAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['options'] = config.get('options', {})
        config['options']['defaultType'] = 'swap'  # For derivatives
        super().__init__('bitmex', config)


class DeribitAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('deribit', config)


class PoloniexAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('poloniex', config)


class BittrexAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('bittrex', config)


class FTXAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Note: FTX is defunct but keeping for historical data
        super().__init__('ftx', config)


class MexcAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('mexc', config)


class CryptocomAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('cryptocom', config)


class PhemexAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('phemex', config)


class AscendexAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('ascendex', config)


class WooAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('woo', config)


class BitgetAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('bitget', config)


class IndependentReserveAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('independentreserve', config)


class WhitebitAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('whitebit', config)


class BingxAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('bingx', config)


class BitbankAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('bitbank', config)


class BithumbAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('bithumb', config)


class UpbitAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('upbit', config)


class CoinoneAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('coinone', config)


class ZaifAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('zaif', config)


class LbankAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('lbank', config)


class ProbitAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('probit', config)


class ExmoAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('exmo', config)


class YobitAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('yobit', config)


class TidexAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('tidex', config)


class BigoneAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('bigone', config)


class OkcoinAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('okcoin', config)


class DigifinexAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('digifinex', config)


class BitsoAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('bitso', config)


class BtctradeimAdapter(CCXTAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('btctradeim', config)


# Registry of all available CEX adapters
CEX_ADAPTERS = {
    'binance': BinanceAdapter,
    'binanceus': BinanceUSAdapter,
    'coinbase': CoinbaseAdapter,
    'kraken': KrakenAdapter,
    'okx': OKXAdapter,
    'bybit': BybitAdapter,
    'gateio': GateioAdapter,
    'huobi': HuobiAdapter,
    'kucoin': KucoinAdapter,
    'bitfinex': BitfinexAdapter,
    'bitstamp': BitstampAdapter,
    'gemini': GeminiAdapter,
    'bitmex': BitMEXAdapter,
    'deribit': DeribitAdapter,
    'poloniex': PoloniexAdapter,
    'bittrex': BittrexAdapter,
    'ftx': FTXAdapter,  # Defunct but kept for historical data
    'mexc': MexcAdapter,
    'cryptocom': CryptocomAdapter,
    'phemex': PhemexAdapter,
    'ascendex': AscendexAdapter,
    'woo': WooAdapter,
    'bitget': BitgetAdapter,
    'independentreserve': IndependentReserveAdapter,
    'whitebit': WhitebitAdapter,
    'bingx': BingxAdapter,
    'bitbank': BitbankAdapter,
    'bithumb': BithumbAdapter,
    'upbit': UpbitAdapter,
    'coinone': CoinoneAdapter,
    'zaif': ZaifAdapter,
    'lbank': LbankAdapter,
    'probit': ProbitAdapter,
    'exmo': ExmoAdapter,
    'yobit': YobitAdapter,
    'tidex': TidexAdapter,
    'bigone': BigoneAdapter,
    'okcoin': OkcoinAdapter,
    'digifinex': DigifinexAdapter,
    'bitso': BitsoAdapter,
    'btctradeim': BtctradeimAdapter,
}


def get_cex_adapter(exchange_id: str, config: Optional[Dict[str, Any]] = None) -> ExchangeInterface:
    """Factory function to get CEX adapter by ID"""
    adapter_class = CEX_ADAPTERS.get(exchange_id)
    if not adapter_class:
        raise ValueError(f"Unknown exchange: {exchange_id}")
    return adapter_class(config)
