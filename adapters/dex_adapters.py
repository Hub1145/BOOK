"""Adapters for research-identified decentralized exchanges"""

from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
import logging
from abc import abstractmethod
from web3 import Web3
from web3.providers.rpc import HTTPProvider
import aiohttp
import json

from core.exchange_interface import (
    ExchangeInterface, NormalizedOrderBook, NormalizedTrade,
    NormalizedTicker, OrderBookLevel
)

logger = logging.getLogger(__name__)


class DEXAdapter(ExchangeInterface):
    """Base adapter for DEX exchanges using Alchemy RPC or Public APIs"""

    def __init__(self, exchange_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(exchange_id, config)
        self.rpc_url = config.get('rpc_url', '')
        self.w3 = None
        self.session = None

    async def connect(self) -> None:
        try:
            if self.rpc_url:
                self.w3 = Web3(HTTPProvider(self.rpc_url))
            if not self.session:
                self.session = aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0'})
            self.connected = True
        except Exception as e:
            logger.error(f"DEX connection error for {self.exchange_id}: {e}")

    async def disconnect(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False

    async def _get(self, url: str, params: Optional[Dict] = None) -> Dict:
        if not self.session: await self.connect()
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200: return await resp.json()
                return {}
        except: return {}

    async def fetch_ticker(self, symbol: str) -> NormalizedTicker:
        return NormalizedTicker(self.exchange_id, symbol, datetime.utcnow(), 0, 0, 0, 0, 0, 0)

    async def fetch_order_book(self, symbol: str) -> NormalizedOrderBook:
        return NormalizedOrderBook(self.exchange_id, symbol, datetime.utcnow(), [], [])

    async def fetch_trades(self, symbol: str, limit: Optional[int] = None) -> List[NormalizedTrade]:
        return []

# --- Specific Implementations ---

class RaydiumAdapter(DEXAdapter):
    def __init__(self, config=None): super().__init__('raydium', config)
    async def fetch_ticker(self, symbol: str) -> NormalizedTicker:
        # Use Raydium public API v2
        data = await self._get("https://api.raydium.io/v2/main/pairs")
        price = 0
        if isinstance(data, list):
            for p in data:
                # Raydium pairs are often like 'SOL-USDC'
                if p.get('name') == symbol or p.get('name') == symbol.replace('/', '-'):
                    price = float(p.get('price', 0))
                    return NormalizedTicker(self.exchange_id, symbol, datetime.utcnow(), price, price, price, float(p.get('volume24h', 0)), 0, 0)
        return NormalizedTicker(self.exchange_id, symbol, datetime.utcnow(), price, price, price, 0, 0, 0)

class PancakeSwapAdapter(DEXAdapter):
    def __init__(self, config=None): super().__init__('pancakeswap', config)
    async def fetch_ticker(self, symbol: str) -> NormalizedTicker:
        data = await self._get("https://api.pancakeswap.info/api/v2/tokens")
        tokens = data.get('data', {})
        price = 0
        for addr, info in tokens.items():
            if info.get('symbol', '').upper() in symbol.upper():
                price = float(info.get('price', 0))
                return NormalizedTicker(self.exchange_id, symbol, datetime.utcnow(), price, price, price, 0, 0, 0)
        return NormalizedTicker(self.exchange_id, symbol, datetime.utcnow(), price, price, price, 0, 0, 0)

class CetusAdapter(DEXAdapter):
    def __init__(self, config=None): super().__init__('cetus', config)
    async def fetch_ticker(self, symbol: str) -> NormalizedTicker:
        data = await self._get("https://api-sui.cetus.zone/v2/sui/pools_info")
        pools = data.get('data', {}).get('pools', [])
        for p in pools:
            if p.get('symbol') == symbol or p.get('symbol') == symbol.replace('/', ''):
                price = float(p.get('price', 0))
                return NormalizedTicker(self.exchange_id, symbol, datetime.utcnow(), price, price, price, float(p.get('vol_in_usd_24h', 0)), 0, 0)
        return NormalizedTicker(self.exchange_id, symbol, datetime.utcnow(), 0, 0, 0, 0, 0, 0)

# Registry
RESEARCH_DEX = [
    'raydium', 'pancakeswap', 'pumpswap', 'cetus', 'aerodrome', 'thruster',
    'sushiswap', 'uniswap_v3', 'orca', 'bisonfi', 'humidifi', 'turbos',
    'aster', 'merchantmoe', 'monoswap', 'baseswap', 'deepbook', 'aftermath',
    'kriya', 'bluemove', 'fenix', 'blasterswap', 'bladeswap', 'hyperblast',
    'defituna', 'tessera', 'phoenix', 'lifinity', 'milkroad', 'ecoportal',
    'hyperion', 'fluid', 'agni', 'tsunamix'
]

DEX_ADAPTERS = {
    'raydium': RaydiumAdapter,
    'pancakeswap': PancakeSwapAdapter,
    'cetus': CetusAdapter,
}

for dex in RESEARCH_DEX:
    if dex not in DEX_ADAPTERS:
        class GenericDEXAdapter(DEXAdapter):
            def __init__(self, config=None, eid=dex): super().__init__(eid, config)
        DEX_ADAPTERS[dex] = GenericDEXAdapter

def get_dex_adapter(exchange_id: str, config: Optional[Dict[str, Any]] = None) -> ExchangeInterface:
    adapter_class = DEX_ADAPTERS.get(exchange_id)
    if not adapter_class: return DEXAdapter(exchange_id, config)
    return adapter_class(config)
