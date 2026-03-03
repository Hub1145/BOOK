"""Adapters for all major decentralized exchanges"""

from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
import logging
from abc import abstractmethod
from web3 import Web3
from web3.providers.rpc import HTTPProvider
# from web3.providers.websocket import WebsocketProvider # Temporarily commented out
import aiohttp
import json

from core.exchange_interface import (
    ExchangeInterface, NormalizedOrderBook, NormalizedTrade, 
    NormalizedTicker, OrderBookLevel
)

logger = logging.getLogger(__name__)


class DEXAdapter(ExchangeInterface):
    """Base adapter for DEX exchanges"""
    
    def __init__(self, exchange_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(exchange_id, config)
        self.rpc_url = config.get('rpc_url', '')
        self.ws_url = config.get('ws_url', '')
        self.w3 = None
        self.contracts = {}
        self.graph_url = config.get('graph_url', '')
        
    async def connect(self) -> None:
        """Initialize connection to DEX"""
        try:
            if self.ws_url:
                # Use HTTPProvider for WebSocket as a workaround for import issues
                self.w3 = Web3(HTTPProvider(self.ws_url))
            else:
                self.w3 = Web3(HTTPProvider(self.rpc_url))
                
            if self.w3.is_connected(): # Changed to is_connected()
                self.connected = True
                logger.info(f"Connected to {self.exchange_id}")
            else:
                raise Exception("Failed to connect to blockchain")
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_id}: {e}")
            raise
            
    async def disconnect(self) -> None:
        """Close connection to DEX"""
        self.connected = False
        logger.info(f"Disconnected from {self.exchange_id}")
        
    @abstractmethod
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        """Get pool data for a trading pair"""
        pass
        
    @abstractmethod
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent swap transactions"""
        pass


class UniswapV2Adapter(DEXAdapter):
    """Adapter for Uniswap V2 compatible DEXes"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('uniswap_v2', config)
        self.factory_address = config.get('factory_address', '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f')
        self.router_address = config.get('router_address', '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D')
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        """Get Uniswap V2 pool data"""
        # Implementation would query the blockchain for pool reserves
        # This is a simplified version
        return {
            'reserve0': 1000000,
            'reserve1': 2000000,
            'token0': 'USDC',
            'token1': 'ETH',
            'fee': 0.003
        }
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent swaps from The Graph or blockchain events"""
        # Implementation would query The Graph or parse blockchain events
        return []
        
    async def subscribe_orderbook(self, symbol: str, callback=None) -> None:
        """Subscribe to pool updates (simulated orderbook)"""
        self.subscriptions['orderbook'].add(symbol)
        
        async def watch_pool():
            while symbol in self.subscriptions['orderbook'] and self.connected:
                try:
                    pool_data = await self.get_pool_data(symbol)
                    orderbook = self._create_orderbook_from_pool(pool_data, symbol)
                    
                    if callback:
                        await callback(orderbook)
                        
                    for cb in self.callbacks.get('orderbook', []):
                        await cb(orderbook)
                        
                    await asyncio.sleep(5)  # Poll every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Error watching pool for {symbol}: {e}")
                    await asyncio.sleep(10)
                    
        asyncio.create_task(watch_pool())
        
    def _create_orderbook_from_pool(self, pool_data: Dict, symbol: str) -> NormalizedOrderBook:
        """Create orderbook representation from AMM pool"""
        reserve0 = pool_data['reserve0']
        reserve1 = pool_data['reserve1']
        current_price = reserve1 / reserve0
        
        # Simulate orderbook levels based on slippage
        bids = []
        asks = []
        
        for i in range(10):
            size = 100 * (i + 1)
            # Calculate price impact for buys and sells
            buy_price = self._calculate_price_impact(reserve0, reserve1, size, True)
            sell_price = self._calculate_price_impact(reserve0, reserve1, size, False)
            
            bids.append(OrderBookLevel(price=sell_price, volume=size))
            asks.append(OrderBookLevel(price=buy_price, volume=size))
            
        return NormalizedOrderBook(
            exchange=self.exchange_id,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price),
            sequence=0
        )
        
    def _calculate_price_impact(self, r0: float, r1: float, amount: float, is_buy: bool) -> float:
        """Calculate price with slippage for AMM"""
        if is_buy:
            return (r1 - (r0 * r1) / (r0 + amount)) / amount
        else:
            return amount / (r0 - (r0 * r1) / (r1 + amount))


class UniswapV3Adapter(DEXAdapter):
    """Adapter for Uniswap V3"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('uniswap_v3', config)
        self.factory_address = config.get('factory_address', '0x1F98431c8aD98523631AE4a59f267346ea31F984')
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        """Get concentrated liquidity pool data"""
        # Would query pool contract for liquidity distribution
        return {
            'liquidity': 1000000000,
            'sqrtPrice': 1000000000000,
            'tick': 0,
            'fee': 0.003
        }
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent swaps from The Graph"""
        return []


class SushiSwapAdapter(UniswapV2Adapter):
    """Adapter for SushiSwap"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['factory_address'] = '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac'
        config['router_address'] = '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F'
        super().__init__(config)
        self.exchange_id = 'sushiswap'


class PancakeSwapAdapter(UniswapV2Adapter):
    """Adapter for PancakeSwap (BSC)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['rpc_url'] = config.get('rpc_url', 'https://bsc-dataseed.binance.org/')
        config['factory_address'] = '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73'
        config['router_address'] = '0x10ED43C718714eb63d5aA57B78B54704E256024E'
        super().__init__(config)
        self.exchange_id = 'pancakeswap'


class CurveAdapter(DEXAdapter):
    """Adapter for Curve Finance"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('curve', config)
        self.registry_address = config.get('registry_address', '0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5')
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        """Get Curve pool data with multiple token balances"""
        return {
            'balances': [1000000, 1000000, 1000000],  # 3pool example
            'A': 100,  # Amplification parameter
            'fee': 0.0004
        }
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        return []


class BalancerAdapter(DEXAdapter):
    """Adapter for Balancer"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('balancer', config)
        self.vault_address = config.get('vault_address', '0xBA12222222228d8Ba445958a75a0704d566BF2C8')
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        """Get Balancer pool data with weights"""
        return {
            'tokens': ['WETH', 'USDC', 'DAI'],
            'balances': [100, 200000, 200000],
            'weights': [0.5, 0.25, 0.25],
            'swapFee': 0.003
        }
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        return []


class DydxAdapter(DEXAdapter):
    """Adapter for dYdX"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('dydx', config)
        self.api_url = config.get('api_url', 'https://api.dydx.exchange')
        
    async def connect(self) -> None:
        """Connect to dYdX API"""
        self.connected = True
        logger.info(f"Connected to {self.exchange_id}")
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        """dYdX uses orderbooks, not pools"""
        return {}
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades from dYdX API"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_url}/v3/trades/{symbol}") as resp:
                data = await resp.json()
                return data.get('trades', [])


class OneInchAdapter(DEXAdapter):
    """Adapter for 1inch (Aggregator)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('1inch', config)
        self.api_url = config.get('api_url', 'https://api.1inch.io/v5.0/1')
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        """1inch aggregates from multiple sources"""
        return {}
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        return []


class KyberSwapAdapter(DEXAdapter):
    """Adapter for KyberSwap"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('kyberswap', config)
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        return {}
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        return []


class TraderJoeAdapter(UniswapV2Adapter):
    """Adapter for TraderJoe (Avalanche)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['rpc_url'] = config.get('rpc_url', 'https://api.avax.network/ext/bc/C/rpc')
        config['factory_address'] = '0x9Ad6C38BE94206cA50bb0d90783181662f0Cfa10'
        super().__init__(config)
        self.exchange_id = 'traderjoe'


class QuickSwapAdapter(UniswapV2Adapter):
    """Adapter for QuickSwap (Polygon)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['rpc_url'] = config.get('rpc_url', 'https://polygon-rpc.com')
        config['factory_address'] = '0x5757371414417b8C6CAad45bAeF941aBc7d3Ab32'
        super().__init__(config)
        self.exchange_id = 'quickswap'


class SpookySwapAdapter(UniswapV2Adapter):
    """Adapter for SpookySwap (Fantom)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['rpc_url'] = config.get('rpc_url', 'https://rpc.ftm.tools')
        config['factory_address'] = '0x152eE697f2E276fA89E96742e9bB9aB1F2E61bE3'
        super().__init__(config)
        self.exchange_id = 'spookyswap'


class RaydiumAdapter(DEXAdapter):
    """Adapter for Raydium (Solana)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('raydium', config)
        self.rpc_url = config.get('rpc_url', 'https://api.mainnet-beta.solana.com')
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        # Would use Solana Web3.py to query pool
        return {}
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        return []


class OrcaAdapter(DEXAdapter):
    """Adapter for Orca (Solana)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('orca', config)
        self.rpc_url = config.get('rpc_url', 'https://api.mainnet-beta.solana.com')
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        return {}
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        return []


class SerumAdapter(DEXAdapter):
    """Adapter for Serum (Solana)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('serum', config)
        self.rpc_url = config.get('rpc_url', 'https://api.mainnet-beta.solana.com')
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        # Serum uses orderbooks, not pools
        return {}
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        return []


class OsmosisAdapter(DEXAdapter):
    """Adapter for Osmosis (Cosmos)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('osmosis', config)
        self.api_url = config.get('api_url', 'https://api-osmosis.imperator.co')
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        return {}
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        return []


class ThorchainAdapter(DEXAdapter):
    """Adapter for THORChain"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('thorchain', config)
        self.api_url = config.get('api_url', 'https://thornode.thorchain.info')
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        return {}
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        return []


class BancorAdapter(DEXAdapter):
    """Adapter for Bancor"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('bancor', config)
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        return {}
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        return []


class SynthetixAdapter(DEXAdapter):
    """Adapter for Synthetix"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('synthetix', config)
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        return {}
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        return []


class GMXAdapter(DEXAdapter):
    """Adapter for GMX (Arbitrum/Avalanche)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('gmx', config)
        
    async def get_pool_data(self, symbol: str) -> Dict[str, Any]:
        return {}
        
    async def get_recent_swaps(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        return []


# Registry of all available DEX adapters
DEX_ADAPTERS = {
    'uniswap_v2': UniswapV2Adapter,
    'uniswap_v3': UniswapV3Adapter,
    'sushiswap': SushiSwapAdapter,
    'pancakeswap': PancakeSwapAdapter,
    'curve': CurveAdapter,
    'balancer': BalancerAdapter,
    'dydx': DydxAdapter,
    '1inch': OneInchAdapter,
    'kyberswap': KyberSwapAdapter,
    'traderjoe': TraderJoeAdapter,
    'quickswap': QuickSwapAdapter,
    'spookyswap': SpookySwapAdapter,
    'raydium': RaydiumAdapter,
    'orca': OrcaAdapter,
    'serum': SerumAdapter,
    'osmosis': OsmosisAdapter,
    'thorchain': ThorchainAdapter,
    'bancor': BancorAdapter,
    'synthetix': SynthetixAdapter,
    'gmx': GMXAdapter,
}


def get_dex_adapter(exchange_id: str, config: Optional[Dict[str, Any]] = None) -> ExchangeInterface:
    """Factory function to get DEX adapter by ID"""
    adapter_class = DEX_ADAPTERS.get(exchange_id)
    if not adapter_class:
        raise ValueError(f"Unknown DEX: {exchange_id}")
    return adapter_class(config)