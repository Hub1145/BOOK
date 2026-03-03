# core/exchange_interface.py
"""Base exchange interface and data models"""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum



@dataclass
class OrderBookLevel:
    """Single level in an orderbook"""
    price: float
    volume: float
    
    
@dataclass
class NormalizedOrderBook:
    """Normalized orderbook across all exchanges"""
    exchange: str
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    sequence: int = 0
    
    def dict(self) -> Dict[str, Any]:
        return {
            'exchange': self.exchange,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'bids': [[b.price, b.volume] for b in self.bids],
            'asks': [[a.price, a.volume] for a in self.asks],
            'sequence': self.sequence
        }



@dataclass
class NormalizedTrade:
    """Normalized trade across all exchanges"""
    exchange: str
    symbol: str
    timestamp: datetime
    id: str
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    taker_side: str
    
    def dict(self) -> Dict[str, Any]:
        return {
            'exchange': self.exchange,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'id': self.id,
            'price': self.price,
            'volume': self.volume,
            'side': self.side,
            'taker_side': self.taker_side
        }



@dataclass
class NormalizedTicker:
    """Normalized ticker data"""
    exchange: str
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume_24h: float
    high_24h: float
    low_24h: float
    vwap_24h: Optional[float] = None
    open_interest: Optional[float] = None
    funding_rate: Optional[float] = None
    
    def dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}



class ExchangeInterface(ABC):
    """Abstract base class for all exchange adapters"""
    
    def __init__(self, exchange_id: str, config: Optional[Dict[str, Any]] = None):
        self.exchange_id = exchange_id
        self.config = config or {}
        self.connected = False
        self.subscriptions: Dict[str, Set[str]] = { # Still keep for future potential WebSocket use
            'orderbook': set(),
            'trades': set(),
            'ticker': set()
        }
        self.callbacks: Dict[str, List[Callable]] = { # Still keep for future potential WebSocket use
            'orderbook': [],
            'trade': [],
            'ticker': []
        }
        
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the exchange"""
        pass
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the exchange"""
        pass

    @abstractmethod
    async def fetch_order_book(self, symbol: str) -> Dict:
        """Fetch orderbook data for a symbol."""
        pass

    @abstractmethod
    async def fetch_trades(self, symbol: str, limit: Optional[int] = None) -> List[Dict]:
        """Fetch trade data for a symbol."""
        pass

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch ticker data for a symbol."""
        pass
            
    def add_callback(self, event_type: str, callback: Callable) -> None:
        """Add a callback for an event type"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            
    def is_connected(self) -> bool:
        """Check if exchange is connected"""
        return self.connected
        
    async def get_instruments(self) -> List[str]:
        """Get list of available instruments (optional)"""
        return []