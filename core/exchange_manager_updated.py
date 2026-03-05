"""Updated Exchange Manager that includes all CEX and DEX adapters"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import yaml
import importlib
import sys

from core.exchange_interface import ExchangeInterface
# Conditional imports for CEX_ADAPTERS and DEX_ADAPTERS to avoid circular dependencies
logger = logging.getLogger(__name__)


class ExchangeManager:
    """Manages multiple exchange connections with all exchanges"""

    def __init__(self, exchanges_config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None): # NEW: added exchanges_config
        self.exchanges: Dict[str, ExchangeInterface] = {}
        # Changed config loading: now processes exchanges_config directly
        self.config = self._load_config_from_file(config_path) if config_path else {} # Renamed and kept for other parts of config
        self.exchanges_config_data = exchanges_config if exchanges_config else {} # NEW: store exchanges_config directly
        self.callbacks = {'orderbook': [], 'trade': [], 'ticker': []}

    def _load_config_from_file(self, config_path: str) -> Dict[str, Any]: # Renamed method
        """Load general configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}

    def register_exchange(self, exchange_id: str, exchange: ExchangeInterface) -> None:
        """Register an exchange adapter"""
        if exchange_id in self.exchanges:
            logger.warning(f"Exchange {exchange_id} already registered, replacing")
        self.exchanges[exchange_id] = exchange
        logger.info(f"Registered exchange: {exchange_id}")

    def auto_register_exchanges(self) -> None:
        """Auto-register all available exchanges from the provided exchanges_config_data"""

        # Dynamically import CEX and DEX adapters here
        cex_adapters = {}
        get_cex_adapter_func = None
        try:
            from adapters.cex_adapters import CEX_ADAPTERS, get_cex_adapter
            cex_adapters = CEX_ADAPTERS
            get_cex_adapter_func = get_cex_adapter
        except ImportError as e:
            logger.warning(f"Could not import CEX adapters: {e}. CEX exchanges will not be available.")

        dex_adapters = {}
        get_dex_adapter_func = None
        try:
            from adapters.dex_adapters import DEX_ADAPTERS, get_dex_adapter
            dex_adapters = DEX_ADAPTERS
            get_dex_adapter_func = get_dex_adapter
        except ImportError as e:
            logger.warning(f"Could not import DEX adapters: {e}. DEX exchanges will not be available.")

        for exchange_id, config in self.exchanges_config_data.items(): # Iterate through all exchanges in the provided config
            if config.get('enabled', False): # Check the 'enabled' flag for each exchange
                if exchange_id in cex_adapters and get_cex_adapter_func:
                    try:
                        adapter = get_cex_adapter_func(exchange_id, config)
                        self.register_exchange(exchange_id, adapter)
                    except Exception as e:
                        logger.error(f"Failed to register CEX {exchange_id}: {e}")
                elif exchange_id in dex_adapters and get_dex_adapter_func:
                    try:
                        adapter = get_dex_adapter_func(exchange_id, config)
                        self.register_exchange(exchange_id, adapter)
                    except Exception as e:
                        logger.error(f"Failed to register DEX {exchange_id}: {e}")
                else:
                    logger.warning(f"Exchange {exchange_id} is enabled but no adapter found or module not available.")


        # Also check for custom adapters in the adapters directory
        self._discover_custom_adapters()

    def _discover_custom_adapters(self) -> None:
        """Discover and load custom exchange adapters"""
        adapters_dir = Path(__file__).parent.parent / "adapters"

        if not adapters_dir.exists():
            return

        # Add adapters directory to Python path
        sys.path.insert(0, str(adapters_dir))

        for adapter_file in adapters_dir.glob("*_adapter.py"):
            if adapter_file.stem in ['cex_adapters', 'dex_adapters']:
                continue  # Skip our main adapter files

            module_name = adapter_file.stem
            exchange_id = module_name.replace('_adapter', '')

            if exchange_id in self.exchanges:
                continue

            try:
                module = importlib.import_module(module_name)

                # Look for adapter class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        issubclass(attr, ExchangeInterface) and
                        attr != ExchangeInterface):

                        # Check if this exchange is enabled in config
                        if exchange_id in self.config.get('exchanges', {}).get('custom', []):
                            adapter = attr(self.config.get('exchanges', {}).get(exchange_id, {}))
                            self.register_exchange(exchange_id, adapter)
                            logger.info(f"Loaded custom adapter: {exchange_id}")
                            break

            except Exception as e:
                logger.error(f"Failed to load adapter {module_name}: {e}")

    async def start_all(self) -> None:
        """Start all registered exchanges concurrently using asyncio.gather"""
        tasks = []
        for exchange_id, exchange in self.exchanges.items():
            tasks.append(self._start_exchange(exchange_id, exchange))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _start_exchange(self, exchange_id: str, exchange: ExchangeInterface) -> None:
        """Start a single exchange with error handling"""
        try:
            await exchange.connect()

            # Register global callbacks
            for event_type, callbacks in self.callbacks.items():
                for callback in callbacks:
                    exchange.add_callback(event_type, callback)

            logger.info(f"Started exchange: {exchange_id}")
        except Exception as e:
            logger.error(f"Failed to start exchange {exchange_id}: {e}")

    async def stop_all(self) -> None:
        """Stop all exchanges"""
        tasks = []
        for exchange_id, exchange in self.exchanges.items():
            tasks.append(self._stop_exchange(exchange_id, exchange))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _stop_exchange(self, exchange_id: str, exchange: ExchangeInterface) -> None:
        """Stop a single exchange with error handling"""
        try:
            await exchange.disconnect()
            if hasattr(exchange, 'close'): # Explicitly call close if available
                await exchange.close()
            logger.info(f"Stopped exchange: {exchange_id}")
        except Exception as e:
            logger.error(f"Failed to stop exchange {exchange_id}: {e}")

    async def subscribe_symbols(self, symbols: List[str], exchanges: Optional[List[str]] = None) -> None:
        """Subscribe to symbols across all or specified exchanges"""
        if exchanges is None:
            exchanges = list(self.exchanges.keys())

        tasks = []
        for exchange_id in exchanges:
            if exchange_id in self.exchanges:
                exchange = self.exchanges[exchange_id]
                for symbol in symbols:
                    # Check if symbol is supported by exchange
                    if self._is_symbol_supported(exchange_id, symbol):
                        tasks.append(exchange.subscribe_orderbook(symbol))
                        tasks.append(exchange.subscribe_trades(symbol))

        await asyncio.gather(*tasks, return_exceptions=True)

    def _is_symbol_supported(self, exchange_id: str, symbol: str) -> bool:
        """Check if a symbol is supported by an exchange"""
        # This could be enhanced with actual market checking
        # For now, use a simple mapping
        symbol_mappings = self.config.get('symbol_mappings', {})

        if exchange_id in symbol_mappings:
            return symbol in symbol_mappings[exchange_id]

        # Default: assume all symbols are supported
        return True

    def add_global_callback(self, event_type: str, callback: Callable) -> None:
        """Add a callback that will be called for all exchanges"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)

        # Add to already connected exchanges
        for exchange in self.exchanges.values():
            exchange.add_callback(event_type, callback)

    def get_exchange(self, exchange_id: str) -> Optional[ExchangeInterface]:
        """Get a specific exchange by ID"""
        return self.exchanges.get(exchange_id)

    async def get_all_instruments(self) -> Dict[str, List[str]]:
        """Get all available instruments from all exchanges concurrently"""
        instruments = {}

        async def fetch_inst(ex_id, ex_adapter):
            try:
                return ex_id, await ex_adapter.get_instruments()
            except Exception as e:
                logger.error(f"Failed to get instruments from {ex_id}: {e}")
                return ex_id, []

        tasks = [fetch_inst(ex_id, ex) for ex_id, ex in self.exchanges.items()]
        results = await asyncio.gather(*tasks)

        for ex_id, inst in results:
            instruments[ex_id] = inst

        return instruments

    def get_exchange_status(self) -> Dict[str, bool]:
        """Get connection status of all exchanges"""
        return {
            exchange_id: exchange.is_connected()
            for exchange_id, exchange in self.exchanges.items()
        }

    def get_all_cex_exchanges(self) -> List[str]:
        """Get list of all available CEX exchanges"""
        try:
            from adapters.cex_adapters import CEX_ADAPTERS
            return list(CEX_ADAPTERS.keys())
        except ImportError:
            return []

    def get_all_dex_exchanges(self) -> List[str]:
        """Get list of all available DEX exchanges"""
        try:
            from adapters.dex_adapters import DEX_ADAPTERS
            return list(DEX_ADAPTERS.keys())
        except ImportError:
            return []

    def get_enabled_exchanges(self) -> Dict[str, List[str]]:
        """Get currently enabled exchanges by type"""
        return {
            'cex': [ex_id for ex_id in self.exchanges.keys() if ex_id in CEX_ADAPTERS],
            'dex': [ex_id for ex_id in self.exchanges.keys() if ex_id in DEX_ADAPTERS],
            'custom': [ex_id for ex_id in self.exchanges.keys()
                      if ex_id not in CEX_ADAPTERS and ex_id not in DEX_ADAPTERS]
        }