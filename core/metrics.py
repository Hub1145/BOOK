# core/metrics.py
"""Metrics tracking for monitoring"""


from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps


# Metrics
orderbook_updates = Counter('orderbook_updates_total', 'Total orderbook updates', ['exchange', 'symbol'])
trade_updates = Counter('trade_updates_total', 'Total trade updates', ['exchange', 'symbol'])
anomalies_detected = Counter('anomalies_detected_total', 'Total anomalies detected', ['severity', 'type'])


processing_time = Histogram('processing_time_seconds', 'Processing time', ['operation'])
active_connections = Gauge('active_connections', 'Active exchange connections', ['exchange'])
last_update_time = Gauge('last_update_timestamp', 'Last update timestamp', ['exchange', 'symbol'])


system_info = Info('pipeline_info', 'Pipeline information')


def track_time(operation: str):
    """Decorator to track execution time"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                processing_time.labels(operation=operation).observe(time.time() - start)
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                processing_time.labels(operation=operation).observe(time.time() - start)
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator