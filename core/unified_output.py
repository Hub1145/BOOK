import os, json
from datetime import datetime
import pandas as pd
from scipy.stats import skew, kurtosis
from typing import Dict, Any, List

def write_unified_record(
    config: Any, # Can be a dict (live) or object (backtest)
    exchange: str,
    symbol: str,
    features: Dict[str, Any],
    feature_stats: Dict[str, Any],
    detectors: Dict[str, Any],
    meta_stats: Dict[str, Any],
    composite: Dict[str, Any]
) -> Dict[str, Any]:
    """Write one unified JSONL row combining features and anomalies."""
    # Handle both object-based config and dict-based config
    if isinstance(config, dict):
        output_config = config.get("output", {})
    else:
        output_config = getattr(config, "output", {})

    output_dir = output_config.get("output_directory", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"unified_{datetime.utcnow().date().isoformat()}.jsonl")
    
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "exchange": exchange,
        "symbol": symbol,
        "features": features,
        "feature_statistics": feature_stats,
        "detectors": detectors,
        "meta_statistics": meta_stats,
        "composite": composite,
    }
    
    with open(filename, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        
    return record