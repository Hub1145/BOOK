# core/config_loader.py
"""Configuration loader utilities"""


import os
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration with environment variable substitution"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Replace environment variables
    _replace_env_vars(config)
    return config
    
def _replace_env_vars(config: Any) -> None:
    """Recursively replace environment variables in config"""
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var, '')
            else:
                _replace_env_vars(value)
    elif isinstance(config, list):
        for item in config:
            _replace_env_vars(item)