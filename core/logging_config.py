"""
Centralized logging configuration for the trading pipeline.
Provides consistent logging setup across all modules with file rotation,
multiple log levels, and customizable formatting.
"""

import logging
import logging.config
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, Any

# Check for optional JSON logger dependency
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGING_AVAILABLE = True
except ImportError:
    JSON_LOGGING_AVAILABLE = False


# Environment variables for configuration
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(exist_ok=True)

# Global placeholder for logging settings, to be populated by _load_config_settings
_global_log_settings = {} # Initialize as empty dict

# Environment variables for configuration
# These will be overwritten by _load_config_settings if pipeline_config is provided
# and has logging settings. Otherwise, they act as fallbacks.
_global_log_settings["level"] = os.getenv("LOG_LEVEL", "INFO").upper()
_global_log_settings["format"] = os.getenv("LOG_FORMAT", "standard")
_global_log_settings["enable_file_logging"] = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
_global_log_settings["file_max_size"] = int(os.getenv("LOG_FILE_MAX_SIZE", "10000000"))  # 10MB default
_global_log_settings["file_backup_count"] = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))


# Define multiple format styles
FORMATTERS = {
    "standard": {
        "format": "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S"
    },
    "detailed": {
        "format": "%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S"
    },
    "simple": {
        "format": "%(levelname)s: %(message)s"
    }
}

# Add JSON formatter only if available
if JSON_LOGGING_AVAILABLE:
    FORMATTERS["json"] = {
        "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
        "format": "%(asctime)s %(levelname)s %(name)s %(message)s"
    }

# Module-level storage for current configuration
_CURRENT_CONFIG = None


def get_logging_config(
    log_level: str = _global_log_settings["level"],
    log_format: str = _global_log_settings["format"],
    enable_file: bool = _global_log_settings["enable_file_logging"],
    file_max_size: int = _global_log_settings["file_max_size"],
    file_backup_count: int = _global_log_settings["file_backup_count"],
    custom_handlers: Optional[Dict[str, Dict[str, Any]]] = None
) -> dict:
    """
    Generate logging configuration dictionary.
    
    Args:
        log_level: Default logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format style to use (standard, detailed, json, simple)
        enable_file: Whether to enable file logging
        custom_handlers: Additional custom handlers to include
    
    Returns:
        Logging configuration dictionary for logging.config.dictConfig()
    """
    # Base configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": FORMATTERS,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": log_format,
                "level": log_level,
                "stream": "ext://sys.stdout"
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console"]
        },
        "loggers": {
            # Suppress noisy libraries
            "urllib3": {"level": "WARNING"},
            "asyncio": {"level": "WARNING"},
            "ccxt": {"level": "INFO"},
            "cryptofeed": {"level": "INFO"}
        }
    }
    
    # Add file handlers if enabled
    if enable_file:
        # Info and above to main log
        config["handlers"]["file_info"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": log_format,
            "level": "INFO",
            "filename": str(LOG_DIR / "app.log"),
            "maxBytes": file_max_size,
            "backupCount": file_backup_count,
            "encoding": "utf-8"
        }
        
        # Errors to separate file
        config["handlers"]["file_error"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",  # Always use detailed format for errors
            "level": "ERROR",
            "filename": str(LOG_DIR / "error.log"),
            "maxBytes": file_max_size // 2,
            "backupCount": file_backup_count,
            "encoding": "utf-8"
        }
        
        # Debug log (only if debug level is set)
        # Uses TimedRotatingFileHandler to rotate daily since debug logs can be verbose
        if log_level == "DEBUG":
            config["handlers"]["file_debug"] = {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "formatter": "detailed",
                "level": "DEBUG",
                "filename": str(LOG_DIR / "debug.log"),
                "when": "midnight",
                "interval": 1,
                "backupCount": 3,
                "encoding": "utf-8"
            }
            config["root"]["handlers"].append("file_debug")
        
        # Add backtest data log
        config["handlers"]["file_backtest_data"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "level": "INFO",
            "filename": str(LOG_DIR / "backtest_data.log"),
            "maxBytes": file_max_size,
            "backupCount": file_backup_count,
            "encoding": "utf-8"
        }
        config["root"]["handlers"].append("file_backtest_data")

        # Add file handlers to root
        config["root"]["handlers"].extend(["file_info", "file_error"])
    
    # Add custom handlers if provided
    if custom_handlers:
        config["handlers"].update(custom_handlers)
        config["root"]["handlers"].extend(custom_handlers.keys())
    
    return config


# Global variable to store loaded logging settings from config.
# This ensures that init_logging can be called without needing to pass config_dict every time
# once the main app has loaded the pipeline_config.
_global_pipeline_logging_config: Optional[Dict[str, Any]] = None

def _load_config_settings(pipeline_config: Optional[Dict[str, Any]] = None) -> None:
    """Loads logging settings from pipeline_config or environment variables."""
    global _global_log_settings, _global_pipeline_logging_config

    if pipeline_config:
        _global_pipeline_logging_config = pipeline_config.get("logging", {})
        
    log_settings = _global_pipeline_logging_config or {}

    _global_log_settings["level"] = log_settings.get("level", os.getenv("LOG_LEVEL", "INFO")).upper()
    _global_log_settings["format"] = log_settings.get("format", os.getenv("LOG_FORMAT", "standard"))
    _global_log_settings["enable_file_logging"] = str(log_settings.get("enable_file_logging", os.getenv("ENABLE_FILE_LOGGING", "true"))).lower() == "true"
    _global_log_settings["file_max_size"] = int(log_settings.get("file_max_size", os.getenv("LOG_FILE_MAX_SIZE", "10000000")))
    _global_log_settings["file_backup_count"] = int(log_settings.get("file_backup_count", os.getenv("LOG_FILE_BACKUP_COUNT", "5")))

# Load initial settings
_load_config_settings()

# Default configuration using potentially updated global settings
LOGGING_CONFIG = get_logging_config()


def init_logging(
    pipeline_config: Optional[Dict[str, Any]] = None,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    enable_file: Optional[bool] = None,
    config_dict: Optional[dict] = None
) -> None:
    """
    Initialize logging configuration for the application.
    Call this once at application startup.

    Args:
        pipeline_config: The full pipeline configuration dictionary, from which
                         logging settings will be extracted.
        log_level: Override default log level
        log_format: Override default format style
        enable_file: Override file logging setting
        config_dict: Complete custom configuration dictionary (overrides all)
    """
    global _CURRENT_CONFIG

    # Load settings from pipeline_config first, then allow explicit overrides
    _load_config_settings(pipeline_config)

    # Determine final settings
    final_log_level = log_level or _global_log_settings["level"]
    final_log_format = log_format or _global_log_settings["format"]
    final_enable_file = enable_file if enable_file is not None else _global_log_settings["enable_file_logging"]
    final_file_max_size = _global_log_settings["file_max_size"]
    final_file_backup_count = _global_log_settings["file_backup_count"]
    
    if config_dict:
        # Use provided configuration dictionary if present (highest precedence)
        _CURRENT_CONFIG = config_dict
        logging.config.dictConfig(config_dict)
    else:
        # Generate configuration with determined settings
        _CURRENT_CONFIG = get_logging_config(
            log_level=final_log_level,
            log_format=final_log_format,
            enable_file=final_enable_file,
            file_max_size=final_file_max_size,
            file_backup_count=final_file_backup_count
        )
        logging.config.dictConfig(_CURRENT_CONFIG)
    
    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {final_log_level}, "
                f"Format: {final_log_format}, "
                f"File logging: {'enabled' if final_enable_file else 'disabled'}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def add_file_handler(
    logger: logging.Logger,
    filename: str,
    level: str = "INFO",
    formatter: str = "standard",
    max_bytes: int = 10_000_000,
    backup_count: int = 5
) -> RotatingFileHandler:
    """
    Add a rotating file handler to a specific logger.
    
    Args:
        logger: Logger to add handler to
        filename: Log file path
        level: Logging level for this handler
        formatter: Formatter style to use
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        The created handler
    """
    # Check if handler already exists to avoid duplicates
    for handler in logger.handlers:
        if hasattr(handler, 'baseFilename') and handler.baseFilename == os.path.abspath(filename):
            logger.warning(f"Handler for {filename} already exists, skipping creation")
            return handler
    
    handler = RotatingFileHandler(
        filename=filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    handler.setLevel(getattr(logging, level.upper()))
    
    # Set formatter
    if formatter in FORMATTERS:
        fmt_config = FORMATTERS[formatter]
        if "()" in fmt_config:
            # Custom formatter class
            handler.setFormatter(logging.Formatter(fmt_config["format"]))
        else:
            handler.setFormatter(
                logging.Formatter(
                    fmt=fmt_config["format"],
                    datefmt=fmt_config.get("datefmt")
                )
            )
    
    logger.addHandler(handler)
    return handler


def setup_module_logger(
    name: str,
    level: Optional[str] = None,
    handlers: Optional[list] = None
) -> logging.Logger:
    """
    Set up a logger with specific configuration for a module.
    
    Args:
        name: Logger name
        level: Logging level (uses default if not specified)
        handlers: List of handler names to use (uses root handlers if not specified)
    
    Returns:
        Configured logger
    """
    global _CURRENT_CONFIG
    logger = logging.getLogger(name)
    
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    
    if handlers is not None and _CURRENT_CONFIG:
        # Remove existing handlers
        logger.handlers = []
        # Add specified handlers from config
        for handler_name in handlers:
            if handler_name in _CURRENT_CONFIG.get("handlers", {}):
                handler_config = _CURRENT_CONFIG["handlers"][handler_name]
                # Create handler from config
                handler_class = handler_config.get("class", "logging.StreamHandler")
                handler = eval(handler_class)(**{k: v for k, v in handler_config.items() 
                                                if k not in ["class", "formatter", "level"]})
                
                # Set level
                if "level" in handler_config:
                    handler.setLevel(getattr(logging, handler_config["level"]))
                
                # Set formatter
                if "formatter" in handler_config:
                    formatter_name = handler_config["formatter"]
                    if formatter_name in FORMATTERS:
                        fmt_config = FORMATTERS[formatter_name]
                        formatter = logging.Formatter(
                            fmt=fmt_config.get("format"),
                            datefmt=fmt_config.get("datefmt")
                        )
                        handler.setFormatter(formatter)
                
                logger.addHandler(handler)
    
    return logger


def log_exception(logger: logging.Logger, exc: Exception, context: str = "") -> None:
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger to use
        exc: Exception to log
        context: Additional context about where the exception occurred
    """
    import traceback
    
    tb = traceback.format_exc()
    logger.error(f"Exception in {context}: {type(exc).__name__}: {str(exc)}\n{tb}")


def create_performance_logger(name: str = "performance") -> logging.Logger:
    """
    Create a specialized logger for performance metrics.
    
    Returns:
        Logger configured for performance logging
    """
    perf_logger = logging.getLogger(f"{name}.performance")
    
    # Create performance log file
    perf_handler = TimedRotatingFileHandler(
        filename=str(LOG_DIR / "performance.log"),
        when="midnight",
        interval=1,
        backupCount=7,
        encoding='utf-8'
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    perf_logger.propagate = False  # Don't propagate to root logger
    
    return perf_logger


class LogContext:
    """
    Context manager for temporary logging configuration changes.
    
    This temporarily changes the logger's level only; handlers remain attached
    and continue to use their configured levels.
    """
    
    def __init__(self, logger: logging.Logger, level: Optional[str] = None):
        self.logger = logger
        self.new_level = level
        self.old_level = None
    
    def __enter__(self):
        if self.new_level:
            self.old_level = self.logger.level
            self.logger.setLevel(getattr(logging, self.new_level.upper()))
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_level is not None:
            self.logger.setLevel(self.old_level)


# Convenience function for debugging
def enable_debug_logging(modules: Optional[list] = None):
    """
    Enable debug logging for specific modules or globally.
    
    Args:
        modules: List of module names to enable debug for (None for root logger only)
    
    Note:
        When modules is None, only the root logger is set to DEBUG.
        Module-specific loggers retain their configured levels unless explicitly listed.
    """
    if modules:
        for module in modules:
            logging.getLogger(module).setLevel(logging.DEBUG)
        modules_str = ', '.join(modules)
    else:
        # Only adjust root logger, not all loggers
        logging.getLogger().setLevel(logging.DEBUG)
        modules_str = 'root logger'
    
    logger = get_logger(__name__)
    logger.info(f"Debug logging enabled for: {modules_str}")


# Example usage in main application
if __name__ == "__main__":
    # Initialize logging
    init_logging(log_level="DEBUG")
    
    # Get loggers
    logger = get_logger(__name__)
    perf_logger = create_performance_logger()
    
    # Example usage
    logger.info("Application starting...")
    logger.debug("Debug information")
    
    try:
        raise ValueError("Example error")
    except Exception as e:
        log_exception(logger, e, "main")
    
    # Performance logging
    perf_logger.info("Operation completed in 0.123s")
    
    # Context manager example
    with LogContext(logger, "DEBUG") as debug_logger:
        debug_logger.debug("This is only logged in debug context")