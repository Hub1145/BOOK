# core/plugins.py
"""Plugin system for extending functionality"""


import importlib
import inspect
from typing import Dict, Any, List
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class PluginManager:
    """Manages plugins for the pipeline"""
    
    def __init__(self, plugin_dir: str = "./plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugins: Dict[str, Any] = {}
        
    def load_plugins(self) -> None:
        """Load all plugins from plugin directory"""
        if not self.plugin_dir.exists():
            logger.warning(f"Plugin directory {self.plugin_dir} does not exist")
            return
            
        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.stem.startswith("_"):
                continue
                
            try:
                module_name = f"plugins.{plugin_file.stem}"
                module = importlib.import_module(module_name)
                
                # Find plugin class
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        hasattr(obj, 'plugin_name') and 
                        obj.__module__ == module_name):
                        
                        plugin_instance = obj()
                        self.plugins[obj.plugin_name] = plugin_instance
                        logger.info(f"Loaded plugin: {obj.plugin_name}")
                        
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")
                
    def get_plugin(self, name: str) -> Any:
        """Get a specific plugin"""
        return self.plugins.get(name)
        
    def list_plugins(self) -> List[str]:
        """List all loaded plugins"""
        return list(self.plugins.keys())