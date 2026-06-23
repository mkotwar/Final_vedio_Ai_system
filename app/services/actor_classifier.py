"""Service for classifying actors based on generic schemas and YAML configs."""

import yaml
from typing import Dict, Any, Set
from pathlib import Path
from loguru import logger
from app.core.config import PROJECT_ROOT

class ActorClassificationService:
    _categories: Dict[str, Set[str]] = {}
    _is_loaded = False

    @classmethod
    def load_categories(cls) -> None:
        if cls._is_loaded:
            return
            
        config_path = PROJECT_ROOT / "config" / "actor_categories.yaml"
        if not config_path.exists():
            logger.warning(f"Actor categories config not found at {config_path}. Using empty mappings.")
            cls._categories = {}
            cls._is_loaded = True
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            cls._categories = {}
            if data:
                for category, types_list in data.items():
                    cat_upper = str(category).upper()
                    cls._categories[cat_upper] = set(str(t).lower() for t in types_list)
            
            logger.info(f"Loaded actor categories from config: {list(cls._categories.keys())}")
            cls._is_loaded = True
        except Exception as e:
            logger.error(f"Failed to load actor categories from {config_path}: {e}")
            cls._categories = {}
            cls._is_loaded = True

    @classmethod
    def classify_object(cls, obj: Dict[str, Any]) -> str:
        """Classifies a metadata object dict into an ActorCategory (e.g., 'HUMAN', 'VEHICLE', 'ANIMAL', 'OBJECT').
        
        Args:
            obj: Dictionary containing 'type' and 'subtype' keys.
            
        Returns:
            str: The semantic actor category. Defaults to 'OBJECT' if no match.
        """
        cls.load_categories()
        
        typ = str(obj.get("type", "")).lower()
        sub = str(obj.get("subtype", "")).lower()
        
        for category, types_set in cls._categories.items():
            if typ in types_set or sub in types_set:
                return category
                
        return "OBJECT"
