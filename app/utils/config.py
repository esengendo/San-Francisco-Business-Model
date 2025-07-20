"""
Configuration management for the SF Business Model Streamlit App
Optimized for production deployment and employer showcase
"""
import os
from pathlib import Path

class AppConfig:
    """Centralized configuration for the Streamlit application"""
    
    def __init__(self):
        # Base directory detection (Docker vs local)
        self.BASE_DIR = self._get_base_dir()
        
        # Data directories (optimized structure)
        self.MODEL_DIR = f"{self.BASE_DIR}/data/models"
        self.DATA_DIR = f"{self.BASE_DIR}/processed/final"
        
        # Fallback paths for legacy compatibility
        self.LEGACY_MODEL_DIR = f"{self.BASE_DIR}/models"
        self.LEGACY_DATA_DIR = f"{self.BASE_DIR}/processed/final"
        
        # Model file paths
        self.MODEL_STATE_DICT_PATH = self._find_model_file("final_business_model.pth")
        self.MODEL_CONFIG_PATH = self._find_config_file("model_config.json")
        self.PREPROCESSING_INFO_PATH = self._find_config_file("preprocessing_info.json")
        
        # Training data paths
        self.X_TRAIN_PATH = self._find_data_file("X_train.parquet")
        self.Y_TRAIN_PATH = self._find_data_file("y_train.parquet")
        
        # App configuration
        self.APP_TITLE = "SF Business Intelligence Platform"
        self.APP_SUBTITLE = "Production ML for Strategic Decision Making"
        self.APP_VERSION = "2.0.0"
        
    def _get_base_dir(self) -> str:
        """Detect base directory based on environment"""
        # Check environment variable first (Docker)
        if base_dir := os.getenv("BASE_DIR"):
            return base_dir
            
        # Check if we're in Docker
        if os.path.exists("/.dockerenv"):
            return "/app"
            
        # Local development - relative to current file
        current_file = Path(__file__).resolve()
        app_dir = current_file.parent.parent
        return str(app_dir.parent)
    
    def _find_model_file(self, filename: str) -> str:
        """Find model file in optimized or legacy location"""
        # Try optimized location first
        optimized_path = f"{self.MODEL_DIR}/{filename}"
        if os.path.exists(optimized_path):
            return optimized_path
            
        # Fall back to legacy location
        legacy_path = f"{self.LEGACY_MODEL_DIR}/{filename}"
        if os.path.exists(legacy_path):
            return legacy_path
            
        # Return optimized path as default
        return optimized_path
    
    def _find_config_file(self, filename: str) -> str:
        """Find config file in optimized or legacy location"""
        return self._find_model_file(filename)
    
    def _find_data_file(self, filename: str) -> str:
        """Find data file in optimized or legacy location"""
        # Try optimized location first
        optimized_path = f"{self.DATA_DIR}/{filename}"
        if os.path.exists(optimized_path):
            return optimized_path
            
        # Fall back to legacy location
        legacy_path = f"{self.LEGACY_DATA_DIR}/{filename}"
        if os.path.exists(legacy_path):
            return legacy_path
            
        # Return optimized path as default
        return optimized_path
    
    def get_environment_info(self) -> dict:
        """Get environment information for display"""
        return {
            "base_dir": self.BASE_DIR,
            "is_docker": os.path.exists("/.dockerenv"),
            "model_path": self.MODEL_STATE_DICT_PATH,
            "data_available": os.path.exists(self.X_TRAIN_PATH),
            "version": self.APP_VERSION
        }

# Global configuration instance
config = AppConfig()