"""
Unified Configuration System for SF Business Model Pipeline
GitHub-ready with consistent directory patterns following logging_config_setup_02.py
"""

import os
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class PipelineConfig:
    """Centralized configuration for the SF Business Model Pipeline"""
    
    def __init__(self):
        # Setup logging first
        self.logger = self._setup_logging()
        
        # Get base directory from environment or auto-detect
        self.base_dir = self._get_base_directory()
        
        # Core directories - following the exact pattern from logging_config_setup_02.py
        self.raw_data_dir = f"{self.base_dir}/raw_data"
        self.processed_dir = f"{self.base_dir}/processed"
        self.model_dir = f"{self.base_dir}/models"
        self.archive_dir = f"{self.base_dir}/archive"
        self.logs_dir = f"{self.base_dir}/logs"  # Added for logging
        
        # Data sources (subdirectories under raw_data and processed)
        # Exact same list as logging_config_setup_02.py
        self.data_sources = [
            "sf_business",
            "economic", 
            "demographic",
            "planning",
            "crime",
            "sf311",
            "mobility",
            "yelp",
            "news",
            "historical",
            "final",
        ]
        
        # Date ranges for data collection - same as logging_config_setup_02.py
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365 * 11)  # 11 years ago for buffer
        
        # Create directories using the same pattern
        self._create_directories()
        
        # Log success message
        self.logger.info(
            f"Directory structure created successfully. Data period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}"
        )
        
    def _get_base_directory(self) -> str:
        """
        Get base directory from environment variable or auto-detect
        Priority:
        1. BASE_DIR environment variable (same as logging_config_setup_02.py)
        2. SF_BUSINESS_BASE_DIR environment variable (backup)
        3. Auto-detect based on environment
        """
        # First check BASE_DIR (same as your logging script)
        if base_dir := os.getenv("BASE_DIR"):
            return base_dir
            
        # Backup check for SF_BUSINESS_BASE_DIR
        if base_dir := os.getenv("SF_BUSINESS_BASE_DIR"):
            return base_dir
        
        # Auto-detect based on environment
        # Check if we're in a Docker container
        if os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER"):
            return "/app"
        
        # For local development, check current directory
        cwd = os.getcwd()
        
        # If we're in a project directory that looks like SF business model
        if any(name in cwd.lower() for name in ["sf", "san", "business"]):
            return cwd
        
        # Default fallback
        return os.path.join(cwd, "sf_business_model")
    
    def _create_directories(self):
        """
        Create all necessary directories using the EXACT same pattern as logging_config_setup_02.py
        """
        # Create main directory structure - same pattern as your script
        for directory in [self.base_dir, self.raw_data_dir, self.processed_dir, self.model_dir, self.archive_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Create subdirectories for different data sources - EXACT same pattern
        for source in self.data_sources:
            os.makedirs(f"{self.raw_data_dir}/{source}", exist_ok=True)
            os.makedirs(f"{self.processed_dir}/{source}", exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup standardized logging following the same pattern as logging_config_setup_02.py"""
        
        # Get log level from environment (default: INFO)
        log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
        
        # Setup logging configuration - same format as your script
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        
        return logging.getLogger("SFBusinessPipeline")
    
    def get_script_logger(self, script_name: str) -> logging.Logger:
        """Get a logger for a specific script"""
        if script_name.endswith('.py'):
            script_name = os.path.basename(script_name)[:-3]  # Remove .py extension
        return logging.getLogger(f"SFBusinessPipeline.{script_name}")
    
    def get_data_dir(self, data_source: str, processed: bool = False) -> str:
        """
        Get directory path for a specific data source
        Returns string path (not Path object) for consistency with your pattern
        """
        if processed:
            return f"{self.processed_dir}/{data_source}"
        else:
            return f"{self.raw_data_dir}/{data_source}"
    
    def ensure_data_dirs(self, data_source: str):
        """
        Ensure specific data source directories exist
        Use this in individual scripts that need specific directories
        """
        os.makedirs(f"{self.raw_data_dir}/{data_source}", exist_ok=True)
        os.makedirs(f"{self.processed_dir}/{data_source}", exist_ok=True)
    
    def get_file_path(self, data_source: str, filename: str, processed: bool = False) -> str:
        """Generate full file path for saving data"""
        data_dir = self.get_data_dir(data_source, processed)
        return os.path.join(data_dir, filename)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Return configuration as dictionary for backward compatibility
        This matches the return format of your logging_config_setup_02.py
        """
        return {
            "base_dir": self.base_dir,
            "raw_data_dir": self.raw_data_dir,
            "processed_dir": self.processed_dir,
            "model_dir": self.model_dir,
            "archive_dir": self.archive_dir,
            "logs_dir": self.logs_dir,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "data_sources": self.data_sources,
        }
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the current environment"""
        return {
            "base_dir": self.base_dir,
            "is_docker": os.path.exists("/.dockerenv"),
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "environment_variables": {
                key: value for key, value in os.environ.items() 
                if key.startswith(("BASE_DIR", "SF_", "PIPELINE_", "DOCKER_", "LOG_"))
            }
        }


# Global configuration instance
config = PipelineConfig()


def get_config() -> PipelineConfig:
    """Get the global configuration instance"""
    return config


def setup_script_logging(script_name: str) -> logging.Logger:
    """
    Standard function for scripts to setup logging
    Usage: logger = setup_script_logging(__file__)
    """
    return config.get_script_logger(script_name)


# Backward compatibility functions - exact same signatures as logging_config_setup_02.py
def setup_logging() -> logging.Logger:
    """Configure logging for the pipeline - same function signature as your script"""
    return config.logger


def setup_directories() -> Dict[str, Any]:
    """
    Create directory structure and return paths and date ranges
    EXACT same function signature and return format as logging_config_setup_02.py
    """
    return config.get_config_dict()


if __name__ == "__main__":
    """Test the configuration system"""
    print("SF Business Model Pipeline Configuration")
    print("=" * 50)
    
    # Test the functions exactly like your logging_config_setup_02.py
    logger = setup_logging()
    config_dict = setup_directories()
    
    print(f"Base Directory: {config_dict['base_dir']}")
    print(f"Raw Data Directory: {config_dict['raw_data_dir']}")
    print(f"Processed Directory: {config_dict['processed_dir']}")
    
    env_info = config.get_environment_info()
    print(f"Running in Docker: {env_info['is_docker']}")
    print(f"Working Directory: {env_info['working_directory']}")
    
    print(f"\nData Collection Period:")
    print(f"  Start: {config_dict['start_date'].strftime('%Y-%m-%d')}")
    print(f"  End: {config_dict['end_date'].strftime('%Y-%m-%d')}")
    
    print(f"\nData Sources:")
    for source in config_dict['data_sources']:
        raw_dir = config.get_data_dir(source, processed=False)
        processed_dir = config.get_data_dir(source, processed=True)
        print(f"  {source}:")
        print(f"    Raw: {raw_dir}")
        print(f"    Processed: {processed_dir}")
    
    print(f"\nEnvironment Variables:")
    for key, value in env_info['environment_variables'].items():
        print(f"  {key}: {value}")
    
    # Test directory creation for a specific source
    config.ensure_data_dirs("crime")
    logger.info("Configuration test completed successfully!")
    
    # Print the same final message as your script
    print(f"Data collection period: {config_dict['start_date'].strftime('%Y-%m-%d')} to {config_dict['end_date'].strftime('%Y-%m-%d')}")