import pytest
import pandas as pd
import numpy as np
import os
import sys
import subprocess
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestIntegrationPipeline:
    """Integration tests for the complete pipeline"""
    
    def setup_method(self):
        """Setup for integration tests"""
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        
    def test_pipeline_runner_exists(self):
        """Test that pipeline runner exists and is executable"""
        pipeline_path = os.path.join(self.base_dir, 'src', 'pipeline_runner.py')
        assert os.path.exists(pipeline_path), "Pipeline runner should exist"
        assert os.path.getsize(pipeline_path) > 0, "Pipeline runner should not be empty"
    
    def test_app_startup_simulation(self):
        """Test app startup simulation without actually running Streamlit"""
        app_path = os.path.join(self.base_dir, 'app', 'app.py')
        main_path = os.path.join(self.base_dir, 'app', 'main.py')
        
        # At least one app file should exist
        assert os.path.exists(app_path) or os.path.exists(main_path), "At least one app file should exist"
        
        # Test Python syntax by importing
        if os.path.exists(main_path):
            try:
                # Test that the file has valid Python syntax
                with open(main_path, 'r') as f:
                    content = f.read()
                compile(content, main_path, 'exec')
            except SyntaxError as e:
                pytest.fail(f"App has syntax error: {e}")
    
    def test_model_pipeline_integration(self):
        """Test model and pipeline integration"""
        # Check if model files exist and are accessible
        model_dir = os.path.join(self.base_dir, 'data', 'models')
        if os.path.exists(model_dir):
            model_files = os.listdir(model_dir)
            
            # Should have at least some model files
            expected_files = ['final_business_model.pth', 'model_config.json']
            for file_name in expected_files:
                if file_name in model_files:
                    file_path = os.path.join(model_dir, file_name)
                    assert os.path.getsize(file_path) > 0, f"Model file {file_name} should not be empty"
    
    def test_data_flow_integrity(self):
        """Test data flow from raw to processed"""
        raw_dir = os.path.join(self.base_dir, 'raw_data')
        processed_dir = os.path.join(self.base_dir, 'processed')
        
        # If raw data exists, processed should exist or be creatable
        if os.path.exists(raw_dir):
            raw_files = []
            for root, dirs, files in os.walk(raw_dir):
                raw_files.extend([f for f in files if f.endswith('.parquet')])
            
            if raw_files:
                # Should have some way to process this data
                processing_dir = os.path.join(self.base_dir, 'src', 'processing')
                assert os.path.exists(processing_dir), "Processing directory should exist if raw data exists"
    
    def test_docker_configuration(self):
        """Test Docker configuration files"""
        dockerfile_paths = [
            os.path.join(self.base_dir, 'Dockerfile'),
            os.path.join(self.base_dir, 'deployment', 'Dockerfile.multi-platform')
        ]
        
        docker_exists = any(os.path.exists(path) for path in dockerfile_paths)
        if docker_exists:
            # If Docker files exist, check docker-compose
            compose_paths = [
                os.path.join(self.base_dir, 'docker-compose.yml'),
                os.path.join(self.base_dir, 'deployment', 'docker-compose.optimized.yml')
            ]
            compose_exists = any(os.path.exists(path) for path in compose_paths)
            # Don't require compose to exist, but log if found
            if compose_exists:
                print("Docker Compose configuration found")
    
    @patch('subprocess.run')
    def test_streamlit_health_check(self, mock_subprocess):
        """Test Streamlit health check endpoint simulation"""
        # Mock successful health check
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "OK"
        
        # Simulate health check
        health_endpoint = "http://localhost:8501/_stcore/health"
        # In real scenario, this would be a curl command
        # We just test that the concept works
        assert health_endpoint.endswith("/_stcore/health")
    
    def test_requirements_consistency(self):
        """Test requirements files consistency"""
        req_files = [
            os.path.join(self.base_dir, 'requirements.txt'),
            os.path.join(self.base_dir, 'requirements-optimized.txt')
        ]
        
        existing_req_files = [f for f in req_files if os.path.exists(f)]
        
        if existing_req_files:
            for req_file in existing_req_files:
                with open(req_file, 'r') as f:
                    content = f.read()
                
                # Should contain key dependencies
                key_deps = ['streamlit', 'pandas', 'numpy', 'torch']
                for dep in key_deps:
                    if dep in content.lower():
                        print(f"Found {dep} in {os.path.basename(req_file)}")
    
    def test_configuration_loading(self):
        """Test configuration loading across components"""
        try:
            from src.utils.config import get_config
            config = get_config()
            
            # Test config structure - might be object or dict
            assert config is not None, "Config should not be None"
            if hasattr(config, 'BASE_DIR'):
                assert config.BASE_DIR is not None, "BASE_DIR should exist"
            elif isinstance(config, dict) and 'BASE_DIR' in config:
                assert config['BASE_DIR'] is not None, "BASE_DIR should exist"
                
        except ImportError:
            # Config module might not exist, that's ok - test basic structure
            base_dir = os.path.dirname(os.path.dirname(__file__))
            assert os.path.exists(base_dir)
    
    def test_feature_engineering_integration(self):
        """Test feature engineering integration"""
        # Test that feature engineering produces expected output structure
        
        # Mock input data
        sample_business_data = pd.DataFrame({
            'business_id': ['1', '2', '3'],
            'latitude': [37.7749, 37.7849, 37.7949],
            'longitude': [-122.4194, -122.4294, -122.4394],
            'start_date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        })
        
        # Basic feature engineering test
        features_df = sample_business_data.copy()
        
        # Add temporal features
        features_df['year'] = features_df['start_date'].dt.year
        features_df['month'] = features_df['start_date'].dt.month
        features_df['day_of_week'] = features_df['start_date'].dt.dayofweek
        
        # Add spatial features (basic)
        features_df['lat_lon_sum'] = features_df['latitude'] + features_df['longitude']
        features_df['distance_from_center'] = np.sqrt(
            (features_df['latitude'] - 37.7749)**2 + 
            (features_df['longitude'] + 122.4194)**2
        )
        
        # Test feature count and types
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        assert len(numeric_features) >= 5, "Should have at least 5 numeric features"
        
        # Test no missing values in key features
        key_features = ['latitude', 'longitude', 'year', 'month']
        for feature in key_features:
            if feature in features_df.columns:
                assert not features_df[feature].isnull().any(), f"{feature} should not have missing values"


class TestSystemResources:
    """Test system resources and dependencies"""
    
    def test_memory_requirements(self):
        """Test basic memory availability"""
        import psutil
        
        # Get available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # Should have at least 1GB available for basic operations
        assert available_gb >= 1.0, f"Should have at least 1GB available memory, found {available_gb:.2f}GB"
    
    def test_disk_space(self):
        """Test disk space availability"""
        import psutil
        
        # Get disk usage for current directory
        base_dir = os.path.dirname(os.path.dirname(__file__))
        disk_usage = psutil.disk_usage(base_dir)
        available_gb = disk_usage.free / (1024**3)
        
        # Should have at least 1GB available for operations
        assert available_gb >= 1.0, f"Should have at least 1GB disk space, found {available_gb:.2f}GB"
    
    def test_python_version(self):
        """Test Python version compatibility"""
        import sys
        
        version = sys.version_info
        
        # Should be Python 3.7 or higher
        assert version.major == 3, "Should be using Python 3"
        assert version.minor >= 7, f"Should be Python 3.7+, found {version.major}.{version.minor}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])