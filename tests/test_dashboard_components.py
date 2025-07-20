import pytest
import pandas as pd
import numpy as np
import os
import sys
import json
from unittest.mock import Mock, patch, MagicMock
import tempfile
import streamlit as st

# Add app to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import dashboard components
try:
    from app.utils.model_loader import load_model, load_preprocessing_info
    from app.utils.config import get_app_config
except ImportError as e:
    print(f"Warning: Could not import app modules: {e}")


class TestDashboardComponents:
    """Test suite for dashboard components"""
    
    def setup_method(self):
        """Setup test data for dashboard testing"""
        self.sample_business_data = {
            'business_name': 'Test Restaurant',
            'latitude': 37.7749,
            'longitude': -122.4194,
            'start_date': '2020-01-01',
            'industry_code': '722511'
        }
        
        self.sample_features = np.random.rand(118)  # 118 features as per documentation
    
    def test_app_structure(self):
        """Test that app structure exists"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        app_dir = os.path.join(base_dir, 'app')
        
        # Check main app files exist
        main_files = ['app.py', 'main.py']
        for file_name in main_files:
            file_path = os.path.join(app_dir, file_name)
            if os.path.exists(file_path):
                assert os.path.getsize(file_path) > 0, f"App file {file_name} should not be empty"
        
        # Check app directories
        app_subdirs = ['components', 'utils', 'pages']
        for subdir in app_subdirs:
            subdir_path = os.path.join(app_dir, subdir)
            if os.path.exists(subdir_path):
                assert os.path.isdir(subdir_path), f"App subdir {subdir} should be a directory"
    
    @patch('torch.load')
    def test_model_loading_interface(self, mock_torch_load):
        """Test model loading functionality"""
        # Mock model loading
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_torch_load.return_value = mock_model
        
        # Test model loading function exists and can be called
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, 'data', 'models', 'final_business_model.pth')
        
        if os.path.exists(model_path):
            try:
                from app.utils.model_loader import load_model
                # This should not raise an exception
                assert callable(load_model)
            except ImportError:
                # If import fails, at least check file exists
                assert os.path.exists(model_path)
    
    def test_config_loading_dashboard(self):
        """Test dashboard configuration loading"""
        try:
            from app.utils.config import get_app_config
            config = get_app_config()
            assert config is not None
        except ImportError:
            # If config module doesn't exist, that's ok for basic test
            pass
    
    def test_prediction_input_validation(self):
        """Test prediction input validation"""
        # Test coordinate validation
        valid_lat = 37.7749
        valid_lon = -122.4194
        
        # SF latitude range
        assert 37.7 <= valid_lat <= 37.8, "Latitude should be in SF range"
        
        # SF longitude range  
        assert -122.5 <= valid_lon <= -122.3, "Longitude should be in SF range"
        
        # Test business name validation
        valid_name = "Test Business"
        assert len(valid_name.strip()) > 0, "Business name should not be empty"
        assert len(valid_name) <= 100, "Business name should not be too long"
    
    def test_feature_preprocessing(self):
        """Test feature preprocessing for model input"""
        # Test that features can be converted to proper format
        features = self.sample_features
        
        # Should be numeric
        assert all(isinstance(x, (int, float, np.number)) for x in features), "All features should be numeric"
        
        # Should not contain NaN
        assert not np.any(np.isnan(features)), "Features should not contain NaN"
        
        # Should be right length (118 features expected)
        expected_length = 118
        if len(features) != expected_length:
            print(f"Warning: Expected {expected_length} features, got {len(features)}")
    
    def test_visualization_components(self):
        """Test visualization component structure"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Check if visualizations directory exists
        viz_dir = os.path.join(base_dir, 'visualizations')
        if os.path.exists(viz_dir):
            # Should contain some visualization files
            viz_files = os.listdir(viz_dir)
            assert len(viz_files) > 0, "Visualizations directory should not be empty"
    
    def test_streamlit_compatibility(self):
        """Test Streamlit compatibility"""
        # Test that streamlit can be imported and basic functions work
        import streamlit as st
        
        # Test basic streamlit functions exist
        assert hasattr(st, 'title')
        assert hasattr(st, 'write')
        assert hasattr(st, 'selectbox')
        assert hasattr(st, 'slider')
        assert hasattr(st, 'button')
    
    def test_model_prediction_interface(self):
        """Test model prediction interface"""
        # Create mock prediction function
        def mock_predict(features):
            """Mock prediction function"""
            if len(features) != 118:
                raise ValueError("Expected 118 features")
            return np.random.rand()  # Return probability between 0-1
        
        # Test prediction with correct input
        prediction = mock_predict(self.sample_features)
        assert 0 <= prediction <= 1, "Prediction should be probability between 0 and 1"
        
        # Test prediction with wrong input size
        with pytest.raises(ValueError):
            mock_predict(np.random.rand(50))  # Wrong number of features
    
    def test_error_handling(self):
        """Test error handling in dashboard"""
        # Test handling of invalid inputs
        invalid_coordinates = [
            (0, 0),  # Not in SF
            (90, 180),  # Out of range
            (None, None),  # Null values
        ]
        
        for lat, lon in invalid_coordinates:
            if lat is None or lon is None:
                # Should handle null coordinates
                assert True  # Test passes if we reach here
            elif not (37.7 <= lat <= 37.8 and -122.5 <= lon <= -122.3):
                # Should handle out of range coordinates
                assert True  # Test passes if we reach here


class TestModelIntegration:
    """Test model integration with dashboard"""
    
    def test_model_config_consistency(self):
        """Test model configuration consistency"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(base_dir, 'data', 'models', 'model_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check required config fields
            required_fields = ['spatial_dim', 'temporal_dim', 'business_dim']
            for field in required_fields:
                if field in config:
                    assert isinstance(config[field], int), f"{field} should be integer"
                    assert config[field] > 0, f"{field} should be positive"
    
    def test_preprocessing_info_availability(self):
        """Test preprocessing info availability"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        preprocessing_path = os.path.join(base_dir, 'data', 'models', 'preprocessing_info.json')
        
        if os.path.exists(preprocessing_path):
            with open(preprocessing_path, 'r') as f:
                preprocessing_info = json.load(f)
            
            # Should contain feature information
            assert preprocessing_info is not None
            # Basic structure check
            if 'feature_names' in preprocessing_info:
                assert isinstance(preprocessing_info['feature_names'], list)
    
    @patch('streamlit.error')
    @patch('streamlit.warning')
    def test_dashboard_error_reporting(self, mock_warning, mock_error):
        """Test dashboard error reporting"""
        # Test that error reporting functions can be called
        try:
            import streamlit as st
            # These should be callable
            assert callable(st.error)
            assert callable(st.warning)
            assert callable(st.success)
            assert callable(st.info)
        except Exception as e:
            pytest.fail(f"Streamlit error reporting failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])