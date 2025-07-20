"""
Model loading and management utilities
Optimized for production deployment and error handling
"""
import json
import torch
import torch.nn as nn
import pytorch_lightning as pl
import streamlit as st
from typing import Tuple, Dict, Any, Optional

from .config import config

class BusinessSurvivalModel(pl.LightningModule):
    """
    Production-ready deep learning model for business survival prediction
    Multi-branch neural architecture for spatial, temporal, and business features
    """

    def __init__(
        self,
        spatial_dim: int,
        temporal_dim: int,
        business_dim: int,
        success_rate: float = 0.5,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Calculate class weights for imbalanced data
        failure_rate = 1 - success_rate
        self.pos_weight = torch.tensor([failure_rate / success_rate])

        # Spatial feature network
        if spatial_dim > 0:
            self.spatial_net = nn.Sequential(
                nn.Linear(spatial_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            spatial_output_dim = 32
        else:
            self.spatial_net = None
            spatial_output_dim = 0

        # Temporal feature network
        if temporal_dim > 0:
            self.temporal_net = nn.Sequential(
                nn.Linear(temporal_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            temporal_output_dim = 32
        else:
            self.temporal_net = None
            temporal_output_dim = 0

        # Business feature network (largest)
        if business_dim > 0:
            self.business_net = nn.Sequential(
                nn.Linear(business_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            business_output_dim = 32
        else:
            self.business_net = None
            business_output_dim = 0

        # Fusion network
        fusion_input_dim = (
            spatial_output_dim + temporal_output_dim + business_output_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, 1),
        )

    def forward(self, spatial_x, temporal_x, business_x):
        """Forward pass through multi-branch architecture"""
        features = []

        if self.spatial_net is not None and spatial_x.size(1) > 0:
            features.append(self.spatial_net(spatial_x))

        if self.temporal_net is not None and temporal_x.size(1) > 0:
            features.append(self.temporal_net(temporal_x))

        if self.business_net is not None and business_x.size(1) > 0:
            features.append(self.business_net(business_x))

        if len(features) == 0:
            raise ValueError("No features to process!")

        fused = torch.cat(features, dim=1)
        logits = self.fusion(fused).squeeze()
        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

@st.cache_resource
def load_model_and_config() -> Tuple[Optional[BusinessSurvivalModel], Optional[Dict], Optional[Dict], bool]:
    """
    Load the trained model and configuration with comprehensive error handling
    Returns: (model, config, preprocessing_info, success_flag)
    """
    try:
        # Load model configuration
        with open(config.MODEL_CONFIG_PATH, "r") as f:
            model_config = json.load(f)

        # Load preprocessing info
        with open(config.PREPROCESSING_INFO_PATH, "r") as f:
            preprocessing_info = json.load(f)

        # Initialize model with saved configuration
        model = BusinessSurvivalModel(
            spatial_dim=model_config["spatial_dim"],
            temporal_dim=model_config["temporal_dim"],
            business_dim=model_config["business_dim"],
            success_rate=model_config["success_rate"],
            dropout_rate=model_config["dropout_rate"],
        )

        # Load trained weights
        model.load_state_dict(torch.load(config.MODEL_STATE_DICT_PATH, map_location="cpu"))
        model.eval()  # Set to evaluation mode

        # Success logging
        st.success(f"✅ Model loaded successfully!")
        
        # Display model info in sidebar
        with st.sidebar:
            st.info(f"""
            **🤖 Model Architecture:**
            - Spatial features: {model_config['spatial_dim']}
            - Temporal features: {model_config['temporal_dim']}
            - Business features: {model_config['business_dim']}
            - Parameters: {model.count_parameters():,}
            - Training date: {model_config.get('training_date', 'Unknown')}
            """)

        return model, model_config, preprocessing_info, True

    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.error(f"Expected locations: {config.MODEL_DIR}")
        return None, None, None, False
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None, None, False

def get_model_info(model: BusinessSurvivalModel, model_config: Dict) -> Dict[str, Any]:
    """Get comprehensive model information for display"""
    if model is None:
        return {}
    
    return {
        "architecture": "Multi-branch Neural Network",
        "framework": "PyTorch Lightning",
        "spatial_features": model_config.get("spatial_dim", 0),
        "temporal_features": model_config.get("temporal_dim", 0),
        "business_features": model_config.get("business_dim", 0),
        "total_parameters": model.count_parameters(),
        "model_size_mb": sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024),  # Approximate MB
        "training_date": model_config.get("training_date", "Unknown"),
        "success_rate": model_config.get("success_rate", 0.5),
        "dropout_rate": model_config.get("dropout_rate", 0.2)
    }