import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION - Update these paths to match your setup
# ============================================================================

# BASE_DIR = os.getenv("BASE_DIR", "/app/San_Francisco_Business_Model")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = f"{BASE_DIR}/models"
DATA_DIR = f"{BASE_DIR}/processed/final"

# Model file paths
MODEL_STATE_DICT_PATH = f"{MODEL_DIR}/final_business_model.pth"
MODEL_CONFIG_PATH = f"{MODEL_DIR}/model_config.json"
PREPROCESSING_INFO_PATH = f"{MODEL_DIR}/preprocessing_info.json"

# Data file paths for context and visualizations
X_TRAIN_PATH = f"{DATA_DIR}/X_train.parquet"
Y_TRAIN_PATH = f"{DATA_DIR}/y_train.parquet"

# ============================================================================
# MODEL ARCHITECTURE - Must match your training code exactly
# ============================================================================


class BusinessSurvivalModel(pl.LightningModule):
    """Deep learning model for business survival prediction - matches training code"""

    def __init__(
        self,
        spatial_dim,
        temporal_dim,
        business_dim,
        success_rate=0.5,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Calculate class weights for imbalanced data
        failure_rate = 1 - success_rate
        self.pos_weight = torch.tensor([failure_rate / success_rate])

        # Build networks for each feature group
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


# ============================================================================
# LOADING FUNCTIONS
# ============================================================================


@st.cache_resource
def load_model_and_config():
    """Load the trained model and configuration"""
    try:
        # Load model configuration
        with open(MODEL_CONFIG_PATH, "r") as f:
            config = json.load(f)

        # Load preprocessing info
        with open(PREPROCESSING_INFO_PATH, "r") as f:
            preprocessing_info = json.load(f)

        # Initialize model with saved configuration
        model = BusinessSurvivalModel(
            spatial_dim=config["spatial_dim"],
            temporal_dim=config["temporal_dim"],
            business_dim=config["business_dim"],
            success_rate=config["success_rate"],
            dropout_rate=config["dropout_rate"],
        )

        # Load trained weights
        model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH, map_location="cpu"))
        model.eval()  # Set to evaluation mode

        print(f"✅ Model loaded successfully!")
        print(f"   Spatial features: {config['spatial_dim']}")
        print(f"   Temporal features: {config['temporal_dim']}")
        print(f"   Business features: {config['business_dim']}")
        print(f"   Training date: {config['training_date']}")

        return model, config, preprocessing_info, True

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None, False


@st.cache_data
def load_training_data():
    """Load training data for visualization and context"""
    try:
        # Load features and targets
        X_train = pd.read_parquet(X_TRAIN_PATH)
        y_train = pd.read_parquet(Y_TRAIN_PATH)

        # Combine for visualization
        df = X_train.copy()
        df["success"] = y_train["success"]

        return df, True

    except Exception as e:
        st.error(f"Failed to load training data: {e}")
        return None, False


# ============================================================================
# FEATURE PREPARATION FUNCTIONS
# ============================================================================


def prepare_input_features(input_data, preprocessing_info, training_data=None):
    """Prepare input features exactly as done during training"""

    # Define feature groups based on preprocessing info
    spatial_features = [
        f for f in preprocessing_info["spatial_features"] if f in input_data.columns
    ]
    temporal_features = [
        f for f in preprocessing_info["temporal_features"] if f in input_data.columns
    ]
    all_features = preprocessing_info["feature_columns"]
    business_features = [
        f for f in all_features if f not in spatial_features + temporal_features
    ]

    def prepare_feature_group(data, features, reference_data=None):
        """Prepare a specific feature group with proper encoding"""
        if not features:
            return np.array([]).reshape(len(data), 0)

        # Extract feature subset
        subset = data[features].copy()

        # Handle each column with proper categorical encoding
        for col in features:
            if not pd.api.types.is_numeric_dtype(subset[col]):
                # For categorical features, use training data to get proper encoding
                if reference_data is not None and col in reference_data.columns:
                    # Get unique values from training data
                    unique_vals = sorted(reference_data[col].unique())
                    value_mapping = {val: float(i) for i, val in enumerate(unique_vals)}

                    # Handle new categories not seen in training
                    current_val = subset[col].iloc[0]
                    if current_val not in value_mapping:
                        # Assign to most common category or create new encoding
                        most_common = (
                            reference_data[col].mode().iloc[0]
                            if len(reference_data[col].mode()) > 0
                            else unique_vals[0]
                        )
                        subset[col] = value_mapping.get(most_common, 0.0)
                    else:
                        subset[col] = value_mapping[current_val]
                else:
                    # Fallback encoding
                    unique_vals = subset[col].unique()
                    value_mapping = {val: float(i) for i, val in enumerate(unique_vals)}
                    subset[col] = subset[col].map(value_mapping).astype(float)
            else:
                subset[col] = pd.to_numeric(subset[col], errors="coerce")
                subset[col] = subset[col].replace([np.inf, -np.inf], np.nan)

        # Fill NaN with 0
        subset = subset.fillna(0.0)

        # Ensure all columns are numeric
        for col in features:
            subset[col] = pd.to_numeric(subset[col], errors="coerce").fillna(0.0)

        return subset.values.astype(np.float64)

    # Prepare feature groups with training data reference for proper encoding
    spatial_array = prepare_feature_group(input_data, spatial_features, training_data)
    temporal_array = prepare_feature_group(input_data, temporal_features, training_data)
    business_array = prepare_feature_group(input_data, business_features, training_data)

    # Use training data statistics for standardization if available
    def smart_standardize(data, features, reference_data=None):
        if data.shape[1] == 0:
            return data

        if reference_data is not None and len(features) > 0:
            # Use training data statistics for standardization
            reference_subset = reference_data[features].select_dtypes(
                include=[np.number]
            )
            if not reference_subset.empty:
                mean = reference_subset.mean().values
                std = reference_subset.std().values
                std = np.where(std == 0, 1, std)  # Avoid division by zero

                # Ensure dimensions match
                if len(mean) == data.shape[1] and len(std) == data.shape[1]:
                    return (data - mean) / std

        # Fallback to simple standardization
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.where(std == 0, 1, std)
        return (data - mean) / std

    spatial_scaled = smart_standardize(spatial_array, spatial_features, training_data)
    temporal_scaled = smart_standardize(
        temporal_array, temporal_features, training_data
    )
    business_scaled = smart_standardize(
        business_array, business_features, training_data
    )

    return spatial_scaled, temporal_scaled, business_scaled


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================


def predict_business_success(model, input_data, preprocessing_info, training_data=None):
    """Make prediction using the trained model"""
    try:
        # Debug: Print input data to check if it's changing
        st.write("🔍 Debug - Current Input:")
        st.write(
            f"Neighborhood: {input_data['neighborhoods_analysis_boundaries'].iloc[0]}"
        )
        st.write(f"Industry: {input_data['business_industry'].iloc[0]}")
        st.write(
            f"Lat/Lon: {input_data['latitude'].iloc[0]:.4f}, {input_data['longitude'].iloc[0]:.4f}"
        )

        # Prepare features with training data reference for proper encoding
        spatial_data, temporal_data, business_data = prepare_input_features(
            input_data, preprocessing_info, training_data
        )

        # Debug: Print feature shapes and some values
        st.write(f"Spatial features shape: {spatial_data.shape}")
        st.write(f"Temporal features shape: {temporal_data.shape}")
        st.write(f"Business features shape: {business_data.shape}")

        if spatial_data.shape[1] > 0:
            st.write(
                f"Sample spatial values: {spatial_data[0][:min(3, spatial_data.shape[1])]}"
            )
        if business_data.shape[1] > 0:
            st.write(
                f"Sample business values: {business_data[0][:min(3, business_data.shape[1])]}"
            )

        # Convert to tensors
        spatial_tensor = torch.FloatTensor(spatial_data)
        temporal_tensor = torch.FloatTensor(temporal_data)
        business_tensor = torch.FloatTensor(business_data)

        # Make prediction
        with torch.no_grad():
            logits = model(spatial_tensor, temporal_tensor, business_tensor)
            probability = torch.sigmoid(logits).item()

        st.write(f"Raw logits: {logits.item():.4f}")
        st.write(f"Final probability: {probability:.4f}")

        return probability

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        import traceback

        st.error(f"Detailed error: {traceback.format_exc()}")
        return 0.5  # Default fallback


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def create_success_gauge(score):
    """Create a gauge chart for the success score"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "5-Year Success Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "red"},
                    {"range": [30, 50], "color": "orange"},
                    {"range": [50, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "green"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": score * 100,
                },
            },
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def create_neighborhood_chart(df, selected_neighborhood=None, selected_industry=None):
    """Create neighborhood success rate chart filtered by industry"""
    if df is None:
        return None

    try:
        # Check if we have neighborhood data
        if "neighborhoods_analysis_boundaries" not in df.columns:
            return None

        # Filter by industry first if specified
        if selected_industry and "business_industry" in df.columns:
            industry_df = df[df["business_industry"] == selected_industry]
            chart_title = f"Top Neighborhoods for {selected_industry} Businesses"

            if len(industry_df) == 0:
                return None
        else:
            industry_df = df
            chart_title = "Top Neighborhoods by 5-Year Business Survival Rate"

        # Group by neighborhood and calculate success rate
        neigh_stats = (
            industry_df.groupby("neighborhoods_analysis_boundaries")["success"]
            .agg(["mean", "count"])
            .reset_index()
        )
        neigh_stats.columns = ["neighborhood", "success_rate", "business_count"]
        neigh_stats = neigh_stats.sort_values("success_rate", ascending=False)

        # Filter for significant sample (adjust threshold based on industry filtering)
        min_sample = 5 if selected_industry else 10
        neigh_stats = neigh_stats[neigh_stats["business_count"] >= min_sample]

        if neigh_stats.empty:
            return None

        # Limit to top 10
        plot_neighs = neigh_stats.head(10)

        # Highlight selected neighborhood
        plot_neighs["selected"] = False
        if (
            selected_neighborhood
            and selected_neighborhood in plot_neighs["neighborhood"].values
        ):
            plot_neighs.loc[
                plot_neighs["neighborhood"] == selected_neighborhood, "selected"
            ] = True
        # If selected neighborhood isn't in top 10, add it to show comparison
        elif selected_neighborhood:
            selected_data = neigh_stats[
                neigh_stats["neighborhood"] == selected_neighborhood
            ]
            if not selected_data.empty:
                selected_data = selected_data.copy()
                selected_data["selected"] = True
                # Add to the plot data
                plot_neighs = pd.concat(
                    [selected_data, plot_neighs.head(9)]
                ).reset_index(drop=True)

        # Create chart with dynamic colors
        colors = [
            "darkblue" if selected else "#1f77b4"
            for selected in plot_neighs["selected"]
        ]

        fig = px.bar(
            plot_neighs,
            x="neighborhood",
            y="success_rate",
            color="selected",
            color_discrete_map={True: "darkblue", False: "#1f77b4"},
            text="business_count",
            title=chart_title,
            labels={
                "neighborhood": "Neighborhood",
                "success_rate": "5-Year Survival Rate",
                "business_count": f'Number of {selected_industry if selected_industry else "All"} Businesses',
            },
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis_tickformat=".0%",
            showlegend=False,
            height=400,
            title_font_size=14,
        )

        fig.update_traces(texttemplate="n=%{text}", textposition="outside")

        # Add annotation if showing industry-specific data
        if selected_industry:
            avg_success_rate = plot_neighs["success_rate"].mean()
            fig.add_annotation(
                text=f"Average {selected_industry} success rate: {avg_success_rate:.1%}",
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                showarrow=False,
                font=dict(size=10, color="gray"),
                bgcolor="rgba(255,255,255,0.8)",
            )

        return fig

    except Exception as e:
        st.warning(f"Could not create neighborhood chart: {e}")
        return None


def create_spatial_heatmap(df, lat, lon):
    """Create spatial heatmap of business success"""
    if df is None:
        return None

    try:
        # Check required columns
        if not all(col in df.columns for col in ["latitude", "longitude", "success"]):
            return None

        # Sample data for performance
        sample_size = min(2000, len(df))
        sample_df = df.sample(sample_size, random_state=42)

        fig = px.scatter_mapbox(
            sample_df,
            lat="latitude",
            lon="longitude",
            color="success",
            color_continuous_scale=["red", "green"],
            size_max=8,
            zoom=12,
            mapbox_style="carto-positron",
            title="Geographic Distribution of Business Success in San Francisco",
            opacity=0.7,
            labels={"success": "5-Year Survival"},
        )

        # Add selected location marker
        fig.add_trace(
            go.Scattermapbox(
                lat=[lat],
                lon=[lon],
                mode="markers",
                marker=dict(size=15, color="blue", opacity=1),
                text="Selected Location",
                hoverinfo="text",
                showlegend=False,
            )
        )

        fig.update_layout(height=500, margin=dict(l=0, r=0, t=50, b=0))

        return fig

    except Exception as e:
        st.warning(f"Could not create spatial heatmap: {e}")
        return None


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================


def main():
    # Page configuration
    st.set_page_config(
        page_title="SF Business Success Predictor", page_icon="📊", layout="wide"
    )

    # Load model and data
    model, config, preprocessing_info, model_loaded = load_model_and_config()
    df, data_loaded = load_training_data()

    # Title
    st.title("📍 San Francisco Business Success Predictor")
    st.markdown(
        "#### Predict 5-year business survival probability using trained ML model"
    )

    # Status indicators
    if model_loaded:
        st.sidebar.success("✅ Trained model loaded successfully")
        st.sidebar.info(f"📅 Model trained: {config['training_date']}")
    else:
        st.sidebar.error("❌ Could not load trained model")
        st.stop()

    if data_loaded:
        st.sidebar.success("✅ Training data loaded for context")
    else:
        st.sidebar.warning("⚠️ Could not load training data (visualizations limited)")

    # Sidebar inputs
    st.sidebar.header("🏢 Business Information")

    # Get unique values from training data if available
    if data_loaded and "business_industry" in df.columns:
        industries = sorted(df["business_industry"].unique())
    else:
        # Fallback list
        industries = [
            "Restaurant",
            "Retail",
            "Technology",
            "Professional Services",
            "Healthcare",
            "Financial Services",
        ]

    selected_industry = st.sidebar.selectbox("Business Industry", industries)

    # Location inputs
    st.sidebar.header("📍 Location")

    # San Francisco neighborhoods with coordinates
    sf_neighborhoods = {
        "Financial District": (37.7946, -122.3999),
        "Marina": (37.8021, -122.4369),
        "Mission": (37.7599, -122.4148),
        "South of Market": (37.7785, -122.4056),
        "Nob Hill": (37.7930, -122.4161),
        "Castro": (37.7609, -122.4351),
        "Chinatown": (37.7941, -122.4078),
        "Hayes Valley": (37.7759, -122.4260),
        "Pacific Heights": (37.7925, -122.4382),
        "Russian Hill": (37.8014, -122.4182),
    }

    selected_neighborhood = st.sidebar.selectbox(
        "Neighborhood", list(sf_neighborhoods.keys()), index=0
    )

    # Get coordinates for selected neighborhood
    default_lat, default_lon = sf_neighborhoods[selected_neighborhood]

    # Allow coordinate adjustment
    col1, col2 = st.sidebar.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=default_lat, format="%.6f")
    with col2:
        longitude = st.number_input("Longitude", value=default_lon, format="%.6f")

    # Additional inputs
    st.sidebar.header("📅 Timing & Context")
    start_year = st.sidebar.slider("Start Year", 2025, 2030, 2025)
    start_month = st.sidebar.slider("Start Month", 1, 12, 6)
    start_quarter = st.sidebar.selectbox("Start Quarter", [1, 2, 3, 4], index=1)

    # Economic factors
    st.sidebar.header("💼 Economic Factors")
    high_crime_area = st.sidebar.checkbox("High Crime Area?", value=False)

    # Build input DataFrame with proper updates
    input_data = pd.DataFrame(
        {
            "latitude": [latitude],
            "longitude": [longitude],
            "business_industry": [selected_industry],
            "neighborhoods_analysis_boundaries": [selected_neighborhood],
            "start_year": [start_year],
            "start_month": [start_month],
            "start_quarter": [start_quarter],
            "high_crime_area": [1 if high_crime_area else 0],
            "supervisor_district": [1],  # These should be updated based on location
            "business_corridor": [0],
            "similar_businesses_count": [50],
            "sf_gdp": [1000],
            "sf_unemployment_rate": [0.042],
            "sf_house_price_index": [400],
            "neighborhood_business_count": [500],
            "district_total_crimes": [100 if high_crime_area else 50],
            "overall_sentiment_mean": [0.5],
        }
    )

    # Add location-specific variations to make predictions more realistic
    # Different neighborhoods should have different characteristics
    neighborhood_adjustments = {
        "Financial District": {
            "sf_gdp": 1200,
            "neighborhood_business_count": 800,
            "similar_businesses_count": 150,
            "overall_sentiment_mean": 0.7,
        },
        "Marina": {
            "sf_gdp": 1100,
            "neighborhood_business_count": 400,
            "similar_businesses_count": 80,
            "overall_sentiment_mean": 0.6,
        },
        "Mission": {
            "sf_gdp": 900,
            "neighborhood_business_count": 600,
            "similar_businesses_count": 120,
            "overall_sentiment_mean": 0.5,
        },
        "South of Market": {
            "sf_gdp": 1000,
            "neighborhood_business_count": 700,
            "similar_businesses_count": 100,
            "overall_sentiment_mean": 0.4,
        },
        "Castro": {
            "sf_gdp": 950,
            "neighborhood_business_count": 300,
            "similar_businesses_count": 60,
            "overall_sentiment_mean": 0.65,
        },
    }

    # Apply neighborhood-specific adjustments
    if selected_neighborhood in neighborhood_adjustments:
        adjustments = neighborhood_adjustments[selected_neighborhood]
        for key, value in adjustments.items():
            if key in input_data.columns:
                input_data[key] = [value]

    # Industry-specific adjustments
    industry_adjustments = {
        "Technology": {"similar_businesses_count": 200, "overall_sentiment_mean": 0.7},
        "Restaurant": {"similar_businesses_count": 80, "overall_sentiment_mean": 0.4},
        "Healthcare": {"similar_businesses_count": 40, "overall_sentiment_mean": 0.8},
        "Retail": {"similar_businesses_count": 100, "overall_sentiment_mean": 0.5},
        "Professional Services": {
            "similar_businesses_count": 150,
            "overall_sentiment_mean": 0.6,
        },
        "Financial Services": {
            "similar_businesses_count": 120,
            "overall_sentiment_mean": 0.75,
        },
    }

    if selected_industry in industry_adjustments:
        adjustments = industry_adjustments[selected_industry]
        for key, value in adjustments.items():
            if key in input_data.columns:
                input_data[key] = [value]

    # Add any missing features with defaults
    for feature in preprocessing_info["feature_columns"]:
        if feature not in input_data.columns:
            input_data[feature] = [0]

    # Create a unique key for caching based on all relevant inputs
    input_key = f"{selected_neighborhood}_{selected_industry}_{latitude:.4f}_{longitude:.4f}_{start_year}_{high_crime_area}"

    # Real-time prediction (updates automatically when inputs change)
    # Use session state to track if inputs have changed
    current_inputs = f"{selected_neighborhood}_{selected_industry}_{latitude:.4f}_{longitude:.4f}_{start_year}_{high_crime_area}"

    # Initialize session state for tracking changes
    if "last_inputs" not in st.session_state:
        st.session_state.last_inputs = ""
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = 0.5

    # Only recalculate if inputs have changed (for performance)
    if current_inputs != st.session_state.last_inputs:
        with st.spinner("🔄 Updating prediction..."):
            # Make prediction based on current inputs with training data for proper encoding
            current_success_score = predict_business_success(
                model, input_data, preprocessing_info, df
            )
            st.session_state.last_prediction = current_success_score
            st.session_state.last_inputs = current_inputs
    else:
        current_success_score = st.session_state.last_prediction

    # Display live prediction in sidebar (updates automatically)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Live Prediction")

    # Dynamic metric with delta showing change from baseline
    baseline_score = 0.5  # You could use city average here
    delta_value = current_success_score - baseline_score

    st.sidebar.metric(
        "Success Probability",
        f"{current_success_score:.1%}",
        delta=f"{delta_value:+.1%} vs baseline",
        help="Updates automatically as you change inputs",
    )

    # Create unique key for this prediction to prevent caching issues
    prediction_key = f"pred_{input_key}_{current_success_score:.3f}"

    # Real-time mini gauge in sidebar
    mini_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current_success_score * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue", "thickness": 0.8},
                "steps": [
                    {"range": [0, 30], "color": "lightpink"},
                    {"range": [30, 50], "color": "lightyellow"},
                    {"range": [50, 70], "color": "lightblue"},
                    {"range": [70, 100], "color": "lightgreen"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 2},
                    "thickness": 0.75,
                    "value": current_success_score * 100,
                },
            },
        )
    )

    mini_gauge.update_layout(
        height=200, margin=dict(l=10, r=10, t=10, b=10), font={"size": 10}
    )

    st.sidebar.plotly_chart(
        mini_gauge, use_container_width=True, key=f"mini_gauge_{prediction_key}"
    )

    # Dynamic color-coded feedback in sidebar (changes with score)
    if current_success_score >= 0.7:
        st.sidebar.success("🟢 **High Success Probability**\nStrong location choice!")
    elif current_success_score >= 0.5:
        st.sidebar.warning("🟡 **Moderate Success Probability**\nProceed with caution")
    else:
        st.sidebar.error("🔴 **Low Success Probability**\nConsider alternatives")

    # Show what's driving the prediction
    st.sidebar.markdown("#### 📍 Current Analysis:")
    st.sidebar.write(f"**Location:** {selected_neighborhood}")
    st.sidebar.write(f"**Industry:** {selected_industry}")
    st.sidebar.write(f"**Year:** {start_year}")

    if high_crime_area:
        st.sidebar.write("⚠️ High crime area adjustment")

    # Quick comparison to neighborhood average (if data available)
    if data_loaded and "neighborhoods_analysis_boundaries" in df.columns:
        try:
            neighborhood_data = df[
                df["neighborhoods_analysis_boundaries"] == selected_neighborhood
            ]
            if len(neighborhood_data) > 0:
                neighborhood_avg = neighborhood_data["success"].mean()
                if current_success_score > neighborhood_avg:
                    st.sidebar.success(
                        f"📈 **{current_success_score - neighborhood_avg:+.1%}** vs neighborhood avg"
                    )
                else:
                    st.sidebar.warning(
                        f"📉 **{current_success_score - neighborhood_avg:+.1%}** vs neighborhood avg"
                    )
        except:
            pass

    # Detailed Analysis button (for comprehensive report)
    if st.sidebar.button("📊 Get Comprehensive Analysis", type="primary"):
        with st.spinner("Generating detailed analysis report..."):

            # Use the current prediction score
            success_score = current_success_score

            # Display results
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### 🎯 Success Score")
                # Use the CURRENT prediction score (already calculated above)
                gauge_chart = create_success_gauge(success_score)

                # Create unique key based on all inputs to force updates
                gauge_key = f"main_gauge_{selected_neighborhood}_{selected_industry}_{success_score:.4f}_{high_crime_area}"
                st.plotly_chart(gauge_chart, use_container_width=True, key=gauge_key)

                # Dynamic interpretation based on actual score
                if success_score >= 0.7:
                    st.success("✅ **High probability** of 5-year business success")
                    confidence_level = "High Confidence"
                    confidence_color = "🟢"
                elif success_score >= 0.5:
                    st.warning("⚠️ **Moderate probability** of 5-year business success")
                    confidence_level = "Moderate Confidence"
                    confidence_color = "🟡"
                else:
                    st.error("❌ **Low probability** of 5-year business success")
                    confidence_level = "Low Confidence"
                    confidence_color = "🔴"

                # Dynamic model info with actual prediction and context
                st.info(
                    f"""
                **🎯 Prediction Summary:**
                - **Location:** {selected_neighborhood}
                - **Industry:** {selected_industry}
                - **Success Score:** {success_score:.1%}
                - **Confidence:** {confidence_color} {confidence_level}
                - **Risk Level:** {"Low" if success_score >= 0.7 else "Moderate" if success_score >= 0.5 else "High"}
                """
                )

                # Show how this prediction compares to benchmarks
                if data_loaded and "neighborhoods_analysis_boundaries" in df.columns:
                    try:
                        # Get neighborhood benchmark for this specific industry
                        neighborhood_industry_data = df[
                            (
                                df["neighborhoods_analysis_boundaries"]
                                == selected_neighborhood
                            )
                            & (df["business_industry"] == selected_industry)
                        ]

                        # Get broader benchmarks
                        neighborhood_all_data = df[
                            df["neighborhoods_analysis_boundaries"]
                            == selected_neighborhood
                        ]
                        citywide_industry_data = df[
                            df["business_industry"] == selected_industry
                        ]
                        city_avg = df["success"].mean()

                        st.markdown("### 📊 Benchmark Comparison")

                        # Create comparison metrics
                        col_a, col_b = st.columns(2)

                        with col_a:
                            if len(neighborhood_all_data) > 0:
                                neighborhood_avg = neighborhood_all_data[
                                    "success"
                                ].mean()
                                delta_neigh = success_score - neighborhood_avg
                                st.metric(
                                    f"{selected_neighborhood} Average",
                                    f"{neighborhood_avg:.1%}",
                                    delta=f"{delta_neigh:+.1%}",
                                )

                        with col_b:
                            if len(citywide_industry_data) > 0:
                                industry_avg = citywide_industry_data["success"].mean()
                                delta_industry = success_score - industry_avg
                                st.metric(
                                    f"{selected_industry} Citywide",
                                    f"{industry_avg:.1%}",
                                    delta=f"{delta_industry:+.1%}",
                                )

                        # Specific combination insight
                        if len(neighborhood_industry_data) >= 5:
                            specific_avg = neighborhood_industry_data["success"].mean()
                            st.success(
                                f"🎯 **{selected_industry} in {selected_neighborhood}** historically: {specific_avg:.1%}"
                            )

                            if success_score > specific_avg:
                                st.success(
                                    f"🚀 Your profile predicts **{success_score - specific_avg:+.1%} better** than historical performance!"
                                )
                            else:
                                st.warning(
                                    f"📉 Your prediction is **{specific_avg - success_score:+.1%} below** historical average."
                                )
                        else:
                            st.info(
                                f"💡 Limited data for {selected_industry} in {selected_neighborhood} - potential market opportunity!"
                            )

                    except Exception as e:
                        st.info("Benchmark comparison not available")
                else:
                    st.info("Historical benchmark data not loaded")

            with col2:
                st.markdown("### 🗺️ Geographic Context")
                spatial_map = create_spatial_heatmap(df, latitude, longitude)
                if spatial_map:
                    st.plotly_chart(
                        spatial_map,
                        use_container_width=True,
                        key=f"map_{selected_neighborhood}",
                    )
                    st.caption(
                        "Green: Successful businesses (5+ years) | Red: Failed businesses | Blue: Your location"
                    )
                else:
                    st.info("Geographic visualization not available")

            # Additional analysis with dynamic content
            st.markdown("### 📊 Contextual Analysis")

            if data_loaded:
                # Dynamic neighborhood comparison highlighting current selection AND filtered by industry
                neighborhood_chart = create_neighborhood_chart(
                    df, selected_neighborhood, selected_industry
                )
                if neighborhood_chart:
                    neighborhood_chart_key = (
                        f"neigh_{selected_neighborhood}_{selected_industry}"
                    )
                    st.plotly_chart(
                        neighborhood_chart,
                        use_container_width=True,
                        key=neighborhood_chart_key,
                    )

                    # Show specific insights for selected neighborhood + industry
                    try:
                        industry_in_neighborhood = df[
                            (
                                df["neighborhoods_analysis_boundaries"]
                                == selected_neighborhood
                            )
                            & (df["business_industry"] == selected_industry)
                        ]

                        if len(industry_in_neighborhood) >= 5:
                            industry_neigh_success = industry_in_neighborhood[
                                "success"
                            ].mean()
                            industry_citywide = df[
                                df["business_industry"] == selected_industry
                            ]["success"].mean()

                            st.markdown(
                                f"### 🎯 {selected_industry} in {selected_neighborhood}"
                            )
                            col_x, col_y, col_z = st.columns(3)

                            with col_x:
                                st.metric("Your Prediction", f"{success_score:.1%}")
                            with col_y:
                                st.metric(
                                    "Historical (This Location)",
                                    f"{industry_neigh_success:.1%}",
                                )
                            with col_z:
                                st.metric(
                                    "Historical (Citywide)", f"{industry_citywide:.1%}"
                                )

                            # Performance indicator
                            if success_score > industry_neigh_success:
                                st.success(
                                    f"🚀 Your business profile predicts **better performance** than historical {selected_industry} businesses in {selected_neighborhood}!"
                                )
                            elif success_score > industry_citywide:
                                st.info(
                                    f"📈 Your prediction is above citywide average for {selected_industry} businesses."
                                )
                            else:
                                st.warning(
                                    f"📉 Your prediction is below historical averages. Consider optimization strategies."
                                )

                        else:
                            st.info(
                                f"Limited historical data for {selected_industry} in {selected_neighborhood}. This could indicate an underserved market opportunity."
                            )

                            # Show general industry data for context
                            industry_citywide = df[
                                df["business_industry"] == selected_industry
                            ]["success"].mean()
                            st.markdown(
                                f"**Citywide {selected_industry} average:** {industry_citywide:.1%}"
                            )

                    except Exception as e:
                        st.warning("Could not generate industry-specific insights.")
                else:
                    st.info(
                        f"Not enough data to show neighborhood comparison for {selected_industry} businesses"
                    )

                # Dynamic business recommendations based on actual score
                st.markdown("### 💡 Recommendations")

                if success_score >= 0.7:
                    st.success(
                        f"""
                    **🎉 Excellent Location Choice for {selected_industry}!**
                    - Your {selected_neighborhood} location shows **{success_score:.1%} success probability**
                    - This is well above the typical threshold for business viability
                    - **Action Items:**
                      * Proceed with confidence in your site selection
                      * Focus on operational excellence and customer service
                      * Consider this a validated market opportunity
                    """
                    )
                elif success_score >= 0.5:
                    st.warning(
                        f"""
                    **⚠️ Moderate Risk for {selected_industry} in {selected_neighborhood}**
                    - Your prediction of **{success_score:.1%}** suggests cautious optimism
                    - **Action Items:**
                      * Conduct additional market research
                      * Develop strategies to mitigate identified risks
                      * Ensure adequate capital reserves (18+ months)
                      * Consider pivoting location or business model if possible
                    """
                    )
                else:
                    st.error(
                        f"""
                    **🚨 High Risk Location for {selected_industry}**
                    - **{success_score:.1%} success probability** indicates significant challenges
                    - **Recommended Actions:**
                      * Strongly consider alternative neighborhoods
                      * If proceeding, develop comprehensive risk mitigation plan
                      * Ensure exceptional business model and execution
                      * Have contingency plans and extra capital reserves
                    """
                    )

                # Location-specific insights
                if high_crime_area:
                    st.warning(
                        "⚠️ **Crime Risk Factor**: High crime areas show reduced business survival rates. Consider security investments and adjusted operating hours."
                    )

                # Industry-specific recommendations
                if selected_industry in ["Restaurant", "Retail"]:
                    st.info(
                        "🍽️ **Customer-Facing Business**: Focus on foot traffic, visibility, and customer experience. Consider delivery options to expand reach."
                    )
                elif selected_industry in ["Technology", "Professional Services"]:
                    st.info(
                        "💼 **Professional Services**: Location matters less than talent and client access. Consider proximity to business districts and transportation."
                    )

            else:
                st.info("Additional analysis requires training data context")

    else:
        # Welcome screen with live preview
        st.markdown(
            """
        ## Welcome to the SF Business Success Predictor
        
        This tool uses a trained machine learning model to predict the probability of 
        5-year business survival at specific locations in San Francisco.
        
        **Features:**
        - Real ML model trained on San Francisco business data
        - Location-specific predictions
        - Industry and timing considerations
        - Data-driven recommendations
        
        **Instructions:**
        1. Select your business industry and location in the sidebar
        2. Watch the live prediction update as you change inputs
        3. Click "Get Detailed Analysis" for comprehensive insights
        
        Configure your business details in the sidebar to see predictions →
        """
        )

        # Show live preview of current prediction (always visible)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🎯 Live Prediction Dashboard")

            # Use the same prediction score that's already calculated
            preview_score = current_success_score

            # Full-size gauge for preview with unique key
            preview_gauge = create_success_gauge(preview_score)
            preview_key = f"main_preview_{input_key}_{preview_score:.3f}"
            st.plotly_chart(preview_gauge, use_container_width=True, key=preview_key)

            # Dynamic status message
            if preview_score >= 0.7:
                st.success("✅ **Excellent Location Choice**")
                recommendation = "Proceed with confidence!"
            elif preview_score >= 0.5:
                st.warning("⚠️ **Moderate Risk Location**")
                recommendation = "Consider additional research"
            else:
                st.error("❌ **High Risk Location**")
                recommendation = "Explore alternatives"

            st.markdown(
                f"""
            **Current Configuration:**
            - 📍 **Location:** {selected_neighborhood}
            - 🏢 **Industry:** {selected_industry}
            - 📅 **Start Year:** {start_year}
            - 🎯 **Success Probability:** **{preview_score:.1%}**
            - 💡 **Quick Recommendation:** {recommendation}
            """
            )

            # Show input changes effect
            st.markdown("##### 🔄 Try Different Options:")
            st.markdown(
                "Change neighborhood or industry in the sidebar to see how predictions update in real-time!"
            )

        with col2:
            if data_loaded:
                st.markdown("### 🗺️ Interactive Location Map")
                preview_map = create_spatial_heatmap(df, latitude, longitude)
                if preview_map:
                    map_key = f"main_map_{selected_neighborhood}_{latitude:.4f}_{longitude:.4f}"
                    st.plotly_chart(preview_map, use_container_width=True, key=map_key)
                    st.caption(
                        "🔵 Your location | 🟢 Successful businesses | 🔴 Failed businesses"
                    )

                    # Live neighborhood insights
                    try:
                        neighborhood_data = df[
                            df["neighborhoods_analysis_boundaries"]
                            == selected_neighborhood
                        ]
                        if len(neighborhood_data) > 0:
                            neighborhood_success_rate = neighborhood_data[
                                "success"
                            ].mean()
                            neighborhood_count = len(neighborhood_data)

                            st.markdown(
                                f"""
                            **📊 {selected_neighborhood} Statistics:**
                            - Historical Success Rate: **{neighborhood_success_rate:.1%}**
                            - Sample Size: **{neighborhood_count} businesses**
                            - Your Prediction: **{preview_score:.1%}**
                            """
                            )

                            if preview_score > neighborhood_success_rate:
                                st.success(
                                    f"🎯 Your business profile shows **{preview_score - neighborhood_success_rate:+.1%}** better odds than average!"
                                )
                            else:
                                st.info(
                                    f"📈 Neighborhood average is **{neighborhood_success_rate - preview_score:+.1%}** higher"
                                )
                    except:
                        pass
            else:
                st.markdown("### 📊 Prediction Insights")
                st.success("✅ Model loaded and making real-time predictions")
                st.info("📈 Predictions update automatically as you adjust inputs")
                st.warning(
                    "📋 Training data not loaded - limited historical context available"
                )

                # Show model confidence
                confidence_level = (
                    "High" if abs(preview_score - 0.5) > 0.2 else "Moderate"
                )
                st.markdown(
                    f"""
                **🤖 Model Confidence:** {confidence_level}
                
                **💡 Interpretation:**
                - Score > 70%: Strong location choice
                - Score 50-70%: Moderate risk, good planning needed  
                - Score < 50%: High risk, consider alternatives
                """
                )

        # Quick comparison section (always visible)
        st.markdown("### ⚡ Quick Comparisons")

        # Create comparison with other neighborhoods for same industry
        if data_loaded:
            try:
                # Get top 3 neighborhoods for this industry
                industry_data = df[df["business_industry"] == selected_industry]
                if len(industry_data) > 0:
                    top_neighborhoods = (
                        industry_data.groupby("neighborhoods_analysis_boundaries")[
                            "success"
                        ]
                        .agg(["mean", "count"])
                        .reset_index()
                    )
                    top_neighborhoods = top_neighborhoods[
                        top_neighborhoods["count"] >= 10
                    ]  # Significant sample
                    top_neighborhoods = top_neighborhoods.sort_values(
                        "mean", ascending=False
                    ).head(3)

                    if not top_neighborhoods.empty:
                        st.markdown(
                            f"**🏆 Top neighborhoods for {selected_industry}:**"
                        )
                        for _, row in top_neighborhoods.iterrows():
                            neighborhood = row["neighborhoods_analysis_boundaries"]
                            success_rate = row["mean"]

                            if neighborhood == selected_neighborhood:
                                st.success(
                                    f"✅ **{neighborhood}**: {success_rate:.1%} (Your Choice)"
                                )
                            else:
                                st.info(f"🏙️ **{neighborhood}**: {success_rate:.1%}")
            except:
                pass

        # Call-to-action for detailed analysis
        st.markdown("---")
        st.info(
            "💡 **Want more insights?** Click 'Get Comprehensive Analysis' in the sidebar for detailed neighborhood comparisons, risk factors, and strategic recommendations!"
        )


if __name__ == "__main__":
    main()
