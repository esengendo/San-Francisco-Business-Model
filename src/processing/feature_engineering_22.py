import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import os
from datetime import datetime

def setup_logging():
    """Initialize logging configuration for feature engineering pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("SFBusinessModel")

def load_merged_data(input_path):
    """Load and validate the merged business dataset
    
    Args:
        input_path: Path to the merged business data file
        
    Returns:
        DataFrame containing validated business data
    """
    logger = setup_logging()
    logger.info("Loading merged business data for feature engineering")
    
    try:
        df = pd.read_parquet(input_path)
        initial_record_count = len(df)
        logger.info(f"Loaded {initial_record_count:,} business records")
        return df
    except Exception as e:
        logger.error(f"Failed to load merged data: {e}")
        raise

def engineer_temporal_features(df):
    """Create time-based features for business success prediction
    
    Features:
    - Start month/quarter (seasonality)
    - Economic cycle indicators
    - Time-based competition metrics
    """
    # Extract temporal components
    df["start_month"] = df["start_date"].dt.month
    df["start_quarter"] = df["start_date"].dt.quarter
    
    # Economic cycle features
    df["recession_start"] = df["start_date"].dt.year.isin([2008, 2020])
    df["post_recession_recovery"] = df["start_date"].dt.year.isin([2009, 2010, 2021])
    
    # Competition features
    df["businesses_started_same_month"] = df.groupby(
        [df["start_date"].dt.to_period("M"), "business_type"]
    )["business_type"].transform("count")
    
    return df

def engineer_spatial_features(df):
    """Create location-based features for business success prediction
    
    Features:
    - Business density metrics
    - Neighborhood characteristics
    - Geographic clustering scores
    """
    # Business density
    df["nearby_business_count"] = df.groupby("neighborhoods_analysis_boundaries")["business_type"].transform("count")
    df["business_type_density"] = df.groupby(["neighborhoods_analysis_boundaries", "business_type"])["business_type"].transform("count")
    df["business_density_ratio"] = df["business_type_density"] / df["nearby_business_count"]
    
    # Neighborhood success rates (with smoothing)
    neighborhood_success = df.groupby("neighborhoods_analysis_boundaries")["success"].agg(["mean", "count"]).reset_index()
    global_mean = df["success"].mean()
    smoothing_factor = 10
    
    neighborhood_success["smoothed_success_rate"] = (
        (neighborhood_success["mean"] * neighborhood_success["count"] + global_mean * smoothing_factor)
        / (neighborhood_success["count"] + smoothing_factor)
    )
    
    df = df.merge(
        neighborhood_success[["neighborhoods_analysis_boundaries", "smoothed_success_rate"]],
        on="neighborhoods_analysis_boundaries",
        how="left"
    )
    
    return df

def engineer_business_features(df):
    """Create business-specific features for success prediction
    
    Features:
    - Industry risk metrics
    - Competition indicators
    - Business type success patterns
    """
    # Industry success rates (with smoothing)
    business_type_success = df.groupby("business_type")["success"].agg(["mean", "count"]).reset_index()
    global_mean = df["success"].mean()
    smoothing_factor = 5
    
    business_type_success["smoothed_type_success_rate"] = (
        (business_type_success["mean"] * business_type_success["count"] + global_mean * smoothing_factor)
        / (business_type_success["count"] + smoothing_factor)
    )
    
    df = df.merge(
        business_type_success[["business_type", "smoothed_type_success_rate"]],
        on="business_type",
        how="left"
    )
    
    # Competition intensity
    df["similar_businesses_nearby"] = df.groupby(
        ["neighborhoods_analysis_boundaries", "business_type"]
    )["business_type"].transform("count")
    
    # Market saturation
    df["market_saturation"] = df["similar_businesses_nearby"] / df["nearby_business_count"]
    
    return df

def engineer_economic_features(df):
    """Create economic indicator features for success prediction
    
    Features:
    - Economic health metrics
    - Market conditions
    - Growth indicators
    """
    # Economic health score
    df["economic_health_score"] = (
        (df["sf_gdp_normalized"] * 0.4) +
        (1 - df["sf_unemployment_rate_normalized"] * 0.3) +
        (df["sf_house_price_index_normalized"] * 0.3)
    )
    
    # Market conditions
    df["favorable_market_conditions"] = (
        (df["economic_health_score"] > df["economic_health_score"].mean()) &
        (df["sf_unemployment_rate"] < df["sf_unemployment_rate"].mean()) &
        (df["overall_sentiment_mean"] > df["overall_sentiment_mean"].mean())
    ).astype(int)
    
    return df

def save_engineered_features(df, output_path):
    """Save the engineered feature dataset
    
    Args:
        df: DataFrame with engineered features
        output_path: Path to save the processed data
    """
    logger = setup_logging()
    
    try:
        df.to_parquet(output_path)
        logger.info(f"Saved engineered features to {output_path}")
        
        # Save feature summary
        feature_summary = {
            "total_features": len(df.columns),
            "numeric_features": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_features": len(df.select_dtypes(include=["object", "category"]).columns),
            "feature_list": df.columns.tolist(),
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_path = output_path.replace(".parquet", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(feature_summary, f, indent=2)
            
        logger.info(f"Saved feature summary to {summary_path}")
        
    except Exception as e:
        logger.error(f"Failed to save engineered features: {e}")
        raise

def main():
    """Execute feature engineering pipeline for business success prediction"""
    logger = setup_logging()
    logger.info("Starting feature engineering pipeline")
    
    # Set up paths
    base_dir = os.getenv("BASE_DIR", "/app/San_Francisco_Business_Model")
    processed_dir = f"{base_dir}/processed"
    
    input_path = f"{processed_dir}/merged_business_data.parquet"
    output_path = f"{processed_dir}/sf_business_success_modeling_data.parquet"
    
    # Load data
    df = load_merged_data(input_path)
    initial_shape = df.shape
    
    # Engineer features
    df = engineer_temporal_features(df)
    df = engineer_spatial_features(df)
    df = engineer_business_features(df)
    df = engineer_economic_features(df)
    
    # Log feature engineering results
    final_shape = df.shape
    logger.info(f"Initial features: {initial_shape[1]}")
    logger.info(f"Engineered features: {final_shape[1]}")
    logger.info(f"New features added: {final_shape[1] - initial_shape[1]}")
    
    # Save results
    save_engineered_features(df, output_path)
    logger.info("Feature engineering pipeline completed successfully")

if __name__ == "__main__":
    main() 