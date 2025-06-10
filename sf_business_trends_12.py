import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging():
    """Initialize logging for business trend analysis pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("SFBusinessTrends")

def load_business_data(input_dir):
    """Load integrated business data for trend analysis
    
    Combines data from:
    - Business Registrations
    - License Applications
    - Economic Indicators
    - Neighborhood Demographics
    
    Args:
        input_dir: Directory containing processed data files
        
    Returns:
        Dictionary containing business DataFrames
    """
    logger = setup_logging()
    
    try:
        # Load business registrations
        registrations_df = pd.read_parquet(f"{input_dir}/business/registrations.parquet")
        logger.info(f"Loaded business registrations: {len(registrations_df):,} records")
        
        # Load license data
        licenses_df = pd.read_parquet(f"{input_dir}/business/licenses.parquet")
        logger.info(f"Loaded business licenses: {len(licenses_df):,} records")
        
        # Load economic indicators
        economic_df = pd.read_parquet(f"{input_dir}/economic/indicators.parquet")
        logger.info(f"Loaded economic indicators: {len(economic_df):,} records")
        
        # Load neighborhood data
        neighborhood_df = pd.read_parquet(f"{input_dir}/demographic/neighborhoods.parquet")
        logger.info(f"Loaded neighborhood data: {len(neighborhood_df):,} records")
        
    except FileNotFoundError as e:
        logger.error(f"Required source file not found: {e}")
        raise
    
    return {
        "registrations": registrations_df,
        "licenses": licenses_df,
        "economic": economic_df,
        "neighborhood": neighborhood_df
    }

def analyze_business_trends(data_dict):
    """Calculate key business performance metrics and trends
    
    Generates insights on:
    - Business Formation Rates
    - Industry Growth Patterns
    - Geographic Distribution
    - Economic Impact Indicators
    
    Args:
        data_dict: Dictionary containing business DataFrames
        
    Returns:
        DataFrame with trend analysis results
    """
    # Standardize date columns
    date_columns = {
        "registrations": "registration_date",
        "licenses": "application_date",
        "economic": "date"
    }
    
    for key, df in data_dict.items():
        if key in date_columns and date_columns[key] in df.columns:
            df["date"] = pd.to_datetime(df[date_columns[key]])
    
    # Calculate business trends
    trend_metrics = pd.DataFrame()
    
    # Analyze business formation
    formation_metrics = data_dict["registrations"].groupby("date").agg({
        "business_id": "count",
        "business_type": "value_counts",
        "neighborhood": "value_counts"
    }).rename(columns={
        "business_id": "new_businesses"
    })
    
    # Analyze license activity
    license_metrics = data_dict["licenses"].groupby("date").agg({
        "license_id": "count",
        "license_type": "value_counts"
    }).rename(columns={
        "license_id": "new_licenses"
    })
    
    # Calculate economic indicators
    economic_metrics = data_dict["economic"].groupby("date").agg({
        "employment_rate": "mean",
        "median_income": "mean",
        "commercial_rent": "mean"
    })
    
    # Combine metrics
    trend_metrics = pd.merge(
        formation_metrics,
        license_metrics,
        on="date",
        how="outer"
    )
    
    trend_metrics = pd.merge(
        trend_metrics,
        economic_metrics,
        on="date",
        how="outer"
    )
    
    # Calculate growth rates
    trend_metrics["business_growth"] = trend_metrics["new_businesses"].pct_change()
    trend_metrics["license_growth"] = trend_metrics["new_licenses"].pct_change()
    
    return trend_metrics

def generate_trend_visualizations(metrics, output_dir):
    """Create business trend visualizations and reports
    
    Generates:
    - Business Formation Charts
    - Industry Distribution Plots
    - Geographic Heat Maps
    - Economic Correlation Analysis
    
    Args:
        metrics: DataFrame containing trend analysis results
        output_dir: Directory for saving outputs
    """
    logger = setup_logging()
    
    # Set visualization style
    plt.style.use("seaborn")
    
    # Business Formation Trends
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics[["new_businesses", "new_licenses"]].plot(ax=ax)
    ax.set_title("Business Formation Trends")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/business_formation_trends.png")
    plt.close()
    
    # Growth Rate Analysis
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics[["business_growth", "license_growth"]].plot(ax=ax)
    ax.set_title("Business Growth Rates")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth Rate (%)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/growth_rate_analysis.png")
    plt.close()
    
    # Economic Correlation
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        metrics[["new_businesses", "employment_rate", "median_income", "commercial_rent"]].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    ax.set_title("Economic Correlation Analysis")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/economic_correlation.png")
    plt.close()
    
    logger.info("Generated trend visualizations")

def save_trend_analysis(metrics, output_dir):
    """Save trend analysis results and generate summary report
    
    Args:
        metrics: DataFrame containing trend analysis results
        output_dir: Directory for saving outputs
    """
    logger = setup_logging()
    
    # Save processed metrics
    metrics_path = f"{output_dir}/business_trends.parquet"
    metrics.to_parquet(metrics_path)
    logger.info(f"Saved trend metrics to {metrics_path}")
    
    # Generate summary report
    latest_month = metrics.tail(30)
    summary = {
        "date_range": {
            "start": metrics.index.min().strftime("%Y-%m-%d"),
            "end": metrics.index.max().strftime("%Y-%m-%d")
        },
        "total_new_businesses": int(metrics["new_businesses"].sum()),
        "avg_monthly_growth": f"{metrics['business_growth'].mean()*100:.1f}%",
        "economic_correlations": {
            "employment": f"{metrics['new_businesses'].corr(metrics['employment_rate']):.2f}",
            "income": f"{metrics['new_businesses'].corr(metrics['median_income']):.2f}",
            "rent": f"{metrics['new_businesses'].corr(metrics['commercial_rent']):.2f}"
        },
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = f"{output_dir}/trend_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved trend analysis summary to {summary_path}")
    return summary

def main():
    """Execute business trend analysis pipeline"""
    base_dir = os.getenv("BASE_DIR", "/app/San_Francisco_Business_Model")
    input_dir = f"{base_dir}/processed"
    output_dir = f"{base_dir}/analysis/trends"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load and analyze business data
        raw_data = load_business_data(input_dir)
        trend_metrics = analyze_business_trends(raw_data)
        generate_trend_visualizations(trend_metrics, output_dir)
        summary = save_trend_analysis(trend_metrics, output_dir)
        
        # Display results
        print("\nBusiness Trend Analysis Complete")
        print("=" * 50)
        print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"Total New Businesses: {summary['total_new_businesses']:,}")
        print(f"Average Monthly Growth: {summary['avg_monthly_growth']}")
        print("\nEconomic Correlations:")
        print(f"  - Employment Rate: {summary['economic_correlations']['employment']}")
        print(f"  - Median Income: {summary['economic_correlations']['income']}")
        print(f"  - Commercial Rent: {summary['economic_correlations']['rent']}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during trend analysis: {e}")
        raise

if __name__ == "__main__":
    main() 