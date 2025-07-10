import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import json
import requests

def setup_logging():
    """Initialize logging for business data collection pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("SFBusinessData")

def fetch_business_data(start_date=None):
    """Collect business registration data from SF Open Data
    
    Retrieves data on:
    - Active Business Registrations
    - Business Types and Categories
    - Location Information
    - Registration Status
    
    Args:
        start_date: Beginning date for data collection (defaults to 1 year ago)
        
    Returns:
        DataFrame containing business registration data
    """
    logger = setup_logging()
    
    if start_date is None:
        start_date = (datetime.now() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    
    # SF Business Data API endpoint
    url = "https://data.sfgov.org/resource/g8m3-pdis.json"
    
    query = f"""
    SELECT business_account_number,
           ownership_name,
           dba_name,
           street_address,
           city,
           state,
           source_zipcode,
           business_start_date,
           business_end_date,
           location_start_date,
           location_end_date,
           business_location,
           business_corridor,
           neighborhoods_analysis_boundaries
    WHERE business_start_date >= '{start_date}'
    """
    
    try:
        logger.info(f"Fetching business data from {start_date}")
        response = requests.get(url, params={"$query": query})
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data)
        logger.info(f"Retrieved {len(df):,} business records")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching business data: {e}")
        return pd.DataFrame()

def process_business_data(df):
    """Transform business data into analysis-ready format
    
    Creates metrics for:
    - Business Demographics
    - Geographic Distribution
    - Industry Composition
    - Temporal Patterns
    
    Args:
        df: Raw business registration DataFrame
        
    Returns:
        DataFrame with processed business metrics
    """
    if df.empty:
        return df
    
    # Standardize dates
    date_columns = [
        "business_start_date",
        "business_end_date",
        "location_start_date",
        "location_end_date"
    ]
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Calculate business metrics
    business_metrics = pd.DataFrame()
    
    # Analyze business patterns
    business_counts = df.groupby("business_start_date").size()
    
    # Calculate neighborhood metrics
    if "neighborhoods_analysis_boundaries" in df.columns:
        neighborhood_counts = df.groupby("neighborhoods_analysis_boundaries").size()
        
    # Calculate corridor metrics
    if "business_corridor" in df.columns:
        corridor_counts = df.groupby("business_corridor").size()
    
    # Combine metrics
    business_metrics["new_businesses"] = business_counts
    business_metrics["neighborhood_distribution"] = neighborhood_counts
    business_metrics["corridor_distribution"] = corridor_counts
    
    return business_metrics

def save_business_data(df, metrics, output_dir):
    """Save processed business data and generate summary report
    
    Args:
        df: Raw business registration DataFrame
        metrics: Processed business metrics DataFrame
        output_dir: Directory for saving outputs
    """
    logger = setup_logging()
    
    # Save raw data
    raw_path = f"{output_dir}/business_registrations.parquet"
    df.to_parquet(raw_path)
    logger.info(f"Saved raw business data to {raw_path}")
    
    # Save metrics
    metrics_path = f"{output_dir}/business_metrics.parquet"
    metrics.to_parquet(metrics_path)
    logger.info(f"Saved business metrics to {metrics_path}")
    
    # Generate summary report
    latest_month = metrics.tail(30)
    summary = {
        "date_range": {
            "start": metrics.index.min().strftime("%Y-%m-%d"),
            "end": metrics.index.max().strftime("%Y-%m-%d")
        },
        "total_businesses": int(metrics["new_businesses"].sum()),
        "top_neighborhoods": metrics["neighborhood_distribution"].nlargest(5).to_dict(),
        "top_corridors": metrics["corridor_distribution"].nlargest(5).to_dict(),
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = f"{output_dir}/business_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved business summary to {summary_path}")
    return summary

def main():
    """Execute business data collection and analysis pipeline"""
    base_dir = os.getenv("BASE_DIR", "/app/San_Francisco_Business_Model")
    output_dir = f"{base_dir}/processed/business"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Collect and process business data
        raw_data = fetch_business_data()
        business_metrics = process_business_data(raw_data)
        summary = save_business_data(raw_data, business_metrics, output_dir)
        
        # Display results
        print("\nBusiness Data Analysis Complete")
        print("=" * 50)
        print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"Total Businesses: {summary['total_businesses']:,}")
        print("\nTop 5 Neighborhoods:")
        for hood, count in summary["top_neighborhoods"].items():
            print(f"  - {hood}: {count:,} businesses")
        print("\nTop 5 Business Corridors:")
        for corridor, count in summary["top_corridors"].items():
            print(f"  - {corridor}: {count:,} businesses")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during business data analysis: {e}")
        raise

if __name__ == "__main__":
    main() 