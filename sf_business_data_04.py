import os
import logging
import pandas as pd
import sodapy

from helper_functions_03 import save_to_parquet

# Setup logging
logger = logging.getLogger(__name__)


def fetch_sf_business_data(raw_data_dir, processed_dir):
    """Fetch Registered Business Locations from DataSF"""
    logger.info("Fetching SF Business Data...")
    # Initialize Socrata client for SF Open Data
    client = sodapy.Socrata("data.sfgov.org", None)
    # Fetch the main business registration dataset
    try:
        results = client.get("g8m3-pdis", limit=500000)
        businesses_df = pd.DataFrame.from_records(results)
        logger.info(
            f"Retrieved {len(businesses_df)} records from current business registry"
        )
        # Save raw data
        save_to_parquet(
            businesses_df, f"{raw_data_dir}/sf_business", "registered_businesses_raw"
        )
    except Exception as e:
        logger.error(f"Error fetching current business data: {e}")
        businesses_df = pd.DataFrame()

    # Process the main business data
    if not businesses_df.empty:
        # Use the full dataset without date filtering
        target_businesses = businesses_df
        # Keep only the specified columns
        columns_to_keep = [
            "uniqueid",
            "ttxid",
            "certificate_number",
            "ownership_name",
            "dba_name",
            "full_business_address",
            "city",
            "state",
            "business_zip",
            "dba_start_date",
            "dba_end_date",
            "location_start_date",
            "location_end_date",
            "parking_tax",
            "transient_occupancy_tax",
            "location",
            "administratively_closed",
            "naic_code",
            "naics_code_description",
            "supervisor_district",
            "neighborhoods_analysis_boundaries",
        ]
        # Filter columns (keep only columns that exist in the dataframe)
        existing_columns = [
            col for col in columns_to_keep if col in target_businesses.columns
        ]
        target_businesses = target_businesses[existing_columns]
        # Save processed data
        save_to_parquet(
            target_businesses, f"{processed_dir}/sf_business", "registered_businesses"
        )
        logger.info(
            f"Processed {len(target_businesses)} businesses within target time range"
        )
        return target_businesses
    else:
        logger.error("Failed to retrieve main business dataset")
        return pd.DataFrame()


if __name__ == "__main__":
    # SF Business Registration Data API - Execution
    from logging_config_setup_02 import setup_logging, setup_directories

    logger = setup_logging()
    config = setup_directories()

    # Execute the function
    sf_business_df = fetch_sf_business_data(
        config["raw_data_dir"], config["processed_dir"]
    )
    print(f"Fetched {len(sf_business_df)} business records")
