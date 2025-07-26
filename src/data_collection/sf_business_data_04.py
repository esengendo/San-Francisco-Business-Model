import os
import logging
import pandas as pd
import sodapy
import sys

# Add project root to path if running directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'src', 'utils'))

# Import unified config - same functions as your _02 script
from config import setup_logging, setup_directories
from helper_functions_03 import save_to_parquet


def fetch_sf_business_data(raw_data_dir, processed_dir):
    """Fetch Registered Business Locations from DataSF"""
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.sf_business_data_04")
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
    # Use unified config - same functions as your _02 script
    logger = setup_logging()
    config = setup_directories()

    logger.info("Starting SF Business data collection")
    logger.info(f"Base directory: {config['base_dir']}")

    # Create directories if they don't exist - same pattern as your _02 script
    os.makedirs(f"{config['raw_data_dir']}/sf_business", exist_ok=True)
    os.makedirs(f"{config['processed_dir']}/sf_business", exist_ok=True)

    # Execute the function
    sf_business_df = fetch_sf_business_data(
        config["raw_data_dir"], config["processed_dir"]
    )
    print(f"Fetched {len(sf_business_df)} business records")
