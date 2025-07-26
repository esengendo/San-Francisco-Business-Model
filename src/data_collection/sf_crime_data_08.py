import os
import time
import logging
import pandas as pd
import sodapy
from datetime import datetime, timedelta
import sys

# Add project root to path if running directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src", "utils"))

# Import unified config - same functions as your _02 script
from config import setup_logging, setup_directories
from helper_functions_03 import save_to_parquet


def fetch_sf_crime_data(raw_data_dir, processed_dir, start_date, end_date):
    """
    Fetch crime data from SF Police Department.
    SFPD provides historical crime data going back many years.

    Parameters:
    -----------
    raw_data_dir : str
        Directory to save raw data
    processed_dir : str
        Directory to save processed data
    start_date : datetime
        Start date for data collection
    end_date : datetime
        End date for data collection

    Returns:
    --------
    pd.DataFrame
        DataFrame containing processed crime data
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.sf_crime_data_08")
    logger.info("Fetching SF Crime Data...")

    # Initialize Socrata client for SF Open Data
    client = sodapy.Socrata("data.sfgov.org", None)

    # Police incident reports dataset - has excellent historical coverage
    dataset_id = "wg3w-h783"  # Police incident reports

    # For 10+ years of data, we'll need to paginate our requests
    all_incidents = []
    offset = 0
    limit = 50000  # Maximum limit per request

    # Create directories if they don't exist
    os.makedirs(f"{raw_data_dir}/crime", exist_ok=True)
    os.makedirs(f"{processed_dir}/crime", exist_ok=True)

    # Fetch data in chunks to get the full 10+ year history
    while True:
        try:
            # Filter by date range
            incidents = client.get(
                dataset_id,
                limit=limit,
                offset=offset,
                where=f"incident_date >= '{start_date.strftime('%Y-%m-%d')}'",
            )

            if not incidents:
                break  # No more data

            incidents_df = pd.DataFrame.from_records(incidents)
            all_incidents.append(incidents_df)
            logger.info(
                f"  - Retrieved {len(incidents)} crime incidents (offset {offset})"
            )

            if len(incidents) < limit:
                break  # Last batch

            offset += limit
            time.sleep(1)  # Rate limiting

        except Exception as e:
            logger.error(f"  - Error fetching crime data: {e}")
            # If API fails, simply log error and break instead of generating synthetic data
            break

    # Combine all batches
    if all_incidents:
        crime_df = pd.concat(all_incidents, ignore_index=True)

        # Save raw data
        raw_file_path = f"{raw_data_dir}/crime/sf_crime_raw.parquet"
        crime_df.to_parquet(raw_file_path)
        logger.info(f"Saved raw crime data to {raw_file_path}")

        # Process data
        # Convert date columns
        crime_df["incident_date"] = pd.to_datetime(
            crime_df["incident_date"], errors="coerce"
        )

        # Extract coordinates
        if "point" in crime_df.columns:
            try:
                crime_df["latitude"] = crime_df["point"].apply(
                    lambda x: (
                        float(x.get("coordinates", [0, 0])[1])
                        if isinstance(x, dict)
                        else None
                    )
                )
                crime_df["longitude"] = crime_df["point"].apply(
                    lambda x: (
                        float(x.get("coordinates", [0, 0])[0])
                        if isinstance(x, dict)
                        else None
                    )
                )
            except Exception as e:
                logger.error(f"Error extracting coordinates: {e}")

        # Aggregate crime by type, neighborhood, and year for trend analysis
        if "incident_date" in crime_df.columns:
            crime_df["year"] = crime_df["incident_date"].dt.year

            # Group by year, neighborhood, and category
            if (
                "police_district" in crime_df.columns
                and "incident_category" in crime_df.columns
            ):
                crime_trends = (
                    crime_df.groupby(["year", "police_district", "incident_category"])
                    .size()
                    .reset_index(name="incident_count")
                )

                # Save crime trends
                trends_file_path = f"{processed_dir}/crime/crime_trends.parquet"
                crime_trends.to_parquet(trends_file_path)
                logger.info(f"Saved crime trends to {trends_file_path}")

        # Calculate crime rates by neighborhood
        if "police_district" in crime_df.columns:
            district_counts = crime_df["police_district"].value_counts().reset_index()
            district_counts.columns = ["police_district", "total_incidents"]

            # Save district crime counts
            district_file_path = f"{processed_dir}/crime/district_crime_counts.parquet"
            district_counts.to_parquet(district_file_path)
            logger.info(f"Saved district crime counts to {district_file_path}")

        # Save processed data
        processed_file_path = f"{processed_dir}/crime/sf_crime.parquet"
        crime_df.to_parquet(processed_file_path)
        logger.info(f"Saved processed crime data to {processed_file_path}")

        logger.info(f"Processed {len(crime_df)} crime incidents")
        return crime_df
    else:
        logger.error("Failed to retrieve crime data from API")
        return pd.DataFrame()


if __name__ == "__main__":
    # Use unified config - same functions as your _02 script
    logger = setup_logging()
    config = setup_directories()
    
    logger.info("Starting SF Crime data collection")
    logger.info(f"Base directory: {config['base_dir']}")
    
    # Create directories if they don't exist - same pattern as your _02 script
    os.makedirs(f"{config['raw_data_dir']}/crime", exist_ok=True)
    os.makedirs(f"{config['processed_dir']}/crime", exist_ok=True)

    # Execute the function
    try:
        crime_data = fetch_sf_crime_data(
            config["raw_data_dir"],
            config["processed_dir"],
            config["start_date"],
            config["end_date"],
        )

        if not crime_data.empty:
            print(f"Successfully retrieved crime data with {len(crime_data)} records")
            print(
                f"Data covers period from {config['start_date'].strftime('%Y-%m-%d')} to {config['end_date'].strftime('%Y-%m-%d')}"
            )
            print(f"Sample of data:")
            print(crime_data.head())
        else:
            print("No crime data retrieved. The API request likely failed.")
    except Exception as e:
        print(f"Error fetching crime data: {e}")
        crime_data = pd.DataFrame()