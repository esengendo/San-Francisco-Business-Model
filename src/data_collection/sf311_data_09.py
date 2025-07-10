import os
import time
import glob
import logging
import pandas as pd
import sodapy
from datetime import datetime, timedelta
import gc
import sys

# Add project root to path if running directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

# Import unified config - same functions as your _02 script
from config import setup_logging, setup_directories


def categorize_service(service_name):
    """Helper function to categorize 311 services into broader groups"""
    if pd.isna(service_name):
        return "Unknown"

    service = str(service_name).lower()

    if any(
        term in service for term in ["street", "road", "pothole", "sidewalk", "curb"]
    ):
        return "Streets & Sidewalks"
    elif any(term in service for term in ["graffiti", "illegal", "dumping", "litter"]):
        return "Graffiti & Illegal Dumping"
    elif any(term in service for term in ["tree", "park", "vegetation"]):
        return "Trees & Parks"
    elif any(term in service for term in ["noise", "music", "construction"]):
        return "Noise Complaints"
    elif any(term in service for term in ["homeless", "encampment"]):
        return "Homeless Concerns"
    elif any(term in service for term in ["parking", "vehicle", "meter"]):
        return "Parking & Vehicles"
    elif any(term in service for term in ["water", "sewer", "drain"]):
        return "Water & Sewer"
    elif any(term in service for term in ["sign", "signal", "light"]):
        return "Signs & Signals"
    else:
        return "Other"


def cleanup_redundant_batches(processed_dir, logger):
    """
    Clean up redundant or corrupted batch files from previous runs.
    """
    logger.info("Starting cleanup of redundant batch files...")

    # Find all batch files
    batch_pattern = f"{processed_dir}/sf311_batch_*.parquet"
    batch_files = glob.glob(batch_pattern)

    if not batch_files:
        logger.info("No batch files found to clean up")
        return

    # Group batch files by year
    year_batches = {}
    corrupted_files = []

    for batch_file in batch_files:
        try:
            # Extract date from filename
            filename = os.path.basename(batch_file)
            # Format: sf311_batch_YYYYMMDD_YYYYMMDD.parquet
            date_part = filename.replace("sf311_batch_", "").replace(".parquet", "")
            start_date_str = date_part.split("_")[0]
            year = start_date_str[:4]

            if year not in year_batches:
                year_batches[year] = []
            year_batches[year].append(batch_file)

        except Exception as e:
            logger.warning(f"Could not parse batch file {batch_file}: {e}")
            corrupted_files.append(batch_file)

    # Remove corrupted files
    for corrupted_file in corrupted_files:
        try:
            os.remove(corrupted_file)
            logger.info(f"Removed corrupted batch file: {corrupted_file}")
        except Exception as e:
            logger.error(f"Error removing corrupted file {corrupted_file}: {e}")

    # Check for duplicate batches within each year
    duplicates_removed = 0
    for year, files in year_batches.items():
        if len(files) > 1:
            # Keep the most recent file, remove older ones
            files_with_mtime = [(f, os.path.getmtime(f)) for f in files]
            files_with_mtime.sort(
                key=lambda x: x[1], reverse=True
            )  # Sort by modification time, newest first

            # Keep the first (newest) file, remove the rest
            for file_path, _ in files_with_mtime[1:]:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed duplicate batch file: {file_path}")
                    duplicates_removed += 1
                except Exception as e:
                    logger.error(f"Error removing duplicate file {file_path}: {e}")

    if duplicates_removed > 0:
        logger.info(f"Cleaned up {duplicates_removed} duplicate batch files")
    else:
        logger.info("No duplicate batch files found")


def validate_batch_file(batch_df, expected_start, expected_end, logger):
    """
    Validate that a batch file contains expected data.
    """
    try:
        # Check if dataframe is empty
        if batch_df.empty:
            logger.warning("Batch file is empty")
            return False

        # Check if required columns exist
        required_columns = ["service_request_id", "requested_datetime"]
        missing_columns = [
            col for col in required_columns if col not in batch_df.columns
        ]
        if missing_columns:
            logger.warning(f"Batch file missing required columns: {missing_columns}")
            return False

        # Check date range if requested_datetime exists
        if "requested_datetime" in batch_df.columns:
            try:
                batch_df["requested_datetime"] = pd.to_datetime(
                    batch_df["requested_datetime"], errors="coerce"
                )
                min_date = batch_df["requested_datetime"].min()
                max_date = batch_df["requested_datetime"].max()

                # Allow some flexibility in date ranges (data might not perfectly align with year boundaries)
                if min_date < expected_start or max_date > expected_end:
                    logger.info(
                        f"Batch date range ({min_date} to {max_date}) extends beyond expected range ({expected_start} to {expected_end}) - this is usually OK"
                    )

            except Exception as e:
                logger.warning(f"Could not validate date range: {e}")

        return True

    except Exception as e:
        logger.warning(f"Error validating batch file: {e}")
        return False


def cleanup_after_combination(
    batch_files, combined_file_path, logger, keep_individual_batches=True
):
    """
    Optional cleanup after successful combination of all batches.

    Parameters:
    -----------
    keep_individual_batches : bool, default=True
        Whether to keep individual batch files after creating combined file
        Set to False only if you're confident you won't need individual years
    """
    if not keep_individual_batches:
        logger.info(
            "Cleaning up individual batch files after successful combination..."
        )

        # Verify the combined file exists and is valid
        if not os.path.exists(combined_file_path):
            logger.error("Combined file does not exist, keeping individual batches")
            return

        try:
            # Quick validation of combined file
            combined_df = pd.read_parquet(combined_file_path)
            if combined_df.empty:
                logger.error("Combined file is empty, keeping individual batches")
                return

            logger.info(f"Combined file validated with {len(combined_df)} records")

            # Remove individual batch files
            removed_count = 0
            for batch_file in batch_files:
                if os.path.exists(batch_file):
                    try:
                        os.remove(batch_file)
                        removed_count += 1
                        logger.info(
                            f"Removed individual batch: {os.path.basename(batch_file)}"
                        )
                    except Exception as e:
                        logger.error(f"Error removing batch file {batch_file}: {e}")

            logger.info(f"Cleaned up {removed_count} individual batch files")

        except Exception as e:
            logger.error(
                f"Error validating combined file, keeping individual batches: {e}"
            )
    else:
        logger.info("Keeping individual batch files for future incremental updates")


def cleanup_old_combined_files(processed_dir, logger, keep_latest=2):
    """
    Clean up old combined data files, keeping only the most recent ones.

    Parameters:
    -----------
    keep_latest : int, default=2
        Number of most recent combined files to keep
    """
    logger.info(f"Cleaning up old combined files, keeping latest {keep_latest}...")

    # Find all combined files
    combined_pattern = f"{processed_dir}/sf311_all_data*.parquet"
    combined_files = glob.glob(combined_pattern)

    if len(combined_files) <= keep_latest:
        logger.info(f"Found {len(combined_files)} combined files, no cleanup needed")
        return

    # Sort by modification time, newest first
    files_with_mtime = [(f, os.path.getmtime(f)) for f in combined_files]
    files_with_mtime.sort(key=lambda x: x[1], reverse=True)

    # Remove old files
    files_to_remove = files_with_mtime[keep_latest:]
    for file_path, _ in files_to_remove:
        try:
            os.remove(file_path)
            logger.info(f"Removed old combined file: {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Error removing old combined file {file_path}: {e}")


def process_batch(batch_df):
    """
    Process a batch of SF311 data with comprehensive transformations.
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.sf311_data_09")
    logger.info(f"Processing batch with {len(batch_df)} records")

    # Process date columns
    date_columns = ["requested_datetime", "closed_date", "updated_datetime"]
    for col in date_columns:
        if col in batch_df.columns:
            batch_df[col] = pd.to_datetime(batch_df[col], errors="coerce")
            logger.info(f"Converted {col} to datetime")

    # Add time dimensions
    if "requested_datetime" in batch_df.columns:
        batch_df["year"] = batch_df["requested_datetime"].dt.year
        batch_df["month"] = batch_df["requested_datetime"].dt.month
        batch_df["day_of_week"] = batch_df["requested_datetime"].dt.day_name()
        batch_df["hour"] = batch_df["requested_datetime"].dt.hour
        logger.info("Added time dimensions")

    # Calculate resolution time
    if "requested_datetime" in batch_df.columns and "closed_date" in batch_df.columns:
        batch_df["resolution_days"] = (
            batch_df["closed_date"] - batch_df["requested_datetime"]
        ).dt.total_seconds() / (24 * 60 * 60)
        logger.info("Calculated resolution time in days")

        # Handle potential negative resolution times (data errors)
        negative_resolution = (batch_df["resolution_days"] < 0).sum()
        if negative_resolution > 0:
            logger.warning(
                f"Found {negative_resolution} cases with negative resolution time - setting to NaN"
            )
            batch_df.loc[batch_df["resolution_days"] < 0, "resolution_days"] = float(
                "nan"
            )

    # Clean and standardize text fields
    text_columns = [
        "service_name",
        "service_subtype",
        "agency_responsible",
        "status_description",
    ]
    for col in text_columns:
        if col in batch_df.columns:
            batch_df[col] = batch_df[col].astype(str).str.strip()

    # Convert coordinates to numeric
    if "lat" in batch_df.columns:
        batch_df["lat"] = pd.to_numeric(batch_df["lat"], errors="coerce")
    if "long" in batch_df.columns:
        batch_df["long"] = pd.to_numeric(batch_df["long"], errors="coerce")

    # Create service category groupings
    if "service_name" in batch_df.columns:
        batch_df["service_category"] = batch_df["service_name"].apply(
            categorize_service
        )

    return batch_df


def fetch_batch(
    client, dataset_id, batch_start, batch_end, max_records=50000, select_columns=None
):
    """
    Fetch a batch of SF311 data with specified columns and date range.
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.sf311_data_09")
    logger.info(
        f"Fetching data from {batch_start.strftime('%Y-%m-%d')} to {batch_end.strftime('%Y-%m-%d')} (limited to {max_records} records)"
    )

    all_cases = []
    offset = 0
    limit = min(1000, max_records)
    total_records = 0

    where_clause = f"requested_datetime >= '{batch_start.strftime('%Y-%m-%d')}' AND requested_datetime <= '{batch_end.strftime('%Y-%m-%d')}'"

    while total_records < max_records:
        try:
            params = {"limit": limit, "offset": offset, "where": where_clause}

            if select_columns:
                params["select"] = ",".join(select_columns)

            cases = client.get(dataset_id, **params)

            if not cases:
                break

            cases_df = pd.DataFrame.from_records(cases)
            all_cases.append(cases_df)

            records_fetched = len(cases)
            total_records += records_fetched

            logger.info(
                f"  - Retrieved {records_fetched} 311 cases (total: {total_records}/{max_records})"
            )

            if records_fetched < limit:
                break

            offset += limit
            time.sleep(1)

            if total_records >= max_records:
                logger.info(f"  - Reached maximum record limit of {max_records}")
                break

        except Exception as e:
            logger.error(f"  - Error fetching 311 data: {e}")
            if total_records > 0:
                logger.info(
                    f"  - Proceeding with {total_records} records retrieved so far"
                )
                break
            else:
                return None

    if all_cases:
        try:
            sf311_df = pd.concat(all_cases, ignore_index=True)
            logger.info(f"Combined {len(sf311_df)} records")
            return sf311_df
        except Exception as e:
            logger.error(f"Error combining data chunks: {e}")
            return None
    else:
        logger.error(
            f"No data retrieved for period {batch_start.strftime('%Y-%m-%d')} to {batch_end.strftime('%Y-%m-%d')}"
        )
        return None


def fetch_sf311_data(
    start_date,
    raw_data_dir,
    processed_dir,
    app_token=None,
    max_records_per_batch=50000,
    select_columns=None,
    cleanup_old_batches=True,
):
    """
    Enhanced function to fetch 311 service requests data from SF311 API.
    Uses year-based batching and caching with automatic cleanup processes.

    Parameters:
    -----------
    start_date : datetime
        Start date for data collection
    raw_data_dir : str
        Directory to save raw data
    processed_dir : str
        Directory to save processed data
    app_token : str, optional
        Socrata app token for higher rate limits
    max_records_per_batch : int, default=50000
        Maximum records to fetch per batch
    select_columns : list, optional
        Specific columns to fetch, defaults to useful subset
    cleanup_old_batches : bool, default=True
        Whether to clean up old/redundant batch files after successful processing

    Returns:
    --------
    pd.DataFrame
        Combined SF311 data from all batches
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.sf311_data_09")
    
    # Default to a useful subset of columns if none specified
    if select_columns is None:
        select_columns = [
            "service_request_id",
            "requested_datetime",
            "closed_date",
            "updated_datetime",
            "status_description",
            "status_notes",
            "agency_responsible",
            "service_name",
            "service_subtype",
            "service_details",
            "address",
            "supervisor_district",
            "neighborhoods_sffind_boundaries",
            "police_district",
            "lat",
            "long",
            "source",
        ]

    logger.info(
        f"Fetching SF311 data with {len(select_columns)} columns: {', '.join(select_columns)}"
    )

    # Initialize Socrata client for SF Open Data
    client = sodapy.Socrata("data.sfgov.org", app_token)

    # 311 Cases dataset
    dataset_id = "vw6y-z8j6"  # SF 311 Cases

    # Create directories if they don't exist
    os.makedirs(f"{raw_data_dir}", exist_ok=True)
    os.makedirs(f"{processed_dir}", exist_ok=True)

    # Clean up old batch files if requested
    if cleanup_old_batches:
        cleanup_redundant_batches(processed_dir, logger)

    # Get current date for end range
    end_date = datetime.now()

    # Calculate batches by year
    current_year = end_date.year
    start_year = start_date.year

    # Create a list of batch periods (start_year to current_year in chunks)
    batch_periods = []
    for year in range(start_year, current_year + 1):
        batch_start = max(datetime(year, 1, 1), start_date)
        batch_end = min(datetime(year, 12, 31), end_date)
        batch_periods.append((batch_start, batch_end))

    logger.info(
        f"Created {len(batch_periods)} batch periods from {start_year} to {current_year}"
    )

    # Process each batch separately
    all_batches = []
    processed_batch_files = []

    for i, (batch_start, batch_end) in enumerate(batch_periods):
        logger.info(
            f"Processing batch {i+1}/{len(batch_periods)}: {batch_start.strftime('%Y-%m-%d')} to {batch_end.strftime('%Y-%m-%d')}"
        )

        batch_filename = f"sf311_batch_{batch_start.strftime('%Y%m%d')}_{batch_end.strftime('%Y%m%d')}.parquet"
        batch_file_path = f"{processed_dir}/{batch_filename}"
        processed_batch_files.append(batch_file_path)

        # Check if batch already processed
        if os.path.exists(batch_file_path):
            logger.info(
                f"Batch {i+1} already processed, loading from disk: {batch_file_path}"
            )
            try:
                batch_df = pd.read_parquet(batch_file_path)

                # Validate the batch file
                if validate_batch_file(batch_df, batch_start, batch_end, logger):
                    all_batches.append(batch_df)
                    continue
                else:
                    logger.warning(
                        f"Batch file {batch_file_path} failed validation, re-fetching..."
                    )
                    # Remove invalid file
                    os.remove(batch_file_path)

            except Exception as e:
                logger.warning(f"Error loading existing batch {batch_file_path}: {e}")
                # Remove corrupted file
                try:
                    os.remove(batch_file_path)
                except:
                    pass

        # Fetch this batch with selected columns
        batch_df = fetch_batch(
            client,
            dataset_id,
            batch_start,
            batch_end,
            max_records=max_records_per_batch,
            select_columns=select_columns,
        )

        if batch_df is not None and not batch_df.empty:
            # Process the batch
            batch_df = process_batch(batch_df)

            # Save the batch
            try:
                batch_df.to_parquet(batch_file_path)
                logger.info(
                    f"Saved batch with {len(batch_df)} records to {batch_file_path}"
                )
            except Exception as e:
                logger.error(f"Error saving batch {batch_file_path}: {e}")

            all_batches.append(batch_df)

        # Clean up memory
        gc.collect()

    # Combine all batches
    if all_batches:
        combined_df = pd.concat(all_batches, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} records from all batches")

        # Save combined data
        combined_file_path = f"{processed_dir}/sf311_all_data.parquet"
        try:
            combined_df.to_parquet(combined_file_path)
            logger.info(f"Saved combined data to {combined_file_path}")

            # Optional: Clean up individual batch files after successful combination
            if cleanup_old_batches:
                cleanup_after_combination(
                    processed_batch_files, combined_file_path, logger
                )

        except Exception as e:
            logger.error(f"Error saving combined data: {e}")

        return combined_df
    else:
        logger.error("No data batches were processed successfully")
        return pd.DataFrame()


def main_sf311_fetch(
    raw_data_dir,
    processed_dir,
    start_date,
    cleanup_old_batches=True,
    cleanup_after_combination=False,
):
    """
    Main function to fetch SF311 data with comprehensive cleanup options

    Parameters:
    -----------
    raw_data_dir : str
        Directory to save raw data
    processed_dir : str
        Directory to save processed data
    start_date : datetime
        Start date for data collection
    cleanup_old_batches : bool, default=True
        Clean up redundant/corrupted batch files before processing
    cleanup_after_combination : bool, default=False
        Remove individual batch files after successful combination (not recommended)

    Returns:
    --------
    pd.DataFrame
        Combined SF311 data
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.sf311_data_09")
    
    # Create subdirectories for SF311 data
    sf311_raw_dir = f"{raw_data_dir}/sf311"
    sf311_processed_dir = f"{processed_dir}/sf311"

    # Call the enhanced function with cleanup
    enhanced_data = fetch_sf311_data(
        start_date=start_date,
        raw_data_dir=sf311_raw_dir,
        processed_dir=sf311_processed_dir,
        app_token=None,
        max_records_per_batch=50000,
        select_columns=None,
        cleanup_old_batches=cleanup_old_batches,
    )

    # Optional: Clean up old combined files
    cleanup_old_combined_files(sf311_processed_dir, logger, keep_latest=2)

    return enhanced_data


if __name__ == "__main__":
    # Use unified config - same functions as your _02 script
    logger = setup_logging()
    config = setup_directories()
    
    logger.info("Starting SF311 Service Request data collection")
    logger.info(f"Base directory: {config['base_dir']}")
    
    # Create directories if they don't exist - same pattern as your _02 script
    os.makedirs(f"{config['raw_data_dir']}/sf311", exist_ok=True)
    os.makedirs(f"{config['processed_dir']}/sf311", exist_ok=True)

    # Execute the main function with cleanup
    try:
        sf311_data = main_sf311_fetch(
            config["raw_data_dir"],
            config["processed_dir"],
            config["start_date"],
            cleanup_old_batches=True,  # Clean up redundant batches
            cleanup_after_combination=False,  # Keep individual batch files
        )

        if not sf311_data.empty:
            print(f"Successfully retrieved SF311 data with {len(sf311_data)} records")
            print(
                f"Data covers period from {config['start_date'].strftime('%Y-%m-%d')} to {config['end_date'].strftime('%Y-%m-%d')}"
            )

            # Show summary statistics
            print(f"\nSF311 Data Summary:")
            print(
                f"- Date range: {sf311_data['requested_datetime'].min()} to {sf311_data['requested_datetime'].max()}"
            )
            print(f"- Total service requests: {len(sf311_data)}")

            if "service_category" in sf311_data.columns:
                print(f"\nTop service categories:")
                print(sf311_data["service_category"].value_counts().head(10))

            print(f"\nSample of data:")
            print(sf311_data.head())

        else:
            print("No SF311 data retrieved. API request likely failed.")

    except Exception as e:
        print(f"Error fetching SF311 data: {e}")
        sf311_data = pd.DataFrame()