"""
SF Planning Department Data Fetcher - Enhanced Version
Enhanced data fetching and processing for SF Planning Department APIs with robust error handling,
user-friendly column names, and comprehensive data processing capabilities.
"""
import os
import pandas as pd
import sodapy
import logging
from datetime import datetime, timedelta
from helper_functions_03 import save_to_parquet

# Setup logging
logger = logging.getLogger(__name__)


def fix_duplicate_columns(df):
    """Fix duplicate column names by adding numeric suffixes"""
    if len(df.columns) != len(set(df.columns)):
        # Find and rename duplicates manually
        new_cols = []
        seen = {}
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols
    return df


def setup_logging():
    """Set up logging configuration"""
    logger = logging.getLogger("sf_planning")
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sf_planning_data.log'),
                logging.StreamHandler()
            ]
        )
    return logger


def categorize_status(status):
    """Helper function to categorize permit statuses into broader groups"""
    if pd.isna(status):
        return "Unknown"

    status = str(status).lower()
    if "approved" in status:
        return "Approved"
    elif "filed" in status or "submitted" in status:
        return "In Process"
    elif "issued" in status:
        return "Issued"
    elif "cancelled" in status or "withdrawn" in status:
        return "Cancelled"
    elif "denied" in status or "disapproved" in status:
        return "Denied"
    elif "completed" in status or "closed" in status:
        return "Completed"
    else:
        return "Other"


def categorize_land_use(land_use):
    """Helper function to categorize land use into broader groups"""
    if pd.isna(land_use):
        return "Unknown"

    land_use = str(land_use).lower()

    if "residential" in land_use:
        if "mixed" in land_use:
            return "Mixed Residential"
        elif "multi" in land_use or "apartment" in land_use:
            return "Multi-Family Residential"
        elif "single" in land_use:
            return "Single-Family Residential"
        else:
            return "Residential"
    elif "commercial" in land_use:
        return "Commercial"
    elif "industrial" in land_use or "production" in land_use:
        return "Industrial"
    elif "office" in land_use:
        return "Office"
    elif "retail" in land_use or "store" in land_use or "shop" in land_use:
        return "Retail"
    elif "institutional" in land_use or "public" in land_use or "civic" in land_use:
        return "Public/Institutional"
    elif "open" in land_use or "park" in land_use or "recreation" in land_use:
        return "Open Space"
    elif "mixed" in land_use:
        return "Mixed Use"
    elif "vacant" in land_use:
        return "Vacant"
    else:
        return "Other"


def categorize_project_type(description):
    """Helper function to categorize permits by project description"""
    if pd.isna(description):
        return "Unknown"

    description = str(description).lower()

    if any(term in description for term in ["new construction", "new building", "new development"]):
        return "New Construction"
    elif any(term in description for term in ["addition", "expand", "extension"]):
        return "Addition"
    elif any(term in description for term in ["renovation", "remodel", "rehab", "rehabilitation"]):
        return "Renovation"
    elif any(term in description for term in ["repair", "replace", "maintenance"]):
        return "Repair"
    elif any(term in description for term in ["demolition", "demolish", "remove"]):
        return "Demolition"
    elif any(term in description for term in ["change of use", "convert", "conversion"]):
        return "Change of Use"
    elif any(term in description for term in ["signage", "sign"]):
        return "Signage"
    elif any(term in description for term in ["solar", "photovoltaic", "pv"]):
        return "Solar"
    elif any(term in description for term in ["interior", "tenant improvement", "ti work"]):
        return "Interior Work"
    else:
        return "Other"


def process_permits_data(df):
    """Process permits data with specific transformations and user-friendly column names"""
    processed_df = df.copy()

    # Make sure column names are unique first
    processed_df = fix_duplicate_columns(processed_df)

    # Clean column names - lowercase and replace spaces with underscores
    processed_df.columns = [col.lower().replace(' ', '_') for col in processed_df.columns]

    # Create user-friendly column names mapping
    friendly_names = {
        'permit_number': 'permit_id',
        'application_number': 'application_id',
        'record_id': 'permit_id',
        'status': 'permit_status',
        'street_number': 'address_number',
        'street_name': 'street_name',
        'street_suffix': 'street_suffix',
        'description': 'project_description',
        'status_date': 'status_update_date',
        'filed_date': 'application_date',
        'first_construction_document_date': 'construction_start_date',
        'structural_notification': 'structural_notification',
        'number_of_existing_stories': 'existing_stories',
        'number_of_proposed_stories': 'proposed_stories',
        'estimated_cost': 'estimated_cost_usd',
        'revised_cost': 'revised_cost_usd',
        'existing_use': 'existing_building_use',
        'existing_units': 'existing_housing_units',
        'proposed_use': 'proposed_building_use',
        'proposed_units': 'proposed_housing_units',
        'existing_construction_type': 'existing_construction_type_code',
        'existing_construction_type_description': 'existing_construction_type',
        'proposed_construction_type': 'proposed_construction_type_code',
        'proposed_construction_type_description': 'proposed_construction_type',
        'site_permit': 'has_site_permit',
        'supervisor_district': 'supervisor_district',
        'neighborhoods_analysis_boundaries': 'neighborhood_code',
        'zipcode': 'zip_code',
        'location': 'location_coordinates'
    }

    # Apply friendly names where possible
    rename_cols = {col: friendly_names.get(col, col) for col in processed_df.columns if col in friendly_names}
    processed_df.rename(columns=rename_cols, inplace=True)
    
    # Fix any remaining duplicates after renaming
    processed_df = fix_duplicate_columns(processed_df)

    # Check for and handle common permit column names with various formats
    # This approach is more resilient to different column naming conventions
    permit_id_cols = [col for col in processed_df.columns if any(x in col.lower() for x in ['permit_id', 'permit_number', 'application_id', 'record_id'])]
    if permit_id_cols:
        # Use the first matching column as the main permit_id
        if 'permit_id' not in processed_df.columns and permit_id_cols:
            processed_df.rename(columns={permit_id_cols[0]: 'permit_id'}, inplace=True)

    # Find and standardize date columns
    date_columns = [col for col in processed_df.columns if any(date_term in col.lower() for date_term in ["date", "filed", "issued", "completed", "approved"])]
    for col in date_columns:
        try:
            processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
        except Exception as e:
            print(f"Could not convert {col} to datetime: {e}")

    # Determine which column to use for filed_date if it doesn't exist already
    if 'application_date' not in processed_df.columns:
        date_priority = ['filed_date', 'application_date', 'permit_creation_date', 'date_filed']
        for date_col in date_priority:
            if date_col in processed_df.columns:
                processed_df['application_date'] = processed_df[date_col]
                break

    # Add time-based columns if application_date exists
    if 'application_date' in processed_df.columns:
        # Extract time components
        processed_df['year'] = processed_df['application_date'].dt.year
        processed_df['month'] = processed_df['application_date'].dt.month
        processed_df['quarter'] = processed_df['application_date'].dt.quarter
        processed_df['week'] = processed_df['application_date'].dt.isocalendar().week

        # Calculate processing times where possible
        if 'issued_date' in processed_df.columns:
            processed_df['days_to_issue'] = (
                processed_df['issued_date'] - processed_df['application_date']
            ).dt.days

        if 'completed_date' in processed_df.columns:
            processed_df['total_processing_days'] = (
                processed_df['completed_date'] - processed_df['application_date']
            ).dt.days

    # Add status categorization if status exists
    status_col = next((col for col in processed_df.columns if 'status' in col.lower() and 'date' not in col.lower()), None)
    if status_col:
        processed_df['status_category'] = processed_df[status_col].apply(categorize_status)

    # Process cost/value data if available
    cost_columns = [col for col in processed_df.columns if any(x in col.lower() for x in ['cost', 'value', 'fee', 'price'])]
    for col in cost_columns:
        try:
            # Convert to numeric, coercing errors to NaN
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        except Exception as e:
            print(f"Could not convert {col} to numeric: {e}")

    # Find any column that might contain project descriptions
    description_cols = [col for col in processed_df.columns if any(x in col.lower() for x in ['description', 'scope', 'work'])]
    if description_cols and 'project_description' not in processed_df.columns:
        # Use the first matching column as the description
        processed_df.rename(columns={description_cols[0]: 'project_description'}, inplace=True)

    # Extract project types from descriptions if available
    if 'project_description' in processed_df.columns:
        processed_df['project_type'] = processed_df['project_description'].apply(categorize_project_type)

    return processed_df


def process_land_use_data(df):
    """Process land use data with specific transformations and user-friendly column names"""
    processed_df = df.copy()

    # Make sure column names are unique
    processed_df = fix_duplicate_columns(processed_df)

    # Clean column names
    processed_df.columns = [col.lower().replace(' ', '_') for col in processed_df.columns]

    # Create user-friendly column names mapping
    friendly_names = {
        # Building identifiers
        'sf16_bldgid': 'building_id',
        'area_id': 'area_id',
        'mblr': 'map_block_lot_record',
        'p2010_name': 'property_name_2010',
        'globalid': 'global_id',

        # Zoning measurements
        'p2010_zminn88ft': 'min_zoning_height_ft',
        'p2010_zmaxn88ft': 'max_zoning_height_ft',

        # Ground measurements
        'gnd_cells50cm': 'ground_measurement_cells',
        'gnd_mincm': 'ground_min_height_cm',
        'gnd_maxcm': 'ground_max_height_cm',
        'gnd_rangecm': 'ground_height_range_cm',
        'gnd_meancm': 'ground_mean_height_cm',
        'gnd_stdcm': 'ground_height_std_cm',
        'gnd_varietycm': 'ground_height_variety_cm',
        'gnd_majoritycm': 'ground_most_common_height_cm',
        'gnd_minoritycm': 'ground_least_common_height_cm',
        'gnd_mediancm': 'ground_median_height_cm',

        # First floor measurements
        'cells50cm_1st': 'first_floor_measurement_cells',
        'mincm_1st': 'first_floor_min_height_cm',
        'maxcm_1st': 'first_floor_max_height_cm',
        'rangecm_1st': 'first_floor_height_range_cm',
        'meancm_1st': 'first_floor_mean_height_cm',
        'stdcm_1st': 'first_floor_height_std_cm',
        'varietycm_1st': 'first_floor_height_variety_cm',
        'majoritycm_1st': 'first_floor_most_common_height_cm',
        'minoritycm_1st': 'first_floor_least_common_height_cm',
        'mediancm_1st': 'first_floor_median_height_cm',

        # Building height measurements
        'hgt_cells50cm': 'building_height_measurement_cells',
        'hgt_mincm': 'building_min_height_cm',
        'hgt_maxcm': 'building_max_height_cm',
        'hgt_rangecm': 'building_height_range_cm',
        'hgt_meancm': 'building_mean_height_cm',
        'hgt_stdcm': 'building_height_std_cm',
        'hgt_varietycm': 'building_height_variety_cm',
        'hgt_majoritycm': 'building_most_common_height_cm',
        'hgt_minoritycm': 'building_least_common_height_cm',
        'hgt_mediancm': 'building_median_height_cm',

        # Derived measurements
        'gnd_min_m': 'ground_min_elevation_m',
        'median_1st_m': 'first_floor_median_height_m',
        'hgt_median_m': 'building_median_height_m',
        'gnd1st_delta': 'ground_to_first_floor_diff_m',
        'peak_1st_m': 'first_floor_peak_height_m',

        # Geospatial data
        'shape': 'geometry',

        # Metadata
        'data_as_of': 'data_as_of',
        'data_loaded_at': 'data_loaded_at'
    }

    # Apply friendly names where possible
    rename_cols = {col: friendly_names.get(col, col) for col in processed_df.columns if col in friendly_names}
    processed_df.rename(columns=rename_cols, inplace=True)

    # Find the land use column if it exists with various naming conventions
    land_use_cols = [col for col in processed_df.columns if any(x in col.lower() for x in ['landuse', 'land_use', 'zoning', 'use'])]
    landuse_col = None

    if land_use_cols:
        # Use the most likely land use column
        for col in ['landuse', 'land_use', 'use_type', 'use', 'zoning']:
            if col in processed_df.columns:
                landuse_col = col
                break

        # If nothing found, use the first matching one
        if not landuse_col and land_use_cols:
            landuse_col = land_use_cols[0]

        # Create categorical groupings
        if landuse_col:
            processed_df['land_use_category'] = processed_df[landuse_col].apply(categorize_land_use)

            # Rename the original land use column if it wasn't already renamed
            if landuse_col != 'land_use' and landuse_col in processed_df.columns:
                processed_df.rename(columns={landuse_col: 'land_use'}, inplace=True)

    # Create simplified subset with most important columns
    important_cols = [
        'building_id', 'area_id', 'property_name_2010', 'map_block_lot_record',
        'min_zoning_height_ft', 'max_zoning_height_ft',
        'building_median_height_m', 'ground_min_elevation_m',
        'first_floor_median_height_m', 'ground_to_first_floor_diff_m',
        'geometry', 'land_use', 'land_use_category',
        'data_as_of', 'data_loaded_at'
    ]

    # Create a subset with just important columns (if they exist in the data)
    existing_important_cols = [col for col in important_cols if col in processed_df.columns]
    if len(existing_important_cols) >= 5:  # If we have at least 5 important columns
        processed_df_simple = processed_df[existing_important_cols].copy()
        # Return both full and simplified versions
        return processed_df, processed_df_simple

    return processed_df


def process_zoning_data(df):
    """Process zoning data with specific transformations"""
    processed_df = df.copy()

    # Make sure column names are unique
    processed_df = fix_duplicate_columns(processed_df)

    # Clean column names
    processed_df.columns = [col.lower().replace(' ', '_') for col in processed_df.columns]

    # Rename columns for consistency
    column_mapping = {
        'objectid': 'object_id',
        'zoning': 'zoning_code',
        'zoning_code': 'zoning_code',
        'zone_class': 'zoning_class',
        'description': 'zoning_description',
        'shape': 'geometry'
    }
    rename_cols = {col: column_mapping.get(col, col) for col in processed_df.columns if col in column_mapping}
    processed_df.rename(columns=rename_cols, inplace=True)

    return processed_df


def process_height_districts_data(df):
    """Process height districts data with specific transformations"""
    processed_df = df.copy()

    # Make sure column names are unique
    processed_df = fix_duplicate_columns(processed_df)

    # Clean column names
    processed_df.columns = [col.lower().replace(' ', '_') for col in processed_df.columns]

    # Process height values - try to extract numeric height values
    height_columns = [col for col in processed_df.columns if 'height' in col.lower()]
    for col in height_columns:
        try:
            # If it's already numeric, convert directly
            processed_df[f"{col}_value"] = pd.to_numeric(processed_df[col], errors='coerce')
        except:
            # If it contains text, keep as is for now
            pass

    # Rename columns for consistency
    column_mapping = {
        'objectid': 'object_id',
        'district': 'height_district',
        'description': 'height_description',
        'shape': 'geometry'
    }
    rename_cols = {col: column_mapping.get(col, col) for col in processed_df.columns if col in column_mapping}
    processed_df.rename(columns=rename_cols, inplace=True)

    return processed_df


def create_aggregated_datasets(processed_planning_data, processed_dir, logger):
    """Create aggregated datasets for analysis"""
    
    # Process permits data to get historical trends (works for any permit-type dataset)
    permit_datasets = [name for name in processed_planning_data.keys() 
                      if any(x in name.lower() for x in ['permit', 'project'])]
    
    for permit_dataset in permit_datasets:
        try:
            logger.info(f"  - Creating {permit_dataset} trends aggregations...")
            permits_df = processed_planning_data[permit_dataset]

            # Group by year and neighborhood to see development trends
            if "year" in permits_df.columns and "neighborhood" in permits_df.columns:
                # By year and neighborhood
                permit_trends_by_neighborhood = permits_df.groupby(['year', 'neighborhood']).size().reset_index(name='permit_count')
                save_to_parquet(permit_trends_by_neighborhood, f"{processed_dir}/planning", f"{permit_dataset}_trends_by_year_neighborhood")
                processed_planning_data[f"{permit_dataset}_trends_by_neighborhood"] = permit_trends_by_neighborhood

                # By year and status
                if "status_category" in permits_df.columns:
                    permit_trends_by_status = permits_df.groupby(['year', 'status_category']).size().reset_index(name='permit_count')
                    save_to_parquet(permit_trends_by_status, f"{processed_dir}/planning", f"{permit_dataset}_trends_by_year_status")
                    processed_planning_data[f"{permit_dataset}_trends_by_status"] = permit_trends_by_status

                # By year and project type if available
                if "project_type" in permits_df.columns:
                    permit_trends_by_project = permits_df.groupby(['year', 'project_type']).size().reset_index(name='permit_count')
                    save_to_parquet(permit_trends_by_project, f"{processed_dir}/planning", f"{permit_dataset}_trends_by_year_project")
                    processed_planning_data[f"{permit_dataset}_trends_by_project"] = permit_trends_by_project

                # By month (time series)
                if "month" in permits_df.columns:
                    monthly_trends = permits_df.groupby(['year', 'month']).size().reset_index(name='permit_count')
                    monthly_trends['date'] = pd.to_datetime(monthly_trends[['year', 'month']].assign(day=1))
                    monthly_trends = monthly_trends.sort_values('date')
                    save_to_parquet(monthly_trends, f"{processed_dir}/planning", f"{permit_dataset}_trends_monthly")
                    processed_planning_data[f"{permit_dataset}_trends_monthly"] = monthly_trends

                logger.info(f"  - Created and saved {permit_dataset} trends aggregations")
        except Exception as e:
            logger.error(f"  - Error creating {permit_dataset} trends: {e}")

    # Create land use summaries if available
    if "land_use" in processed_planning_data:
        try:
            logger.info("  - Creating land use aggregations...")
            land_use_df = processed_planning_data["land_use"]

            if "land_use_category" in land_use_df.columns:
                # Count by land use category
                land_use_summary = land_use_df["land_use_category"].value_counts().reset_index()
                land_use_summary.columns = ["land_use_category", "count"]
                save_to_parquet(land_use_summary, f"{processed_dir}/planning", "land_use_summary")
                processed_planning_data["land_use_summary"] = land_use_summary

                logger.info(f"  - Created and saved land use aggregations")
        except Exception as e:
            logger.error(f"  - Error creating land use summaries: {e}")
    
    return processed_planning_data


def fetch_sf_planning_data(raw_data_dir, processed_dir, start_date):
    """
    Fetch planning and zoning data from SF Planning Department API, save to raw data directory,
    and then process it, saving to processed directory with user-friendly column names.

    Args:
        raw_data_dir (str): Directory to save raw data
        processed_dir (str): Directory to save processed data  
        start_date (datetime): Start date for filtering permit data

    Returns:
        dict: Dictionary of all processed datasets
    """
    logger.info(f"Fetching SF Planning Department Data from {start_date.strftime('%Y-%m-%d')}...")

    # Initialize Socrata client for SF Planning Data
    # Note: For production use, you should use an app_token to avoid throttling
    client = sodapy.Socrata("data.sfgov.org", None)

    # Planning datasets to collect - verified working IDs (updated January 2025)
    planning_datasets = [
        {"id": "i98e-djp9", "name": "building_permits"},     # Building Permits - WORKING
        {"id": "qvu5-m3a2", "name": "planning_projects"},    # Planning Projects - NEW (replaces kncr-c6jh)
        {"id": "y673-d69b", "name": "planning_non_projects"}, # Planning Non-Projects - NEW
        {"id": "ynuv-fyni", "name": "land_use"},             # Land use - WORKING
        # Note: Other datasets have 404 errors, commented out for now
        # {"id": "gm9s-s9qv", "name": "zoning"},             # 404 Error  
        # {"id": "5baz-2hbr", "name": "height_districts"},   # 404 Error
    ]

    all_planning_data = {}
    processed_planning_data = {}

    # Create directories if they don't exist
    os.makedirs(f"{raw_data_dir}/planning", exist_ok=True)
    os.makedirs(f"{processed_dir}/planning", exist_ok=True)

    # Fetch each current dataset
    for dataset in planning_datasets:
        try:
            # For permits, we can filter by date to get historical data
            if "permits" in dataset["name"]:
                results = client.get(
                    dataset["id"],
                    limit=500000,
                    where=f"filed_date >= '{start_date.strftime('%Y-%m-%d')}'"
                )
            else:
                # For zoning/land use, we get current state
                results = client.get(dataset["id"], limit=500000)

            df = pd.DataFrame.from_records(results)

            # Clean column names to avoid duplicates when processing
            df.columns = [f"{col}" for col in df.columns]

            # Save raw data
            save_to_parquet(df, f"{raw_data_dir}/planning", f"{dataset['name']}_raw")

            # Store in our collection
            all_planning_data[dataset["name"]] = df

            logger.info(f"  - Retrieved {len(df)} records from {dataset['name']}")
        except Exception as e:
            logger.error(f"  - Error retrieving {dataset['name']}: {e}")

            # Try to load from existing raw file if fetch fails
            try:
                logger.info(f"  - Attempting to load {dataset['name']} from existing raw file...")
                raw_file_path = f"{raw_data_dir}/planning/{dataset['name']}_raw.parquet"
                df = pd.read_parquet(raw_file_path)
                all_planning_data[dataset["name"]] = df
                logger.info(f"  - Loaded {len(df)} records from existing {dataset['name']}_raw.parquet")
            except Exception as load_err:
                logger.error(f"  - Could not load from raw file either: {load_err}")

    # Now process each dataset
    logger.info("Processing datasets...")
    for dataset_name, df in all_planning_data.items():
        try:
            logger.info(f"  - Processing {dataset_name} dataset...")

            # Check for duplicate column names before processing
            if len(df.columns) != len(set(df.columns)):
                # Find duplicate columns
                cols = pd.Series(df.columns)
                duplicates = cols[cols.duplicated()].unique()
                logger.warning(f"Found duplicate columns in {dataset_name}: {list(duplicates)}")

                # Fix duplicates using our custom function
                df = fix_duplicate_columns(df)
                logger.info(f"Renamed duplicate columns in {dataset_name}")

            # Process based on dataset type
            if "permits" in dataset_name or "projects" in dataset_name:
                processed_df = process_permits_data(df)
                save_to_parquet(processed_df, f"{processed_dir}/planning", f"{dataset_name}_processed")
                processed_planning_data[dataset_name] = processed_df
                logger.info(f"  - Processed and saved {dataset_name} data with {len(processed_df)} rows")
        except Exception as e:
            logger.error(f"  - Error processing {dataset_name}: {e}")
            # If processing fails, store the raw data as processed but with cleaned column names
            try:
                # At least try to fix column names
                df_clean = df.copy()
                df_clean = fix_duplicate_columns(df_clean)
                df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
                processed_planning_data[dataset_name] = df_clean
                save_to_parquet(df_clean, f"{processed_dir}/planning", f"{dataset_name}_basic_processed")
                logger.info(f"  - Saved basic processed version of {dataset_name} with clean column names")
            except Exception as basic_err:
                logger.error(f"  - Could not even save basic processed version of {dataset_name}: {basic_err}")
                processed_planning_data[dataset_name] = df

    # Create special aggregated datasets
    processed_planning_data = create_aggregated_datasets(processed_planning_data, processed_dir, logger)

    logger.info("Completed SF Planning data processing")

    # Return all processed datasets
    return processed_planning_data


if __name__ == "__main__":
    # SF Planning Department Data API - Execution
    from logging_config_setup_02 import setup_logging, setup_directories
    
    logger = setup_logging()
    config = setup_directories()
    
    # Execute the function
    planning_data = fetch_sf_planning_data(
        config['raw_data_dir'],
        config['processed_dir'], 
        config['start_date']
    )
    
    # Print summary of what was collected
    print("\n=== SF Planning Data Collection Summary ===")
    for dataset_name, df in planning_data.items():
        print(f"{dataset_name}: {len(df)} rows")
        
    print(f"\nTotal datasets collected: {len(planning_data)}")
    
    print(f"\nFiles saved to: {config['processed_dir']}/planning/")
    print("Ready for business success modeling!")(processed_df)} rows")
            elif dataset_name == "land_use":
                # For land_use, we get both processed and simplified versions
                processed_result = process_land_use_data(df)
                if isinstance(processed_result, tuple) and len(processed_result) == 2:
                    # We got both full and simplified versions
                    processed_df, processed_df_simple = processed_result
                    # Save both versions
                    save_to_parquet(processed_df, f"{processed_dir}/planning", f"{dataset_name}_processed")
                    save_to_parquet(processed_df_simple, f"{processed_dir}/planning", f"{dataset_name}_simplified")
                    # Store both in our processed data collection
                    processed_planning_data[dataset_name] = processed_df
                    processed_planning_data[f"{dataset_name}_simplified"] = processed_df_simple
                    logger.info(f"  - Processed and saved {dataset_name} data with {len(processed_df)} rows and simplified version with {len(processed_df_simple)} rows")
                else:
                    # We just got the full version
                    processed_df = processed_result
                    save_to_parquet(processed_df, f"{processed_dir}/planning", f"{dataset_name}_processed")
                    processed_planning_data[dataset_name] = processed_df
                    logger.info(f"  - Processed and saved {dataset_name} data with {len(processed_df)} rows")
            elif dataset_name == "zoning":
                processed_df = process_zoning_data(df)
                save_to_parquet(processed_df, f"{processed_dir}/planning", f"{dataset_name}_processed")
                processed_planning_data[dataset_name] = processed_df
                logger.info(f"  - Processed and saved {dataset_name} data with {len(processed_df)} rows")
            elif dataset_name == "height_districts":
                processed_df = process_height_districts_data(df)
                save_to_parquet(processed_df, f"{processed_dir}/planning", f"{dataset_name}_processed")
                processed_planning_data[dataset_name] = processed_df
                logger.info(f"  - Processed and saved {dataset_name} data with {len(processed_df)} rows")
            else:
                # Basic processing for other datasets
                processed_df = df.copy()
                processed_df.columns = [col.lower().replace(' ', '_') for col in processed_df.columns]
                save_to_parquet(processed_df, f"{processed_dir}/planning", f"{dataset_name}_processed")
                processed_planning_data[dataset_name] = processed_df
                logger.info(f"  - Processed and saved {dataset_name} data with {len