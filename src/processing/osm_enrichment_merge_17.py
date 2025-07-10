# osm_enrichment_pipeline_17.py
# OSM Business Data Enrichment Pipeline
# Enriches business success data with OpenStreetMap amenity and feature data

import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import logging


def setup_logging_and_directories():
    """Configure logging and set up directory structure"""
    # Configure logging and set global variables
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("SFBusinessPipeline")

    # Set base directory to local MacBook folder
    base_dir = os.getenv("BASE_DIR", "/app/San_Francisco_Business_Model")
    # base_dir = "/Users/baboo/Documents/San Francisco Business Model"
    raw_data_dir = f"{base_dir}/raw_data"
    processed_dir = f"{base_dir}/processed"
    model_dir = f"{base_dir}/models"
    archive_dir = f"{base_dir}/archive"

    # Create directory structure
    for directory in [base_dir, raw_data_dir, processed_dir, model_dir, archive_dir]:
        os.makedirs(directory, exist_ok=True)

    # Create subdirectories for different data sources
    data_sources = [
        "sf_business",
        "economic",
        "demographic",
        "planning",
        "crime",
        "sf311",
        "mobility",
        "yelp",
        "news",
        "historical",
        "final",
        "osm",
    ]
    for source in data_sources:
        os.makedirs(f"{raw_data_dir}/{source}", exist_ok=True)
        os.makedirs(f"{processed_dir}/{source}", exist_ok=True)

    # Specific directories
    final_processed_dir = f"{processed_dir}/final"
    osm_processed_dir = f"{processed_dir}/osm"

    # Set pandas display options
    pd.set_option("display.max_columns", None)

    return logger, final_processed_dir, osm_processed_dir


def load_datasets(final_processed_dir, osm_processed_dir, logger):
    """Load the main business and OSM datasets"""
    logger.info("Loading datasets...")

    # Load main SF business dataset from the parquet file (keeping exact original path)
    business_path = f"{final_processed_dir}/sf_business_success_modeling_data.parquet"
    osm_path = f"{osm_processed_dir}/osm_businesses.parquet"

    if not os.path.exists(business_path):
        logger.error(f"Business dataset not found at: {business_path}")
        raise FileNotFoundError(f"Business dataset not found at: {business_path}")

    if not os.path.exists(osm_path):
        logger.error(f"OSM dataset not found at: {osm_path}")
        raise FileNotFoundError(f"OSM dataset not found at: {osm_path}")

    # Load main SF business dataset
    sf_business_success_df = pd.read_parquet(business_path)

    # Load OSM dataset with specific columns (keeping original column selection)
    use_cols = [
        "osm_id",
        "osm_type",
        "business_name",
        "business_type",
        "business_group",
        "latitude",
        "longitude",
        "street",
        "housenumber",
        "city",
        "postcode",
        "phone",
        "website",
        "email",
        "opening_hours",
        "wheelchair",
        "level",
        "cuisine",
        "delivery",
        "takeaway",
        "outdoor_seating",
        "internet_access",
        "tag_amenity",
        "tag_name",
        "tag_source",
    ]

    try:
        # First check which columns actually exist
        osm_sample = pd.read_parquet(osm_path, nrows=1)
        available_cols = [col for col in use_cols if col in osm_sample.columns]

        if len(available_cols) < len(use_cols):
            missing_cols = set(use_cols) - set(available_cols)
            logger.warning(f"Missing OSM columns: {missing_cols}")

        osm_df = pd.read_parquet(osm_path, columns=available_cols)

    except Exception as e:
        logger.warning(f"Error loading OSM data with specific columns: {e}")
        logger.info("Loading OSM data with all columns...")
        osm_df = pd.read_parquet(osm_path)

    # Display basic information about both datasets
    logger.info(
        f"SF Business Success Data: {len(sf_business_success_df)} records, {len(sf_business_success_df.columns)} columns"
    )
    logger.info(f"OSM Data: {len(osm_df)} records, {len(osm_df.columns)} columns")

    print("SF Business Success Data Information:")
    print(f"Number of records: {len(sf_business_success_df)}")
    print(f"Number of columns: {len(sf_business_success_df.columns)}")
    print(f"Success column present: {'success' in sf_business_success_df.columns}")

    print("\nOSM Data Information:")
    print(f"Number of records: {len(osm_df)}")
    print(f"Number of columns: {len(osm_df.columns)}")

    return sf_business_success_df, osm_df


def assess_data_quality(sf_business_success_df, osm_df, logger):
    """Data Quality Assessment"""
    logger.info("Performing data quality assessment...")
    print("\n=== Data Quality Assessment ===")

    # Check for missing values in both datasets
    print("\nMissing values in SF Business Success Data:")
    sf_missing = sf_business_success_df.isnull().sum()
    sf_missing_top = sf_missing[sf_missing > 0].sort_values(ascending=False).head(10)
    print(sf_missing_top)

    print("\nMissing values in OSM Data:")
    osm_missing = osm_df.isnull().sum()
    osm_missing_top = osm_missing[osm_missing > 0].sort_values(ascending=False).head(10)
    print(osm_missing_top)

    # Examine the success variable distribution
    if "success" in sf_business_success_df.columns:
        print(f"\nSuccess rate statistics:")
        success_stats = sf_business_success_df["success"].describe()
        print(success_stats)

        return success_stats

    return None


def analyze_business_locations(sf_business_success_df, osm_df, logger):
    """Analyze Business Locations"""
    logger.info("Analyzing business location data...")
    print("\n=== Analyzing Business Locations ===")

    # Sample data for analysis
    sf_sample = sf_business_success_df.sample(
        min(1000, len(sf_business_success_df)), random_state=42
    )
    osm_sample = osm_df.sample(min(500, len(osm_df)), random_state=42)

    # Filter out records with missing coordinates
    sf_sample = sf_sample.dropna(subset=["longitude", "latitude"])
    osm_sample = osm_sample.dropna(subset=["longitude", "latitude"])

    print(f"SF Business records with coordinates: {len(sf_sample)}")
    print(f"OSM records with coordinates: {len(osm_sample)}")

    # Calculate coordinate ranges
    if len(sf_sample) > 0:
        print(f"\nSF Business coordinate ranges:")
        print(
            f"Latitude: {sf_sample['latitude'].min():.4f} to {sf_sample['latitude'].max():.4f}"
        )
        print(
            f"Longitude: {sf_sample['longitude'].min():.4f} to {sf_sample['longitude'].max():.4f}"
        )

    if len(osm_sample) > 0:
        print(f"\nOSM coordinate ranges:")
        print(
            f"Latitude: {osm_sample['latitude'].min():.4f} to {osm_sample['latitude'].max():.4f}"
        )
        print(
            f"Longitude: {osm_sample['longitude'].min():.4f} to {osm_sample['longitude'].max():.4f}"
        )

    return sf_sample, osm_sample


def spatial_left_join(main_df, osm_df, threshold=0.001, logger=None):
    """
    Perform a left join from main business dataframe to OSM dataframe based on spatial proximity

    Parameters:
    main_df: Main SF business dataframe (all records preserved)
    osm_df: OSM dataframe (only used to enrich main_df)
    threshold: maximum distance in degrees for a match (roughly 100m)

    Returns:
    main_df with joined columns from osm_df
    """
    if logger:
        logger.info(f"Performing spatial join with threshold {threshold} degrees...")

    # Filter out records with missing coordinates
    main_valid = main_df.dropna(subset=["latitude", "longitude"])
    osm_valid = osm_df.dropna(subset=["latitude", "longitude"])

    if len(main_valid) == 0:
        if logger:
            logger.error("No valid coordinates in main dataset")
        return main_df

    if len(osm_valid) == 0:
        if logger:
            logger.error("No valid coordinates in OSM dataset")
        return main_df

    # Create coordinate arrays
    coords_main = np.vstack(
        [main_valid["latitude"].values, main_valid["longitude"].values]
    ).T
    coords_osm = np.vstack(
        [osm_valid["latitude"].values, osm_valid["longitude"].values]
    ).T

    # Build KD-tree
    tree = cKDTree(coords_osm)

    # Find nearest neighbors
    distances, indices = tree.query(coords_main, k=1)

    # Create a new dataframe with joined data (preserve all main_df records)
    result = main_df.copy()

    # Select useful amenity features from OSM data
    # Keep only these specific OSM features that are relevant to business success
    osm_features_to_keep = [
        "cuisine",
        "delivery",
        "takeaway",
        "outdoor_seating",
        "wheelchair",
        "level",
        "internet_access",
        "opening_hours",
        "website",
        "phone",
        "tag_amenity",
        "tag_name",
    ]

    # Filter to features that actually exist in the OSM dataset
    osm_features_to_keep = [
        col for col in osm_features_to_keep if col in osm_df.columns
    ]
    if logger:
        logger.info(f"Available OSM features to join: {osm_features_to_keep}")

    # Add a suffix to osm columns to avoid name conflicts
    osm_cols = {col: f"osm_{col}" for col in osm_features_to_keep}

    # Initialize new columns
    for new_col in osm_cols.values():
        result[new_col] = np.nan

    # Add matched data where distance is below threshold
    main_valid_indices = main_valid.index
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist <= threshold:
            main_idx = main_valid_indices[i]
            osm_row = osm_valid.iloc[idx]
            for col, new_col in osm_cols.items():
                result.loc[main_idx, new_col] = osm_row[col]

    # Add match distance column and flag for valid coordinate records
    result["osm_match_distance"] = np.nan
    result["has_osm_match"] = False

    # Set match info for records with valid coordinates
    for i, main_idx in enumerate(main_valid_indices):
        result.loc[main_idx, "osm_match_distance"] = distances[i]
        result.loc[main_idx, "has_osm_match"] = distances[i] <= threshold

    if logger:
        logger.info(
            f"Spatial join completed. Matches found: {result['has_osm_match'].sum()}"
        )
    return result


def merge_datasets(sf_business_success_df, osm_df, logger):
    """Merge Datasets Based on Geographic Proximity"""
    logger.info("Merging datasets based on geographic proximity...")
    print("\n=== Merging Datasets Based on Geographic Proximity ===")

    # Apply the merge - preserving ALL records from the main business dataframe
    merged_business_success_df = spatial_left_join(
        sf_business_success_df, osm_df, logger=logger
    )

    # Check the merge results
    total_businesses = len(sf_business_success_df)
    matched_businesses = merged_business_success_df["has_osm_match"].sum()
    match_rate = (matched_businesses / total_businesses) * 100

    print(f"\nMerging Results:")
    print(f"Total SF businesses: {total_businesses}")
    print(f"Businesses with OSM matches: {matched_businesses}")
    print(f"Match rate: {match_rate:.2f}%")

    logger.info(
        f"Merge completed: {matched_businesses}/{total_businesses} businesses matched ({match_rate:.2f}%)"
    )

    return merged_business_success_df


def explore_merged_dataset(merged_business_success_df, logger):
    """Explore the Merged Dataset"""
    logger.info("Exploring merged dataset...")
    print("\n=== Exploring the Merged Dataset ===")

    # Explore how many businesses were matched
    match_count = merged_business_success_df["has_osm_match"].sum()
    match_percentage = (match_count / len(merged_business_success_df)) * 100

    # Explore additional OSM features that were added
    osm_cols = [
        col for col in merged_business_success_df.columns if col.startswith("osm_")
    ]
    print(f"\nNew OSM features added: {len(osm_cols)}")
    print(f"Features: {osm_cols}")

    # Analyze which business types have the most OSM matches
    business_type_cols = [
        "business_industry",
        "business_type",
        "naics_code_description",
    ]
    business_type_col = None

    for col in business_type_cols:
        if col in merged_business_success_df.columns:
            business_type_col = col
            break

    if business_type_col:
        business_type_match = merged_business_success_df.groupby(business_type_col)[
            "has_osm_match"
        ].agg(["count", "sum", "mean"])
        business_type_match = business_type_match.sort_values("mean", ascending=False)
        business_type_match = business_type_match.rename(columns={"mean": "match_rate"})
        business_type_match["match_rate"] = business_type_match["match_rate"] * 100

        print(f"\nTop 10 business types by OSM match rate (minimum 10 businesses):")
        top_matches = business_type_match[business_type_match["count"] >= 10].head(10)
        print(top_matches)

    # Check if success rates differ between matched and unmatched businesses
    if "success" in merged_business_success_df.columns:
        # Calculate success rate statistics by match status
        print("\nSuccess rate by OSM match status:")
        success_by_match = merged_business_success_df.groupby("has_osm_match")[
            "success"
        ].describe()
        print(success_by_match)

        # Calculate mean success rates
        matched_success = merged_business_success_df[
            merged_business_success_df["has_osm_match"] == True
        ]["success"].mean()
        unmatched_success = merged_business_success_df[
            merged_business_success_df["has_osm_match"] == False
        ]["success"].mean()

        print(f"\nSuccess rate comparison:")
        print(f"Businesses with OSM match: {matched_success:.2%}")
        print(f"Businesses without OSM match: {unmatched_success:.2%}")
        print(f"Difference: {matched_success - unmatched_success:.2%}")

    return osm_cols


def save_final_dataset(
    merged_business_success_df, osm_cols, final_processed_dir, logger
):
    """Save the final merged dataset"""
    logger.info("Saving final merged dataset...")

    # Save the final merged dataset (keeping exact original filename and path)
    output_path = f"{final_processed_dir}/sf_business_success_osm_enriched.parquet"
    merged_business_success_df.to_parquet(output_path)

    print("\nFinal merged dataset:")
    print(f"Number of records: {len(merged_business_success_df)}")
    print(f"Number of features: {len(merged_business_success_df.columns)}")
    print(f"OSM features added: {len(osm_cols)}")
    print(f"\nMerged dataset saved as: {output_path}")

    logger.info(f"Final dataset saved to: {output_path}")

    return output_path


def main():
    """Main execution function for OSM business enrichment"""
    # Setup logging and directories
    logger, final_processed_dir, osm_processed_dir = setup_logging_and_directories()

    logger.info("Starting OSM Business Data Enrichment Pipeline")
    print("=" * 80)
    print("OSM BUSINESS DATA ENRICHMENT PIPELINE")
    print("=" * 80)

    try:
        # Load datasets
        sf_business_success_df, osm_df = load_datasets(
            final_processed_dir, osm_processed_dir, logger
        )

        # Assess data quality
        success_stats = assess_data_quality(sf_business_success_df, osm_df, logger)

        # Analyze business locations
        sf_sample, osm_sample = analyze_business_locations(
            sf_business_success_df, osm_df, logger
        )

        # Merge datasets based on geographic proximity
        merged_business_success_df = merge_datasets(
            sf_business_success_df, osm_df, logger
        )

        # Explore the merged dataset
        osm_cols = explore_merged_dataset(merged_business_success_df, logger)

        # Save the final dataset
        output_path = save_final_dataset(
            merged_business_success_df, osm_cols, final_processed_dir, logger
        )

        # Summary Report
        print("\n" + "=" * 80)
        print("OSM ENRICHMENT COMPLETE")
        print("=" * 80)

        print(f"\n📊 FINAL ENRICHED DATASET:")
        print(f"   • Records: {len(merged_business_success_df):,}")
        print(f"   • Features: {len(merged_business_success_df.columns)}")
        print(f"   • OSM Features Added: {len(osm_cols)}")
        print(
            f"   • OSM Match Rate: {merged_business_success_df['has_osm_match'].mean()*100:.2f}%"
        )

        if "success" in merged_business_success_df.columns:
            print(
                f"   • Success Rate: {merged_business_success_df['success'].mean():.2%}"
            )

        print(f"\n💾 OUTPUT FILE:")
        print(f"   • sf_business_success_osm_enriched.parquet")

        print(f"\n📁 SAVED TO:")
        print(f"   • {final_processed_dir}/")

        logger.info("OSM Business Data Enrichment Pipeline completed successfully!")

        return merged_business_success_df

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"❌ Error in OSM enrichment pipeline: {e}")
        raise


# Execute if run as main script
if __name__ == "__main__":
    print("OSM Business Data Enrichment Pipeline Starting...")
    enriched_dataset = main()
