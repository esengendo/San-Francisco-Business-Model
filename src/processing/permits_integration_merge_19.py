# permits_integration_pipeline_19.py
# Building Permits Data Integration Pipeline
# Integrates business success data with building permits information
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings("ignore")


def setup_logging_and_directories():
    """Configure logging and set up directory structure"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("SFBusinessPipeline")

    # Configure paths for MacBook structure - using correct available file
    base_dir = os.getenv("BASE_DIR", "/app/San_Francisco_Business_Model")
    # base_dir = "/Users/baboo/Documents/San Francisco Business Model"
    business_df_path = (
        f"{base_dir}/processed/final/sf_business_success_with_land_use.parquet"
    )
    permits_df_path = f"{base_dir}/processed/planning/building_permits_processed.parquet"  # Using actual file
    output_path = f"{base_dir}/processed/final/sf_business_success_with_permits.parquet"

    # Set pandas display options
    pd.set_option("display.max_columns", None)

    return logger, base_dir, business_df_path, permits_df_path, output_path


def load_datasets(business_df_path, permits_df_path, logger):
    """Load the business and permits datasets"""
    logger.info("Loading datasets...")
    print("🔄 Loading datasets...")

    try:
        df_business = pd.read_parquet(business_df_path)
        logger.info(f"Loaded business data: {df_business.shape}")
        print(f"✅ Business dataframe: {df_business.shape}")
    except FileNotFoundError:
        logger.error(f"Business data not found at: {business_df_path}")
        print(f"❌ Business data not found at: {business_df_path}")
        df_business = pd.DataFrame()

    try:
        df_permits = pd.read_parquet(permits_df_path)
        logger.info(f"Loaded permits data: {df_permits.shape}")
        print(f"✅ Permits dataframe: {df_permits.shape}")
    except FileNotFoundError:
        logger.error(f"Permits data not found at: {permits_df_path}")
        print(f"❌ Permits data not found at: {permits_df_path}")
        df_permits = pd.DataFrame()

    if df_business.empty or df_permits.empty:
        logger.error("Cannot proceed without both datasets")
        print("⚠️  Cannot proceed without both datasets")
        raise FileNotFoundError("Required datasets not found")

    return df_business, df_permits


def inspect_merge_compatibility(df_business, df_permits, logger):
    """Inspect data types for merge compatibility"""
    logger.info("Checking merge key compatibility...")
    print("\n🔍 Checking merge key compatibility...")

    # Let's first see what columns we actually have
    print(
        f"📊 Business columns ({len(df_business.columns)}): {list(df_business.columns)[:10]}..."
    )
    print(
        f"📊 Permits columns ({len(df_permits.columns)}): {list(df_permits.columns)[:10]}..."
    )

    # Check for common location-based columns
    location_keys = [
        "supervisor_district",
        "neighborhoods_analysis_boundaries",
        "neighborhood",
        "district",
        "zipcode",
        "address",
    ]

    print(f"\n🗺️  Checking for location merge keys:")
    merge_keys = []
    for key in location_keys:
        biz_has = key in df_business.columns
        permit_has = key in df_permits.columns
        if biz_has and permit_has:
            print(f"  ✅ {key}: Available in both datasets")
            merge_keys.append(key)
            print(f"  🔗 Will use {key} as merge key")
            print(
                f"    Business: {df_business[key].dtype} (sample: {df_business[key].dropna().iloc[0] if not df_business[key].dropna().empty else 'N/A'})"
            )
            print(
                f"    Permits: {df_permits[key].dtype} (sample: {df_permits[key].dropna().iloc[0] if not df_permits[key].dropna().empty else 'N/A'})"
            )
        elif biz_has:
            print(f"  📊 {key}: Only in business data")
        elif permit_has:
            print(f"  🏗️  {key}: Only in permits data")

    if not merge_keys:
        print(
            "  ⚠️  No common location keys found - will need to explore alternative merge strategies"
        )

    # Find all potential merge keys
    business_columns = set(df_business.columns)
    permits_columns = set(df_permits.columns)
    common_columns = business_columns.intersection(permits_columns)
    print(f"\n📋 Found {len(common_columns)} potential merge keys:")
    for col in sorted(common_columns):
        print(f"  • {col}")

    return merge_keys


def standardize_data_types(df_business, df_permits, merge_keys, logger):
    """Fix Data Types for Merge Keys"""
    logger.info("Standardizing data types for merge...")
    print("\n🔧 Standardizing data types for merge...")

    # Convert merge keys to consistent string format
    if merge_keys:
        for key in merge_keys:
            if df_business[key].dtype != df_permits[key].dtype:
                print(f"  Converting {key} to string in both datasets")
                df_business[key] = df_business[key].astype(str)
                df_permits[key] = df_permits[key].astype(str)
    else:
        print("  ⚠️  No merge keys to standardize - will create aggregate features only")

    return df_business, df_permits


def clean_permits_data(df_permits, logger):
    """Clean and Prepare Permits Data"""
    logger.info("Cleaning permits data...")
    print("\n🧹 Cleaning permits data...")

    # Convert numeric columns
    numeric_cols = [
        "estimated_cost",
        "revised_cost",
        "number_of_proposed_stories",
        "proposed_units",
        "number_of_existing_stories",
    ]

    for col in numeric_cols:
        if col in df_permits.columns:
            if df_permits[col].dtype == "object":
                print(f"  Converting {col} to numeric")
                df_permits[col] = pd.to_numeric(df_permits[col], errors="coerce")

    # Convert date columns
    date_cols = [
        "completed_date",
        "status_date",
        "filed_date",
        "approved_date",
        "issued_date",
        "first_construction_document_date",
    ]

    for col in date_cols:
        if col in df_permits.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_permits[col]):
                print(f"  Converting {col} to datetime")
                df_permits[col] = pd.to_datetime(df_permits[col], errors="coerce")

    return df_permits


def create_district_features(df_permits, logger):
    """Create District-Level Permit Features"""
    logger.info("Creating district-level permit features...")
    print("\n🏛️  Creating district-level permit features...")

    # Find the best district column to use
    district_cols = [col for col in df_permits.columns if "district" in col.lower()]
    if district_cols:
        district_col = district_cols[0]  # Use first available district column
        print(f"  Using district column: {district_col}")

        # Basic permit counts
        permits_by_district = (
            df_permits.groupby(district_col)
            .agg(
                {
                    df_permits.columns[
                        0
                    ]: "count"  # Count using first column (usually permit_number or similar)
                }
            )
            .reset_index()
        )
        permits_by_district.columns = [district_col, "district_permits_count"]

        print(
            f"  ✅ Created district features for {len(permits_by_district)} districts"
        )
        print(f"  📊 District features: {list(permits_by_district.columns)}")
        return permits_by_district, district_col
    else:
        print("  ⚠️  No district column found in permits data")
        return pd.DataFrame(), None


def create_neighborhood_features(df_permits, logger):
    """Create Neighborhood-Level Permit Features"""
    logger.info("Creating neighborhood-level permit features...")
    print("\n🏘️  Creating neighborhood-level permit features...")

    # Find the best neighborhood column to use
    neighborhood_cols = [
        col
        for col in df_permits.columns
        if any(term in col.lower() for term in ["neighborhood", "area", "zone"])
    ]
    if neighborhood_cols:
        neighborhood_col = neighborhood_cols[
            0
        ]  # Use first available neighborhood column
        print(f"  Using neighborhood column: {neighborhood_col}")

        # Basic neighborhood counts
        permits_by_neighborhood = (
            df_permits.groupby(neighborhood_col)
            .agg({df_permits.columns[0]: "count"})  # Count using first column
            .reset_index()
        )
        permits_by_neighborhood.columns = [
            neighborhood_col,
            "neighborhood_permits_count",
        ]

        print(
            f"  ✅ Created neighborhood features for {len(permits_by_neighborhood)} neighborhoods"
        )
        print(f"  📊 Neighborhood features: {list(permits_by_neighborhood.columns)}")
        return permits_by_neighborhood, neighborhood_col
    else:
        print("  ⚠️  No neighborhood column found in permits data")
        return pd.DataFrame(), None


def create_time_based_features(df_permits, logger):
    """Create Time-Based Permit Features"""
    logger.info("Creating time-based permit features...")
    print("\n📅 Creating time-based permit features...")

    try:
        if "status_date" in df_permits.columns:
            # Extract year for time-based analysis
            df_permits["year"] = df_permits["status_date"].dt.year

            # Remove invalid years
            valid_permits = df_permits.dropna(
                subset=["year", "supervisor_district"]
            ).copy()

            if len(valid_permits) > 0:
                # Calculate annual permit counts by district
                annual_permits = (
                    valid_permits.groupby(["year", "supervisor_district"])[
                        "permit_number"
                    ]
                    .count()
                    .reset_index()
                )

                # Create yearly trend features
                district_yearly_permits = annual_permits.pivot_table(
                    index="supervisor_district",
                    columns="year",
                    values="permit_number",
                    fill_value=0,
                ).reset_index()

                # Clean column names
                year_cols = [
                    str(int(col)) if isinstance(col, (int, float)) else str(col)
                    for col in district_yearly_permits.columns[1:]
                ]
                district_yearly_permits.columns = ["supervisor_district"] + [
                    f"district_permits_{col}" for col in year_cols
                ]

                print(f"  ✅ Created yearly trends for {len(year_cols)} years")
                return district_yearly_permits
            else:
                print("  ⚠️  No valid permit data for time analysis")
                return pd.DataFrame()
        else:
            print("  ⚠️  No status_date column found")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error creating time-based features: {e}")
        print(f"  ⚠️  Error creating time-based features: {e}")
        return pd.DataFrame()


def merge_permit_features(
    df_business,
    permits_by_district,
    permits_by_neighborhood,
    district_yearly_permits,
    merge_keys,
    df_permits,
    logger,
):
    """Merge All Features with Business Data"""
    logger.info("Merging all permit features with business data...")
    print("\n🔗 Merging all permit features with business data...")

    df_merged = df_business.copy()
    initial_shape = df_merged.shape

    # Merge district features if we have matching columns
    if not permits_by_district.empty and merge_keys:
        district_merge_key = [key for key in merge_keys if "district" in key.lower()]
        if district_merge_key:
            merge_key = district_merge_key[0]
            df_merged = df_merged.merge(permits_by_district, on=merge_key, how="left")
            print(f"  ✅ Added district features using {merge_key}: {df_merged.shape}")

    # Merge neighborhood features if we have matching columns
    if not permits_by_neighborhood.empty and merge_keys:
        neighborhood_merge_key = [
            key
            for key in merge_keys
            if any(term in key.lower() for term in ["neighborhood", "area"])
        ]
        if neighborhood_merge_key:
            merge_key = neighborhood_merge_key[0]
            df_merged = df_merged.merge(
                permits_by_neighborhood, on=merge_key, how="left"
            )
            print(
                f"  ✅ Added neighborhood features using {merge_key}: {df_merged.shape}"
            )

    # If no direct merges possible, create city-wide aggregate features
    if df_merged.shape[1] == initial_shape[1]:
        print("  📊 No direct merge possible - creating city-wide permit statistics...")

        # Create overall permit statistics
        total_permits = len(df_permits)
        avg_permits_per_area = (
            total_permits / len(df_business["supervisor_district"].unique())
            if "supervisor_district" in df_business.columns
            else total_permits / 100
        )

        df_merged["citywide_total_permits"] = total_permits
        df_merged["avg_permits_per_district"] = avg_permits_per_area

        # Add permit activity indicators
        if "estimated_cost_usd" in df_permits.columns:
            df_merged["citywide_avg_permit_cost"] = df_permits[
                "estimated_cost_usd"
            ].mean()

        print(f"  ✅ Added city-wide permit features: {df_merged.shape}")

    # Merge yearly trends if available
    if not district_yearly_permits.empty and merge_keys:
        district_merge_key = [key for key in merge_keys if "district" in key.lower()]
        if district_merge_key:
            merge_key = district_merge_key[0]
            df_merged = df_merged.merge(
                district_yearly_permits, on=merge_key, how="left"
            )
            print(f"  ✅ Added yearly trends: {df_merged.shape}")

    # Fill missing permit values with zeros
    permit_columns = [col for col in df_merged.columns if "permit" in col.lower()]
    if permit_columns:
        df_merged[permit_columns] = df_merged[permit_columns].fillna(0)
        print(f"  🔧 Filled missing values in {len(permit_columns)} permit columns")

    # Create high permit activity flag if we have permit counts
    permit_count_cols = [col for col in df_merged.columns if "permits_count" in col]
    if permit_count_cols:
        count_col = permit_count_cols[0]
        median_permits = df_merged[count_col].median()
        df_merged["high_permit_activity_area"] = df_merged[count_col] > median_permits
        high_activity_count = df_merged["high_permit_activity_area"].sum()
        print(
            f"  🏗️  Created high_permit_activity_area flag ({high_activity_count:,} businesses in high-activity areas)"
        )

    return df_merged, initial_shape


def save_results(df_merged, initial_shape, output_path, base_dir, logger):
    """Save Results"""
    logger.info("Saving merged dataset...")
    print(f"\n💾 Saving merged dataset...")
    print(f"  Original business data: {initial_shape}")
    print(f"  Final merged data: {df_merged.shape}")
    print(f"  New features added: {df_merged.shape[1] - initial_shape[1]}")

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_merged.to_parquet(output_path)
        logger.info(f"Saved to: {output_path}")
        print(f"✅ Saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving: {e}")
        print(f"❌ Error saving: {e}")
        # Try backup location
        backup_path = f"{base_dir}/sf_business_success_with_permits.parquet"
        df_merged.to_parquet(backup_path)
        logger.info(f"Saved to backup location: {backup_path}")
        print(f"✅ Saved to backup location: {backup_path}")
        return backup_path


def summarize_features(df_merged, logger):
    """Summary of Key Features"""
    logger.info("Summarizing features for modeling...")
    print(f"\n📋 Summary of features for modeling:")

    # Categorize features
    location_features = ["supervisor_district", "neighborhoods_analysis_boundaries"]
    business_features = ["business_type", "business_corridor", "success"]
    permit_features = [col for col in df_merged.columns if "permit" in col.lower()]

    print(
        f"  📍 Location features ({len([f for f in location_features if f in df_merged.columns])}): {', '.join(f for f in location_features if f in df_merged.columns)}"
    )
    print(
        f"  🏢 Business features ({len([f for f in business_features if f in df_merged.columns])}): {', '.join(f for f in business_features if f in df_merged.columns)}"
    )
    print(
        f"  🏗️  Permit features ({len(permit_features)}): {', '.join(permit_features[:5])}{'...' if len(permit_features) > 5 else ''}"
    )

    if permit_features:
        print(f"\n🎯 Key permit indicators:")
        key_indicators = [
            "district_permits_count",
            "neighborhood_permits_count",
            "district_permit_completion_rate",
            "high_permit_activity_area",
        ]
        for indicator in key_indicators:
            if indicator in df_merged.columns:
                if df_merged[indicator].dtype in ["int64", "float64"]:
                    print(f"  • {indicator}: {df_merged[indicator].mean():.2f} (avg)")
                else:
                    print(
                        f"  • {indicator}: {df_merged[indicator].value_counts().iloc[0]} most common"
                    )


def main():
    """Main execution function for permits integration"""
    # Setup logging and directories
    logger, base_dir, business_df_path, permits_df_path, output_path = (
        setup_logging_and_directories()
    )

    logger.info("Starting Building Permits Data Integration Pipeline")
    print("=" * 80)
    print("BUILDING PERMITS DATA INTEGRATION PIPELINE")
    print("=" * 80)

    try:
        # Load datasets
        df_business, df_permits = load_datasets(
            business_df_path, permits_df_path, logger
        )

        # Inspect merge compatibility
        merge_keys = inspect_merge_compatibility(df_business, df_permits, logger)

        # Standardize data types
        df_business, df_permits = standardize_data_types(
            df_business, df_permits, merge_keys, logger
        )

        # Clean permits data
        df_permits = clean_permits_data(df_permits, logger)

        # Create district-level features
        permits_by_district, district_col = create_district_features(df_permits, logger)

        # Create neighborhood-level features
        permits_by_neighborhood, neighborhood_col = create_neighborhood_features(
            df_permits, logger
        )

        # Create time-based features
        district_yearly_permits = create_time_based_features(df_permits, logger)

        # Merge all features
        df_merged, initial_shape = merge_permit_features(
            df_business,
            permits_by_district,
            permits_by_neighborhood,
            district_yearly_permits,
            merge_keys,
            df_permits,
            logger,
        )

        # Save results
        final_output_path = save_results(
            df_merged, initial_shape, output_path, base_dir, logger
        )

        # Summarize features
        summarize_features(df_merged, logger)

        print(f"\n🎉 Integration complete! Ready for business success modeling.")

        # Summary Report
        print("\n" + "=" * 80)
        print("PERMITS INTEGRATION COMPLETE")
        print("=" * 80)

        print(f"\n📊 DATASET CREATED:")
        print(f"   • sf_business_success_with_permits.parquet")

        print(f"\n📋 FINAL DATASET SUMMARY:")
        print(f"   • Records: {df_merged.shape[0]:,}")
        print(f"   • Features: {df_merged.shape[1]}")
        print(f"   • New Permit Features: {df_merged.shape[1] - initial_shape[1]}")

        if "success" in df_merged.columns:
            print(f"   • Success Rate: {df_merged['success'].mean():.2%}")

        permit_features = [col for col in df_merged.columns if "permit" in col.lower()]
        if permit_features:
            print(f"   • Permit Features Added: {len(permit_features)}")

        print(f"\n💾 OUTPUT FILE:")
        print(f"   • {final_output_path}")

        logger.info(
            "Building Permits Data Integration Pipeline completed successfully!"
        )

        return df_merged

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"❌ Error in permits integration pipeline: {e}")
        raise


# Execute if run as main script
if __name__ == "__main__":
    print("Building Permits Data Integration Pipeline Starting...")
    integrated_dataset = main()
