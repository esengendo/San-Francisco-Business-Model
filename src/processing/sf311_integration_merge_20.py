import os
import pandas as pd
import numpy as np
from pathlib import Path
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

    # Configure paths for MacBook structure - using correct file names
    base_dir = os.getenv("BASE_DIR", "/app/San_Francisco_Business_Model")
    # base_dir = Path("/Users/baboo/Documents/San Francisco Business Model")
    processed_dir = os.path.join(base_dir, "processed")
    final_dir = os.path.join(processed_dir, "final")
    sf311_dir = os.path.join(processed_dir, "sf311")

    # Set pandas display options
    pd.set_option("display.max_columns", None)

    return logger, base_dir, final_dir, sf311_dir


def load_datasets(final_dir, sf311_dir, logger):
    """Load the business and SF 311 datasets"""
    logger.info("Loading datasets...")
    print("🔄 Loading datasets...")

    # Load dataframes
    business_df = pd.read_parquet(
        os.path.join(final_dir, "sf_business_success_with_permits.parquet")
    )
    sf311_df = pd.read_parquet(
        os.path.join(sf311_dir, "sf311_all_data.parquet")
    )  # Using correct filename

    logger.info(f"Business dataframe shape: {business_df.shape}")
    logger.info(f"SF 311 calls dataframe shape: {sf311_df.shape}")

    print(f"✅ Business dataframe shape: {business_df.shape}")
    print(f"✅ SF 311 calls dataframe shape: {sf311_df.shape}")
    print(f"✅ Business dataframe has {business_df.shape[1]} columns")

    return business_df, sf311_df


def explore_sf311_data(sf311_df, logger):
    """Exploring SF 311 Calls Data"""
    logger.info("Exploring SF 311 calls data...")
    print("\n🔍 Exploring SF 311 calls data...")

    # Check columns in SF 311 dataset
    print("📋 SF 311 dataset columns:")
    print(sf311_df.columns.tolist())

    # Check basic statistics for resolution days
    print("\n📈 Resolution days statistics:")
    print(sf311_df["resolution_days"].describe())

    # Most common service types
    print("\n🔝 Top 10 most common service types:")
    print(sf311_df["service_name"].value_counts().head(10))

    # Distribution of status descriptions
    print("\n📊 Status distribution:")
    print(sf311_df["status_description"].value_counts())

    # Check distribution of calls by supervisor district
    print("\n🗺️  Calls by supervisor district:")
    district_counts = sf311_df["supervisor_district"].value_counts()
    print(district_counts)


def check_district_formats(business_df, sf311_df, logger):
    """Check district format compatibility"""
    logger.info("Checking district format compatibility...")

    # Print unique values to understand the format discrepancy
    print("\n🔍 Unique supervisor districts in business dataframe:")
    print(business_df["supervisor_district"].unique())
    print("\n🔍 Unique supervisor districts in 311 dataframe:")
    print(sf311_df["supervisor_district"].unique())


def standardize_district(district):
    """Function to standardize district format"""
    if pd.isna(district) or district is None:
        return None
    try:
        # Extract just the numeric portion and format consistently
        if isinstance(district, str):
            district_num = float(district.split(".")[0])
        else:
            district_num = float(str(district).split(".")[0])
        return f"{district_num:.1f}"  # Format as "X.0"
    except (ValueError, TypeError, AttributeError):
        return None


def prepare_datasets_for_merge(business_df, sf311_df, logger):
    """Feature Engineering for SF 311 Data with Standardized Districts"""
    logger.info("Feature Engineering for SF 311 Data...")
    print("\n🔧 Feature Engineering for SF 311 Data...")

    # Create copies to avoid modifying originals
    business_df_copy = business_df.copy()
    sf311_df_copy = sf311_df.copy()

    # Standardize district format in both dataframes
    print("🔧 Standardizing supervisor_district format in both dataframes")
    business_df_copy["supervisor_district_std"] = business_df_copy[
        "supervisor_district"
    ].apply(standardize_district)
    sf311_df_copy["supervisor_district_std"] = sf311_df_copy[
        "supervisor_district"
    ].apply(standardize_district)

    # Convert datetime columns to datetime format if needed
    if not pd.api.types.is_datetime64_any_dtype(sf311_df_copy["requested_datetime"]):
        sf311_df_copy["requested_datetime"] = pd.to_datetime(
            sf311_df_copy["requested_datetime"]
        )

    if not pd.api.types.is_datetime64_any_dtype(sf311_df_copy["closed_date"]):
        sf311_df_copy["closed_date"] = pd.to_datetime(sf311_df_copy["closed_date"])

    return business_df_copy, sf311_df_copy


def create_district_311_features(sf311_df_copy, logger):
    """Create district-level 311 features"""
    logger.info("Creating district-level 311 features...")
    print("📊 Creating district-level 311 features...")

    # Group by standardized supervisor district and create aggregated features
    district_311_features = (
        sf311_df_copy.groupby("supervisor_district_std")
        .agg(
            # Volume of 311 calls (could indicate neighborhood issues)
            sf311_total_calls=("service_request_id", "count"),
            # Average resolution time (could indicate city responsiveness)
            sf311_avg_resolution_days=("resolution_days", "mean"),
            # Percentage of unresolved cases
            sf311_pct_unresolved=(
                "status_description",
                lambda x: sum(x != "Closed") / len(x) * 100,
            ),
            # Most common issue type
            sf311_most_common_issue=(
                "service_name",
                lambda x: x.value_counts().index[0],
            ),
            # Diversity of issues (more diverse might indicate complex neighborhood)
            sf311_issue_diversity=("service_name", lambda x: len(x.unique())),
            # Recent calls (last 6 months) - assuming we have a date range that makes sense
            sf311_recent_calls=(
                "requested_datetime",
                lambda x: sum(x > (x.max() - pd.DateOffset(months=6))),
            ),
        )
        .reset_index()
    )

    # Create heat index for each district (combining multiple negative factors)
    district_311_features["sf311_district_heat_index"] = (
        district_311_features["sf311_total_calls"]
        / district_311_features["sf311_total_calls"].max()
        * 0.4
        + district_311_features["sf311_avg_resolution_days"]
        / district_311_features["sf311_avg_resolution_days"].max()
        * 0.3
        + district_311_features["sf311_pct_unresolved"]
        / district_311_features["sf311_pct_unresolved"].max()
        * 0.3
    )

    # Categorize districts based on 311 call profile
    district_311_features["sf311_district_profile"] = pd.qcut(
        district_311_features["sf311_district_heat_index"],
        q=4,
        labels=[
            "Low Issue Area",
            "Moderate Issue Area",
            "High Issue Area",
            "Critical Issue Area",
        ],
    )

    print("✅ District 311 features created:")
    print(district_311_features.head())

    return district_311_features


def create_time_based_features(sf311_df_copy, logger):
    """Create time-based features by district and year"""
    logger.info("Creating time-based features...")
    print("📅 Creating time-based features...")

    # Year column already exists in your data, no need to extract
    time_district_features = (
        sf311_df_copy.groupby(["supervisor_district_std", "year"])
        .agg(
            sf311_yearly_calls=("service_request_id", "count"),
            sf311_yearly_avg_resolution=("resolution_days", "mean"),
        )
        .reset_index()
    )

    # Create year-over-year change in calls volume
    time_district_features = time_district_features.sort_values(
        ["supervisor_district_std", "year"]
    )
    time_district_features["sf311_prev_yearly_calls"] = time_district_features.groupby(
        "supervisor_district_std"
    )["sf311_yearly_calls"].shift(1)
    time_district_features["sf311_yoy_calls_change"] = (
        (
            time_district_features["sf311_yearly_calls"]
            - time_district_features["sf311_prev_yearly_calls"]
        )
        / time_district_features["sf311_prev_yearly_calls"]
        * 100
    )

    # Drop NaN values which occur for the first year in each district
    time_district_features = time_district_features.dropna(
        subset=["sf311_yoy_calls_change"]
    )
    print(
        f"✅ Time-based features created for {len(time_district_features)} district-year combinations"
    )

    return time_district_features


def merge_business_with_311_data(
    business_df_copy, district_311_features, time_district_features, logger
):
    """Merging Business Data with 311 Data"""
    logger.info("Merging business data with 311 features...")
    print("\n🔗 Merging business data with 311 features...")

    # First, check the number of unique standardized supervisor districts in each dataframe
    print(
        f"📊 Unique standardized supervisor districts in business_df: {business_df_copy['supervisor_district_std'].nunique()}"
    )
    print(
        f"📊 Unique standardized supervisor districts in district_311_features: {district_311_features['supervisor_district_std'].nunique()}"
    )
    print("\n🔍 Standardized districts in business dataframe:")
    print(sorted(business_df_copy["supervisor_district_std"].unique()))
    print("\n🔍 Standardized districts in 311 features:")
    print(sorted(district_311_features["supervisor_district_std"].unique()))

    # Make a copy of the business dataframe to ensure we preserve all columns
    business_with_311 = business_df_copy.copy()

    # Merge business data with district 311 features - KEEPING ALL COLUMNS using standardized district
    business_with_311 = pd.merge(
        business_with_311,
        district_311_features,
        on="supervisor_district_std",
        how="left",
    )

    # Check if we maintained all the original business columns
    print(f"\n📈 Original business dataframe columns: {business_df_copy.shape[1]}")
    print(f"📈 Merged dataframe columns: {business_with_311.shape[1]}")
    print(
        f"📈 Additional columns added: {business_with_311.shape[1] - business_df_copy.shape[1]}"
    )

    # Merge time-based features with our business data
    # We need to match on both district and year
    if "year" in business_with_311.columns:
        business_with_311 = pd.merge(
            business_with_311,
            time_district_features,
            left_on=["supervisor_district_std", "year"],
            right_on=["supervisor_district_std", "year"],
            how="left",
        )

        # Verify we still have all columns
        print(
            f"📈 Merged dataframe with time features columns: {business_with_311.shape[1]}"
        )
        print(
            f"📈 Additional columns added: {business_with_311.shape[1] - business_df_copy.shape[1]}"
        )
    else:
        print(
            "⚠️  Warning: 'year' column not found in business dataframe. Skipping time-based feature merge."
        )

    # Check for any NaN values after merge
    print(f"\n🔢 Rows before merge: {business_df_copy.shape[0]}")
    print(f"🔢 Rows after merge: {business_with_311.shape[0]}")
    print(
        f"🔢 NaN count in sf311_district_heat_index: {business_with_311['sf311_district_heat_index'].isna().sum()}"
    )

    return business_with_311


def handle_missing_values(business_with_311, logger):
    """Handling Missing Values"""
    logger.info("Handling missing values...")
    print("\n🧹 Handling missing values...")

    # If there are missing values, we can fill them with median values
    if (
        "sf311_district_heat_index" in business_with_311.columns
        and business_with_311["sf311_district_heat_index"].isna().sum() > 0
    ):
        print("🔧 Filling missing values...")

        # For numeric columns, fill with median
        numeric_cols = [
            "sf311_total_calls",
            "sf311_avg_resolution_days",
            "sf311_pct_unresolved",
            "sf311_issue_diversity",
            "sf311_recent_calls",
            "sf311_district_heat_index",
            "sf311_yearly_calls",
            "sf311_yearly_avg_resolution",
            "sf311_yoy_calls_change",
        ]

        for col in numeric_cols:
            if (
                col in business_with_311.columns
                and business_with_311[col].isna().sum() > 0
            ):
                median_val = business_with_311[col].median()
                business_with_311[col] = business_with_311[col].fillna(median_val)
                print(f"  ✅ Filled {col} with median: {median_val:.2f}")

        # For categorical columns, fill with most common value
        cat_cols = ["sf311_most_common_issue", "sf311_district_profile"]
        for col in cat_cols:
            if (
                col in business_with_311.columns
                and business_with_311[col].isna().sum() > 0
            ):
                mode_val = business_with_311[col].mode()[0]
                business_with_311[col] = business_with_311[col].fillna(mode_val)
                print(f"  ✅ Filled {col} with mode: {mode_val}")

        # Check remaining NaN values
        sf311_cols = [col for col in business_with_311.columns if "sf311" in col]
        remaining_nans = {
            col: business_with_311[col].isna().sum() for col in sf311_cols
        }
        print("📊 Remaining NaN values after filling:")
        for col, count in remaining_nans.items():
            if count > 0:
                print(f"  {col}: {count}")
        if sum(remaining_nans.values()) == 0:
            print("  ✅ All missing values filled!")
    else:
        print("✅ No missing values found in 311 features")

    return business_with_311


def analyze_relationships(business_with_311, logger):
    """Exploring Relationships between 311 Features and Business Success (without visualizations)"""
    logger.info("Analyzing relationships between 311 features and business success...")
    print("\n📊 Analyzing relationships between 311 features and business success...")

    # Success rate by district 311 profile
    if (
        "sf311_district_profile" in business_with_311.columns
        and "success" in business_with_311.columns
    ):
        success_by_profile = (
            business_with_311.groupby("sf311_district_profile")["success"].mean() * 100
        )
        print("\n🏆 Business Success Rate by District 311 Profile:")
        for profile, rate in success_by_profile.items():
            print(f"  {profile}: {rate:.2f}%")

    # Correlation between 311 heat index and success rate
    if (
        "sf311_district_heat_index" in business_with_311.columns
        and "success" in business_with_311.columns
    ):
        correlation = business_with_311["sf311_district_heat_index"].corr(
            business_with_311["success"]
        )
        print(
            f"\n🔗 Correlation between District 311 Heat Index and Business Success: {correlation:.4f}"
        )

    # Correlation analysis for new features
    sf311_numeric_cols = [
        col
        for col in business_with_311.columns
        if "sf311" in col and business_with_311[col].dtype in ["int64", "float64"]
    ]
    if sf311_numeric_cols and "success" in business_with_311.columns:
        print("\n📈 Correlations between 311 features and business success:")
        for col in sf311_numeric_cols:
            correlation = business_with_311[col].corr(business_with_311["success"])
            print(f"  {col}: {correlation:.4f}")


def save_final_dataset(business_with_311, business_df, final_dir, logger):
    """Save Final Dataset"""
    logger.info("Saving final dataset...")
    print("\n💾 Saving final dataset...")

    # Remove the temporary standardization column before saving
    if "supervisor_district_std" in business_with_311.columns:
        business_with_311 = business_with_311.drop(columns=["supervisor_district_std"])
        print("✅ Removed temporary standardization column")

    # Verify final columns
    print(f"📊 Final Dataset Summary:")
    print(f"  Original business dataframe columns: {business_df.shape[1]}")
    print(f"  Final merged dataframe columns: {business_with_311.shape[1]}")
    print(
        f"  Added 311 feature columns: {business_with_311.shape[1] - business_df.shape[1]}"
    )

    # List the 311 feature columns added
    sf311_cols = [col for col in business_with_311.columns if "sf311" in col]
    print(f"\n🏗️  Added 311 feature columns ({len(sf311_cols)}):")
    for col in sf311_cols:
        print(f"  • {col}")

    # Save to parquet file with all columns preserved
    output_path = output_path = os.path.join(
        final_dir, "sf_business_success_with_311.parquet"
    )
    try:
        business_with_311.to_parquet(output_path)
        logger.info(f"Final merged dataset saved to: {output_path}")
        print(f"✅ Final merged dataset saved to: {output_path}")
        print(f"   Shape: {business_with_311.shape}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        print(f"❌ Error saving dataset: {e}")
        return None


def summarize_features(business_with_311, logger):
    """Summary and feature analysis"""
    logger.info("Summarizing key 311 features...")
    print("\n📈 Summary of key 311 features:")

    # Print summary statistics for our new 311 features
    sf311_cols = [col for col in business_with_311.columns if "sf311" in col]
    if sf311_cols:
        for feature in sf311_cols:
            if feature in business_with_311.columns and business_with_311[
                feature
            ].dtype in ["int64", "float64"]:
                corr_with_success = (
                    business_with_311[feature].corr(business_with_311["success"])
                    if "success" in business_with_311.columns
                    else 0
                )
                print(f"🔹 {feature}:")
                print(f"   Mean: {business_with_311[feature].mean():.2f}")
                print(f"   Min: {business_with_311[feature].min():.2f}")
                print(f"   Max: {business_with_311[feature].max():.2f}")
                print(f"   Correlation with success: {corr_with_success:.4f}")
            else:
                if feature in business_with_311.columns:
                    most_common = (
                        business_with_311[feature].value_counts().iloc[0]
                        if len(business_with_311[feature].value_counts()) > 0
                        else "N/A"
                    )
                    print(f"🔹 {feature}: {most_common} (most common)")
            print()


def main():
    """Main execution function for SF 311 integration"""
    # Setup logging and directories
    logger, base_dir, final_dir, sf311_dir = setup_logging_and_directories()

    logger.info("Starting SF 311 Service Calls Data Integration Pipeline")
    print("=" * 80)
    print("SF 311 SERVICE CALLS DATA INTEGRATION PIPELINE")
    print("=" * 80)

    try:
        # Load datasets
        business_df, sf311_df = load_datasets(final_dir, sf311_dir, logger)

        # Explore SF 311 data
        explore_sf311_data(sf311_df, logger)

        # Check district formats
        check_district_formats(business_df, sf311_df, logger)

        # Prepare datasets for merge
        business_df_copy, sf311_df_copy = prepare_datasets_for_merge(
            business_df, sf311_df, logger
        )

        # Create district-level 311 features
        district_311_features = create_district_311_features(sf311_df_copy, logger)

        # Create time-based features
        time_district_features = create_time_based_features(sf311_df_copy, logger)

        # Merge business data with 311 data
        business_with_311 = merge_business_with_311_data(
            business_df_copy, district_311_features, time_district_features, logger
        )

        # Handle missing values
        business_with_311 = handle_missing_values(business_with_311, logger)

        # Analyze relationships
        analyze_relationships(business_with_311, logger)

        # Save final dataset
        output_path = save_final_dataset(
            business_with_311, business_df, final_dir, logger
        )

        # Summarize features
        summarize_features(business_with_311, logger)

        print("\n🎯 Next steps for business success prediction:")
        print("  1. ✅ SF 311 service data integrated")
        print("  2. 🔄 Further feature engineering using 311 insights")
        print("  3. 📊 Prepare all features for deep learning model")
        print("  4. 🎲 Split data into train/validation/test sets")
        print("  5. 🤖 Build initial model to predict business success probability")
        print("  6. 📈 Evaluate model performance and iterate")

        # Summary Report
        print("\n" + "=" * 80)
        print("SF 311 INTEGRATION COMPLETE")
        print("=" * 80)

        print(f"\n📊 DATASET CREATED:")
        print(f"   • sf_business_success_with_311.parquet")

        print(f"\n📋 FINAL DATASET SUMMARY:")
        print(f"   • Records: {business_with_311.shape[0]:,}")
        print(f"   • Features: {business_with_311.shape[1]}")
        print(
            f"   • New 311 Features: {business_with_311.shape[1] - business_df.shape[1]}"
        )

        if "success" in business_with_311.columns:
            print(f"   • Success Rate: {business_with_311['success'].mean():.2%}")

        sf311_cols = [col for col in business_with_311.columns if "sf311" in col]
        if sf311_cols:
            print(f"   • 311 Features Added: {len(sf311_cols)}")

        print(f"\n💾 OUTPUT FILE:")
        if output_path:
            print(f"   • {output_path}")

        print(
            f"\n🎉 Integration complete! Dataset ready for business success modeling."
        )
        print(f"📁 Final dataset location: {output_path}")
        print(f"📊 Final dataset dimensions: {business_with_311.shape}")

        logger.info(
            "SF 311 Service Calls Data Integration Pipeline completed successfully!"
        )

        return business_with_311

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"❌ Error in SF 311 integration pipeline: {e}")
        raise


# Execute if run as main script
if __name__ == "__main__":
    print("SF 311 Service Calls Data Integration Pipeline Starting...")
    integrated_dataset = main()
