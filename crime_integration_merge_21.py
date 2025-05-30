# crime_integration_pipeline_21.py
# Crime Data Integration Pipeline
# Integrates business success data with crime statistics

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

    # Configure paths for local MacBook structure
    base_dir = os.getenv("BASE_DIR", "/app/San_Francisco_Business_Model")
    # base_dir = Path("/Users/baboo/Documents/San Francisco Business Model")
    processed_dir = base_dir / "processed"
    final_dir = processed_dir / "final"
    crime_dir = processed_dir / "crime"

    # Define file paths - keeping same file names
    business_df_path = final_dir / "sf_business_success_with_311.parquet"
    crime_df_path = crime_dir / "sf_crime.parquet"
    output_path = final_dir / "sf_business_success_with_crime.parquet"

    # Set pandas display options
    pd.set_option("display.max_columns", None)

    return logger, base_dir, business_df_path, crime_df_path, output_path


def load_datasets(business_df_path, crime_df_path, logger):
    """Load the business and crime datasets"""
    logger.info("Loading datasets...")
    print("🔄 Loading datasets...")

    # Load dataframes
    df_business = pd.read_parquet(business_df_path)
    df_crime = pd.read_parquet(crime_df_path)

    logger.info(f"Business dataframe shape: {df_business.shape}")
    logger.info(f"Crime dataframe shape: {df_crime.shape}")

    print(f"✅ Business dataframe shape: {df_business.shape}")
    print(f"✅ Crime dataframe shape: {df_crime.shape}")

    return df_business, df_crime


def check_merge_compatibility(df_business, df_crime, logger):
    """Check common columns for potential merge keys"""
    logger.info("Checking merge compatibility...")

    # Check common columns for potential merge keys
    business_columns = set(df_business.columns)
    crime_columns = set(df_crime.columns)
    common_columns = business_columns.intersection(crime_columns)

    print("\n🔍 Potential merge keys (common columns):")
    for col in sorted(common_columns):
        print(f"  • {col}")

    return common_columns


def standardize_supervisor_districts(df_business, df_crime, logger):
    """Data Preparation - Fix Supervisor District Format Issues"""
    logger.info("Standardizing supervisor district formats...")
    print("\n🔧 Standardizing supervisor district formats...")

    # Check unique values in supervisor district columns
    print("\n📊 Unique values in business dataframe supervisor_district:")
    print(df_business["supervisor_district"].unique())

    print("\n📊 Unique values in crime dataframe supervisor_district:")
    print(df_crime["supervisor_district"].unique())

    # Fix format issues in the supervisor_district column
    # Business dataframe has float-like strings ('3.0') while crime has integer strings ('3')
    if "supervisor_district" in df_business.columns:
        # Convert to string first
        df_business["supervisor_district"] = df_business["supervisor_district"].astype(
            str
        )

        # Convert float-like strings ('3.0') to integers ('3')
        df_business["supervisor_district"] = df_business["supervisor_district"].apply(
            lambda x: (
                str(int(float(x)))
                if x != "nan" and x.replace(".", "", 1).isdigit()
                else x
            )
        )

        # Replace '0.0' or '0' with appropriate value
        if "0" in df_business["supervisor_district"].values:
            print("📝 Note: Found '0' value in business district, treating as missing")
            df_business["supervisor_district"] = df_business[
                "supervisor_district"
            ].replace("0", "Unknown")

    # Ensure crime dataframe has consistent format
    if "supervisor_district" in df_crime.columns:
        # Convert to string type
        df_crime["supervisor_district"] = df_crime["supervisor_district"].astype(str)

        # Handle 'nan' values
        df_crime["supervisor_district"] = df_crime["supervisor_district"].replace(
            "nan", "Unknown"
        )

    # Check the standardized values
    print("\n✅ Standardized business supervisor_district values:")
    print(df_business["supervisor_district"].unique())

    print("\n✅ Standardized crime supervisor_district values:")
    print(df_crime["supervisor_district"].unique())

    return df_business, df_crime


def create_district_crime_features(df_crime, logger):
    """Feature Engineering: District-Level Crime Aggregation"""
    logger.info("Creating district-level crime features...")
    print("\n🏗️ Creating district-level crime features...")

    # Create crime count by district
    crime_by_district = (
        df_crime.groupby("supervisor_district")
        .size()
        .reset_index(name="district_total_crimes")
    )

    # Add crime type distribution
    if "incident_category" in df_crime.columns:
        # Get top crime categories
        top_crime_categories = (
            df_crime["incident_category"].value_counts().head(5).index.tolist()
        )
        print(f"🔝 Top crime categories: {top_crime_categories}")

        # Create counts for each category
        for category in top_crime_categories:
            # Create safe column name
            safe_name = category.lower().replace(" ", "_").replace("-", "_")

            # Count category by district
            category_count = (
                df_crime[df_crime["incident_category"] == category]
                .groupby("supervisor_district")
                .size()
                .reset_index(name=f"district_{safe_name}_count")
            )

            # Merge with the district totals
            crime_by_district = crime_by_district.merge(
                category_count, on="supervisor_district", how="left"
            )

        # Calculate percentages of each crime type
        for category in top_crime_categories:
            safe_name = category.lower().replace(" ", "_").replace("-", "_")
            crime_by_district[f"district_{safe_name}_pct"] = (
                crime_by_district[f"district_{safe_name}_count"]
                / crime_by_district["district_total_crimes"]
                * 100
            ).fillna(0)

    return crime_by_district


def create_yearly_crime_trends(df_crime, logger):
    """Calculate yearly crime trends"""
    logger.info("Calculating yearly crime trends by district...")

    # Calculate yearly crime trends
    district_yearly_crimes = None
    if "year" in df_crime.columns:
        print("\n📅 Calculating yearly crime trends by district...")

        # Get yearly crime counts
        yearly_crimes = (
            df_crime.groupby(["supervisor_district", "year"])
            .size()
            .reset_index(name="crime_count")
        )

        # Pivot to get crime by year for each district
        district_yearly_crimes = yearly_crimes.pivot_table(
            index="supervisor_district",
            columns="year",
            values="crime_count",
            fill_value=0,
        ).reset_index()

        # Fix column names
        year_cols = [str(col) for col in district_yearly_crimes.columns[1:]]
        district_yearly_crimes.columns = ["supervisor_district"] + [
            f"district_crimes_{year}" for year in year_cols
        ]

        # Calculate year-over-year crime change for the most recent years
        if len(year_cols) >= 2:
            sorted_years = sorted(map(int, year_cols))
            recent_year = str(sorted_years[-1])
            previous_year = str(sorted_years[-2])

            district_yearly_crimes[f"district_crime_yoy_change"] = (
                (
                    district_yearly_crimes[f"district_crimes_{recent_year}"]
                    - district_yearly_crimes[f"district_crimes_{previous_year}"]
                )
                / district_yearly_crimes[f"district_crimes_{previous_year}"].replace(
                    0, 1
                )
                * 100
            )

            print(
                f"✅ Added year-over-year crime change from {previous_year} to {recent_year}"
            )

    return district_yearly_crimes


def create_spatial_crime_features(df_business, df_crime, logger):
    """Feature Engineering: Spatial Crime Features"""
    logger.info("Creating spatial crime features...")
    print("\n🗺️ Creating spatial crime features...")

    # If we have latitude and longitude, we can create proximity-based features
    if all(col in df_business.columns for col in ["latitude", "longitude"]) and all(
        col in df_crime.columns for col in ["latitude", "longitude"]
    ):

        try:
            from sklearn.neighbors import BallTree

            print("📍 Creating crime proximity features based on coordinates...")

            # Function to calculate crime density around business locations
            def calculate_crime_density(business_coords, crime_coords, radius_km=0.5):
                """Calculate number of crimes within radius_km of each business"""
                # Convert to radians for BallTree
                earth_radius = 6371  # km
                business_coords_rad = np.radians(business_coords)
                crime_coords_rad = np.radians(crime_coords)

                # Create BallTree for efficient nearest neighbor search
                tree = BallTree(crime_coords_rad, metric="haversine")

                # Count points within radius
                counts = tree.query_radius(
                    business_coords_rad, r=radius_km / earth_radius, count_only=True
                )

                return counts

            # Sample a subset of crime data if it's very large to improve performance
            max_crime_sample = 100000
            if len(df_crime) > max_crime_sample:
                crime_sample = df_crime.sample(max_crime_sample, random_state=42)
                print(
                    f"📊 Using a sample of {max_crime_sample} crimes for proximity calculation"
                )
            else:
                crime_sample = df_crime

            # Extract coordinates, dropping any rows with missing coordinates
            business_coords = df_business[["latitude", "longitude"]].dropna().values
            crime_coords = crime_sample[["latitude", "longitude"]].dropna().values

            # If we have sufficient data points, proceed with calculation
            if len(business_coords) > 0 and len(crime_coords) > 0:
                # Create a temporary dataframe with business index and coordinates
                temp_df = df_business[["latitude", "longitude"]].reset_index()
                temp_df = temp_df.dropna(subset=["latitude", "longitude"])

                # Calculate crime density
                temp_df["crimes_within_500m"] = calculate_crime_density(
                    temp_df[["latitude", "longitude"]].values, crime_coords, 0.5
                )

                # Merge back to main business dataframe
                df_business = df_business.merge(
                    temp_df[["index", "crimes_within_500m"]],
                    left_on=df_business.index,
                    right_on="index",
                    how="left",
                )

                # Clean up
                df_business = df_business.drop("index", axis=1)

                print(
                    f"✅ Created crime proximity features for {len(temp_df)} businesses with valid coordinates"
                )
            else:
                print("⚠️ Insufficient coordinate data for proximity calculation")
        except ImportError:
            logger.warning("sklearn not available for spatial analysis")
            print(
                "⚠️ sklearn not available for spatial analysis - skipping proximity features"
            )
        except Exception as e:
            logger.error(f"Error calculating crime proximity: {e}")
            print(f"❌ Error calculating crime proximity: {e}")
            print("⏭️ Skipping crime proximity features")

    return df_business


def merge_crime_features(
    df_business, crime_by_district, district_yearly_crimes, df_crime, logger
):
    """Merging Crime Features with Business Data"""
    logger.info("Merging district-level crime features with business data...")
    print("\n🔗 Merging district-level crime features with business data...")

    # First merge the district crime totals with business data - using LEFT join
    df_merged = df_business.merge(
        crime_by_district,
        on="supervisor_district",
        how="left",  # This preserves all business records
    )

    # If we have yearly crime trends, merge those as well - also using LEFT join
    if district_yearly_crimes is not None:
        print("📅 Merging yearly crime trends (LEFT join)...")
        df_merged = df_merged.merge(
            district_yearly_crimes,
            on="supervisor_district",
            how="left",  # This preserves all business records
        )

    # Verify row count is preserved
    print(f"\n📊 Original business dataframe rows: {len(df_business)}")
    print(f"📊 Merged dataframe rows: {len(df_merged)}")
    if len(df_business) == len(df_merged):
        print("✅ Row count preserved - successful LEFT merge")
    else:
        print("⚠️ Warning: Row count changed after merge!")

    # Fill missing crime data with zeros (assuming no crime data means no recorded crime)
    crime_columns = [col for col in df_merged.columns if "crime" in col.lower()]
    df_merged[crime_columns] = df_merged[crime_columns].fillna(0)

    return df_merged, crime_columns


def create_derived_crime_features(df_merged, logger):
    """Create normalized crime rates and flags"""
    logger.info("Creating derived crime features...")

    # Create normalized crime rates (per business)
    if (
        "district_total_crimes" in df_merged.columns
        and "businesses_in_neighborhood" in df_merged.columns
    ):
        df_merged["crime_per_business"] = df_merged[
            "district_total_crimes"
        ] / df_merged["businesses_in_neighborhood"].replace(0, 1)
        print("✅ Created crime_per_business ratio")

    # Create high crime area flag
    if "district_total_crimes" in df_merged.columns:
        median_crimes = df_merged["district_total_crimes"].median()
        df_merged["high_crime_area"] = (
            df_merged["district_total_crimes"] > median_crimes
        )
        print("✅ Created high_crime_area flag")

    # Create crime trend indicators
    if "district_crime_yoy_change" in df_merged.columns:
        df_merged["crime_increasing"] = df_merged["district_crime_yoy_change"] > 0
        print("✅ Created crime_increasing flag")

    return df_merged


def save_merged_dataset(df_merged, output_path, base_dir, logger):
    """Save the merged dataframe"""
    logger.info("Saving merged dataset...")
    print("\n💾 Saving merged dataset...")

    try:
        df_merged.to_parquet(output_path)
        logger.info(f"Merged dataframe saved to: {output_path}")
        print(f"✅ Merged dataframe saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving merged dataframe: {e}")
        print(f"❌ Error saving merged dataframe: {e}")
        # Try alternate path in base directory
        alternate_path = base_dir / "sf_business_success_with_crime.parquet"
        df_merged.to_parquet(alternate_path)
        logger.info(f"Merged dataframe saved to alternate path: {alternate_path}")
        print(f"✅ Merged dataframe saved to alternate path: {alternate_path}")
        return alternate_path


def analyze_crime_relationships(df_merged, logger):
    """Quick Exploratory Analysis (without visualizations)"""
    logger.info("Analyzing crime relationships with business success...")
    print("\n📈 Analyzing crime relationships with business success...")

    # Analyze relationship between crime and business success
    if all(col in df_merged.columns for col in ["success", "high_crime_area"]):
        success_by_crime = df_merged.groupby("high_crime_area")["success"].mean()
        print("\n🏆 Business Success Rate by Crime Level:")
        for crime_level, rate in success_by_crime.items():
            crime_label = "High Crime Area" if crime_level else "Low Crime Area"
            print(f"  {crime_label}: {rate:.2%}")

    # Success rate by district with crime overlay
    if all(
        col in df_merged.columns
        for col in ["success", "supervisor_district", "district_total_crimes"]
    ):
        district_summary = (
            df_merged.groupby("supervisor_district")
            .agg({"success": "mean", "district_total_crimes": "first"})
            .reset_index()
        )

        district_summary = district_summary.sort_values(
            "district_total_crimes", ascending=False
        )

        print("\n🗺️ Top 5 Districts by Crime Volume and Business Success:")
        for _, row in district_summary.head().iterrows():
            print(
                f"  District {row['supervisor_district']}: {row['success']:.2%} success, {row['district_total_crimes']:.0f} crimes"
            )

    # Correlation analysis
    if all(col in df_merged.columns for col in ["success", "district_total_crimes"]):
        # Select key columns for correlation analysis
        key_columns = ["success", "district_total_crimes", "business_age_years"]

        # Add crime proximity if available
        if "crimes_within_500m" in df_merged.columns:
            key_columns.append("crimes_within_500m")

        # Add other relevant columns
        for col in ["crime_per_business", "high_crime_area"]:
            if col in df_merged.columns:
                key_columns.append(col)

        # Calculate correlations with success
        print("\n🔗 Correlations with Business Success:")
        for col in key_columns:
            if col != "success" and col in df_merged.columns:
                if df_merged[col].dtype in ["int64", "float64", "bool"]:
                    correlation = df_merged[col].corr(df_merged["success"])
                    print(f"  {col}: {correlation:.4f}")

    # Success by business type and crime level
    if all(
        col in df_merged.columns
        for col in ["success", "high_crime_area", "business_industry"]
    ):
        # Get top business industries
        top_industries = df_merged["business_industry"].value_counts().head(5).index
        industry_df = df_merged[df_merged["business_industry"].isin(top_industries)]

        # Group by industry and crime level
        success_by_industry_crime = industry_df.groupby(
            ["business_industry", "high_crime_area"]
        )["success"].mean()

        print("\n🏢 Success Rate by Top Industries and Crime Level:")
        for (industry, high_crime), rate in success_by_industry_crime.items():
            crime_label = "High Crime" if high_crime else "Low Crime"
            print(f"  {industry} ({crime_label}): {rate:.2%}")


def summarize_crime_features(df_merged, logger):
    """Summarize crime features and their relationships"""
    logger.info("Summarizing crime features...")

    # List all crime-related features created
    crime_features = [col for col in df_merged.columns if "crime" in col.lower()]
    print(f"\n🏗️ Crime features created ({len(crime_features)}):")
    for feature in crime_features:
        print(f"  • {feature}")

    # Print summary statistics for crime features
    print(f"\n📈 Summary of key crime features:")
    for feature in crime_features:
        if feature in df_merged.columns and df_merged[feature].dtype in [
            "int64",
            "float64",
        ]:
            corr_with_success = (
                df_merged[feature].corr(df_merged["success"])
                if "success" in df_merged.columns
                else 0
            )
            print(f"🔹 {feature}:")
            print(f"   Mean: {df_merged[feature].mean():.2f}")
            print(f"   Min: {df_merged[feature].min():.2f}")
            print(f"   Max: {df_merged[feature].max():.2f}")
            print(f"   Correlation with success: {corr_with_success:.4f}")
            print()

    # Print recommendations for modeling
    print("\n🎯 Recommended features for business success prediction with crime data:")
    key_features = [
        # Location features
        "supervisor_district",
        # Business characteristics
        "business_industry",
        "business_age_years",
        # Crime features
        "district_total_crimes",
        "high_crime_area",
        "crime_per_business",
    ]

    if "crimes_within_500m" in df_merged.columns:
        key_features.append("crimes_within_500m")

    if "district_crime_yoy_change" in df_merged.columns:
        key_features.append("district_crime_yoy_change")

    available_features = [f for f in key_features if f in df_merged.columns]
    print("  " + ", ".join(available_features))


def main():
    """Main execution function for crime integration"""
    # Setup logging and directories
    logger, base_dir, business_df_path, crime_df_path, output_path = (
        setup_logging_and_directories()
    )

    logger.info("Starting Crime Data Integration Pipeline")
    print("=" * 80)
    print("CRIME DATA INTEGRATION PIPELINE")
    print("=" * 80)

    try:
        # Load datasets
        df_business, df_crime = load_datasets(business_df_path, crime_df_path, logger)

        # Check merge compatibility
        common_columns = check_merge_compatibility(df_business, df_crime, logger)

        # Standardize supervisor districts
        df_business, df_crime = standardize_supervisor_districts(
            df_business, df_crime, logger
        )

        # Create district-level crime features
        crime_by_district = create_district_crime_features(df_crime, logger)

        # Create yearly crime trends
        district_yearly_crimes = create_yearly_crime_trends(df_crime, logger)

        # Create spatial crime features
        df_business = create_spatial_crime_features(df_business, df_crime, logger)

        # Merge crime features with business data
        df_merged, crime_columns = merge_crime_features(
            df_business, crime_by_district, district_yearly_crimes, df_crime, logger
        )

        # Create derived crime features
        df_merged = create_derived_crime_features(df_merged, logger)

        # Print final statistics
        print(f"\n📊 Final merged dataframe shape: {df_merged.shape}")
        print(f"📊 Number of new crime-related features: {len(crime_columns)}")

        # Save the merged dataframe
        final_output_path = save_merged_dataset(
            df_merged, output_path, base_dir, logger
        )

        # Analyze crime relationships
        analyze_crime_relationships(df_merged, logger)

        # Summarize crime features
        summarize_crime_features(df_merged, logger)

        # Summary Report
        print("\n" + "=" * 80)
        print("CRIME INTEGRATION COMPLETE")
        print("=" * 80)

        print(f"\n📊 DATASET CREATED:")
        print(f"   • sf_business_success_with_crime.parquet")

        print(f"\n📋 FINAL DATASET SUMMARY:")
        print(f"   • Records: {df_merged.shape[0]:,}")
        print(f"   • Features: {df_merged.shape[1]}")
        print(f"   • New Crime Features: {len(crime_columns)}")

        if "success" in df_merged.columns:
            print(f"   • Success Rate: {df_merged['success'].mean():.2%}")

        crime_features = [col for col in df_merged.columns if "crime" in col.lower()]
        if crime_features:
            print(f"   • Crime Features Added: {len(crime_features)}")

        print(f"\n💾 OUTPUT FILE:")
        print(f"   • {final_output_path}")

        print(
            f"\n🎉 Crime integration complete! Dataset ready for business success modeling."
        )
        print(f"📁 Final dataset location: {final_output_path}")
        print(f"📊 Final dataset dimensions: {df_merged.shape}")

        logger.info("Crime Data Integration Pipeline completed successfully!")

        return df_merged

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"❌ Error in crime integration pipeline: {e}")
        raise


# Execute if run as main script
if __name__ == "__main__":
    print("Crime Data Integration Pipeline Starting...")
    integrated_dataset = main()
