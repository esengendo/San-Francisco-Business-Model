# land_use_integration_pipeline_18.py
# Land Use Data Integration Pipeline
# Integrates business success data with land use and zoning information

import os
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, Polygon, MultiPolygon, shape
import numpy as np
import json
from shapely.geometry import mapping, shape
import logging


def setup_logging_and_directories():
    """Configure logging and set up directory structure"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("SFBusinessPipeline")

    # Configure paths for MacBook structure
    base_dir = os.getenv("BASE_DIR", "/app/San_Francisco_Business_Model")
    #base_dir = "/Users/baboo/Documents/San Francisco Business Model"
    processed_dir = f"{base_dir}/processed"

    # Set pandas display options
    pd.set_option("display.max_columns", None)

    return logger, base_dir, processed_dir


def load_datasets(processed_dir, logger):
    """Load the datasets - keeping exact same file paths as original"""
    logger.info("Loading datasets...")
    print("Loading datasets...")

    try:
        business_df = pd.read_parquet(
            f"{processed_dir}/final/sf_business_success_osm_enriched.parquet"
        )
        logger.info(f"Loaded business data: {business_df.shape}")
        print(f"✅ Loaded business data: {business_df.shape}")
    except FileNotFoundError:
        logger.error("Business data not found")
        print("❌ Business data not found. Check if the file exists at:")
        print(f"   {processed_dir}/final/sf_business_success_osm_enriched.parquet")
        business_df = pd.DataFrame()

    try:
        land_use_df = pd.read_parquet(
            f"{processed_dir}/planning/land_use_simplified.parquet"
        )
        logger.info(f"Loaded land use data: {land_use_df.shape}")
        print(f"✅ Loaded land use data: {land_use_df.shape}")
    except FileNotFoundError:
        logger.error("Land use data not found")
        print("❌ Land use data not found. Check if the file exists at:")
        print(f"   {processed_dir}/planning/land_use_simplified.parquet")
        land_use_df = pd.DataFrame()

    if business_df.empty or land_use_df.empty:
        logger.error("Cannot proceed without both datasets")
        print("⚠️  Cannot proceed without both datasets")
        raise FileNotFoundError("Required datasets not found")

    # Display basic information
    print(
        f"\nBusiness DataFrame: {business_df.shape[0]:,} rows, {business_df.shape[1]} columns"
    )
    print(f"Memory usage: {business_df.memory_usage().sum() / 1024**2:.2f} MB")
    print(
        f"\nLand Use DataFrame: {land_use_df.shape[0]:,} rows, {land_use_df.shape[1]} columns"
    )
    print(f"Memory usage: {land_use_df.memory_usage().sum() / 1024**2:.2f} MB")

    return business_df, land_use_df


def convert_to_geographic_data(business_df, land_use_df, logger):
    """Convert business dataframe to GeoDataFrame"""
    logger.info("Converting to geographic data...")
    print("\n🗺️  Converting to geographic data...")

    business_gdf = business_df.copy()
    business_gdf["geometry"] = gpd.points_from_xy(
        business_df["longitude"], business_df["latitude"]
    )
    business_gdf = gpd.GeoDataFrame(business_gdf, geometry="geometry", crs="EPSG:4326")

    # Handle the complex geometry format in land_use_df
    land_use_gdf = land_use_df.copy()

    # Check geometry column
    print(f"\nGeometry column type: {type(land_use_df['geometry'].iloc[0])}")
    sample_geom = land_use_df["geometry"].iloc[0]
    print(f"Sample geometry: {str(sample_geom)[:200]}...")

    return business_gdf, land_use_gdf


def convert_to_shapely(geom_obj):
    """Convert various geometry formats to shapely geometry"""
    try:
        if isinstance(geom_obj, str):
            return wkt.loads(geom_obj)
        elif isinstance(geom_obj, dict):
            return shape(geom_obj)
        elif hasattr(geom_obj, "__geo_interface__"):
            return shape(geom_obj.__geo_interface__)
        else:
            geom_str = str(geom_obj)
            if "POLYGON" in geom_str or "MULTIPOLYGON" in geom_str:
                return wkt.loads(geom_str)
            else:
                return Point(0, 0).buffer(0.01)
    except Exception as e:
        print(f"⚠️  Error converting geometry: {e}")
        return None


def perform_spatial_join(business_gdf, land_use_gdf, logger):
    """Perform spatial join between business and land use data"""
    logger.info("Performing spatial join...")
    spatial_join_success = True

    try:
        print("\n🔄 Converting land use geometries...")
        land_use_gdf["geometry"] = land_use_gdf["geometry"].apply(
            lambda x: convert_to_shapely(x) if x is not None else None
        )

        # Remove invalid geometries
        valid_geoms = land_use_gdf["geometry"].notna()
        invalid_count = len(land_use_gdf) - valid_geoms.sum()
        land_use_gdf = land_use_gdf[valid_geoms]

        if invalid_count > 0:
            print(f"⚠️  Removed {invalid_count:,} rows with invalid geometries")

        # Convert to GeoDataFrame
        land_use_gdf = gpd.GeoDataFrame(
            land_use_gdf, geometry="geometry", crs="EPSG:4326"
        )

        print("🔗 Performing spatial join...")
        # Try 'within' predicate first
        merged_gdf = gpd.sjoin(
            business_gdf, land_use_gdf, how="left", predicate="within"
        )

        # Check success
        land_use_matches = (
            merged_gdf.index_right.notna().sum()
            if "index_right" in merged_gdf.columns
            else 0
        )

        if land_use_matches > 0:
            print(
                f"✅ Spatial join successful: {land_use_matches:,} businesses matched with land use data"
            )
            join_method = "within predicate"
        else:
            print("⚠️  No matches with 'within'. Trying nearest join...")
            merged_gdf = gpd.sjoin_nearest(
                business_gdf, land_use_gdf, how="left", max_distance=100
            )
            land_use_matches = (
                merged_gdf.index_right.notna().sum()
                if "index_right" in merged_gdf.columns
                else 0
            )

            if land_use_matches > 0:
                print(
                    f"✅ Nearest join successful: {land_use_matches:,} matches within 100m"
                )
                join_method = "nearest predicate (100m max)"
            else:
                print("❌ Spatial join failed")
                spatial_join_success = False

    except Exception as e:
        logger.error(f"Error in spatial join: {e}")
        print(f"❌ Error in spatial join: {e}")
        spatial_join_success = False
        merged_gdf = business_gdf.copy()
        land_use_matches = 0

    return merged_gdf, land_use_matches, spatial_join_success


def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate distance between two points using Haversine formula"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371 * 1000  # Earth radius in km, convert to meters


def find_nearest_land_use(business_row, land_use_df, max_distance=500):
    """Find nearest land use for a business location"""
    biz_lon, biz_lat = business_row["longitude"], business_row["latitude"]

    # If land use has lat/lon columns, use them
    if "longitude" in land_use_df.columns and "latitude" in land_use_df.columns:
        distances = land_use_df.apply(
            lambda row: haversine_distance(
                biz_lon, biz_lat, row["longitude"], row["latitude"]
            ),
            axis=1,
        )

        min_distance_idx = distances.idxmin()
        min_distance = distances[min_distance_idx]

        if min_distance <= max_distance:
            nearest_row = land_use_df.loc[min_distance_idx].to_dict()
            nearest_row["distance_to_land_use"] = min_distance
            return nearest_row

    return None


def fallback_distance_matching(business_df, land_use_df, logger):
    """Fallback to manual distance-based matching if spatial join failed"""
    logger.info("Falling back to manual distance-based matching...")
    print("\n🔄 Falling back to manual distance-based matching...")

    # Process in batches for memory efficiency
    batch_size = 1000
    all_results = []
    total_batches = (len(business_df) + batch_size - 1) // batch_size

    for batch_num, start_idx in enumerate(range(0, len(business_df), batch_size)):
        end_idx = min(start_idx + batch_size, len(business_df))
        print(
            f"Processing batch {batch_num + 1}/{total_batches}: rows {start_idx}-{end_idx}"
        )

        batch = business_df.iloc[start_idx:end_idx]
        batch_results = []

        for idx, row in batch.iterrows():
            nearest = find_nearest_land_use(row, land_use_df)
            combined = {**row.to_dict(), **(nearest if nearest else {})}
            batch_results.append(combined)

        all_results.extend(batch_results)

    merged_gdf = pd.DataFrame(all_results)
    land_use_matches = merged_gdf.get("building_id", pd.Series()).notna().sum()
    print(f"Manual matching completed: {land_use_matches:,} matches found")

    return merged_gdf, land_use_matches


def clean_merged_data(merged_gdf, logger):
    """Clean up merged dataframe"""
    logger.info("Cleaning merged data...")
    print("\n🧹 Cleaning merged data...")

    if "index_right" in merged_gdf.columns:
        merged_gdf = merged_gdf.drop(columns=["index_right"])

    # Handle duplicate columns from join
    duplicate_cols = [
        col for col in merged_gdf.columns if col.endswith(("_right", "_left"))
    ]
    if duplicate_cols:
        print(f"Handling {len(duplicate_cols)} duplicate columns")
        for col in duplicate_cols:
            base_col = col.replace("_left", "").replace("_right", "")
            if f"{base_col}_left" in merged_gdf.columns:
                merged_gdf[base_col] = merged_gdf[f"{base_col}_left"]
            merged_gdf = merged_gdf.drop(columns=[col])

    return merged_gdf


def save_merged_data(merged_gdf, base_dir, logger):
    """Save merged data - keeping exact same output filename as original"""
    output_path = (
        f"{base_dir}/processed/final/sf_business_success_with_land_use.parquet"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_gdf.to_parquet(output_path)
    logger.info(f"Saved merged data to: {output_path}")
    print(f"💾 Saved merged data to: {output_path}")
    return output_path


def analyze_land_use_impact(merged_df, logger):
    """Analyze the impact of land use on business success"""
    logger.info("Analyzing land use impact on business success...")
    print("\n📈 Analyzing land use impact on business success...")

    # Define analysis columns
    land_use_cols = [
        "land_use",
        "land_use_category",
        "max_zoning_height_ft",
        "building_median_height_m",
        "ground_min_elevation_m",
    ]
    success_cols = ["success", "business_age_years"]

    # Filter to existing columns
    valid_land_use_cols = [col for col in land_use_cols if col in merged_df.columns]
    valid_success_cols = [col for col in success_cols if col in merged_df.columns]

    if not valid_land_use_cols or not valid_success_cols:
        logger.warning("Missing required columns for analysis")
        print("❌ Missing required columns for analysis")
        return merged_df, None

    # Create analysis subset
    analysis_df = merged_df.dropna(subset=valid_land_use_cols + valid_success_cols)
    print(f"Analysis dataset: {len(analysis_df):,} rows with complete data")

    # Success rate by land use category
    if "land_use_category" in analysis_df.columns and "success" in analysis_df.columns:
        success_by_land_use = (
            analysis_df.groupby("land_use_category")["success"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        success_by_land_use.columns = ["Land Use Category", "Success Rate", "Count"]

        # Filter to significant categories (min 100 businesses)
        significant_categories = success_by_land_use[
            success_by_land_use["Count"] >= 100
        ]

        if not significant_categories.empty:
            print(f"\n🏆 Success rate by land use category (min 100 businesses):")
            for _, row in significant_categories.head(10).iterrows():
                print(
                    f"  {row['Land Use Category']}: {row['Success Rate']:.3f} ({row['Count']:,} businesses)"
                )
        else:
            significant_categories = None
    else:
        significant_categories = None
        print("⚠️  Cannot analyze success by land use - missing columns")

    # Correlation analysis
    numeric_cols = [
        "success",
        "business_age_years",
        "max_zoning_height_ft",
        "building_median_height_m",
        "ground_min_elevation_m",
        "sf_gdp",
        "sf_unemployment_rate",
        "sf_median_household_income",
    ]

    valid_numeric_cols = [col for col in numeric_cols if col in analysis_df.columns]

    if len(valid_numeric_cols) >= 2 and "success" in valid_numeric_cols:
        corr_matrix = analysis_df[valid_numeric_cols].corr()
        success_corr = (
            corr_matrix["success"].drop("success").sort_values(key=abs, ascending=False)
        )

        print(f"\n🔗 Top correlations with business success:")
        for feature, corr in success_corr.head(5).items():
            print(f"  {feature}: {corr:.3f}")
    else:
        print("⚠️  Cannot calculate correlations - insufficient data")

    return analysis_df, significant_categories


def engineer_land_use_features(merged_df, base_dir, logger):
    """Create new features from land use data"""
    logger.info("Engineering land use features...")
    print("\n🔧 Engineering land use features...")

    feature_df = merged_df.copy()
    features_added = 0

    # 1. Land use category dummies
    if "land_use_category" in feature_df.columns:
        land_use_dummies = pd.get_dummies(
            feature_df["land_use_category"], prefix="land_use"
        )
        feature_df = pd.concat([feature_df, land_use_dummies], axis=1)
        features_added += len(land_use_dummies.columns)
        print(f"  ✅ Added {len(land_use_dummies.columns)} land use category features")

    # 2. Height utilization ratio
    height_cols = ["building_median_height_m", "max_zoning_height_ft"]
    if all(col in feature_df.columns for col in height_cols):
        feature_df["max_zoning_height_m"] = (
            pd.to_numeric(feature_df["max_zoning_height_ft"], errors="coerce") * 0.3048
        )
        feature_df["building_median_height_m"] = pd.to_numeric(
            feature_df["building_median_height_m"], errors="coerce"
        )

        # Avoid division by zero
        mask = (feature_df["max_zoning_height_m"] > 0) & (
            feature_df["building_median_height_m"] > 0
        )
        feature_df.loc[mask, "height_utilization_ratio"] = (
            feature_df.loc[mask, "building_median_height_m"]
            / feature_df.loc[mask, "max_zoning_height_m"]
        )
        features_added += 1
        print(f"  ✅ Added height utilization ratio")

    # 3. Elevation tiers
    if "ground_min_elevation_m" in feature_df.columns:
        feature_df["ground_min_elevation_m"] = pd.to_numeric(
            feature_df["ground_min_elevation_m"], errors="coerce"
        )

        try:
            # Fill NaN with median for quartile calculation
            elevation_filled = feature_df["ground_min_elevation_m"].fillna(
                feature_df["ground_min_elevation_m"].median()
            )

            feature_df["elevation_tier"] = pd.qcut(
                elevation_filled,
                q=5,
                labels=["Very Low", "Low", "Medium", "High", "Very High"],
                duplicates="drop",
            )

            elevation_dummies = pd.get_dummies(
                feature_df["elevation_tier"], prefix="elevation"
            )
            feature_df = pd.concat([feature_df, elevation_dummies], axis=1)
            features_added += len(elevation_dummies.columns)
            print(
                f"  ✅ Added elevation tier and {len(elevation_dummies.columns)} elevation features"
            )
        except Exception as e:
            print(f"  ⚠️  Could not create elevation tiers: {e}")

    # 4. Business density features
    density_cols = ["land_use_category", "building_id"]
    if all(col in feature_df.columns for col in density_cols):
        try:
            # Businesses per building
            building_counts = (
                feature_df.groupby("building_id")
                .size()
                .reset_index(name="biz_count_in_building")
            )
            feature_df = feature_df.merge(building_counts, on="building_id", how="left")

            # Average density by land use
            land_use_density = (
                feature_df.groupby("land_use_category")["biz_count_in_building"]
                .mean()
                .reset_index()
            )
            land_use_density.columns = ["land_use_category", "avg_biz_density"]

            feature_df = feature_df.merge(
                land_use_density, on="land_use_category", how="left"
            )

            # Relative density
            mask = feature_df["avg_biz_density"] > 0
            feature_df.loc[mask, "relative_biz_density"] = (
                feature_df.loc[mask, "biz_count_in_building"]
                / feature_df.loc[mask, "avg_biz_density"]
            )

            features_added += 3
            print(f"  ✅ Added business density features")
        except Exception as e:
            print(f"  ⚠️  Could not create density features: {e}")

    print(f"🎯 Total new features added: {features_added}")

    # Save engineered features - keeping exact same output filename as original
    output_path = f"{base_dir}/processed/final/sf_business_success_engineered.parquet"
    feature_df.to_parquet(output_path)
    logger.info(f"Saved engineered features to: {output_path}")
    print(f"💾 Saved engineered features to: {output_path}")

    return feature_df


def main():
    """Main execution function for land use integration"""
    # Setup logging and directories
    logger, base_dir, processed_dir = setup_logging_and_directories()

    logger.info("Starting Land Use Data Integration Pipeline")
    print("=" * 80)
    print("LAND USE DATA INTEGRATION PIPELINE")
    print("=" * 80)

    try:
        # Load datasets
        business_df, land_use_df = load_datasets(processed_dir, logger)

        # Convert to geographic data
        business_gdf, land_use_gdf = convert_to_geographic_data(
            business_df, land_use_df, logger
        )

        # Perform spatial join
        merged_gdf, land_use_matches, spatial_join_success = perform_spatial_join(
            business_gdf, land_use_gdf, logger
        )

        # Fallback to manual distance-based matching if spatial join failed
        if not spatial_join_success:
            merged_gdf, land_use_matches = fallback_distance_matching(
                business_df, land_use_df, logger
            )

        # Clean up merged dataframe
        merged_gdf = clean_merged_data(merged_gdf, logger)

        # Report results
        print(f"\n📊 Integration Results:")
        print(f"Original businesses: {len(business_df):,}")
        print(f"Final merged dataset: {len(merged_gdf):,}")

        if land_use_matches > 0:
            match_percent = (land_use_matches / len(business_df)) * 100
            print(
                f"Businesses with land use data: {land_use_matches:,} ({match_percent:.1f}%)"
            )
        else:
            print("⚠️  No land use matches found")

        # Save merged data
        merged_output_path = save_merged_data(merged_gdf, base_dir, logger)

        # Run analysis and feature engineering
        analysis_df, success_by_land_use = analyze_land_use_impact(merged_gdf, logger)
        engineered_df = engineer_land_use_features(merged_gdf, base_dir, logger)

        # Final summary
        print(f"\n🎉 Data Integration Complete!")
        print(f"📊 Dataset Evolution:")
        print(
            f"  Original business data: {business_df.shape[0]:,} × {business_df.shape[1]} columns"
        )
        print(
            f"  Land use data: {land_use_df.shape[0]:,} × {land_use_df.shape[1]} columns"
        )
        print(f"  Merged data: {merged_gdf.shape[0]:,} × {merged_gdf.shape[1]} columns")
        print(
            f"  Final engineered data: {engineered_df.shape[0]:,} × {engineered_df.shape[1]} columns"
        )

        if land_use_matches > 0:
            match_rate = (land_use_matches / len(business_df)) * 100
            print(f"🎯 Land use match rate: {match_rate:.1f}%")

        print(f"\n🔑 Ready for modeling with features:")
        print(f"  • Original business features (type, location, economic indicators)")
        print(f"  • Land use features (category, zoning, building height)")
        print(f"  • Engineered features (ratios, tiers, density metrics)")
        print(f"\n📁 Output files saved to: {processed_dir}/final/")

        # Summary Report
        print("\n" + "=" * 80)
        print("LAND USE INTEGRATION COMPLETE")
        print("=" * 80)

        print(f"\n📊 DATASETS CREATED:")
        print(f"   • sf_business_success_with_land_use.parquet")
        print(f"   • sf_business_success_engineered.parquet")

        print(f"\n📋 FINAL DATASET SUMMARY:")
        print(f"   • Records: {engineered_df.shape[0]:,}")
        print(f"   • Features: {engineered_df.shape[1]}")
        print(
            f"   • Land Use Match Rate: {match_rate:.1f}%"
            if land_use_matches > 0
            else "   • Land Use Match Rate: 0%"
        )

        if "success" in engineered_df.columns:
            print(f"   • Success Rate: {engineered_df['success'].mean():.2%}")

        logger.info("Land Use Data Integration Pipeline completed successfully!")

        return engineered_df

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"❌ Error in land use integration pipeline: {e}")
        raise


# Execute if run as main script
if __name__ == "__main__":
    print("Land Use Data Integration Pipeline Starting...")
    integrated_dataset = main()
