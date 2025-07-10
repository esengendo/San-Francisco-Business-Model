import os
import json
import logging
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ADD after existing imports:
import sys

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
from config import setup_logging, setup_directories

# Setup logging
logger = logging.getLogger(__name__)


def categorize_comprehensive_business(business_type):
    """
    Categorize business types into broader categories for analysis.

    Parameters:
    -----------
    business_type : str
        The specific business type from OSM

    Returns:
    --------
    str
        Broader business category
    """

    category_mapping = {
        # Food & Drink
        "restaurant": "Food & Dining",
        "cafe": "Food & Dining",
        "fast_food": "Food & Dining",
        "bar": "Food & Dining",
        "pub": "Food & Dining",
        "food_court": "Food & Dining",
        "ice_cream": "Food & Dining",
        # Retail & Shopping
        "shop": "Retail & Shopping",
        "supermarket": "Retail & Shopping",
        "convenience": "Retail & Shopping",
        "department_store": "Retail & Shopping",
        "mall": "Retail & Shopping",
        "clothes": "Retail & Shopping",
        "shoes": "Retail & Shopping",
        "electronics": "Retail & Shopping",
        "books": "Retail & Shopping",
        "jewelry": "Retail & Shopping",
        "hardware": "Retail & Shopping",
        "furniture": "Retail & Shopping",
        "bicycle": "Retail & Shopping",
        "car": "Automotive",
        # Financial Services
        "bank": "Financial Services",
        "atm": "Financial Services",
        "credit_union": "Financial Services",
        # Healthcare & Services
        "pharmacy": "Healthcare",
        "hospital": "Healthcare",
        "doctors": "Healthcare",
        "dentist": "Healthcare",
        "veterinary": "Healthcare",
        "optician": "Healthcare",
        # Personal Services
        "hairdresser": "Personal Services",
        "beauty_salon": "Personal Services",
        "laundry": "Personal Services",
        "dry_cleaning": "Personal Services",
        "tattoo": "Personal Services",
        # Entertainment & Culture
        "cinema": "Entertainment & Culture",
        "theatre": "Entertainment & Culture",
        "nightclub": "Entertainment & Culture",
        "casino": "Entertainment & Culture",
        "arts_centre": "Entertainment & Culture",
        "museum": "Entertainment & Culture",
        "gallery": "Entertainment & Culture",
        # Accommodation
        "hotel": "Accommodation",
        "hostel": "Accommodation",
        "guest_house": "Accommodation",
        "motel": "Accommodation",
        # Education
        "school": "Education",
        "college": "Education",
        "university": "Education",
        "library": "Education",
        "kindergarten": "Education",
        # Automotive
        "fuel": "Automotive",
        "car_wash": "Automotive",
        "car_repair": "Automotive",
        "parking": "Automotive",
        # Government & Public Services
        "courthouse": "Government & Public",
        "fire_station": "Government & Public",
        "police": "Government & Public",
        "post_office": "Government & Public",
        "townhall": "Government & Public",
        # Fitness & Recreation
        "gym": "Fitness & Recreation",
        "sports_centre": "Fitness & Recreation",
        "swimming_pool": "Fitness & Recreation",
    }

    return category_mapping.get(business_type, "Other")


def create_comprehensive_analysis(df):
    """
    Create comprehensive analysis of the business data.

    Parameters:
    -----------
    df : pd.DataFrame
        Combined business DataFrame

    Returns:
    --------
    dict
        Analysis results
    """

    analysis = {
        "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_businesses": len(df),
        "unique_business_types": df["business_type"].nunique(),
        "business_categories": df["business_category"].value_counts().to_dict(),
        "business_types": df["business_type"].value_counts().head(20).to_dict(),
        "businesses_with_phone": df["phone"].notna().sum(),
        "businesses_with_website": df["website"].notna().sum(),
        "businesses_with_hours": df["opening_hours"].notna().sum(),
        "businesses_with_address": df["address"].notna().sum(),
        "geographic_coverage": {
            "min_lat": (
                float(df["latitude"].min()) if "latitude" in df.columns else None
            ),
            "max_lat": (
                float(df["latitude"].max()) if "latitude" in df.columns else None
            ),
            "min_lon": (
                float(df["longitude"].min()) if "longitude" in df.columns else None
            ),
            "max_lon": (
                float(df["longitude"].max()) if "longitude" in df.columns else None
            ),
        },
    }

    # Add cuisine analysis for restaurants
    if "cuisine" in df.columns and df["cuisine"].notna().sum() > 0:
        analysis["top_cuisines"] = (
            df[df["cuisine"].notna()]["cuisine"].value_counts().head(10).to_dict()
        )

    return analysis


def save_category_datasets(df, processed_dir):
    """
    Save separate datasets for each business category.

    Parameters:
    -----------
    df : pd.DataFrame
        Combined business DataFrame
    processed_dir : str
        Directory to save category datasets
    """

    logger.info("Saving category-specific datasets...")

    for category in df["business_category"].unique():
        if pd.notna(category):
            category_df = df[df["business_category"] == category].copy()

            if not category_df.empty:
                # Create safe filename
                safe_category = category.lower().replace(" & ", "_").replace(" ", "_")
                category_file = f"{processed_dir}/sf_businesses_{safe_category}.parquet"

                category_df.to_parquet(category_file)
                logger.info(
                    f"Saved {len(category_df)} {category} businesses to {category_file}"
                )


def print_comprehensive_summary(analysis):
    """
    Print a comprehensive summary of the business data collection.

    Parameters:
    -----------
    analysis : dict
        Analysis results dictionary
    """

    print(f"\n{'='*60}")
    print("COMPREHENSIVE SF BUSINESS DATA COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Collection Date: {analysis['collection_date']}")
    print(f"Total Businesses: {analysis['total_businesses']:,}")
    print(f"Unique Business Types: {analysis['unique_business_types']}")

    print(f"\nTop Business Categories:")
    for category, count in list(analysis["business_categories"].items())[:10]:
        print(f"  • {category}: {count:,}")

    print(f"\nTop Business Types:")
    for biz_type, count in list(analysis["business_types"].items())[:10]:
        print(f"  • {biz_type}: {count:,}")

    print(f"\nData Completeness:")
    print(
        f"  • With Phone: {analysis['businesses_with_phone']:,} ({analysis['businesses_with_phone']/analysis['total_businesses']*100:.1f}%)"
    )
    print(
        f"  • With Website: {analysis['businesses_with_website']:,} ({analysis['businesses_with_website']/analysis['total_businesses']*100:.1f}%)"
    )
    print(
        f"  • With Hours: {analysis['businesses_with_hours']:,} ({analysis['businesses_with_hours']/analysis['total_businesses']*100:.1f}%)"
    )
    print(
        f"  • With Address: {analysis['businesses_with_address']:,} ({analysis['businesses_with_address']/analysis['total_businesses']*100:.1f}%)"
    )

    if "top_cuisines" in analysis:
        print(f"\nTop Cuisines:")
        for cuisine, count in list(analysis["top_cuisines"].items())[:5]:
            print(f"  • {cuisine}: {count}")

    print(f"{'='*60}")


def create_comprehensive_visualizations(sf_all_businesses):
    """
    Create comprehensive visualizations for the business data.

    Parameters:
    -----------
    sf_all_businesses : pd.DataFrame
        Combined business DataFrame
    """

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))

    # 1. Business Categories (Top 10)
    plt.subplot(2, 3, 1)
    top_categories = sf_all_businesses["business_category"].value_counts().head(10)
    top_categories.plot(kind="barh")
    plt.title("Top 10 Business Categories")
    plt.xlabel("Number of Businesses")

    # 2. Business Types (Top 15)
    plt.subplot(2, 3, 2)
    top_types = sf_all_businesses["business_type"].value_counts().head(15)
    top_types.plot(kind="bar")
    plt.title("Top 15 Business Types")
    plt.xlabel("Business Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")

    # 3. Geographic Distribution
    plt.subplot(2, 3, 3)
    if (
        "latitude" in sf_all_businesses.columns
        and "longitude" in sf_all_businesses.columns
    ):
        # Sample for better visualization
        sample_size = min(5000, len(sf_all_businesses))
        geo_sample = sf_all_businesses.sample(sample_size)

        plt.scatter(
            geo_sample["longitude"],
            geo_sample["latitude"],
            alpha=0.6,
            s=5,
            c=geo_sample["business_category"].astype("category").cat.codes,
            cmap="tab20",
        )
        plt.title("Business Geographic Distribution")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

    # 4. Data Completeness
    plt.subplot(2, 3, 4)
    completeness_data = {
        "Phone": sf_all_businesses["phone"].notna().sum(),
        "Website": sf_all_businesses["website"].notna().sum(),
        "Hours": sf_all_businesses["opening_hours"].notna().sum(),
        "Address": sf_all_businesses["address"].notna().sum(),
    }

    plt.bar(
        completeness_data.keys(),
        [v / len(sf_all_businesses) * 100 for v in completeness_data.values()],
    )
    plt.title("Data Completeness (%)")
    plt.ylabel("Percentage of Businesses")
    plt.ylim(0, 100)

    # 5. Top Cuisines (if available)
    plt.subplot(2, 3, 5)
    if (
        "cuisine" in sf_all_businesses.columns
        and sf_all_businesses["cuisine"].notna().sum() > 0
    ):
        top_cuisines = (
            sf_all_businesses[sf_all_businesses["cuisine"].notna()]["cuisine"]
            .value_counts()
            .head(10)
        )
        top_cuisines.plot(kind="barh")
        plt.title("Top 10 Cuisines")
        plt.xlabel("Number of Restaurants")
    else:
        plt.text(
            0.5,
            0.5,
            "No Cuisine Data\nAvailable",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.title("Cuisine Data")

    # 6. Category Distribution (Pie Chart)
    plt.subplot(2, 3, 6)
    category_counts = sf_all_businesses["business_category"].value_counts().head(8)
    plt.pie(category_counts.values, labels=category_counts.index, autopct="%1.1f%%")
    plt.title("Business Category Distribution")

    plt.tight_layout()
    plt.show()


def cleanup_old_osm_files(base_dir):
    """
    Clean up old OSM files from previous runs.

    Parameters:
    -----------
    base_dir : str
        Base directory of the project
    """
    logger.info("Cleaning up old OSM files...")
    # This function would be implemented based on the OSM helper functions
    # For now, just log that cleanup would happen here
    logger.info("OSM file cleanup completed")


def get_multiple_business_types(
    business_types, base_dir, cache_hours=24, rate_limit=True
):
    """
    Placeholder function for fetching multiple business types from OSM.
    This would typically call the OSM API functions defined elsewhere.

    Parameters:
    -----------
    business_types : list
        List of business types to fetch
    base_dir : str
        Base directory for caching
    cache_hours : int
        Hours to cache results
    rate_limit : bool
        Whether to apply rate limiting

    Returns:
    --------
    dict
        Dictionary of business type DataFrames
    """
    logger.info(f"Fetching {len(business_types)} business types from OSM API...")

    # Placeholder - in real implementation, this would call OSM API functions
    # For now, return empty dict to avoid dependency on OSM functions not defined in this cell
    sf_businesses_dict = {}

    for business_type in business_types:
        logger.info(f"  - Fetching {business_type} businesses...")
        # Placeholder for actual OSM API call
        # df = fetch_osm_business_type(business_type, base_dir, cache_hours, rate_limit)
        # sf_businesses_dict[business_type] = df

        # For demonstration, create empty DataFrame
        sf_businesses_dict[business_type] = pd.DataFrame()

    logger.warning("OSM API functions not available - returning empty results")
    return sf_businesses_dict


def save_comprehensive_business_data(sf_businesses_dict, base_dir):
    """
    Save comprehensive business data with detailed analysis and categorization.

    Parameters:
    -----------
    sf_businesses_dict : dict
        Dictionary of business type DataFrames
    base_dir : str
        Base directory of the project

    Returns:
    --------
    pd.DataFrame
        Combined comprehensive business DataFrame
    """

    processed_dir = f"{base_dir}/processed/sf_business"
    os.makedirs(processed_dir, exist_ok=True)

    # Combine all business types into one DataFrame
    all_businesses = []
    for business_type, df in sf_businesses_dict.items():
        if not df.empty:
            # Add business category grouping
            df["business_category"] = categorize_comprehensive_business(business_type)
            all_businesses.append(df)

    if all_businesses:
        sf_all_businesses = pd.concat(all_businesses, ignore_index=True)
        logger.info(f"Combined {len(sf_all_businesses)} total businesses")

        # Create comprehensive analysis
        analysis_results = create_comprehensive_analysis(sf_all_businesses)

        # Save main combined data
        combined_file = f"{processed_dir}/sf_comprehensive_businesses.parquet"
        sf_all_businesses.to_parquet(combined_file)
        logger.info(f"Saved comprehensive business data to: {combined_file}")

        # Save as CSV for easy viewing
        csv_file = f"{processed_dir}/sf_comprehensive_businesses.csv"
        sf_all_businesses.to_csv(csv_file, index=False)

        # Save analysis results
        analysis_file = f"{processed_dir}/sf_business_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis_results, f, indent=2, default=str)
        logger.info(f"Saved business analysis to: {analysis_file}")

        # Save category-specific datasets
        save_category_datasets(sf_all_businesses, processed_dir)

        # Print comprehensive summary
        print_comprehensive_summary(analysis_results)

        return sf_all_businesses

    else:
        logger.warning("No comprehensive business data to save")
        return pd.DataFrame()


def get_comprehensive_sf_businesses(base_dir, cache_hours=24, rate_limit=True):
    """
    Fetch comprehensive SF business data from OpenStreetMap with all business types.
    Uses the enhanced functions but with expanded business categories.

    Parameters:
    -----------
    base_dir : str
        Base directory of the project
    cache_hours : int, default=24
        Number of hours to cache results before refreshing
    rate_limit : bool, default=True
        Whether to apply rate limiting to respect API limits

    Returns:
    --------
    tuple
        (sf_businesses_dict, sf_all_businesses) - dictionary and combined DataFrame
    """

    logger.info("Starting comprehensive OSM business data collection...")

    # Comprehensive business types - organized by category
    comprehensive_business_types = [
        # Food & Drink
        "restaurant",
        "cafe",
        "fast_food",
        "bar",
        "pub",
        "food_court",
        "ice_cream",
        # Retail & Shopping
        "shop",  # General shops
        "supermarket",
        "convenience",
        "department_store",
        "mall",
        "clothes",
        "shoes",
        "electronics",
        "books",
        "jewelry",
        "hardware",
        "furniture",
        "bicycle",
        "car",
        # Financial Services
        "bank",
        "atm",
        "credit_union",
        # Healthcare & Services
        "pharmacy",
        "hospital",
        "doctors",
        "dentist",
        "veterinary",
        "optician",
        # Personal Services
        "hairdresser",
        "beauty_salon",
        "laundry",
        "dry_cleaning",
        "tattoo",
        # Entertainment & Culture
        "cinema",
        "theatre",
        "nightclub",
        "casino",
        "arts_centre",
        "museum",
        "gallery",
        # Accommodation
        "hotel",
        "hostel",
        "guest_house",
        "motel",
        # Education
        "school",
        "college",
        "university",
        "library",
        "kindergarten",
        # Automotive
        "fuel",
        "car_wash",
        "car_repair",
        "parking",
        # Government & Public
        "courthouse",
        "fire_station",
        "police",
        "post_office",
        "townhall",
        # Fitness & Recreation
        "gym",
        "sports_centre",
        "swimming_pool",
    ]

    logger.info(f"Fetching {len(comprehensive_business_types)} business types...")

    # Use the enhanced function with comprehensive business types
    sf_businesses_dict = get_multiple_business_types(
        business_types=comprehensive_business_types,
        base_dir=base_dir,
        cache_hours=cache_hours,
        rate_limit=rate_limit,
    )

    # Save comprehensive combined data
    sf_all_businesses = save_comprehensive_business_data(sf_businesses_dict, base_dir)

    return sf_businesses_dict, sf_all_businesses


def main_comprehensive_osm_collection(base_dir):
    """
    Main function to execute comprehensive OSM business data collection.

    Parameters:
    -----------
    base_dir : str
        Base directory of the project

    Returns:
    --------
    tuple
        (sf_businesses_dict, sf_all_businesses) - dictionary and combined DataFrame
    """

    try:
        # Clean up old files first
        cleanup_old_osm_files(base_dir)

        # Fetch comprehensive business data
        sf_businesses_dict, sf_all_businesses = get_comprehensive_sf_businesses(
            base_dir=base_dir, cache_hours=24, rate_limit=True
        )

        # Create visualizations if data exists
        if not sf_all_businesses.empty:
            logger.info("Creating comprehensive business visualizations...")
            create_comprehensive_visualizations(sf_all_businesses)

        logger.info("Comprehensive OSM business collection completed successfully")
        return sf_businesses_dict, sf_all_businesses

    except Exception as e:
        logger.error(f"Error in comprehensive OSM business collection: {e}")
        return {}, pd.DataFrame()


# REPLACE the main section with:
if __name__ == "__main__":
    logger = setup_logging()
    config = setup_directories()

    logger.info("Starting OSM Business Collector data processing")
    logger.info(f"Base directory: {config['base_dir']}")

    # Keep your existing main code
    logger.info(
        "Starting comprehensive SF business data collection from OpenStreetMap..."
    )

    # Execute the comprehensive collection
    sf_businesses_dict, sf_all_businesses = main_comprehensive_osm_collection(
        config["base_dir"]
    )

    # Display results if successful
    if not sf_all_businesses.empty:
        print(f"\nComprehensive OpenStreetMap Business Data Sample:")
        print(sf_all_businesses.head())

        print(f"\nDataFrame Info:")
        print(f"Shape: {sf_all_businesses.shape}")
        print(f"Columns: {list(sf_all_businesses.columns)}")

    else:
        logger.warning("No comprehensive business data was collected")
        print("Note: This module requires OSM API helper functions to be available.")
        print("The structure is in place for when OSM integration is added.")

    logger.info("Comprehensive OSM business data collection process completed")
