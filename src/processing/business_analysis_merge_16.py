# business_analysis_pipeline_16.py
# Comprehensive Business Economic Analysis and Modeling Pipeline
# Combines all merging strategies, creates multiple datasets, and performs deep analysis
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm


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
    #base_dir = "/Users/baboo/Documents/San Francisco Business Model"
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
    ]
    for source in data_sources:
        os.makedirs(f"{raw_data_dir}/{source}", exist_ok=True)
        os.makedirs(f"{processed_dir}/{source}", exist_ok=True)

    # Specific directories
    business_processed_dir = f"{processed_dir}/sf_business"
    economic_raw_dir = f"{raw_data_dir}/economic"
    OUTPUT_DIR = f"{processed_dir}/final"

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set pandas display options
    pd.set_option("display.max_columns", None)

    return logger, business_processed_dir, economic_raw_dir, OUTPUT_DIR


def load_datasets(business_processed_dir, economic_raw_dir, logger):
    """Load the main business and economic datasets"""
    logger.info("Loading datasets...")

    # Load main SF business registrations dataset
    business_path = f"{business_processed_dir}/enriched_registered_businesses.parquet"
    economic_path = f"{economic_raw_dir}/fred_economic_indicators.parquet"

    if not os.path.exists(business_path):
        logger.error(f"Business dataset not found at: {business_path}")
        raise FileNotFoundError(f"Business dataset not found at: {business_path}")

    if not os.path.exists(economic_path):
        logger.error(f"Economic dataset not found at: {economic_path}")
        raise FileNotFoundError(f"Economic dataset not found at: {economic_path}")

    sf_businesses = pd.read_parquet(business_path)
    sf_econ_indicators = pd.read_parquet(economic_path)

    # Brief overview of datasets
    logger.info(
        f"Business dataset: {sf_businesses.shape[0]:,} rows, {sf_businesses.shape[1]} columns"
    )
    logger.info(
        f"Economic dataset: {sf_econ_indicators.shape[0]:,} rows, {sf_econ_indicators.shape[1]} columns"
    )

    print(
        f"Business dataset: {sf_businesses.shape[0]:,} rows, {sf_businesses.shape[1]} columns"
    )
    print(
        f"Economic dataset: {sf_econ_indicators.shape[0]:,} rows, {sf_econ_indicators.shape[1]} columns"
    )

    print("\nSF Business Registration Sample:")
    print(sf_businesses.head(3))

    print("\nSF Economic Indicators Sample:")
    print(sf_econ_indicators.head(3))

    return sf_businesses, sf_econ_indicators


def prepare_data_for_merging(sf_businesses, sf_econ_indicators, logger):
    """Prepare both datasets for merging"""
    logger.info("Preparing data for merging...")

    # Make copies to preserve original data
    biz_data = sf_businesses.copy()
    econ_data = sf_econ_indicators.copy()

    # Convert date columns to datetime in business dataset
    date_columns = [
        "dba_start_date",
        "dba_end_date",
        "location_start_date",
        "location_end_date",
        "Dba_start_date",
        "Dba_end_date",
        "Location_start_date",
        "Location_end_date",
    ]

    for col in date_columns:
        if col in biz_data.columns:
            biz_data[col] = pd.to_datetime(biz_data[col], errors="coerce")

    # Create a unified start_date column (using lowercase variant as primary)
    biz_data["start_date"] = biz_data["dba_start_date"].fillna(
        biz_data["Dba_start_date"]
    )

    # Define business success metric
    # A business is considered successful if:
    # 1. It is still open, OR
    # 2. It was open for at least 5 years before closing
    biz_data["success"] = (
        (biz_data["is_open"] == True)
        | ((biz_data["is_open"] == False) & (biz_data["business_age_years"] >= 5))
    ).astype(int)

    # Extract time period information for merging
    biz_data["year_quarter"] = pd.to_datetime(biz_data["start_date"]).dt.to_period("Q")
    biz_data["year_quarter_str"] = biz_data["year_quarter"].astype(str)
    biz_data["start_year"] = pd.to_datetime(biz_data["start_date"]).dt.year

    # Prepare economic dataset
    econ_data["date"] = pd.to_datetime(econ_data["date"])
    econ_data["year_quarter"] = econ_data["date"].dt.to_period("Q")
    econ_data["year_quarter_str"] = econ_data["year_quarter"].astype(str)
    econ_data["year"] = econ_data["date"].dt.year

    # Fill missing values in economic data
    numeric_columns = econ_data.select_dtypes(include=[np.number]).columns
    econ_data[numeric_columns] = econ_data[numeric_columns].fillna(method="ffill")

    # Display the success rate distribution
    success_rate = biz_data["success"].mean()
    logger.info(f"Overall business success rate: {success_rate:.2%}")
    print(f"\nOverall business success rate: {success_rate:.2%}")

    return biz_data, econ_data


def strategy_1_time_based_context(biz_data, econ_data, OUTPUT_DIR, logger):
    """Strategy 1: Time-Based Business Opening Context"""
    logger.info("Executing Strategy 1: Time-Based Business Opening Context")
    print("\n--- Strategy 1: Time-Based Business Opening Context ---")

    # Sort both datasets by date for merge_asof
    biz_data_sorted = biz_data.sort_values("start_date")
    econ_data_sorted = econ_data.sort_values("date")

    # Merge each business with the nearest economic data point
    biz_with_econ_context = pd.merge_asof(
        biz_data_sorted,
        econ_data_sorted,
        left_on="start_date",
        right_on="date",
        direction="nearest",
    )

    print(
        f"Merged dataset shape: {biz_with_econ_context.shape[0]:,} rows, {biz_with_econ_context.shape[1]} columns"
    )

    # Save strategy 1 result
    output_path = f"{OUTPUT_DIR}/business_economic_at_opening.parquet"
    biz_with_econ_context.to_parquet(output_path)
    logger.info(f"Saved time-based merged dataset to: {output_path}")
    print(f"Saved time-based merged dataset to: {output_path}")

    # Display sample of merged data
    print("\nSample of businesses with economic context at opening time:")
    sample_cols = ["ttxid", "business_industry", "start_date", "success"]
    econ_cols = ["sf_unemployment_rate", "sf_house_price_index", "sf_gdp"]
    available_cols = [
        col for col in sample_cols + econ_cols if col in biz_with_econ_context.columns
    ]
    print(biz_with_econ_context[available_cols].head())

    return biz_with_econ_context


def prepare_economic_aggregations(econ_data, logger):
    """Prepare economic data aggregations for various analyses"""
    logger.info("Preparing economic data aggregations...")

    # Yearly economic indicators (for longer-term trends)
    yearly_econ = (
        econ_data.groupby("year")
        .agg(
            {
                "sf_unemployment_rate": "mean",
                "sf_house_price_index": "mean",
                "sf_gdp": "mean",
                "sf_real_gdp": "mean",
                "sf_per_capita_income": "mean",
                "sf_median_household_income": "mean",
                "sf_active_listings": "mean",
                "us_cpi": "mean",
            }
        )
        .reset_index()
    )

    # Quarterly economic indicators (for finer temporal resolution)
    quarterly_econ = (
        econ_data.groupby("year_quarter_str")
        .agg(
            {
                "sf_unemployment_rate": "mean",
                "sf_house_price_index": "mean",
                "sf_gdp": "mean",
                "sf_real_gdp": "mean",
                "sf_per_capita_income": "mean",
                "sf_median_household_income": "mean",
                "sf_active_listings": "mean",
                "us_cpi": "mean",
            }
        )
        .reset_index()
    )

    return yearly_econ, quarterly_econ


def strategy_2_business_type_success(biz_data, yearly_econ, OUTPUT_DIR, logger):
    """Strategy 2: Business Type Success by Economic Conditions"""
    logger.info("Executing Strategy 2: Business Type Success by Economic Conditions")
    print("\n--- Strategy 2: Business Type Success by Economic Conditions ---")

    # Determine business type column
    if "business_industry" in biz_data.columns:
        business_type_col = "business_industry"
    else:
        business_type_col = "naics_code_description"

    # Calculate success metrics by business type and year
    business_type_success = (
        biz_data.groupby([business_type_col, "start_year"])
        .agg({"success": "mean", "ttxid": "count"})  # Business count
        .reset_index()
    )
    business_type_success.columns = [
        business_type_col,
        "year",
        "success_rate",
        "business_count",
    ]

    # Merge with yearly economic data
    industry_econ_performance = pd.merge(business_type_success, yearly_econ, on="year")

    print(
        f"Business type by economic conditions dataset shape: {industry_econ_performance.shape[0]:,} rows, {industry_econ_performance.shape[1]} columns"
    )

    # Save strategy 2 result
    output_path = f"{OUTPUT_DIR}/business_type_economic_success.parquet"
    industry_econ_performance.to_parquet(output_path)
    logger.info(f"Saved industry economic performance dataset to: {output_path}")
    print(f"Saved industry economic performance dataset to: {output_path}")

    print("\nSample of business types performance by economic conditions:")
    print(industry_econ_performance.head())

    return industry_econ_performance, business_type_col


def strategy_3_neighborhood_context(biz_data, quarterly_econ, OUTPUT_DIR, logger):
    """Strategy 3: Neighborhood Economic Context"""
    logger.info("Executing Strategy 3: Neighborhood Economic Context")
    print("\n--- Strategy 3: Neighborhood Economic Context ---")

    # Determine neighborhood column
    if "supervisor_district" in biz_data.columns:
        neighborhood_col = "supervisor_district"
    elif "neighborhoods_analysis_boundaries" in biz_data.columns:
        neighborhood_col = "neighborhoods_analysis_boundaries"
    else:
        neighborhood_col = None
        logger.warning(
            "No neighborhood column found, skipping neighborhood context merging."
        )
        print("Warning: No neighborhood column found. Skipping neighborhood analysis.")
        return None, None

    # Group by district and quarter
    district_success = (
        biz_data.groupby([neighborhood_col, "year_quarter_str"])
        .agg({"success": "mean", "ttxid": "count"})  # Business count
        .reset_index()
    )
    district_success.columns = [
        neighborhood_col,
        "year_quarter_str",
        "success_rate",
        "business_count",
    ]

    # Merge district success with economic indicators
    district_econ_performance = pd.merge(
        district_success, quarterly_econ, on="year_quarter_str"
    )

    print(
        f"Neighborhood economic context dataset shape: {district_econ_performance.shape[0]:,} rows, {district_econ_performance.shape[1]} columns"
    )

    # Save strategy 3 result
    output_path = f"{OUTPUT_DIR}/district_economic_success.parquet"
    district_econ_performance.to_parquet(output_path)
    logger.info(f"Saved district economic performance dataset to: {output_path}")
    print(f"Saved district economic performance dataset to: {output_path}")

    # Also save as neighborhood_economic_success.parquet for compatibility
    output_path_alt = f"{OUTPUT_DIR}/neighborhood_economic_success.parquet"
    district_econ_performance.to_parquet(output_path_alt)
    logger.info(
        f"Saved neighborhood economic performance dataset to: {output_path_alt}"
    )

    print("\nSample of district performance by economic conditions:")
    print(district_econ_performance.head())

    return district_econ_performance, neighborhood_col


def combined_business_neighborhood_analysis(
    biz_data, yearly_econ, business_type_col, neighborhood_col, OUTPUT_DIR, logger
):
    """Combined Business Type and Neighborhood Analysis"""
    if not neighborhood_col:
        logger.info("Skipping combined analysis - no neighborhood column available")
        return None

    logger.info("Executing Combined Business Type and Neighborhood Analysis")
    print("\n--- Combined Business Type and Neighborhood Analysis ---")

    # Calculate success metrics by business type, neighborhood, and year
    combined_success = (
        biz_data.groupby([business_type_col, neighborhood_col, "start_year"])
        .agg({"success": "mean", "ttxid": "count"})  # Business count
        .reset_index()
    )
    combined_success.columns = [
        business_type_col,
        neighborhood_col,
        "year",
        "success_rate",
        "business_count",
    ]

    # Merge with yearly economic data
    combined_econ_performance = pd.merge(combined_success, yearly_econ, on="year")

    # Filter to combinations with sufficient data (at least 5 businesses)
    combined_econ_performance = combined_econ_performance[
        combined_econ_performance["business_count"] >= 5
    ]

    print(
        f"Combined business type and neighborhood dataset shape: {combined_econ_performance.shape[0]:,} rows, {combined_econ_performance.shape[1]} columns"
    )

    # Save combined analysis
    output_path = f"{OUTPUT_DIR}/business_type_neighborhood_economic_success.parquet"
    combined_econ_performance.to_parquet(output_path)
    logger.info(
        f"Saved combined business type and neighborhood dataset to: {output_path}"
    )
    print(f"Saved combined business type and neighborhood dataset to: {output_path}")

    return combined_econ_performance


def add_economic_features(row, econ_df):
    """
    For a given business, extract economic indicators at start date
    and calculate trajectory features.
    """
    # Skip if start date is missing
    if pd.isna(row["start_date"]):
        return pd.Series(
            {
                "unemployment_at_opening": np.nan,
                "house_price_idx_at_opening": np.nan,
                "gdp_at_opening": np.nan,
                "cpi_at_opening": np.nan,
                "unemployment_1yr_trend": np.nan,
                "house_price_1yr_trend": np.nan,
            }
        )

    start_date = row["start_date"]

    # Get economic data at or just before start date
    opening_econ = (
        econ_df[econ_df["date"] <= start_date]
        .sort_values("date")
        .iloc[-1:]
        .reset_index(drop=True)
    )

    if len(opening_econ) == 0:
        return pd.Series(
            {
                "unemployment_at_opening": np.nan,
                "house_price_idx_at_opening": np.nan,
                "gdp_at_opening": np.nan,
                "cpi_at_opening": np.nan,
                "unemployment_1yr_trend": np.nan,
                "house_price_1yr_trend": np.nan,
            }
        )

    # Get economic data 1 year before opening to calculate trends
    year_before = econ_df[
        (econ_df["date"] <= start_date)
        & (econ_df["date"] >= start_date - pd.DateOffset(years=1))
    ]

    # Calculate 1-year trends for key indicators
    unemployment_trend = np.nan
    house_price_trend = np.nan

    if len(year_before) > 1:
        if "sf_unemployment_rate" in year_before.columns:
            unemployment_series = year_before["sf_unemployment_rate"].dropna()
            if len(unemployment_series) > 1:
                unemployment_trend = (
                    unemployment_series.iloc[-1] / unemployment_series.iloc[0]
                ) - 1

        if "sf_house_price_index" in year_before.columns:
            house_price_series = year_before["sf_house_price_index"].dropna()
            if len(house_price_series) > 1:
                house_price_trend = (
                    house_price_series.iloc[-1] / house_price_series.iloc[0]
                ) - 1

    # Create feature dictionary
    return pd.Series(
        {
            "unemployment_at_opening": (
                opening_econ["sf_unemployment_rate"].iloc[0]
                if "sf_unemployment_rate" in opening_econ
                else np.nan
            ),
            "house_price_idx_at_opening": (
                opening_econ["sf_house_price_index"].iloc[0]
                if "sf_house_price_index" in opening_econ
                else np.nan
            ),
            "gdp_at_opening": (
                opening_econ["sf_gdp"].iloc[0] if "sf_gdp" in opening_econ else np.nan
            ),
            "cpi_at_opening": (
                opening_econ["us_cpi"].iloc[0] if "us_cpi" in opening_econ else np.nan
            ),
            "unemployment_1yr_trend": unemployment_trend,
            "house_price_1yr_trend": house_price_trend,
        }
    )


def strategy_4_economic_trajectory_features(biz_data, econ_data, OUTPUT_DIR, logger):
    """Strategy 4: Economic Trajectory Feature Engineering"""
    logger.info("Executing Strategy 4: Economic Trajectory Feature Engineering")
    print("\n--- Strategy 4: Economic Trajectory Feature Engineering ---")

    # Process in batches to handle large dataframe
    batch_size = 10000
    businesses_with_econ_features = pd.DataFrame()

    print(
        "Adding economic trajectory features to businesses (this may take some time)..."
    )

    # Using tqdm to track progress
    for i in tqdm(range(0, len(biz_data), batch_size), desc="Processing batches"):
        batch = biz_data.iloc[i : i + batch_size].copy()
        economic_features = batch.apply(
            lambda row: add_economic_features(row, econ_data), axis=1
        )
        batch = pd.concat([batch, economic_features], axis=1)
        businesses_with_econ_features = pd.concat(
            [businesses_with_econ_features, batch]
        )

    print(
        f"Economic trajectory features dataset shape: {businesses_with_econ_features.shape[0]:,} rows, {businesses_with_econ_features.shape[1]} columns"
    )

    # Save strategy 4 result
    output_path = f"{OUTPUT_DIR}/business_economic_trajectories.parquet"
    businesses_with_econ_features.to_parquet(output_path)
    logger.info(f"Saved businesses with economic trajectory features to: {output_path}")
    print(f"Saved businesses with economic trajectory features to: {output_path}")

    print("\nSample of businesses with economic trajectory features:")
    sample_cols = [
        "ttxid",
        "business_industry",
        "start_date",
        "success",
        "unemployment_at_opening",
        "house_price_idx_at_opening",
        "unemployment_1yr_trend",
        "house_price_1yr_trend",
    ]
    available_cols = [
        col for col in sample_cols if col in businesses_with_econ_features.columns
    ]
    print(businesses_with_econ_features[available_cols].head())

    return businesses_with_econ_features


def create_comprehensive_final_dataset(
    biz_data, econ_data, business_type_col, neighborhood_col, OUTPUT_DIR, logger
):
    """Create Comprehensive Final Dataset for Modeling"""
    logger.info(
        "Creating Comprehensive Final Dataset for Business Success Prediction Modeling"
    )
    print(
        "\n--- Creating Comprehensive Final Dataset for Business Success Prediction Modeling ---"
    )

    # Start with original business data
    sf_business_success_model_data = biz_data.copy()

    # Add economic indicators at time of business opening using merge_asof
    biz_data_sorted = sf_business_success_model_data.sort_values("start_date")
    econ_data_sorted = econ_data.sort_values("date")

    sf_business_success_model_data = pd.merge_asof(
        biz_data_sorted,
        econ_data_sorted,
        left_on="start_date",
        right_on="date",
        direction="nearest",
    )

    # Calculate success metrics for each business type
    business_type_stats = (
        sf_business_success_model_data.groupby(business_type_col)
        .agg({"success": "mean", "ttxid": "count"})
        .reset_index()
    )
    business_type_stats.columns = [
        business_type_col,
        "industry_success_rate",
        "industry_business_count",
    ]

    # Add industry-level success metrics to each business
    sf_business_success_model_data = pd.merge(
        sf_business_success_model_data,
        business_type_stats,
        on=business_type_col,
        how="left",
    )

    # Add neighborhood-level features if available
    if neighborhood_col:
        # Calculate success metrics for each neighborhood
        neighborhood_stats = (
            sf_business_success_model_data.groupby(neighborhood_col)
            .agg({"success": "mean", "ttxid": "count"})
            .reset_index()
        )
        neighborhood_stats.columns = [
            neighborhood_col,
            "neighborhood_success_rate",
            "neighborhood_business_count",
        ]

        # Add neighborhood-level success metrics to each business
        sf_business_success_model_data = pd.merge(
            sf_business_success_model_data,
            neighborhood_stats,
            on=neighborhood_col,
            how="left",
        )

        # Calculate business density features
        neighborhood_counts = (
            sf_business_success_model_data.groupby(neighborhood_col)
            .size()
            .reset_index(name="businesses_in_neighborhood")
        )
        neighborhood_type_counts = (
            sf_business_success_model_data.groupby(
                [neighborhood_col, business_type_col]
            )
            .size()
            .reset_index(name="similar_businesses_count")
        )

        # Add counts to the final dataframe
        sf_business_success_model_data = pd.merge(
            sf_business_success_model_data,
            neighborhood_counts,
            on=neighborhood_col,
            how="left",
        )
        sf_business_success_model_data = pd.merge(
            sf_business_success_model_data,
            neighborhood_type_counts,
            on=[neighborhood_col, business_type_col],
            how="left",
        )

        # Calculate concentration ratio
        sf_business_success_model_data["business_type_concentration"] = (
            sf_business_success_model_data["similar_businesses_count"]
            / sf_business_success_model_data["businesses_in_neighborhood"]
        )

    # Add features for economic conditions relative to historical averages
    for col in ["sf_unemployment_rate", "sf_house_price_index", "sf_gdp"]:
        if col in sf_business_success_model_data.columns:
            # Calculate historical average and std dev
            hist_avg = econ_data[col].mean()
            hist_std = econ_data[col].std()

            # Add z-score feature
            if hist_std > 0:
                sf_business_success_model_data[f"{col}_zscore"] = (
                    sf_business_success_model_data[col] - hist_avg
                ) / hist_std

    print(
        f"Comprehensive final modeling dataset shape: {sf_business_success_model_data.shape[0]:,} rows, {sf_business_success_model_data.shape[1]} columns"
    )

    # Save the comprehensive final merged dataset
    output_path = f"{OUTPUT_DIR}/sf_business_success_modeling_data.parquet"
    print(
        f"Saving comprehensive final modeling dataset with {sf_business_success_model_data.shape[1]} features for {sf_business_success_model_data.shape[0]:,} businesses..."
    )
    sf_business_success_model_data.to_parquet(output_path)
    logger.info(f"Comprehensive final dataset saved to: {output_path}")
    print(f"Comprehensive final dataset saved to: {output_path}")

    print("\nSample of comprehensive final business success prediction dataset:")
    print(sf_business_success_model_data.head())

    return sf_business_success_model_data


def analyze_business_success_patterns(
    industry_econ_performance,
    neighborhood_econ_performance,
    business_type_col,
    neighborhood_col,
    OUTPUT_DIR,
    logger,
):
    """Comprehensive Analysis of Business Success Patterns"""
    logger.info("Performing comprehensive analysis of business success patterns")
    print("\n--- Comprehensive Analysis of Business Success Patterns ---")

    # 1. Top and Bottom Performing Business Types
    print("\n=== Business Type Performance Analysis ===")

    # Get business types with at least 50 instances
    business_types_count = industry_econ_performance.groupby(business_type_col)[
        "business_count"
    ].sum()
    valid_business_types = business_types_count[business_types_count >= 50].index

    # Calculate average success rate by business type
    business_type_avg_success = (
        industry_econ_performance[
            industry_econ_performance[business_type_col].isin(valid_business_types)
        ]
        .groupby(business_type_col)["success_rate"]
        .mean()
        .sort_values(ascending=False)
    )

    print("Top 5 Business Types by Success Rate:")
    print(business_type_avg_success.head(5))

    print("\nBottom 5 Business Types by Success Rate:")
    print(business_type_avg_success.tail(5))

    # 2. Business Success vs. Economic Indicators
    print("\n=== Economic Indicators Correlation Analysis ===")

    # Calculate correlation between success rate and economic indicators
    econ_cols = [
        "sf_unemployment_rate",
        "sf_house_price_index",
        "sf_gdp",
        "sf_per_capita_income",
        "us_cpi",
    ]
    available_econ_cols = [
        col for col in econ_cols if col in industry_econ_performance.columns
    ]

    success_econ_corr = (
        industry_econ_performance[["success_rate"] + available_econ_cols]
        .corr()["success_rate"]
        .sort_values(ascending=False)
    )

    print("Correlation between Business Success and Economic Indicators:")
    print(success_econ_corr)

    # 3. Business Success by Neighborhood (if available)
    if neighborhood_econ_performance is not None and neighborhood_col:
        print(f"\n=== Neighborhood Performance Analysis ===")

        # Calculate average success rate by neighborhood
        neighborhood_avg_success = (
            neighborhood_econ_performance.groupby(neighborhood_col)["success_rate"]
            .mean()
            .sort_values(ascending=False)
        )

        # Get neighborhoods with sufficient data
        valid_neighborhoods = neighborhood_econ_performance.groupby(neighborhood_col)[
            "business_count"
        ].sum()
        valid_neighborhoods = valid_neighborhoods[valid_neighborhoods >= 50].index
        neighborhood_avg_success = neighborhood_avg_success[
            neighborhood_avg_success.index.isin(valid_neighborhoods)
        ]

        print(f"Business Success Rates by {neighborhood_col}:")
        print(neighborhood_avg_success)

    # 4. Business Type Success Rate Over Time
    print("\n=== Business Type Success Over Time Analysis ===")

    # Select top 5 most common business types
    top_business_types = (
        industry_econ_performance.groupby(business_type_col)["business_count"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    # Filter data to these business types
    top_types_data = industry_econ_performance[
        industry_econ_performance[business_type_col].isin(top_business_types)
    ]

    # Log success rate trends for top business types
    print("Success Rate Trends for Top Business Types:")
    for business_type in top_business_types:
        data = top_types_data[top_types_data[business_type_col] == business_type]
        avg_success = data["success_rate"].mean()
        print(f"{business_type}: Average Success Rate = {avg_success:.2%}")


def create_preliminary_analysis_visualizations(
    sf_business_success_model_data, OUTPUT_DIR, logger
):
    """Create preliminary analysis of merged data (without visualizations for CI/CD)"""
    logger.info("Creating preliminary analysis of business success factors")
    print("\n=== Preliminary Analysis of Business Success Factors ===")

    # Success rate overall
    overall_success_rate = sf_business_success_model_data["success"].mean()
    print(f"Overall business success rate: {overall_success_rate:.2%}")

    # Analyze success rate by economic condition (unemployment)
    if "unemployment_at_opening" in sf_business_success_model_data.columns:
        # Create unemployment rate bins
        unemployment_data = sf_business_success_model_data[
            "unemployment_at_opening"
        ].dropna()
        if len(unemployment_data) > 0:
            sf_business_success_model_data["unemployment_bin"] = pd.qcut(
                sf_business_success_model_data["unemployment_at_opening"].dropna(),
                4,
                labels=["Low", "Medium-Low", "Medium-High", "High"],
                duplicates="drop",
            )

            # Calculate success rate by unemployment bin
            unemployment_success = sf_business_success_model_data.groupby(
                "unemployment_bin"
            )["success"].agg(["mean", "count"])
            unemployment_success.columns = ["success_rate", "business_count"]

            print("\nBusiness Success Rate by Unemployment Level at Opening:")
            for idx, row in unemployment_success.iterrows():
                print(f"{idx}: {row['success_rate']:.2%} (n={row['business_count']:,})")

    # Analyze success rate by house price trend
    if "house_price_1yr_trend" in sf_business_success_model_data.columns:
        # Remove NaN values and outliers
        trend_data = sf_business_success_model_data[
            sf_business_success_model_data["house_price_1yr_trend"].notna()
        ]
        if len(trend_data) > 0:
            q1 = trend_data["house_price_1yr_trend"].quantile(0.01)
            q3 = trend_data["house_price_1yr_trend"].quantile(0.99)
            trend_data = trend_data[
                (trend_data["house_price_1yr_trend"] >= q1)
                & (trend_data["house_price_1yr_trend"] <= q3)
            ]

            if len(trend_data) > 0:
                # Create trend bins
                trend_data["price_trend_bin"] = pd.qcut(
                    trend_data["house_price_1yr_trend"],
                    4,
                    labels=[
                        "Declining",
                        "Slow Growth",
                        "Moderate Growth",
                        "Rapid Growth",
                    ],
                    duplicates="drop",
                )

                # Calculate success rate by trend bin
                trend_success = trend_data.groupby("price_trend_bin")["success"].agg(
                    ["mean", "count"]
                )
                trend_success.columns = ["success_rate", "business_count"]

                print("\nBusiness Success Rate by House Price Trend Before Opening:")
                for idx, row in trend_success.iterrows():
                    print(
                        f"{idx}: {row['success_rate']:.2%} (n={row['business_count']:,})"
                    )

    # If business type data is available, analyze success by industry
    if "business_industry" in sf_business_success_model_data.columns:
        # Get top industries by count
        top_industries = (
            sf_business_success_model_data["business_industry"]
            .value_counts()
            .head(10)
            .index
        )

        # Calculate success rate for top industries
        industry_success = (
            sf_business_success_model_data[
                sf_business_success_model_data["business_industry"].isin(top_industries)
            ]
            .groupby("business_industry")["success"]
            .agg(["mean", "count"])
        )
        industry_success.columns = ["success_rate", "business_count"]
        industry_success = industry_success.sort_values("success_rate", ascending=False)

        print("\nBusiness Success Rate by Industry (Top 10 by Count):")
        for idx, row in industry_success.iterrows():
            print(f"{idx}: {row['success_rate']:.2%} (n={row['business_count']:,})")


def main():
    """Main execution function for comprehensive business economic analysis"""
    # Setup logging and directories
    logger, business_processed_dir, economic_raw_dir, OUTPUT_DIR = (
        setup_logging_and_directories()
    )

    logger.info("Starting comprehensive business economic analysis pipeline")
    print("=" * 80)
    print("COMPREHENSIVE BUSINESS ECONOMIC ANALYSIS PIPELINE")
    print("=" * 80)

    # Load datasets
    sf_businesses, sf_econ_indicators = load_datasets(
        business_processed_dir, economic_raw_dir, logger
    )

    # Prepare data for merging
    biz_data, econ_data = prepare_data_for_merging(
        sf_businesses, sf_econ_indicators, logger
    )

    # Prepare economic aggregations
    yearly_econ, quarterly_econ = prepare_economic_aggregations(econ_data, logger)

    # Execute Strategy 1: Time-Based Business Opening Context
    strategy_1_result = strategy_1_time_based_context(
        biz_data, econ_data, OUTPUT_DIR, logger
    )

    # Execute Strategy 2: Business Type Success by Economic Conditions
    industry_econ_performance, business_type_col = strategy_2_business_type_success(
        biz_data, yearly_econ, OUTPUT_DIR, logger
    )

    # Execute Strategy 3: Neighborhood Economic Context
    neighborhood_econ_performance, neighborhood_col = strategy_3_neighborhood_context(
        biz_data, quarterly_econ, OUTPUT_DIR, logger
    )

    # Execute Combined Business Type and Neighborhood Analysis
    combined_econ_performance = combined_business_neighborhood_analysis(
        biz_data, yearly_econ, business_type_col, neighborhood_col, OUTPUT_DIR, logger
    )

    # Execute Strategy 4: Economic Trajectory Feature Engineering
    strategy_4_result = strategy_4_economic_trajectory_features(
        biz_data, econ_data, OUTPUT_DIR, logger
    )

    # Create Comprehensive Final Dataset for Modeling
    final_dataset = create_comprehensive_final_dataset(
        biz_data, econ_data, business_type_col, neighborhood_col, OUTPUT_DIR, logger
    )

    # Perform Comprehensive Analysis of Business Success Patterns
    analyze_business_success_patterns(
        industry_econ_performance,
        neighborhood_econ_performance,
        business_type_col,
        neighborhood_col,
        OUTPUT_DIR,
        logger,
    )

    # Create Preliminary Analysis Visualizations
    create_preliminary_analysis_visualizations(final_dataset, OUTPUT_DIR, logger)

    # Summary Report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 80)

    print("\n📊 DATASETS CREATED:")
    print(f"   • business_economic_at_opening.parquet")
    print(f"   • business_type_economic_success.parquet")
    print(f"   • district_economic_success.parquet")
    print(f"   • neighborhood_economic_success.parquet")
    if combined_econ_performance is not None:
        print(f"   • business_type_neighborhood_economic_success.parquet")
    print(f"   • business_economic_trajectories.parquet")
    print(f"   • sf_business_success_modeling_data.parquet")

    print(f"\n📋 FINAL MODELING DATASET:")
    print(f"   • Records: {final_dataset.shape[0]:,}")
    print(f"   • Features: {final_dataset.shape[1]}")
    print(f"   • Success Rate: {final_dataset['success'].mean():.2%}")

    logger.info(
        "Comprehensive business economic analysis pipeline completed successfully!"
    )

    return {
        "strategy_1": strategy_1_result,
        "industry_performance": industry_econ_performance,
        "neighborhood_performance": neighborhood_econ_performance,
        "combined_performance": combined_econ_performance,
        "trajectory_features": strategy_4_result,
        "final_dataset": final_dataset,
    }


# Execute if run as main script
if __name__ == "__main__":
    print("Comprehensive Business Economic Analysis Pipeline Starting...")
    results = main()
