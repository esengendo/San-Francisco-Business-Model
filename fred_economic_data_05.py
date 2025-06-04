import os
import logging
import pandas as pd
from fredapi import Fred
import sys

# Add project root to path if running directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

# Import unified config - same functions as your _02 script
from config import setup_logging, setup_directories
from helper_functions_03 import save_to_parquet
from api_keys_validation_01 import validate_api_keys


def fetch_fred_economic_data(
    raw_data_dir, processed_dir, start_date, end_date, fred_api_key=None
):
    """
    Fetch economic indicators from FRED API for San Francisco region.
    FRED provides excellent historical data going back decades.
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.fred_economic_data_05")
    logger.info("Fetching FRED Economic Data...")

    # Use the API key provided or load from environment
    if fred_api_key is None:
        fred_api_key, _ = validate_api_keys()

    # Use the API key loaded from environment variables
    fred = Fred(api_key=fred_api_key)

    # Series IDs for San Francisco MSA economic indicators
    # These series have long histories, typically 10+ years
    series_ids = {
        # GDP and Growth
        "NGMP41860": "sf_gdp",  # SF GDP
        "RGMP41860": "sf_real_gdp",  # SF Real GDP
        # Employment
        "SANF806UR": "sf_unemployment_rate",  # SF Unemployment Rate
        "SMU06419207000000001": "sf_total_employment",  # Total Employment
        "LFSA41860": "sf_civilian_labor_force",  # Civilian Labor Force
        # Income
        "PCPI06075": "sf_per_capita_income",  # Per Capita Personal Income
        "MHICA06075A052NCEN": "sf_median_household_income",  # Median Household Income
        # Housing and Real Estate
        "SFXRSA": "sf_house_price_index",  # House Price Index
        "ACTLISCOU41860": "sf_active_listings",  # Active Listings Count
        "MEDLISPRIA41860": "sf_median_listing_price",  # Median Listing Price
        # Business Activity
        "BPPRIV06075": "sf_private_housing_permits",  # Private Housing Building Permits
        "RRVRUSQ934SBOG": "sf_retail_vacancy_rate",  # Retail Vacancy Rate
        # Consumer Behavior
        "SMU06419207072200001SA": "sf_food_service_employment",  # Food Service Employment
        "CPIAUCSL": "us_cpi",  # Consumer Price Index (US)
    }

    # Fetch each series and compile into dataframe
    economic_data = {}
    for series_id, name in series_ids.items():
        try:
            # FRED API allows fetching data for the entire 10+ year period
            series = fred.get_series(
                series_id,
                observation_start=start_date.strftime("%Y-%m-%d"),
                observation_end=end_date.strftime("%Y-%m-%d"),
            )
            # Check if we got data
            if len(series) > 0:
                economic_data[name] = series
                logger.info(f"  - Retrieved {name} ({len(series)} observations)")
            else:
                logger.warning(f"  - No data for {name}")
        except Exception as e:
            logger.error(f"  - Error retrieving {name}: {e}")

    # Convert to DataFrame
    econ_df = pd.DataFrame(economic_data)
    econ_df.index.name = "date"
    econ_df = econ_df.reset_index()

    # Save raw data
    save_to_parquet(econ_df, f"{raw_data_dir}/economic", "fred_economic_indicators")

    # Process for any missing values or anomalies
    # For economic time series, interpolation is appropriate for handling missing values
    econ_df = econ_df.interpolate(method="linear")

    # Calculate additional derived metrics
    if (
        "sf_unemployment_rate" in econ_df.columns
        and "sf_median_household_income" in econ_df.columns
    ):
        # Example: Economic health index (simplified)
        econ_df["economic_health_index"] = (100 - econ_df["sf_unemployment_rate"]) * (
            econ_df["sf_median_household_income"]
            / econ_df["sf_median_household_income"].max()
        )

    # Resample to monthly frequency if needed (FRED data comes in various frequencies)
    econ_df["date"] = pd.to_datetime(econ_df["date"])
    econ_df.set_index("date", inplace=True)

    # Fill any remaining gaps
    econ_df = econ_df.fillna(method="ffill").fillna(method="bfill")
    econ_df = econ_df.reset_index()

    # Save processed data
    save_to_parquet(econ_df, f"{processed_dir}/economic", "fred_economic_indicators")

    return econ_df


if __name__ == "__main__":
    # Use unified config - same functions as your _02 script
    logger = setup_logging()
    config = setup_directories()
    
    logger.info("Starting FRED Economic data collection")
    logger.info(f"Base directory: {config['base_dir']}")
    
    # Create directories if they don't exist - same pattern as your _02 script
    os.makedirs(f"{config['raw_data_dir']}/economic", exist_ok=True)
    os.makedirs(f"{config['processed_dir']}/economic", exist_ok=True)

    # Execute the function
    fred_economic_df = fetch_fred_economic_data(
        config["raw_data_dir"],
        config["processed_dir"],
        config["start_date"],
        config["end_date"],
    )
    print(f"Fetched {len(fred_economic_df)} economic indicator records")
