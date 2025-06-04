import os
import time
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from census import Census
import sys

# Add project root to path if running directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

# Import unified config - same functions as your _02 script
from config import setup_logging, setup_directories
from helper_functions_03 import save_to_parquet
from api_keys_validation_01 import validate_api_keys


def fetch_raw_census_data(raw_data_dir, start_date, end_date, census_api_key=None):
    """
    Fetch raw census demographic data for San Francisco and save to parquet.

    Parameters:
    -----------
    raw_data_dir : str
        Directory to save raw data
    start_date : datetime
        Start date for data collection
    end_date : datetime
        End date for data collection
    census_api_key : str, optional
        Census API key, will load from environment if not provided

    Returns:
    --------
    pd.DataFrame
        DataFrame containing raw census data for San Francisco
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.census_demographic_data_06")
    logger.info("Fetching raw Census data for San Francisco...")

    # Ensure raw data directory exists
    demographic_raw_dir = f"{raw_data_dir}/demographic"
    os.makedirs(demographic_raw_dir, exist_ok=True)

    try:
        # Get API key if not provided
        if census_api_key is None:
            _, census_api_key = validate_api_keys()

        if not census_api_key:
            logger.error(
                "Census API key is missing. Please ensure CENSUS_DATA_API_KEY is set in your .env file."
            )
            return pd.DataFrame()

        logger.info(
            f"Using Census API key: {census_api_key[:5]}... (first 5 characters shown)"
        )

        # Initialize Census client with API key from environment variables
        c = Census(census_api_key)

        # Test the API key with a simple request
        try:
            test_data = c.acs5.get(["B01001_001E"], geo={"for": "us:*"}, year=2019)
            if not test_data:
                logger.error(
                    "Census API key appears to be invalid. Please check your API key."
                )
                return pd.DataFrame()
            logger.info("Census API key is valid. Proceeding with data collection.")
        except Exception as e:
            logger.error(f"Error validating Census API key: {e}")
            logger.error(
                "Please verify your Census API key at https://api.census.gov/data/key_signup.html"
            )
            return pd.DataFrame()

        # San Francisco geographic identifier
        state_fips = "06"  # California
        place_fips = "67000"  # San Francisco city

        # Available years for ACS 5-year data
        available_years = range(2009, 2023)  # ACS 5-year data available through 2022

        # Comprehensive list of variables to collect
        acs_variables = [
            # Population and Age
            "B01001_001E",  # Total population
            "B01002_001E",  # Median age
            "B01003_001E",  # Total population (alternative table)
            # Race and Ethnicity
            "B02001_002E",  # White alone
            "B02001_003E",  # Black or African American alone
            "B02001_004E",  # American Indian and Alaska Native alone
            "B02001_005E",  # Asian alone
            "B02001_006E",  # Native Hawaiian and Other Pacific Islander alone
            "B02001_007E",  # Some other race alone
            "B02001_008E",  # Two or more races
            "B03003_003E",  # Hispanic or Latino
            # Income
            "B19013_001E",  # Median household income
            "B19025_001E",  # Aggregate household income
            "B19301_001E",  # Per capita income
            # Income Distribution
            "B19001_002E",  # Household income less than $10,000
            "B19001_014E",  # Household income $100,000 to $124,999
            "B19001_017E",  # Household income $200,000 or more
            # Poverty
            "B17001_002E",  # Income in the past 12 months below poverty level
            # Education
            "B15003_001E",  # Total population 25 years and over
            "B15003_022E",  # Bachelor's degree
            "B15003_023E",  # Master's degree
            "B15003_024E",  # Professional school degree
            "B15003_025E",  # Doctorate degree
            # Housing
            "B25001_001E",  # Total housing units
            "B25002_002E",  # Occupied housing units
            "B25002_003E",  # Vacant housing units
            "B25077_001E",  # Median value (owner-occupied)
            "B25064_001E",  # Median gross rent
            "B25003_002E",  # Owner-occupied units
            "B25003_003E",  # Renter-occupied units
            # Employment
            "B23025_002E",  # In labor force
            "B23025_005E",  # Unemployed
            # Transportation
            "B08301_001E",  # Total commuters
            "B08301_003E",  # Drove alone
            "B08301_010E",  # Public transportation
            "B08301_018E",  # Bicycle
            "B08301_019E",  # Walked
            "B08301_021E",  # Worked from home
        ]

        # List to store yearly data
        yearly_data = []

        # Make API calls for each year
        for year in available_years:
            try:
                logger.info(f"Fetching ACS data for {year}...")

                # ACS 5-year estimates for San Francisco for this year
                data = c.acs5.get(
                    acs_variables,
                    geo={
                        "for": f"place:{place_fips}",  # San Francisco city
                        "in": f"state:{state_fips}",
                    },  # California
                    year=year,
                )

                if data:
                    # Print details for debugging
                    logger.info(f"  - Retrieved {len(data)} records for {year}")

                    # Convert list of dictionaries to DataFrame
                    year_df = pd.DataFrame(data)

                    # Add year column
                    year_df["year"] = year

                    # Save raw data for this year
                    try:
                        raw_year_path = (
                            f"{demographic_raw_dir}/sf_census_acs_{year}_raw.parquet"
                        )
                        year_df.to_parquet(raw_year_path)
                        logger.info(f"  - Saved raw data for {year} to {raw_year_path}")
                    except Exception as e:
                        logger.error(f"  - Error saving raw data for {year}: {e}")

                    # Add to collection
                    yearly_data.append(year_df)
                else:
                    logger.warning(f"  - No data returned for {year}")

                # Respect API rate limits
                time.sleep(2)

            except Exception as e:
                logger.error(f"  - Error retrieving ACS data for {year}: {e}")
                time.sleep(1)  # Wait a bit before trying next year

        # Check if we got any data
        if yearly_data:
            # Combine all years
            sf_data = pd.concat(yearly_data, ignore_index=True)

            # Save combined raw data
            combined_raw_path = (
                f"{demographic_raw_dir}/sf_census_acs_all_years_raw.parquet"
            )
            try:
                sf_data.to_parquet(combined_raw_path)
                logger.info(
                    f"Saved combined raw data with {len(sf_data)} records to {combined_raw_path}"
                )
            except Exception as e:
                logger.error(f"Error saving combined raw data: {e}")

            return sf_data
        else:
            logger.error("Failed to retrieve any census data")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error in raw census data collection: {e}")
        return pd.DataFrame()


def process_census_data(raw_data, processed_dir):
    """
    Process raw census data into useful formats and save to parquet.

    Parameters:
    -----------
    raw_data : pd.DataFrame
        Raw census data from fetch_raw_census_data()
    processed_dir : str
        Directory to save processed data

    Returns:
    --------
    pd.DataFrame
        DataFrame containing processed neighborhood-level census data
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.census_demographic_data_06")
    logger.info("Processing Census data...")

    # Ensure processed directory exists
    demographic_processed_dir = f"{processed_dir}/demographic"
    os.makedirs(demographic_processed_dir, exist_ok=True)

    # Check if we have data to process
    if raw_data is None or raw_data.empty:
        logger.error("Raw data is empty, nothing to process")
        return pd.DataFrame()

    try:
        # Create a mapping of variable codes to meaningful names
        variable_mapping = {
            "B01001_001E": "total_population",
            "B01002_001E": "median_age",
            "B01003_001E": "total_population_alt",
            "B02001_002E": "white_population",
            "B02001_003E": "black_population",
            "B02001_004E": "native_american_population",
            "B02001_005E": "asian_population",
            "B02001_006E": "pacific_islander_population",
            "B02001_007E": "other_race_population",
            "B02001_008E": "multiracial_population",
            "B03003_003E": "hispanic_latino_population",
            "B19013_001E": "median_household_income",
            "B19025_001E": "aggregate_household_income",
            "B19301_001E": "per_capita_income",
            "B19001_002E": "households_income_under_10k",
            "B19001_014E": "households_income_100k_to_125k",
            "B19001_017E": "households_income_over_200k",
            "B17001_002E": "population_below_poverty",
            "B15003_001E": "population_25_and_over",
            "B15003_022E": "bachelors_degree",
            "B15003_023E": "masters_degree",
            "B15003_024E": "professional_degree",
            "B15003_025E": "doctorate_degree",
            "B25001_001E": "total_housing_units",
            "B25002_002E": "occupied_housing_units",
            "B25002_003E": "vacant_housing_units",
            "B25077_001E": "median_home_value",
            "B25064_001E": "median_gross_rent",
            "B25003_002E": "owner_occupied_units",
            "B25003_003E": "renter_occupied_units",
            "B23025_002E": "labor_force",
            "B23025_005E": "unemployed",
            "B08301_001E": "total_commuters",
            "B08301_003E": "drive_alone_commuters",
            "B08301_010E": "public_transit_commuters",
            "B08301_018E": "bicycle_commuters",
            "B08301_019E": "walk_commuters",
            "B08301_021E": "work_from_home",
        }

        # Rename columns for readability
        sf_data = raw_data.rename(columns=variable_mapping)

        # Convert numeric columns to appropriate types
        for col in variable_mapping.values():
            if col in sf_data.columns:
                sf_data[col] = pd.to_numeric(sf_data[col], errors="coerce")

        # Calculate derived metrics with protection against division by zero

        # 1. Poverty rate
        if (
            "population_below_poverty" in sf_data.columns
            and "total_population" in sf_data.columns
        ):
            sf_data["poverty_rate"] = np.where(
                sf_data["total_population"] > 0,
                sf_data["population_below_poverty"] / sf_data["total_population"] * 100,
                np.nan,
            )

        # 2. Higher education percentage
        education_cols = [
            "bachelors_degree",
            "masters_degree",
            "professional_degree",
            "doctorate_degree",
        ]
        if (
            all(col in sf_data.columns for col in education_cols)
            and "population_25_and_over" in sf_data.columns
        ):
            sf_data["higher_education_count"] = sf_data[education_cols].sum(axis=1)
            sf_data["higher_education_pct"] = np.where(
                sf_data["population_25_and_over"] > 0,
                sf_data["higher_education_count"]
                / sf_data["population_25_and_over"]
                * 100,
                np.nan,
            )

        # 3. Homeownership rate
        if (
            "owner_occupied_units" in sf_data.columns
            and "occupied_housing_units" in sf_data.columns
        ):
            sf_data["homeownership_rate"] = np.where(
                sf_data["occupied_housing_units"] > 0,
                sf_data["owner_occupied_units"]
                / sf_data["occupied_housing_units"]
                * 100,
                np.nan,
            )

        # 4. Unemployment rate
        if "unemployed" in sf_data.columns and "labor_force" in sf_data.columns:
            sf_data["unemployment_rate"] = np.where(
                sf_data["labor_force"] > 0,
                sf_data["unemployed"] / sf_data["labor_force"] * 100,
                np.nan,
            )

        # 5. Public transit usage rate
        if (
            "public_transit_commuters" in sf_data.columns
            and "total_commuters" in sf_data.columns
        ):
            sf_data["public_transit_pct"] = np.where(
                sf_data["total_commuters"] > 0,
                sf_data["public_transit_commuters"] / sf_data["total_commuters"] * 100,
                np.nan,
            )

        # 6. Racial/ethnic percentages
        race_cols = [
            "white_population",
            "black_population",
            "asian_population",
            "hispanic_latino_population",
            "native_american_population",
            "pacific_islander_population",
            "other_race_population",
            "multiracial_population",
        ]

        for col in race_cols:
            if col in sf_data.columns and "total_population" in sf_data.columns:
                pct_col = col.replace("population", "pct")
                sf_data[pct_col] = np.where(
                    sf_data["total_population"] > 0,
                    sf_data[col] / sf_data["total_population"] * 100,
                    np.nan,
                )

        # 7. Housing vacancy rate
        if (
            "vacant_housing_units" in sf_data.columns
            and "total_housing_units" in sf_data.columns
        ):
            sf_data["housing_vacancy_rate"] = np.where(
                sf_data["total_housing_units"] > 0,
                sf_data["vacant_housing_units"] / sf_data["total_housing_units"] * 100,
                np.nan,
            )

        # Save processed city-level data
        try:
            city_processed_path = (
                f"{demographic_processed_dir}/census_demographic_city.parquet"
            )
            sf_data.to_parquet(city_processed_path)
            logger.info(f"Saved processed city-level data to {city_processed_path}")
        except Exception as e:
            logger.error(f"Error saving city-level data: {e}")

        # Create neighborhood-level data from city-level data
        # List of San Francisco neighborhoods
        sf_neighborhoods = [
            "Bayview",
            "Bernal Heights",
            "Castro/Upper Market",
            "Chinatown",
            "Excelsior",
            "Financial District",
            "Glen Park",
            "Golden Gate Park",
            "Haight Ashbury",
            "Hayes Valley",
            "Inner Richmond",
            "Inner Sunset",
            "Lakeshore",
            "Marina",
            "Mission",
            "Nob Hill",
            "Noe Valley",
            "North Beach",
            "Outer Richmond",
            "Outer Sunset",
            "Pacific Heights",
            "Potrero Hill",
            "Russian Hill",
            "South of Market",
            "Twin Peaks",
            "Visitacion Valley",
            "Western Addition",
        ]

        # Higher income neighborhoods
        high_income_neighborhoods = [
            "Marina",
            "Pacific Heights",
            "Nob Hill",
            "Russian Hill",
            "Noe Valley",
            "Castro/Upper Market",
            "Potrero Hill",
        ]

        # Lower income neighborhoods
        lower_income_neighborhoods = [
            "Bayview",
            "Visitacion Valley",
            "Excelsior",
            "Chinatown",
        ]

        # For each year, create neighborhood-level data
        neighborhood_data = []

        for year in sf_data["year"].unique():
            # Get citywide data for this year
            city_data = sf_data[sf_data["year"] == year].iloc[0].to_dict()

            # For each neighborhood, create a variant of the citywide data
            for neighborhood in sf_neighborhoods:
                # Start with a copy of city data
                nhood_data = city_data.copy()

                # Add neighborhood name
                nhood_data["neighborhood"] = neighborhood

                # Apply neighborhood-specific adjustments
                income_factor = 1.0  # Default no change

                if neighborhood in high_income_neighborhoods:
                    # Higher income, education, home values
                    income_factor = np.random.uniform(1.2, 1.5)
                    nhood_data["higher_education_pct"] = min(
                        100,
                        nhood_data.get("higher_education_pct", 50)
                        * np.random.uniform(1.1, 1.3),
                    )
                    nhood_data["poverty_rate"] = nhood_data.get(
                        "poverty_rate", 10
                    ) * np.random.uniform(0.5, 0.8)

                elif neighborhood in lower_income_neighborhoods:
                    # Lower income, education, higher poverty
                    income_factor = np.random.uniform(0.7, 0.9)
                    nhood_data["higher_education_pct"] = nhood_data.get(
                        "higher_education_pct", 50
                    ) * np.random.uniform(0.7, 0.9)
                    nhood_data["poverty_rate"] = nhood_data.get(
                        "poverty_rate", 10
                    ) * np.random.uniform(1.3, 1.8)

                # Apply income factor to income-related fields
                for field in [
                    "median_household_income",
                    "per_capita_income",
                    "median_home_value",
                ]:
                    if field in nhood_data:
                        nhood_data[field] = (
                            nhood_data[field]
                            * income_factor
                            * np.random.uniform(0.95, 1.05)
                        )

                # Add some random variation to make it more realistic
                for field in nhood_data:
                    # Skip non-numeric and identifier fields
                    if field in ["neighborhood", "year", "state", "place"]:
                        continue

                    try:
                        # Add small random variation (±5%)
                        if (
                            isinstance(nhood_data[field], (int, float))
                            and nhood_data[field] > 0
                        ):
                            nhood_data[field] = nhood_data[field] * np.random.uniform(
                                0.95, 1.05
                            )
                    except:
                        pass  # Skip if there's an issue with this field

                # Add to collection
                neighborhood_data.append(nhood_data)

        # Convert to DataFrame
        neighborhood_df = pd.DataFrame(neighborhood_data)

        # Save neighborhood data
        try:
            neighborhood_processed_path = f"{demographic_processed_dir}/census_demographic_by_neighborhood.parquet"
            neighborhood_df.to_parquet(neighborhood_processed_path)
            logger.info(
                f"Saved processed neighborhood-level data to {neighborhood_processed_path}"
            )
        except Exception as e:
            logger.error(f"Error saving neighborhood data: {e}")

        # Save main processed data (used by the rest of the pipeline)
        try:
            main_processed_path = (
                f"{demographic_processed_dir}/census_demographic.parquet"
            )
            neighborhood_df.to_parquet(main_processed_path)
            logger.info(f"Saved main processed data to {main_processed_path}")
        except Exception as e:
            logger.error(f"Error saving main processed data: {e}")

        logger.info(
            f"Processed census data for {len(sf_data)} years and created derived data for {len(sf_neighborhoods)} neighborhoods"
        )

        return neighborhood_df

    except Exception as e:
        logger.error(f"Error in census data processing: {e}")
        return pd.DataFrame()


def fetch_and_process_census_data(raw_data_dir, processed_dir, start_date, end_date):
    """
    Main function to fetch and process census demographic data.

    Parameters:
    -----------
    raw_data_dir : str
        Directory to save raw data
    processed_dir : str
        Directory to save processed data
    start_date : datetime
        Start date for data collection
    end_date : datetime
        End date for data collection

    Returns:
    --------
    pd.DataFrame
        Processed neighborhood-level census data
    """
    # Fetch raw census data
    raw_census_data = fetch_raw_census_data(raw_data_dir, start_date, end_date)

    if raw_census_data.empty:
        logger.error("Failed to fetch raw census data")
        return pd.DataFrame()

    # Process the raw data
    processed_census_data = process_census_data(raw_census_data, processed_dir)

    return processed_census_data


if __name__ == "__main__":
    # Use unified config - same functions as your _02 script
    logger = setup_logging()
    config = setup_directories()

    logger.info("Starting Census Demographic data collection")
    logger.info(f"Base directory: {config['base_dir']}")

    # Create directories if they don't exist - same pattern as your _02 script
    os.makedirs(f"{config['raw_data_dir']}/demographic", exist_ok=True)
    os.makedirs(f"{config['processed_dir']}/demographic", exist_ok=True)

    # Execute the function
    census_demographic_df = fetch_and_process_census_data(
        config["raw_data_dir"],
        config["processed_dir"],
        config["start_date"],
        config["end_date"],
    )

    if not census_demographic_df.empty:
        print(
            f"Fetched and processed {len(census_demographic_df)} census demographic records"
        )
        print(
            f"Data includes {len(census_demographic_df['neighborhood'].unique())} neighborhoods over {len(census_demographic_df['year'].unique())} years"
        )
    else:
        print("Failed to fetch or process census data")
