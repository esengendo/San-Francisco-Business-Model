import os
import re
import json
import ast
import logging
import pandas as pd
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import sys

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src", "utils"))

# Import unified config - same functions as your _02 script
from config import setup_logging, setup_directories
from helper_functions_03 import save_to_parquet

# Define the columns we want to read from the source file
COLUMNS_TO_READ = [
    "ttxid",
    "certificate_number",
    "ownership_name",
    "dba_name",
    "full_business_address",
    "city",
    "business_zip",
    "dba_start_date",
    "dba_end_date",
    "location_start_date",
    "location_end_date",
    "parking_tax",
    "transient_occupancy_tax",
    "location",
    "administratively_closed",
    "naic_code",
    "naic_code_description",
    "supervisor_district",
    "neighborhoods_analysis_boundaries",
    "business_corridor",
]


def load_data(file_path):
    """
    Load data from a parquet file, reading only the specified columns.

    Parameters:
    -----------
    file_path : str
        Path to the parquet file

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the data with selected columns
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.business_data_processing_15")
    
    # First check which columns actually exist in the file
    try:
        available_columns = pd.read_parquet(file_path, columns=None).columns.tolist()
        columns_to_read = [col for col in COLUMNS_TO_READ if col in available_columns]

        logger.info(f"Reading {len(columns_to_read)} columns from parquet file...")
        return pd.read_parquet(file_path, columns=columns_to_read)
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def create_naics_mapping():
    """
    Create mappings from NAICS codes to industry sectors.

    Returns:
    --------
    tuple
        (naics_mapping, range_mapping) - Maps 2-digit codes to primary sectors and code ranges to sectors
    """
    # Primary sectors mapping (first 2 digits)
    naics_mapping = {
        "11": "Agriculture, Forestry, Fishing and Hunting",
        "21": "Mining, Quarrying, and Oil and Gas Extraction",
        "22": "Utilities",
        "23": "Construction",
        "31": "Manufacturing",
        "32": "Manufacturing",
        "33": "Manufacturing",
        "42": "Wholesale Trade",
        "44": "Retail Trade",
        "45": "Retail Trade",
        "48": "Transportation and Warehousing",
        "49": "Transportation and Warehousing",
        "51": "Information",
        "52": "Finance and Insurance",
        "53": "Real Estate and Rental and Leasing",
        "54": "Professional, Scientific, and Technical Services",
        "55": "Management of Companies and Enterprises",
        "56": "Administrative and Support and Waste Management and Remediation Services",
        "61": "Educational Services",
        "62": "Health Care and Social Assistance",
        "71": "Arts, Entertainment, and Recreation",
        "72": "Accommodation and Food Services",
        "81": "Other Services (except Public Administration)",
        "92": "Public Administration",
    }

    # Add mappings for specific code ranges found in the dataset
    range_mapping = {
        "2300-2399": "Construction",
        "5400-5499": "Professional, Scientific, and Technical Services",
        "5300-5399": "Real Estate and Rental and Leasing",
        "4400-4599": "Retail Trade",
        "5100-5199": "Information",
        "7220-7229": "Accommodation and Food Services",
        "5600-5699": "Administrative and Support and Waste Management and Remediation Services",
        "6100-6199": "Educational Services",
        "6200-6299": "Health Care and Social Assistance",
        "7210-7219": "Accommodation and Food Services",
        "7100-7199": "Arts, Entertainment, and Recreation",
        "4200-4299": "Wholesale Trade",
        "2200-2299": "Utilities",
        "5210-5239": "Finance and Insurance",
        "4800-4999": "Transportation and Warehousing",
        "8100-8139": "Other Services (except Public Administration)",
        "3100-3399": "Manufacturing",
        "5240-5249": "Finance and Insurance",
    }

    return naics_mapping, range_mapping


def create_keyword_industry_mapping():
    """
    Create a mapping from business name keywords to likely industries.
    This allows us to infer industry from business name when NAICS code is missing.

    Returns:
    --------
    dict
        Mapping of keywords to industry categories
    """
    keyword_mapping = {
        # Real Estate related
        "property": "Real Estate and Rental and Leasing",
        "properties": "Real Estate and Rental and Leasing",
        "realty": "Real Estate and Rental and Leasing",
        "estate": "Real Estate and Rental and Leasing",
        "bldg": "Real Estate and Rental and Leasing",
        "apartments": "Real Estate and Rental and Leasing",
        "rental": "Real Estate and Rental and Leasing",
        "leasing": "Real Estate and Rental and Leasing",
        "breather": "Real Estate and Rental and Leasing",
        # Construction
        "construction": "Construction",
        "builders": "Construction",
        "contractor": "Construction",
        "contractors": "Construction",
        "building": "Construction",
        "development": "Construction",
        # Professional Services
        "law": "Professional, Scientific, and Technical Services",
        "legal": "Professional, Scientific, and Technical Services",
        "attorney": "Professional, Scientific, and Technical Services",
        "consulting": "Professional, Scientific, and Technical Services",
        "consultants": "Professional, Scientific, and Technical Services",
        "architect": "Professional, Scientific, and Technical Services",
        "design": "Professional, Scientific, and Technical Services",
        "accounting": "Professional, Scientific, and Technical Services",
        "tax": "Professional, Scientific, and Technical Services",
        "technologies": "Professional, Scientific, and Technical Services",
        "technology": "Professional, Scientific, and Technical Services",
        "software": "Professional, Scientific, and Technical Services",
        "office": "Professional, Scientific, and Technical Services",
        "creative": "Professional, Scientific, and Technical Services",
        "studio": "Professional, Scientific, and Technical Services",
        # Retail
        "store": "Retail Trade",
        "shop": "Retail Trade",
        "market": "Retail Trade",
        "retail": "Retail Trade",
        "sales": "Retail Trade",
        "jewelry": "Retail Trade",
        "clothing": "Retail Trade",
        "apparel": "Retail Trade",
        "fashion": "Retail Trade",
        # Food Service and Accommodation
        "restaurant": "Accommodation and Food Services",
        "cafe": "Accommodation and Food Services",
        "bakery": "Accommodation and Food Services",
        "coffee": "Accommodation and Food Services",
        "catering": "Accommodation and Food Services",
        "bar": "Accommodation and Food Services",
        "grill": "Accommodation and Food Services",
        "food": "Accommodation and Food Services",
        "pizza": "Accommodation and Food Services",
        "burger": "Accommodation and Food Services",
        "kitchen": "Accommodation and Food Services",
        "hotel": "Accommodation and Food Services",
        "inn": "Accommodation and Food Services",
        "motel": "Accommodation and Food Services",
        "resort": "Accommodation and Food Services",
        "suites": "Accommodation and Food Services",
        "lodging": "Accommodation and Food Services",
        # Health Services
        "medical": "Health Care and Social Assistance",
        "health": "Health Care and Social Assistance",
        "clinic": "Health Care and Social Assistance",
        "hospital": "Health Care and Social Assistance",
        "dental": "Health Care and Social Assistance",
        "doctor": "Health Care and Social Assistance",
        "therapy": "Health Care and Social Assistance",
        "therapeutic": "Health Care and Social Assistance",
        "medicine": "Health Care and Social Assistance",
        "wellness": "Health Care and Social Assistance",
        "care": "Health Care and Social Assistance",
        "nursing": "Health Care and Social Assistance",
        "physician": "Health Care and Social Assistance",
        "surgeon": "Health Care and Social Assistance",
        "patient": "Health Care and Social Assistance",
        "pharmacy": "Health Care and Social Assistance",
        "rehab": "Health Care and Social Assistance",
        "physical": "Health Care and Social Assistance",
        "mental": "Health Care and Social Assistance",
        "counseling": "Health Care and Social Assistance",
        "pilates": "Health Care and Social Assistance",
        "yoga": "Health Care and Social Assistance",
        "fitness": "Health Care and Social Assistance",
        # Education
        "school": "Educational Services",
        "academy": "Educational Services",
        "institute": "Educational Services",
        "college": "Educational Services",
        "university": "Educational Services",
        "education": "Educational Services",
        "learning": "Educational Services",
        "training": "Educational Services",
        "tutor": "Educational Services",
        "teach": "Educational Services",
        "class": "Educational Services",
        "course": "Educational Services",
        "instruction": "Educational Services",
        "study": "Educational Services",
        "student": "Educational Services",
        # Transportation
        "transport": "Transportation and Warehousing",
        "logistics": "Transportation and Warehousing",
        "shipping": "Transportation and Warehousing",
        "delivery": "Transportation and Warehousing",
        "freight": "Transportation and Warehousing",
        "truck": "Transportation and Warehousing",
        "taxi": "Transportation and Warehousing",
        "limo": "Transportation and Warehousing",
        # Finance and Insurance
        "financial": "Finance and Insurance",
        "finance": "Finance and Insurance",
        "bank": "Finance and Insurance",
        "investment": "Finance and Insurance",
        "investing": "Finance and Insurance",
        "capital": "Finance and Insurance",
        "wealth": "Finance and Insurance",
        "asset": "Finance and Insurance",
        "insurance": "Finance and Insurance",
        "assurance": "Finance and Insurance",
        "underwriting": "Finance and Insurance",
        # Manufacturing
        "manufacturing": "Manufacturing",
        "factory": "Manufacturing",
        "production": "Manufacturing",
        "fabrication": "Manufacturing",
        "industrial": "Manufacturing",
        "maker": "Manufacturing",
        # Information/Media
        "media": "Information",
        "productions": "Information",
        "digital": "Information",
        "film": "Information",
        "video": "Information",
        "entertainment": "Information",
        "publishing": "Information",
        "communication": "Information",
        # Administrative and Support Services
        "cleaning": "Administrative and Support and Waste Management and Remediation Services",
        "cleaners": "Administrative and Support and Waste Management and Remediation Services",
        "janitorial": "Administrative and Support and Waste Management and Remediation Services",
        "maintenance": "Administrative and Support and Waste Management and Remediation Services",
        "service": "Administrative and Support and Waste Management and Remediation Services",
        "staffing": "Administrative and Support and Waste Management and Remediation Services",
        "employment": "Administrative and Support and Waste Management and Remediation Services",
        "recruitment": "Administrative and Support and Waste Management and Remediation Services",
        "security": "Administrative and Support and Waste Management and Remediation Services",
        # Arts and Recreation
        "gallery": "Arts, Entertainment, and Recreation",
        "arts": "Arts, Entertainment, and Recreation",
        "theater": "Arts, Entertainment, and Recreation",
        "theatre": "Arts, Entertainment, and Recreation",
        "recreation": "Arts, Entertainment, and Recreation",
        "gaming": "Arts, Entertainment, and Recreation",
        "sports": "Arts, Entertainment, and Recreation",
        # Wholesale Trade
        "wholesale": "Wholesale Trade",
        "distributing": "Wholesale Trade",
        "distributor": "Wholesale Trade",
        "supply": "Wholesale Trade",
        "supplies": "Wholesale Trade",
        # Other Services
        "salon": "Other Services (except Public Administration)",
        "barber": "Other Services (except Public Administration)",
        "beauty": "Other Services (except Public Administration)",
        "spa": "Other Services (except Public Administration)",
        "repair": "Other Services (except Public Administration)",
        "auto": "Other Services (except Public Administration)",
        "automotive": "Other Services (except Public Administration)",
        "car": "Other Services (except Public Administration)",
    }

    return keyword_mapping


def classify_by_naics(naics_code, naics_mapping, range_mapping):
    """
    Classify a business based on its NAICS code or code range.

    Parameters:
    -----------
    naics_code : str or numeric
        NAICS code or range for the business
    naics_mapping : dict
        Mapping of 2-digit codes to sectors
    range_mapping : dict
        Mapping of code ranges to sectors

    Returns:
    --------
    str or None
        Industry classification or None if unknown
    """
    # Skip missing codes
    if not naics_code or pd.isna(naics_code) or naics_code == "nan":
        return None

    # Check if it's a simple NAICS code and use the first 2 digits
    if str(naics_code).isdigit():
        sector_code = str(naics_code)[:2]
        if sector_code in naics_mapping:
            return naics_mapping[sector_code]

    # Check if it's a range like "2300-2399"
    if "-" in str(naics_code):
        if str(naics_code) in range_mapping:
            return range_mapping[str(naics_code)]

    # Check if it has multiple ranges like "2300-2399 3100-3399"
    if " " in str(naics_code):
        return "Multiple Industries"

    return None


def classify_by_name(business_name, keyword_mapping):
    """
    Classify a business based on keywords in its name.

    Parameters:
    -----------
    business_name : str
        Name of the business
    keyword_mapping : dict
        Mapping of keywords to industry categories

    Returns:
    --------
    str or None
        Industry classification or None if no matches
    """
    # Skip missing names
    if not business_name or pd.isna(business_name):
        return None

    # Convert to lowercase for matching
    name = str(business_name).lower()

    # Check for keyword matches
    matches = defaultdict(int)
    words = re.findall(r"\b\w+\b", name)  # Split into words

    # Count occurrences of each industry in the business name
    for word in words:
        if word in keyword_mapping:
            matches[keyword_mapping[word]] += 1

    # Return the industry with the most matches
    if matches:
        return max(matches.items(), key=lambda x: x[1])[0]

    return None


def classify_business(row, naics_mapping, range_mapping, keyword_mapping):
    """
    Classify a business using multiple methods in priority order:
    1. If NAICS code description exists, use that directly
    2. If NAICS code exists, determine the primary sector from that
    3. If neither exists, use business name to infer the closest primary sector

    Parameters:
    -----------
    row : pandas Series
        A row from the business dataframe
    naics_mapping : dict
        Mapping of 2-digit codes to sectors
    range_mapping : dict
        Mapping of code ranges to sectors
    keyword_mapping : dict
        Mapping of keywords to industry categories

    Returns:
    --------
    str
        Industry classification
    """
    # First check if we already have a NAICS description
    if (
        row["naic_code_description"]
        and not pd.isna(row["naic_code_description"])
        and row["naic_code_description"] != "nan"
    ):
        if row["naic_code_description"] == "Multiple":
            return "Multiple Industries"

        # Special handling for Private Education and Health Services - split into separate categories
        if row["naic_code_description"] == "Private Education and Health Services":
            # Try to determine if it's more education or health based on the business name
            if row["dba_name"] and not pd.isna(row["dba_name"]):
                name = str(row["dba_name"]).lower()
                edu_keywords = [
                    "school",
                    "academy",
                    "institute",
                    "college",
                    "university",
                    "education",
                    "learning",
                    "training",
                    "tutor",
                    "teach",
                    "class",
                    "course",
                    "instruction",
                    "study",
                    "student",
                ]
                health_keywords = [
                    "medical",
                    "health",
                    "clinic",
                    "hospital",
                    "dental",
                    "doctor",
                    "therapy",
                    "therapeutic",
                    "medicine",
                    "wellness",
                    "care",
                    "nursing",
                    "physician",
                    "surgeon",
                    "patient",
                    "pharmacy",
                    "rehab",
                    "physical",
                    "mental",
                    "counseling",
                    "pilates",
                    "yoga",
                    "fitness",
                ]

                # Count keyword matches in each category
                edu_matches = sum(1 for kw in edu_keywords if kw in name)
                health_matches = sum(1 for kw in health_keywords if kw in name)

                # Classify based on keyword matches
                if edu_matches > health_matches:
                    return "Educational Services"
                elif health_matches > edu_matches:
                    return "Health Care and Social Assistance"
                # If no clear distinction, try using NAICS code if available
                elif row["naic_code"] and not pd.isna(row["naic_code"]):
                    code_str = str(row["naic_code"])
                    if "61" in code_str[:2]:
                        return "Educational Services"
                    elif "62" in code_str[:2]:
                        return "Health Care and Social Assistance"

            # Default fallback if we can't determine
            return (
                "Educational Services"
                if "education" in str(row).lower()
                else "Health Care and Social Assistance"
            )

        # Return the existing description (already standardized)
        return row["naic_code_description"]

    # Try to classify by NAICS code if available
    if (
        row["naic_code"]
        and not pd.isna(row["naic_code"])
        and str(row["naic_code"]) != "nan"
    ):
        # For numeric NAICS codes, use first two digits
        if isinstance(row["naic_code"], (int, float)) or (
            str(row["naic_code"]).isdigit()
        ):
            sector_code = str(row["naic_code"])[:2]
            if sector_code in naics_mapping:
                return naics_mapping[sector_code]

        # Handle range format like "2300-2399"
        naics_str = str(row["naic_code"])

        # Direct range match
        if naics_str in range_mapping:
            return range_mapping[naics_str]

        # Handle multiple ranges like "2300-2399 3100-3399"
        if " " in naics_str:
            ranges = naics_str.split()
            for r in ranges:
                if r in range_mapping:
                    # Return the first valid range mapping
                    return range_mapping[r]
            return "Multiple Industries"

    # Try to classify by business name if NAICS info is missing
    name_classification = classify_by_name(row["dba_name"], keyword_mapping)
    if name_classification:
        return name_classification

    # Try owner name if business name doesn't help
    owner_classification = classify_by_name(row["ownership_name"], keyword_mapping)
    if owner_classification:
        return owner_classification

    # If all else fails
    return "Unclassified"


def create_business_industry_column(df):
    """
    Create a new business_industry column in the dataframe using all classification methods.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with business information

    Returns:
    --------
    pandas DataFrame
        DataFrame with new business_industry column
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.business_data_processing_15")
    
    # Get mappings
    naics_mapping, range_mapping = create_naics_mapping()
    keyword_mapping = create_keyword_industry_mapping()

    # Apply classification to each row
    logger.info("Classifying businesses by industry...")
    df["business_industry"] = df.apply(
        lambda row: classify_business(
            row, naics_mapping, range_mapping, keyword_mapping
        ),
        axis=1,
    )

    return df


def clean_date_columns(df):
    """
    Clean and standardize date columns to MM/DD/YYYY format.
    Creates new columns with capitalized first letters, preserving original columns.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing date columns

    Returns:
    --------
    pandas DataFrame
        The dataframe with additional standardized date columns
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.business_data_processing_15")
    
    # List of date columns to process
    date_columns = [
        "dba_start_date",
        "dba_end_date",
        "location_start_date",
        "location_end_date",
    ]

    for col in date_columns:
        if col in df.columns:
            # Create new column name with capitalized first letter
            new_col = col[0].upper() + col[1:]

            # Convert to datetime, coerce errors to NaT
            # Then format dates as MM/DD/YYYY for non-null values
            df[new_col] = pd.to_datetime(df[col], errors="coerce").apply(
                lambda x: x.strftime("%m/%d/%Y") if pd.notnull(x) else None
            )

            logger.info(
                f"Created column '{new_col}' with standardized dates from '{col}'"
            )

    return df


def extract_lat_long(df):
    """
    Extract latitude and longitude from the location field and flag coordinates
    within San Francisco.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing a 'location' column

    Returns:
    --------
    pandas DataFrame
        DataFrame with new columns: 'latitude', 'longitude', and 'in_sf'
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.business_data_processing_15")
    
    # Initialize new columns
    df["longitude"] = None
    df["latitude"] = None
    df["in_sf"] = False  # Column to flag whether coordinates are in SF

    # San Francisco bounding box (slightly expanded to catch edge neighborhoods)
    sf_bounds = {
        "min_lat": 37.695,  # Southern boundary
        "max_lat": 37.845,  # Northern boundary
        "min_lon": -122.525,  # Western boundary
        "max_lon": -122.345,  # Eastern boundary
    }

    # Function to safely extract coordinates
    def extract_coordinates(loc_str):
        if pd.isna(loc_str) or loc_str is None or loc_str == "":
            return None, None

        try:
            # Try to parse as JSON
            try:
                loc_data = json.loads(loc_str)
            except:
                # If not valid JSON, try to parse as Python dict literal
                try:
                    loc_data = ast.literal_eval(loc_str)
                except:
                    # If that fails too, try some basic string parsing
                    if "coordinates" in loc_str and "[" in loc_str and "]" in loc_str:
                        coords_str = (
                            loc_str.split("coordinates")[1].split("[")[1].split("]")[0]
                        )
                        coords = [float(x.strip()) for x in coords_str.split(",")]
                        return coords[0], coords[1]
                    return None, None

            # Extract coordinates from parsed data (GeoJSON format)
            if isinstance(loc_data, dict) and "coordinates" in loc_data:
                coords = loc_data["coordinates"]
                if isinstance(coords, list) and len(coords) >= 2:
                    # GeoJSON format typically has [longitude, latitude]
                    return coords[0], coords[1]

            return None, None
        except Exception as e:
            logger.warning(f"Error parsing location data: {e}")
            return None, None

    # Apply extraction to each row
    logger.info("Extracting latitude and longitude from location field...")

    # Check for location column
    if "location" not in df.columns:
        logger.warning("'location' column not found in the dataset.")
        return df

    # Process in batches to show progress
    total_rows = len(df)
    batch_size = min(1000, total_rows)
    sf_count = 0
    non_sf_count = 0

    for i in range(0, total_rows, batch_size):
        end_idx = min(i + batch_size, total_rows)
        if i % 10000 == 0:
            logger.info(f"Processing rows {i} to {end_idx} of {total_rows}...")

        batch = df.iloc[i:end_idx]
        for idx, row in batch.iterrows():
            lon, lat = extract_coordinates(row["location"])
            df.at[idx, "longitude"] = lon
            df.at[idx, "latitude"] = lat

            # Check if coordinates are within San Francisco
            if lon is not None and lat is not None:
                try:
                    lon_float = float(lon)
                    lat_float = float(lat)
                    in_sf = (
                        sf_bounds["min_lat"] <= lat_float <= sf_bounds["max_lat"]
                        and sf_bounds["min_lon"] <= lon_float <= sf_bounds["max_lon"]
                    )
                    df.at[idx, "in_sf"] = in_sf
                    if in_sf:
                        sf_count += 1
                    else:
                        non_sf_count += 1
                except (ValueError, TypeError):
                    # If conversion fails, mark as not in SF
                    df.at[idx, "in_sf"] = False
                    non_sf_count += 1

    # Report stats on extraction success
    valid_coords = df[df["latitude"].notna() & df["longitude"].notna()].shape[0]
    logger.info(
        f"Successfully extracted coordinates for {valid_coords} out of {total_rows} records ({valid_coords/total_rows*100:.2f}%)"
    )
    logger.info(
        f"Coordinates in San Francisco: {sf_count} ({sf_count/valid_coords*100:.2f}% of valid coordinates)"
    )
    logger.info(
        f"Coordinates outside San Francisco: {non_sf_count} ({non_sf_count/valid_coords*100:.2f}% of valid coordinates)"
    )

    return df


def fill_missing_geo_data(df):
    """
    Fill missing neighborhood and supervisor district information for San Francisco coordinates.
    Uses a simplified approach with geographic lookup tables.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with latitude, longitude columns, in_sf flag, and potentially
        missing neighborhoods_analysis_boundaries and supervisor_district columns

    Returns:
    --------
    pandas DataFrame
        DataFrame with filled neighborhood and district information for SF records
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.business_data_processing_15")
    
    logger.info(
        "Filling missing geographic information for San Francisco businesses..."
    )

    # Filter to only include records within San Francisco
    sf_df = df[df["in_sf"] == True].copy() if "in_sf" in df.columns else df.copy()

    # Count missing values before processing - handle 'nan' strings
    missing_neighborhoods = (
        (sf_df["neighborhoods_analysis_boundaries"] == "nan")
        | sf_df["neighborhoods_analysis_boundaries"].isna()
    ).sum()
    missing_districts = (
        (sf_df["supervisor_district"] == "nan") | sf_df["supervisor_district"].isna()
    ).sum()

    logger.info(
        f"Before: Missing SF neighborhoods: {missing_neighborhoods}, Missing SF districts: {missing_districts}"
    )

    # Skip if no missing values
    if missing_neighborhoods == 0 and missing_districts == 0:
        logger.info(
            "No missing geographic information to fill for San Francisco businesses."
        )
        return df

    # Create simplified lookup for districts based on coordinates
    district_lookup = [
        # District 1 (Richmond) - Northwest
        {
            "min_lat": 37.769,
            "max_lat": 37.789,
            "min_lon": -122.51,
            "max_lon": -122.471,
            "district": 1.0,
        },
        # District 2 (Marina/Pacific Heights) - North
        {
            "min_lat": 37.789,
            "max_lat": 37.810,
            "min_lon": -122.452,
            "max_lon": -122.38,
            "district": 2.0,
        },
        # District 3 (North Beach/Chinatown) - Northeast
        {
            "min_lat": 37.789,
            "max_lat": 37.810,
            "min_lon": -122.414,
            "max_lon": -122.38,
            "district": 3.0,
        },
        # District 4 (Sunset) - West
        {
            "min_lat": 37.740,
            "max_lat": 37.769,
            "min_lon": -122.51,
            "max_lon": -122.471,
            "district": 4.0,
        },
        # District 5 (Haight/Western Addition) - Central North
        {
            "min_lat": 37.769,
            "max_lat": 37.785,
            "min_lon": -122.452,
            "max_lon": -122.414,
            "district": 5.0,
        },
        # District 6 (SOMA/Tenderloin) - East Central
        {
            "min_lat": 37.765,
            "max_lat": 37.789,
            "min_lon": -122.414,
            "max_lon": -122.38,
            "district": 6.0,
        },
        # District 7 (Twin Peaks/West of Twin Peaks) - Central West
        {
            "min_lat": 37.730,
            "max_lat": 37.755,
            "min_lon": -122.48,
            "max_lon": -122.44,
            "district": 7.0,
        },
        # District 8 (Castro/Noe Valley) - Central
        {
            "min_lat": 37.740,
            "max_lat": 37.765,
            "min_lon": -122.44,
            "max_lon": -122.414,
            "district": 8.0,
        },
        # District 9 (Mission/Bernal Heights) - Central East
        {
            "min_lat": 37.730,
            "max_lat": 37.765,
            "min_lon": -122.414,
            "max_lon": -122.38,
            "district": 9.0,
        },
        # District 10 (Bayview/Hunters Point) - Southeast
        {
            "min_lat": 37.710,
            "max_lat": 37.740,
            "min_lon": -122.414,
            "max_lon": -122.36,
            "district": 10.0,
        },
        # District 11 (Excelsior/Oceanview) - South
        {
            "min_lat": 37.710,
            "max_lat": 37.730,
            "min_lon": -122.47,
            "max_lon": -122.414,
            "district": 11.0,
        },
    ]

    # Create simplified lookup for major neighborhoods
    neighborhood_lookup = [
        # Major neighborhoods and their approximate boundaries
        {
            "min_lat": 37.769,
            "max_lat": 37.789,
            "min_lon": -122.51,
            "max_lon": -122.471,
            "neighborhood": "Richmond",
        },
        {
            "min_lat": 37.789,
            "max_lat": 37.810,
            "min_lon": -122.452,
            "max_lon": -122.43,
            "neighborhood": "Marina",
        },
        {
            "min_lat": 37.789,
            "max_lat": 37.805,
            "min_lon": -122.43,
            "max_lon": -122.40,
            "neighborhood": "Russian Hill",
        },
        {
            "min_lat": 37.790,
            "max_lat": 37.810,
            "min_lon": -122.414,
            "max_lon": -122.38,
            "neighborhood": "North Beach",
        },
        {
            "min_lat": 37.782,
            "max_lat": 37.790,
            "min_lon": -122.414,
            "max_lon": -122.39,
            "neighborhood": "Chinatown",
        },
        {
            "min_lat": 37.740,
            "max_lat": 37.769,
            "min_lon": -122.51,
            "max_lon": -122.471,
            "neighborhood": "Sunset",
        },
        {
            "min_lat": 37.765,
            "max_lat": 37.780,
            "min_lon": -122.452,
            "max_lon": -122.435,
            "neighborhood": "Western Addition",
        },
        {
            "min_lat": 37.768,
            "max_lat": 37.778,
            "min_lon": -122.452,
            "max_lon": -122.435,
            "neighborhood": "Haight Ashbury",
        },
        {
            "min_lat": 37.765,
            "max_lat": 37.789,
            "min_lon": -122.414,
            "max_lon": -122.39,
            "neighborhood": "Tenderloin",
        },
        {
            "min_lat": 37.775,
            "max_lat": 37.795,
            "min_lon": -122.40,
            "max_lon": -122.38,
            "neighborhood": "Financial District",
        },
        {
            "min_lat": 37.765,
            "max_lat": 37.780,
            "min_lon": -122.40,
            "max_lon": -122.38,
            "neighborhood": "South of Market",
        },
        {
            "min_lat": 37.740,
            "max_lat": 37.765,
            "min_lon": -122.44,
            "max_lon": -122.414,
            "neighborhood": "Castro/Noe Valley",
        },
        {
            "min_lat": 37.745,
            "max_lat": 37.765,
            "min_lon": -122.414,
            "max_lon": -122.395,
            "neighborhood": "Mission",
        },
        {
            "min_lat": 37.730,
            "max_lat": 37.745,
            "min_lon": -122.414,
            "max_lon": -122.39,
            "neighborhood": "Bernal Heights",
        },
        {
            "min_lat": 37.710,
            "max_lat": 37.740,
            "min_lon": -122.414,
            "max_lon": -122.36,
            "neighborhood": "Bayview/Hunters Point",
        },
        {
            "min_lat": 37.710,
            "max_lat": 37.730,
            "min_lon": -122.47,
            "max_lon": -122.414,
            "neighborhood": "Excelsior",
        },
    ]

    # Filter to only process rows with valid coordinates and missing neighborhood or district
    rows_to_process = sf_df[
        (sf_df["latitude"].notna())
        & (sf_df["longitude"].notna())
        & (
            (
                (sf_df["neighborhoods_analysis_boundaries"] == "nan")
                | sf_df["neighborhoods_analysis_boundaries"].isna()
            )
            | (
                (sf_df["supervisor_district"] == "nan")
                | sf_df["supervisor_district"].isna()
            )
        )
    ]

    if len(rows_to_process) == 0:
        logger.info(
            "No SF rows with valid coordinates and missing geographic information."
        )
        return df

    logger.info(
        f"Processing {len(rows_to_process)} SF rows with missing geographic info..."
    )

    filled_neighborhoods = 0
    filled_districts = 0

    # Process each row with missing data
    for idx, row in rows_to_process.iterrows():
        # Convert coordinates to numeric to ensure they're valid
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except (ValueError, TypeError):
            continue

        # Fill missing neighborhood if needed
        if row["neighborhoods_analysis_boundaries"] == "nan" or pd.isna(
            row["neighborhoods_analysis_boundaries"]
        ):

            # Find matching neighborhood from lookup
            for area in neighborhood_lookup:
                if (
                    area["min_lat"] <= lat <= area["max_lat"]
                    and area["min_lon"] <= lon <= area["max_lon"]
                ):
                    neighborhood = area["neighborhood"]
                    # Update both dataframes
                    sf_df.at[idx, "neighborhoods_analysis_boundaries"] = neighborhood
                    df.at[idx, "neighborhoods_analysis_boundaries"] = neighborhood
                    filled_neighborhoods += 1
                    break

            # If no match found, use "San Francisco" as default
            if row["neighborhoods_analysis_boundaries"] == "nan" or pd.isna(
                row["neighborhoods_analysis_boundaries"]
            ):
                sf_df.at[idx, "neighborhoods_analysis_boundaries"] = "San Francisco"
                df.at[idx, "neighborhoods_analysis_boundaries"] = "San Francisco"
                filled_neighborhoods += 1

        # Fill missing supervisor district if needed
        if row["supervisor_district"] == "nan" or pd.isna(row["supervisor_district"]):

            # Find matching district from lookup
            for area in district_lookup:
                if (
                    area["min_lat"] <= lat <= area["max_lat"]
                    and area["min_lon"] <= lon <= area["max_lon"]
                ):
                    district = area["district"]
                    # Update both dataframes
                    sf_df.at[idx, "supervisor_district"] = district
                    df.at[idx, "supervisor_district"] = district
                    filled_districts += 1
                    break

            # If no match found, use 0 as default (unspecified district)
            if row["supervisor_district"] == "nan" or pd.isna(
                row["supervisor_district"]
            ):
                sf_df.at[idx, "supervisor_district"] = 0.0
                df.at[idx, "supervisor_district"] = 0.0
                filled_districts += 1

    # Count missing values after processing
    missing_neighborhoods_after = (
        (sf_df["neighborhoods_analysis_boundaries"] == "nan")
        | sf_df["neighborhoods_analysis_boundaries"].isna()
    ).sum()
    missing_districts_after = (
        (sf_df["supervisor_district"] == "nan") | sf_df["supervisor_district"].isna()
    ).sum()

    logger.info(
        f"After: Missing SF neighborhoods: {missing_neighborhoods_after}, Missing SF districts: {missing_districts_after}"
    )
    logger.info(
        f"Filled {filled_neighborhoods} neighborhoods and {filled_districts} districts"
    )

    return df


def add_business_status_and_age(df):
    """
    Add columns for business status (open/closed) and business age.
    Works with date columns in either MM/DD/YYYY string format or datetime format.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with date columns

    Returns:
    --------
    pandas DataFrame
        DataFrame with new columns: is_open and business_age_years
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.business_data_processing_15")
    
    logger.info("Adding business status and age columns...")

    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()

    # Get current date for calculations
    current_date = pd.Timestamp.now()

    # Determine if business is open based on end dates
    # A business is considered closed if it has an end date in the past
    df_copy["is_open"] = True  # Default to open

    # Check DBA end date (if exists)
    dba_end_col = "Dba_end_date"
    if dba_end_col in df_copy.columns:
        # Convert date strings to datetime objects for comparison
        end_dates = pd.to_datetime(df_copy[dba_end_col], errors="coerce")
        mask_dba_closed = (end_dates.notna()) & (end_dates < current_date)
        df_copy.loc[mask_dba_closed, "is_open"] = False

    # Check location end date (if exists)
    loc_end_col = "Location_end_date"
    if loc_end_col in df_copy.columns:
        # Convert date strings to datetime objects for comparison
        end_dates = pd.to_datetime(df_copy[loc_end_col], errors="coerce")
        mask_location_closed = (end_dates.notna()) & (end_dates < current_date)
        df_copy.loc[mask_location_closed, "is_open"] = False

    # Check administratively closed flag if it exists
    if "administratively_closed" in df_copy.columns:
        mask_admin_closed = df_copy["administratively_closed"] == "Y"
        df_copy.loc[mask_admin_closed, "is_open"] = False

    # Calculate business age at location
    # Use location_start_date if available, otherwise use dba_start_date
    loc_start_col = "Location_start_date"
    dba_start_col = "Dba_start_date"

    start_date_col = None
    if loc_start_col in df_copy.columns and df_copy[loc_start_col].notna().any():
        start_date_col = loc_start_col
    elif dba_start_col in df_copy.columns and df_copy[dba_start_col].notna().any():
        start_date_col = dba_start_col

    if start_date_col is not None:
        # Calculate business age in years
        df_copy["business_age_years"] = None

        # Convert date strings to datetime for calculations
        start_dates = pd.to_datetime(df_copy[start_date_col], errors="coerce")

        # For open businesses, calculate age from start to current date
        open_mask = df_copy["is_open"] & start_dates.notna()
        if open_mask.any():
            df_copy.loc[open_mask, "business_age_years"] = (
                (current_date - start_dates.loc[open_mask]).dt.days / 365.25
            ).round(1)

        # For closed businesses, calculate age from start to end date
        closed_mask = ~df_copy["is_open"] & start_dates.notna()
        if closed_mask.any():
            # Find end date (use the earlier of dba_end_date and location_end_date if both exist)
            for idx in df_copy.loc[closed_mask].index:
                end_date = current_date  # Default to current date

                # Check both end dates
                dba_end = (
                    pd.to_datetime(df_copy.at[idx, dba_end_col])
                    if dba_end_col in df_copy.columns
                    else pd.NaT
                )
                loc_end = (
                    pd.to_datetime(df_copy.at[idx, loc_end_col])
                    if loc_end_col in df_copy.columns
                    else pd.NaT
                )

                # Use the earlier non-null end date
                if pd.notna(dba_end) and pd.notna(loc_end):
                    end_date = min(dba_end, loc_end)
                elif pd.notna(dba_end):
                    end_date = dba_end
                elif pd.notna(loc_end):
                    end_date = loc_end

                # Calculate age
                start_date = pd.to_datetime(df_copy.at[idx, start_date_col])
                if pd.notna(start_date):
                    age_years = (end_date - start_date).days / 365.25
                    df_copy.at[idx, "business_age_years"] = round(age_years, 1)

    # Print statistics
    open_count = df_copy["is_open"].sum()
    closed_count = len(df_copy) - open_count
    logger.info(f"Business status: {open_count} open, {closed_count} closed")

    avg_age = df_copy.loc[
        df_copy["business_age_years"].notna(), "business_age_years"
    ].mean()
    if not pd.isna(avg_age):
        logger.info(f"Average business age: {avg_age:.1f} years")
    else:
        logger.info("Could not calculate average business age due to missing data")

    return df_copy


def create_processing_summary(df, summary_path, input_path, sf_only):
    """Create a summary report of the business data processing"""
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.business_data_processing_15")
    
    try:
        with open(summary_path, "w") as f:
            f.write(f"SF Business Data Processing Report\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(
                f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Input file: {input_path}\n")
            f.write(f"SF only filter: {sf_only}\n")
            f.write(f"Total records processed: {len(df):,}\n\n")

            # Industry distribution
            f.write("Industry Distribution (Top 15):\n")
            industry_counts = df["business_industry"].value_counts().head(15)
            for industry, count in industry_counts.items():
                f.write(f"  {industry}: {count:,} ({count/len(df)*100:.1f}%)\n")
            f.write("\n")

            # Business status
            if "is_open" in df.columns:
                open_count = df["is_open"].sum()
                closed_count = len(df) - open_count
                f.write(f"Business Status:\n")
                f.write(f"  Open: {open_count:,} ({open_count/len(df)*100:.1f}%)\n")
                f.write(
                    f"  Closed: {closed_count:,} ({closed_count/len(df)*100:.1f}%)\n\n"
                )

            # Geographic coverage
            if "in_sf" in df.columns:
                sf_count = df["in_sf"].sum()
                f.write(f"Geographic Distribution:\n")
                f.write(
                    f"  San Francisco: {sf_count:,} ({sf_count/len(df)*100:.1f}%)\n"
                )
                f.write(
                    f"  Other locations: {len(df)-sf_count:,} ({(len(df)-sf_count)/len(df)*100:.1f}%)\n\n"
                )

            # Data completeness
            f.write("Data Completeness:\n")
            for col in [
                "business_industry",
                "neighborhoods_analysis_boundaries",
                "supervisor_district",
                "latitude",
                "longitude",
            ]:
                if col in df.columns:
                    non_null = df[col].notna().sum()
                    f.write(f"  {col}: {non_null:,} ({non_null/len(df)*100:.1f}%)\n")

        logger.info(f"Processing summary saved to {summary_path}")

    except Exception as e:
        logger.error(f"Error creating processing summary: {e}")


def process_businesses_from_drive(
    input_path, output_path, business_processed_dir, sf_only=True
):
    """
    Process a registered_businesses parquet file:
    1. Add industry classification
    2. Clean date columns to MM/DD/YYYY format (with capitalized column names)
    3. Extract latitude and longitude from location field and flag SF businesses
    4. Fill missing neighborhood and supervisor district data using simplified lookup
    5. Add business status (open/closed) and age information
    6. Filter to SF-only businesses if requested
    7. Save the output as a parquet file

    Parameters:
    -----------
    input_path : str
        Path to the input parquet file
    output_path : str
        Path where to save the output parquet file
    business_processed_dir : str
        Directory for processed business data
    sf_only : bool
        If True, filter the final dataset to only include San Francisco businesses

    Returns:
    --------
    pandas DataFrame
        Processed DataFrame
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.business_data_processing_15")
    
    logger.info(f"Loading data from {input_path}...")

    # Load data
    df = load_data(file_path=input_path)
    logger.info(f"Data loaded successfully with {len(df)} records.")

    # Create business industry column
    logger.info("Adding business industry classification...")
    df = create_business_industry_column(df)

    # Clean date columns
    logger.info("Standardizing date columns...")
    df = clean_date_columns(df)

    # Extract latitude and longitude and flag SF businesses
    logger.info("Extracting geographic coordinates...")
    df = extract_lat_long(df)

    # Fill missing neighborhood and supervisor district data with simplified approach
    logger.info("Filling missing neighborhood and district data...")
    df = fill_missing_geo_data(df)

    # Add business status and age information
    logger.info("Adding business status and age information...")
    df = add_business_status_and_age(df)

    # Filter to SF-only businesses if requested
    if sf_only:
        sf_businesses = df[df["in_sf"] == True].copy()
        logger.info(
            f"Filtered to {len(sf_businesses)} San Francisco businesses (from original {len(df)} records)"
        )
        df = sf_businesses

    # Print statistics
    logger.info(f"\nTotal records in final dataset: {len(df)}")
    logger.info(f"\nIndustry classification counts:")
    industry_counts = df["business_industry"].value_counts().head(10)
    for industry, count in industry_counts.items():
        logger.info(f"  {industry}: {count}")

    # Calculate improvement statistics
    previously_unclassified = df[
        df["naic_code_description"].isin(["nan", None])
        | pd.isna(df["naic_code_description"])
    ].shape[0]
    now_classified = df[~df["business_industry"].isin(["Unclassified"])].shape[0]

    logger.info(f"\nPreviously unclassified records: {previously_unclassified}")
    logger.info(
        f"Records with industry classification after processing: {now_classified}"
    )
    logger.info(
        f"Improvement: {now_classified - (len(df) - previously_unclassified)} additional records classified"
    )

    # Calculate percentage of records with complete geographic data
    has_neighborhood = (
        ~(df["neighborhoods_analysis_boundaries"] == "nan")
        & ~df["neighborhoods_analysis_boundaries"].isna()
    ).sum()
    has_district = (
        ~(df["supervisor_district"] == "nan") & ~df["supervisor_district"].isna()
    ).sum()

    logger.info(f"\nGeographic data coverage:")
    logger.info(
        f"Records with neighborhood: {has_neighborhood} ({has_neighborhood/len(df)*100:.1f}%)"
    )
    logger.info(
        f"Records with district: {has_district} ({has_district/len(df)*100:.1f}%)"
    )

    # Fix data types before saving
    logger.info("Converting data types for compatibility...")

    # Ensure supervisor_district is float type
    try:
        df["supervisor_district"] = pd.to_numeric(
            df["supervisor_district"], errors="coerce"
        )
        logger.info("Converted supervisor_district to numeric type")
    except Exception as e:
        logger.error(f"Error converting supervisor_district: {e}")

    # Convert other numeric columns that might have mixed types
    for col in ["latitude", "longitude", "business_age_years"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as parquet file using the exact output path
    logger.info(f"Saving enriched file to {output_path}...")
    try:
        df.to_parquet(output_path, index=False)
        logger.info(f"File saved successfully.")
    except Exception as e:
        logger.error(f"Error saving to parquet: {e}")
        # Fallback: try to save as CSV if parquet fails
        csv_path = output_path.replace(".parquet", ".csv")
        logger.info(f"Attempting to save as CSV to {csv_path}...")
        df.to_csv(csv_path, index=False)
        logger.info("CSV file saved as fallback.")

    # Create processing summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = (
        f"{business_processed_dir}/business_processing_summary_{timestamp}.txt"
    )
    create_processing_summary(df, summary_path, input_path, sf_only)

    return df


def main_business_processing(raw_data_dir, processed_dir, sf_only=True):
    """
    Main function to process business data

    Parameters:
    -----------
    raw_data_dir : str
        Raw data directory
    processed_dir : str
        Processed data directory
    sf_only : bool
        Whether to filter to SF businesses only

    Returns:
    --------
    pd.DataFrame or None
        Processed business DataFrame
    """
    # Use the unified logging system
    logger = logging.getLogger("SFBusinessPipeline.business_data_processing_15")
    
    # Define business specific directories
    business_raw_dir = f"{raw_data_dir}/sf_business"
    business_processed_dir = f"{processed_dir}/sf_business"

    # Create directories
    os.makedirs(business_raw_dir, exist_ok=True)
    os.makedirs(business_processed_dir, exist_ok=True)

    # Use the exact original file paths from the Google Colab version
    input_path = f"{business_raw_dir}/registered_businesses_raw.parquet"
    output_path = f"{business_processed_dir}/enriched_registered_businesses.parquet"

    # Check if input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        logger.info(f"Available files in {business_raw_dir}:")
        try:
            files = os.listdir(business_raw_dir)
            for file in files:
                if file.endswith(".parquet"):
                    logger.info(f"  - {file}")
        except:
            logger.info("  No files found or directory doesn't exist")
        return None

    # Process the businesses
    result_df = process_businesses_from_drive(
        input_path=input_path,
        output_path=output_path,
        business_processed_dir=business_processed_dir,
        sf_only=sf_only,
    )

    logger.info("Business data processing complete!")
    return result_df


if __name__ == "__main__":
    # Use unified config - same functions as your _02 script
    logger = setup_logging()
    config = setup_directories()
    
    logger.info("Starting Business Data Processing pipeline")
    logger.info(f"Base directory: {config['base_dir']}")
    
    # Create directories if they don't exist - same pattern as your _02 script
    os.makedirs(f"{config['raw_data_dir']}/sf_business", exist_ok=True)
    os.makedirs(f"{config['processed_dir']}/sf_business", exist_ok=True)

    logger.info("Starting business data processing pipeline...")

    # Execute the main function
    result = main_business_processing(
        config["raw_data_dir"],
        config["processed_dir"],
        sf_only=True,  # Keep original behavior - SF businesses only
    )

    if result is not None and not result.empty:
        print(f"Successfully processed business data with {len(result)} records")

        # Show industry distribution
        if "business_industry" in result.columns:
            print(f"\nTop 10 industries:")
            industry_counts = result["business_industry"].value_counts().head(10)
            for industry, count in industry_counts.items():
                print(f"  {industry}: {count:,}")

        # Show business status
        if "is_open" in result.columns:
            open_count = result["is_open"].sum()
            closed_count = len(result) - open_count
            print(f"\nBusiness status:")
            print(f"  Open: {open_count:,}")
            print(f"  Closed: {closed_count:,}")

        # Show geographic coverage
        if "neighborhoods_analysis_boundaries" in result.columns:
            neighborhoods_with_data = (
                result["neighborhoods_analysis_boundaries"].notna().sum()
            )
            print(f"\nGeographic data:")
            print(
                f"  Records with neighborhood data: {neighborhoods_with_data:,} ({neighborhoods_with_data/len(result)*100:.1f}%)"
            )

        print(f"\nSample of processed data:")
        display_cols = [
            "dba_name",
            "business_industry",
            "neighborhoods_analysis_boundaries",
            "is_open",
        ]
        available_cols = [col for col in display_cols if col in result.columns]
        if available_cols:
            print(result[available_cols].head())
    else:
        print("Business data processing failed - no data to display")

    logger.info("Business data processing pipeline completed")