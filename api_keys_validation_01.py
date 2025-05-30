import os
import logging
from dotenv import load_dotenv

# Setup logging
logger = logging.getLogger(__name__)


def validate_api_keys():
    """Load and validate API keys from environment variables."""
    # Load environment variables from .env file
    load_dotenv()

    # Access your API keys
    fred_api_key = os.getenv("FRED_API_KEY")
    census_data_api_key = os.getenv("CENSUS_DATA_API_KEY")

    # Validate keys exist
    if not fred_api_key:
        logger.error(
            "FRED API key not found. Please set FRED_API_KEY environment variable."
        )
        raise ValueError("Missing FRED API key")

    if not census_data_api_key:
        logger.error(
            "Census Data API key not found. Please set CENSUS_DATA_API_KEY environment variable."
        )
        raise ValueError("Missing Census Data API key")

    logger.info("API keys successfully loaded and validated")
    return fred_api_key, census_data_api_key


if __name__ == "__main__":
    fred_api_key, census_data_api_key = validate_api_keys()
