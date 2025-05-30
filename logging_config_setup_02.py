import os
import logging
from datetime import datetime, timedelta


def setup_logging():
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("SFBusinessPipeline")


def setup_directories():
    """Create directory structure and return paths and date ranges."""
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

    # Calculate date ranges for 10+ years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 11)  # 11 years ago for buffer

    logger.info(
        f"Directory structure created successfully. Data period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )

    return {
        "base_dir": base_dir,
        "raw_data_dir": raw_data_dir,
        "processed_dir": processed_dir,
        "model_dir": model_dir,
        "archive_dir": archive_dir,
        "start_date": start_date,
        "end_date": end_date,
        "data_sources": data_sources,
    }


if __name__ == "__main__":
    logger = setup_logging()
    config = setup_directories()
    print(
        f"Data collection period: {config['start_date'].strftime('%Y-%m-%d')} to {config['end_date'].strftime('%Y-%m-%d')}"
    )
