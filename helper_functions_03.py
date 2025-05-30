import requests
import time
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Setup logging
logger = logging.getLogger(__name__)


def make_api_request(url, headers=None, params=None, max_retries=3, backoff_factor=2):
    """Generic API request function with retry logic"""
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            wait_time = backoff_factor**retries
            logger.warning(f"Error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    logger.error(f"Failed after {max_retries} retries.")
    return None


def save_to_parquet(df, directory, filename):
    """Save DataFrame to Parquet format with appropriate schema"""
    full_path = f"{directory}/{filename}.parquet"
    # Ensure all string columns are properly encoded
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("str")
    # Convert to PyArrow table and write to Parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, full_path, compression="snappy")
    logger.info(f"Saved to {full_path}")
    return full_path


if __name__ == "__main__":
    # Test functions when run directly
    logger.info("Helper functions module loaded successfully")
    print("Available functions: make_api_request, save_to_parquet")
