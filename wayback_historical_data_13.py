import os
import time
import random
import pickle
import logging
import pandas as pd
import requests
import sys
import feedparser
import threading
import gc
from datetime import datetime, timedelta
from tqdm import tqdm
from urllib.parse import urljoin
import psutil

# Add project root to path if running directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

# Import unified config - same functions as your _02 script
from config import setup_logging, setup_directories
# Setup logging
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('wayback_collection.log')
        ]
    )
    return logging.getLogger(__name__)



def save_to_parquet(df, directory, filename):
    """Save DataFrame to Parquet format with appropriate schema"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        # Fallback to pandas parquet if pyarrow not available
        logging.warning("PyArrow not available, using pandas parquet engine")
        df.to_parquet(f"{directory}/{filename}.parquet", index=False)
        return f"{directory}/{filename}.parquet"

    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)

    full_path = f"{directory}/{filename}.parquet"

    try:
        # Ensure all string columns are properly encoded
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('str')

        # Convert to PyArrow table and write to Parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, full_path, compression='snappy')

        logger.info(f"Saved to {full_path}")
        return full_path
    except Exception as e:
        logger.error(f"Error saving with PyArrow: {e}")
        # Fallback to pandas
        df.to_parquet(full_path, index=False)
        logger.info(f"Saved to {full_path} using pandas")
        return full_path

# Initialize logger
logger = setup_logging()

def save_progress(data, filepath, archive_dir):
    """Save progress data with backup"""
    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        # Make a backup copy with timestamp in archive directory
        backup_name = os.path.basename(filepath).replace(
            ".pkl", f'_backup_{datetime.now().strftime("%H%M%S")}.pkl'
        )
        backup_path = os.path.join(archive_dir, backup_name)

        # Ensure archive directory exists
        os.makedirs(archive_dir, exist_ok=True)

        with open(backup_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Progress saved to {filepath} with backup at {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving progress: {e}")
        return False

def auto_save_thread(stop_event, save_data_func, interval=300):
    """Thread function to auto-save progress periodically"""
    while not stop_event.is_set():
        try:
            save_data_func()
            logger.info(f"Auto-saved progress (periodic save)")
        except Exception as e:
            logger.error(f"Error in auto-save thread: {e}")

        # Sleep for the specified interval (in small chunks to check stop_event frequently)
        for _ in range(interval):
            if stop_event.is_set():
                break
            time.sleep(1)

def get_wayback_snapshots(url, from_year, to_year):
    """
    Get available snapshots from Wayback Machine for a given URL and date range

    Parameters:
    -----------
    url : str
        The URL to search for
    from_year : str
        Start year
    to_year : str
        End year

    Returns:
    --------
    list
        List of (timestamp, url) tuples for available snapshots
    """
    wayback_api_url = "http://web.archive.org/cdx/search/cdx"

    params = {
        "url": url,
        "from": f"{from_year}0101",
        "to": f"{to_year}1231",
        "output": "json",
        "fl": "timestamp,original",
        "filter": "statuscode:200",
        "collapse": "timestamp:8",  # Collapse to daily snapshots
    }

    try:
        response = requests.get(wayback_api_url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Skip the header row
        if len(data) > 1:
            snapshots = [(row[0], row[1]) for row in data[1:]]
            logger.info(
                f"Found {len(snapshots)} snapshots for {url} from {from_year} to {to_year}"
            )
            return snapshots
        else:
            logger.warning(f"No snapshots found for {url} in the specified date range")
            return []

    except Exception as e:
        logger.error(f"Error fetching snapshots from Wayback Machine: {e}")
        return []

def construct_archived_url(snapshot):
    """
    Construct the full Wayback Machine URL from a snapshot tuple

    Parameters:
    -----------
    snapshot : tuple
        (timestamp, original_url) tuple

    Returns:
    --------
    str
        Full Wayback Machine URL
    """
    timestamp, original_url = snapshot
    return f"http://web.archive.org/web/{timestamp}/{original_url}"

def fetch_archived_rss(archived_url):
    """
    Fetch and parse an archived RSS feed

    Parameters:
    -----------
    archived_url : str
        Full Wayback Machine URL

    Returns:
    --------
    feedparser.FeedParserDict or None
        Parsed feed or None if failed
    """
    try:
        # Set a reasonable timeout and user agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(archived_url, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse the RSS content
        feed = feedparser.parse(response.content)

        if hasattr(feed, "bozo") and feed.bozo:
            logger.warning(f"Malformed feed detected for {archived_url}")

        return feed

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching {archived_url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request error for {archived_url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {archived_url}: {e}")
        return None

def deduplicate_articles(articles):
    """
    Remove duplicate articles based on title and link

    Parameters:
    -----------
    articles : list
        List of article dictionaries

    Returns:
    --------
    list
        Deduplicated list of articles
    """
    seen = set()
    unique_articles = []

    for article in articles:
        # Create a key based on title and link
        key = (
            article.get("title", "").strip().lower(),
            article.get("link", "").strip(),
        )

        if key not in seen and key[0] and key[1]:  # Ensure both title and link exist
            seen.add(key)
            unique_articles.append(article)

    logger.info(
        f"Deduplicated {len(articles)} articles to {len(unique_articles)} unique articles"
    )
    return unique_articles

def collect_rss_data(
    rss_url,
    from_year,
    to_year,
    news_raw_dir,
    archive_dir,
    batch_size=10,
    resume_file=None,
    max_snapshots=None,
):
    """
    Collect historical RSS data from Wayback Machine with resumable functionality.

    Parameters:
    -----------
    rss_url : str
        URL of the RSS feed to collect from
    from_year : str
        Start year for collection
    to_year : str
        End year for collection
    news_raw_dir : str
        Directory to save raw news data
    archive_dir : str
        Directory to save archive and backup data
    batch_size : int
        Number of snapshots to process before taking a longer break
    resume_file : str, optional
        Path to resume file if resuming a previous run
    max_snapshots : int, optional
        Maximum number of snapshots to process (for testing or limiting runtime)

    Returns:
    --------
    pd.DataFrame
        DataFrame of collected articles
    """
    # Initialize variables for resume functionality
    all_articles = []
    start_batch = 0
    snapshots = []
    current_batch_idx = 0

    # Check if we're resuming from a previous run
    if resume_file and os.path.exists(resume_file):
        try:
            with open(resume_file, "rb") as f:
                resume_data = pickle.load(f)
                all_articles = resume_data.get("articles", [])
                start_batch = resume_data.get("next_batch", 0)
                snapshots = resume_data.get("snapshots", [])
                logger.info(
                    f"Resuming from batch {start_batch} with {len(all_articles)} articles already collected"
                )
        except Exception as e:
            logger.error(f"Error loading resume file: {e}")
            start_batch = 0
            snapshots = get_wayback_snapshots(rss_url, from_year, to_year)
    else:
        # Get snapshots from the Wayback Machine for the given RSS feed
        snapshots = get_wayback_snapshots(rss_url, from_year, to_year)

    if not snapshots:
        logger.warning("No snapshots found. Cannot proceed with collection.")
        return pd.DataFrame()

    # Limit snapshots if max_snapshots is specified
    if max_snapshots and len(snapshots) > max_snapshots:
        logger.info(
            f"Limiting to {max_snapshots} snapshots (out of {len(snapshots)} available)"
        )
        snapshots = snapshots[:max_snapshots]

    logger.info(f"Total snapshots to process: {len(snapshots)}")

    # Create a stop event for the auto-save thread
    stop_event = threading.Event()

    # Define the auto-save function
    def auto_save_func():
        nonlocal current_batch_idx
        progress_file = f"{news_raw_dir}/collection_progress_{from_year}_{to_year}.pkl"
        next_batch = (
            start_batch + ((current_batch_idx - start_batch) // batch_size) * batch_size
        )
        resume_data = {
            "articles": all_articles,
            "next_batch": next_batch,
            "snapshots": snapshots,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "from_year": from_year,
            "to_year": to_year,
            "rss_url": rss_url,
        }
        save_progress(resume_data, progress_file, archive_dir)

    # Start the auto-save thread - save every 5 minutes
    auto_save_thread_obj = threading.Thread(
        target=auto_save_thread, args=(stop_event, auto_save_func, 300)
    )
    auto_save_thread_obj.daemon = True
    auto_save_thread_obj.start()

    try:
        # Process snapshots in batches with frequent saves
        for batch_idx in range(start_batch, len(snapshots), batch_size):
            current_batch_idx = batch_idx

            batch_end = min(batch_idx + batch_size, len(snapshots))
            batch = snapshots[batch_idx:batch_end]

            batch_num = batch_idx // batch_size + 1
            total_batches = (len(snapshots) + batch_size - 1) // batch_size
            logger.info(f"Processing batch {batch_num}/{total_batches}")

            batch_articles = []

            # Process this batch of snapshots
            for idx, snapshot in enumerate(
                tqdm(batch, desc=f"Processing batch {batch_num}/{total_batches}")
            ):
                archived_url = construct_archived_url(snapshot)

                try:
                    feed = fetch_archived_rss(archived_url)

                    # If the feed is successfully parsed, extract entries (articles)
                    if feed and hasattr(feed, "entries") and feed.entries:
                        for entry in feed.entries:
                            article = {
                                "title": entry.get("title", ""),
                                "link": entry.get("link", ""),
                                "published": entry.get("published", ""),
                                "description": entry.get("description", ""),
                                "snapshot_date": snapshot[1],  # Add the snapshot date
                                "categories": (
                                    ",".join(
                                        [
                                            cat.get("term", "")
                                            for cat in entry.get("tags", [])
                                        ]
                                    )
                                    if hasattr(entry, "tags")
                                    else ""
                                ),
                                "author": (
                                    entry.get("author", "")
                                    if hasattr(entry, "author")
                                    else ""
                                ),
                                "source_url": rss_url,
                                "wayback_url": archived_url,
                            }
                            batch_articles.append(article)
                    else:
                        logger.warning(f"No entries found for snapshot {snapshot[1]}")
                except Exception as e:
                    logger.error(f"Error processing snapshot {snapshot[1]}: {e}")

                # Short sleep between snapshots within a batch
                time.sleep(1)

            # Add batch articles to all_articles
            all_articles.extend(batch_articles)

            # Save progress after each batch
            progress_file = (
                f"{news_raw_dir}/collection_progress_{from_year}_{to_year}.pkl"
            )
            resume_data = {
                "articles": all_articles,
                "next_batch": batch_idx + batch_size,
                "snapshots": snapshots,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "from_year": from_year,
                "to_year": to_year,
                "rss_url": rss_url,
            }

            save_progress(resume_data, progress_file, archive_dir)
            logger.info(
                f"Progress saved. {len(all_articles)} articles collected so far."
            )

            # Save intermediate results every 5 batches or if we have a significant number of articles
            if batch_num % 5 == 0 or len(all_articles) > 1000:
                try:
                    # First deduplicate to save space
                    unique_articles = deduplicate_articles(all_articles)
                    intermediate_df = pd.DataFrame(unique_articles)

                    if not intermediate_df.empty:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        intermediate_file = f"intermediate_{from_year}_{to_year}_batch{batch_num}_{timestamp}"
                        save_to_parquet(
                            intermediate_df, news_raw_dir, intermediate_file
                        )

                        # Force garbage collection to free memory
                        del intermediate_df
                        gc.collect()
                except Exception as e:
                    logger.error(f"Error saving intermediate results: {e}")

            # Take a break between batches to avoid rate limiting
            if batch_end < len(snapshots):
                pause_time = 15 + random.uniform(0, 5)  # 15-20 second pause
                logger.info(f"Taking a {pause_time:.1f} second break between batches")
                time.sleep(pause_time)

    finally:
        # Stop the auto-save thread
        stop_event.set()
        auto_save_thread_obj.join(timeout=5)

    logger.info(f"Total articles collected: {len(all_articles)}")

    if not all_articles:
        logger.warning("No articles collected. Returning empty DataFrame.")
        return pd.DataFrame()

    # Deduplicate articles
    unique_articles = deduplicate_articles(all_articles)

    # Force garbage collection before creating DataFrame
    gc.collect()

    # Convert the collected articles into a Pandas DataFrame
    df_articles = pd.DataFrame(unique_articles)

    # Convert the 'published' field to a datetime column if possible
    if not df_articles.empty and "published" in df_articles.columns:
        try:
            # Handle common timezone issues
            df_articles["published"] = df_articles["published"].str.replace(
                " UT", " UTC", regex=False
            )
            df_articles["published"] = df_articles["published"].str.replace(
                " GMT", " UTC", regex=False
            )

            df_articles["published_date"] = pd.to_datetime(
                df_articles["published"], errors="coerce"
            )

            # Add new columns for year, month, day for easier analysis
            df_articles["year"] = df_articles["published_date"].dt.year
            df_articles["month"] = df_articles["published_date"].dt.month
            df_articles["day"] = df_articles["published_date"].dt.day

            # Add data source for tracking
            df_articles["data_source"] = "wayback_machine"

        except Exception as e:
            logger.error(f"Error converting published dates: {e}")

    return df_articles

def check_system_resources():
    """
    Check system resources and estimate remaining capacity.
    Returns memory usage percentage and available disk space.
    """
    try:
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Get disk usage for the base directory
        disk = psutil.disk_usage("/")  # Use root for disk check
        disk_free_gb = disk.free / (1024**3)

        return memory_percent, disk_free_gb
    except Exception as e:
        logger.warning(f"Error checking system resources: {e}")
        return 50, 100  # Conservative defaults

def estimate_collection_time(
    date_ranges, snapshots_per_year=150, seconds_per_batch=240, batch_size=10
):
    """
    Estimate how long collection will take.

    Parameters:
    -----------
    date_ranges : list
        List of (from_year, to_year) tuples
    snapshots_per_year : int
        Estimated snapshots per year
    seconds_per_batch : int
        Average seconds per batch
    batch_size : int
        Batch size

    Returns:
    --------
    float
        Estimated total hours
    """
    total_snapshots = 0

    for from_year, to_year in date_ranges:
        years = int(to_year) - int(from_year) + 1
        total_snapshots += years * snapshots_per_year

    total_batches = (total_snapshots + batch_size - 1) // batch_size
    total_seconds = total_batches * seconds_per_batch
    total_hours = total_seconds / 3600

    return total_hours

def resume_collection(
    from_year, to_year, news_raw_dir, archive_dir, rss_url=None, max_snapshots=None
):
    """
    Resume a collection that was paused or interrupted.

    Parameters:
    -----------
    from_year : str
        Start year of the original collection
    to_year : str
        End year of the original collection
    news_raw_dir : str
        Directory for raw news data
    archive_dir : str
        Directory for archive data
    rss_url : str, optional
        RSS URL to collect from
    max_snapshots : int, optional
        Maximum number of snapshots to process

    Returns:
    --------
    pd.DataFrame
        DataFrame of collected articles
    """
    if rss_url is None:
        rss_url = "https://www.sfgate.com/bayarea/feed/Bay-Area-News-429.php"

    progress_file = f"{news_raw_dir}/collection_progress_{from_year}_{to_year}.pkl"

    if not os.path.exists(progress_file):
        logger.warning(
            f"No progress file found for {from_year}-{to_year}. Starting new collection."
        )
        return collect_rss_data(
            rss_url,
            from_year,
            to_year,
            news_raw_dir,
            archive_dir,
            batch_size=10,
            max_snapshots=max_snapshots,
        )

    logger.info(f"Resuming collection for {from_year}-{to_year}")
    return collect_rss_data(
        rss_url,
        from_year,
        to_year,
        news_raw_dir,
        archive_dir,
        batch_size=10,
        resume_file=progress_file,
        max_snapshots=max_snapshots,
    )

def smart_collect_data(
    date_ranges,
    raw_data_dir,
    archive_dir,
    rss_url=None,
    batch_size=10,
    prioritize_recent=True,
):
    """
    Collect data efficiently with system resource monitoring.

    Parameters:
    -----------
    date_ranges : list
        List of (from_year, to_year) tuples
    raw_data_dir : str
        Directory for raw data
    archive_dir : str
        Directory for archive data
    rss_url : str, optional
        RSS URL to collect from
    batch_size : int
        Batch size for collection
    prioritize_recent : bool
        If True, collect most recent data first

    Returns:
    --------
    pd.DataFrame
        DataFrame of collected articles
    """
    if rss_url is None:
        rss_url = "https://www.sfgate.com/bayarea/feed/Bay-Area-News-429.php"

    news_raw_dir = f"{raw_data_dir}/news"
    os.makedirs(news_raw_dir, exist_ok=True)

    # Check estimated runtime
    hours_estimate = estimate_collection_time(date_ranges, batch_size=batch_size)
    memory_percent, disk_free_gb = check_system_resources()

    logger.info(f"Estimated collection time: {hours_estimate:.1f} hours")
    logger.info(f"System memory usage: {memory_percent:.1f}%")
    logger.info(f"Available disk space: {disk_free_gb:.1f} GB")

    # Create empty DataFrame to store all results
    all_df = pd.DataFrame()

    # If prioritizing recent data, reverse the order
    if prioritize_recent:
        date_ranges = list(reversed(date_ranges))

    # Process each date range
    for from_year, to_year in date_ranges:
        # Check system resources before each range - be more lenient with memory
        memory_percent, disk_free_gb = check_system_resources()
        if memory_percent > 95 or disk_free_gb < 2:
            logger.warning(
                f"System resources critically low (Memory: {memory_percent:.1f}%, Disk: {disk_free_gb:.1f}GB). Stopping collection."
            )
            break
        elif memory_percent > 90:
            logger.warning(
                f"High memory usage detected ({memory_percent:.1f}%). Running garbage collection..."
            )
            gc.collect()
            # Check again after cleanup
            memory_percent, _ = check_system_resources()
            logger.info(f"Memory usage after cleanup: {memory_percent:.1f}%")

        logger.info(f"Collecting data for period {from_year}-{to_year}")

        # Check if we already have complete data for this range
        complete_file = f"{news_raw_dir}/historical_rss_{from_year}_{to_year}.parquet"
        if os.path.exists(complete_file):
            logger.info(
                f"Found complete data file for {from_year}-{to_year}. Loading existing data."
            )
            try:
                df_existing = pd.read_parquet(complete_file)
                all_df = pd.concat([all_df, df_existing], ignore_index=True)
                logger.info(
                    f"Added {len(df_existing)} existing articles from {from_year}-{to_year}"
                )
                continue
            except Exception as e:
                logger.error(f"Error loading existing data: {e}")

        # Collect the data for this period with adaptive batch size
        max_snapshots = None
        if memory_percent > 90:
            # Limit snapshots if memory is high
            years_in_range = int(to_year) - int(from_year) + 1
            max_snapshots = min(100, years_in_range * 20)  # Conservative limit
            logger.info(
                f"High memory usage - limiting to {max_snapshots} snapshots for {from_year}-{to_year}"
            )

        df_period = resume_collection(
            from_year, to_year, news_raw_dir, archive_dir, rss_url, max_snapshots
        )

        if not df_period.empty:
            # Append to the full dataset
            all_df = pd.concat([all_df, df_period], ignore_index=True)

            # Save period results
            interim_filename = f"historical_rss_{from_year}_{to_year}"
            save_to_parquet(df_period, news_raw_dir, interim_filename)
            logger.info(f"Saved results for {from_year}-{to_year}")

        # Force garbage collection
        del df_period
        gc.collect()

        # Take a break between date ranges
        if (from_year, to_year) != date_ranges[-1]:  # If not the last range
            pause_time = 30 + random.uniform(0, 10)  # 30-40 second pause
            logger.info(
                f"Taking a {pause_time:.1f} second break before processing next date range"
            )
            time.sleep(pause_time)

    return all_df

def test_collection(news_raw_dir, archive_dir, test_snapshots=5):
    """
    Run a small test collection to verify the system works

    Parameters:
    -----------
    news_raw_dir : str
        Directory for raw news data
    archive_dir : str
        Directory for archive data
    test_snapshots : int
        Number of snapshots to test with

    Returns:
    --------
    pd.DataFrame
        Test results
    """
    logger.info(f"Running test collection with {test_snapshots} snapshots")

    # Use just one recent year for testing
    rss_url = "https://www.sfgate.com/bayarea/feed/Bay-Area-News-429.php"

    test_df = collect_rss_data(
        rss_url=rss_url,
        from_year="2023",
        to_year="2023",
        news_raw_dir=news_raw_dir,
        archive_dir=archive_dir,
        batch_size=2,
        max_snapshots=test_snapshots,
    )

    if not test_df.empty:
        logger.info(f"Test collection successful! Collected {len(test_df)} articles")
        print(f"Test results: {len(test_df)} articles collected")
        print("Sample article titles:")
        for title in test_df["title"].head(3):
            print(f"  - {title}")
        return test_df
    else:
        logger.warning("Test collection failed - no articles collected")
        return pd.DataFrame()

def main_wayback_collection(test_mode=True, test_snapshots=5):
    """
    Main execution function using unified configuration system
    """
    # Use unified config - same functions as your _02 script
    logger = setup_logging()
    config = setup_directories()
    
    # Extract paths from config
    raw_data_dir = config["raw_data_dir"]
    processed_dir = config["processed_dir"]
    archive_dir = config["archive_dir"]
    
    logger.info("Starting Wayback Machine data collection")
    logger.info(f"Base directory: {config['base_dir']}")

    # Create news subdirectories
    news_raw_dir = f"{raw_data_dir}/news"
    news_processed_dir = f"{processed_dir}/news"

    os.makedirs(news_raw_dir, exist_ok=True)
    os.makedirs(news_processed_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)

    # Force garbage collection at start to free up memory
    gc.collect()

    # Check initial system state
    memory_percent, disk_free_gb = check_system_resources()
    logger.info(
        f"Initial system state - Memory: {memory_percent:.1f}%, Disk: {disk_free_gb:.1f}GB"
    )

    # Run test mode if requested
    if test_mode:
        return test_collection(news_raw_dir, archive_dir, test_snapshots)

    # If memory is very high, try to clear some before starting
    if memory_percent > 90:
        logger.warning(
            "High initial memory usage detected. Consider closing other applications."
        )
        logger.info("Proceeding with reduced batch size and more frequent cleanup...")
        batch_size = 5  # Smaller batches for high memory situations
    else:
        batch_size = 10

    # Define date ranges to collect data in smaller chunks (most recent first)
    date_ranges = [
        ("2023", "2024"),  # Most recent first
        ("2021", "2022"),
        ("2019", "2020"),
        ("2017", "2018"),
        ("2015", "2016"),
        ("2013", "2014"),  # Oldest last
    ]

    # RSS feed to collect from
    rss_url = "https://www.sfgate.com/bayarea/feed/Bay-Area-News-429.php"

    # Run the smart collection
    logger.info("Starting smart Wayback Machine data collection")
    all_df = smart_collect_data(
        date_ranges,
        raw_data_dir,
        archive_dir,
        rss_url,
        batch_size=batch_size,
        prioritize_recent=True,
    )

    # Display summary of all collected data and save results
    if not all_df.empty:
        print("Summary of all collected articles:")
        print(f"Total articles: {len(all_df)}")

        if "published_date" in all_df.columns:
            print(
                f"Date range: {all_df['published_date'].min()} to {all_df['published_date'].max()}"
            )

        print("\nSample of collected articles:")
        print(all_df.head())

        # Save the final combined dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"historical_wayback_articles_{timestamp}"

        save_to_parquet(all_df, news_raw_dir, final_filename)
        logger.info(f"Saved all articles to {final_filename}.parquet")

        # Also save as CSV if not too large
        if len(all_df) < 100000:
            csv_path = f"{news_raw_dir}/{final_filename}.csv"
            all_df.to_csv(csv_path, index=False)
            logger.info(f"Saved all articles as CSV to {csv_path}")
        else:
            logger.info(
                f"Dataset too large for CSV format ({len(all_df)} rows). Saved as Parquet only."
            )

        # Save summary statistics
        summary_filename = (
            f"{news_processed_dir}/wayback_collection_summary_{timestamp}.txt"
        )
        with open(summary_filename, "w") as f:
            f.write(
                f"Wayback Machine data collection completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"RSS Feed URL: {rss_url}\n")
            f.write(f"Date Range: {date_ranges[-1][0]} to {date_ranges[0][1]}\n")
            f.write(f"Total articles collected: {len(all_df)}\n")

            if "published_date" in all_df.columns:
                f.write(f"Earliest article date: {all_df['published_date'].min()}\n")
                f.write(f"Latest article date: {all_df['published_date'].max()}\n")

                # Count articles by year
                year_counts = all_df["year"].value_counts().sort_index()
                f.write("\nArticles by year:\n")
                for year, count in year_counts.items():
                    f.write(f"  {year}: {count}\n")

        logger.info(f"Summary statistics saved to {summary_filename}")
        logger.info("Wayback Machine data collection and processing complete!")

        return all_df
    else:
        logger.warning("No articles were collected. No files saved.")
        return pd.DataFrame()

if __name__ == "__main__":
    # Wayback Machine Historical Data Collection - Execution
    logger.info("Starting Wayback Machine historical data collection...")

    # You can run in test mode first to verify the system works:
    # collected_data = main_wayback_collection(test_mode=True, test_snapshots=3)

    # Or run the full collection:
    collected_data = main_wayback_collection(test_mode=False)

    if not collected_data.empty:
        print(
            f"Successfully collected Wayback Machine data with {len(collected_data)} articles"
        )

        if "published_date" in collected_data.columns:
            print(
                f"Date range: {collected_data['published_date'].min()} to {collected_data['published_date'].max()}"
            )

        if "year" in collected_data.columns:
            print(f"\nArticles by year:")
            year_counts = collected_data["year"].value_counts().sort_index()
            for year, count in year_counts.items():
                if pd.notna(year):
                    print(f"  {int(year)}: {count:,}")

        print(f"\nSample data:")
        print(collected_data[["title", "published_date", "source_url"]].head())
    else:
        print("No historical data collected")

    logger.info("Wayback Machine historical data collection completed")
