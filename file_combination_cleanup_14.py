import os
import glob
import logging
import pandas as pd
import pytz
import shutil
from datetime import datetime, timedelta
from tqdm import tqdm
from helper_functions_03 import save_to_parquet

# Setup logging
logger = logging.getLogger(__name__)


def normalize_timezone_columns(df):
    """
    Fix timezone issues in datetime columns by ensuring consistency.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with potential timezone issues

    Returns:
    --------
    pd.DataFrame
        DataFrame with normalized timezone handling
    """
    # Check if published_date column exists
    if "published_date" in df.columns:
        try:
            # Convert to datetime if not already (with error handling)
            df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")

            # Drop rows where published_date couldn't be parsed
            original_rows = len(df)
            df = df.dropna(subset=["published_date"])
            dropped_rows = original_rows - len(df)
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with unparseable dates")

            # Check if any timezone info exists
            has_timezone = False
            sample_size = min(100, len(df))
            if sample_size > 0:
                for date in df["published_date"].sample(sample_size).dropna():
                    if date.tzinfo is not None:
                        has_timezone = True
                        break

            # Normalize timezone handling
            if has_timezone:
                # Some dates have timezone info - make all dates timezone-aware (UTC)
                # First, handle timezone-naive dates by localizing to UTC
                naive_mask = df["published_date"].dt.tz is None
                if naive_mask.any():
                    df.loc[naive_mask, "published_date"] = df.loc[
                        naive_mask, "published_date"
                    ].dt.tz_localize("UTC")

                # Then, convert any timezone-aware dates to UTC
                df["published_date"] = df["published_date"].dt.tz_convert("UTC")
            else:
                # No timezone info - leave all dates as timezone-naive
                # This is safer than adding timezone info when none exists
                pass

            # Regenerate year/month/day columns to ensure consistency
            df["year"] = df["published_date"].dt.year
            df["month"] = df["published_date"].dt.month
            df["day"] = df["published_date"].dt.day

        except Exception as e:
            logger.warning(f"Error normalizing published_date: {e}")
            # If we encounter errors, drop the problematic column
            if "published_date" in df.columns:
                logger.warning(f"Dropping problematic published_date column")
                df = df.drop(columns=["published_date"])
                # Also drop derived columns if they exist
                for col in ["year", "month", "day"]:
                    if col in df.columns:
                        df = df.drop(columns=[col])

    return df


def find_news_files(raw_data_dir, processed_dir, archive_dir, file_types=None):
    """
    Find all news-related files across all directories.

    Parameters:
    -----------
    raw_data_dir : str
        Raw data directory
    processed_dir : str
        Processed data directory
    archive_dir : str
        Archive directory
    file_types : list, optional
        List of file types to look for

    Returns:
    --------
    list
        List of file paths to parquet files
    """
    if file_types is None:
        file_types = ["rss", "gdelt", "wayback", "historical", "news", "articles"]

    # Look in all relevant directories
    news_raw_dir = f"{raw_data_dir}/news"
    news_processed_dir = f"{processed_dir}/news"

    locations = [news_raw_dir, news_processed_dir, archive_dir]

    # Target specific patterns for news files
    patterns = []
    for file_type in file_types:
        patterns.extend(
            [
                f"*{file_type}*.parquet",
                f"{file_type}_*.parquet",
                f"*_{file_type}_*.parquet",
            ]
        )

    # Add specific patterns we know about
    patterns.extend(
        [
            "intermediate_*.parquet",
            "*_articles_*.parquet",
            "sf_news_*.parquet",
            "combined_*.parquet",
        ]
    )

    all_files = []

    # Search in each location with each pattern
    for location in locations:
        if not os.path.exists(location):
            continue

        for pattern in patterns:
            search_pattern = os.path.join(location, pattern)
            matching_files = glob.glob(search_pattern)

            if matching_files:
                logger.info(
                    f"Found {len(matching_files)} files matching pattern {pattern} in {location}"
                )
                all_files.extend(matching_files)

    # Remove duplicates and sort
    all_files = sorted(list(set(all_files)))

    # Filter out files that don't appear to be part of the news collection
    filtered_files = []
    for file in all_files:
        filename = os.path.basename(file).lower()

        # Keep files that match our patterns
        if any(file_type in filename for file_type in file_types) or any(
            pattern in filename
            for pattern in ["intermediate", "articles", "sf_news", "combined"]
        ):
            filtered_files.append(file)

    logger.info(f"Found {len(filtered_files)} news-related parquet files to process")

    # Print found files for verification
    if filtered_files:
        logger.info("Files found:")
        for file in filtered_files:
            logger.info(f"  - {os.path.basename(file)}")

    return filtered_files


def combine_parquet_files(
    raw_data_dir, processed_dir, archive_dir, output_filename, file_types=None
):
    """
    Combine all news parquet files into one and fix timezone issues.

    Parameters:
    -----------
    raw_data_dir : str
        Raw data directory
    processed_dir : str
        Processed data directory
    archive_dir : str
        Archive directory
    output_filename : str
        Base filename for output (without extension)
    file_types : list, optional
        Types of files to combine

    Returns:
    --------
    pd.DataFrame or None
        Combined DataFrame, or None if no files found
    """
    logger.info("Starting combination of news parquet files")

    # Find all parquet files
    all_files = find_news_files(raw_data_dir, processed_dir, archive_dir, file_types)

    if not all_files:
        logger.error("No news parquet files found")
        return None

    # Read and combine all files
    dfs = []
    skipped_files = []

    for file in tqdm(all_files, desc="Reading parquet files"):
        try:
            df = pd.read_parquet(file)

            # Check if DataFrame is valid
            if df.empty:
                logger.warning(f"Skipping empty file: {file}")
                skipped_files.append(file)
                continue

            # Fix any timezone issues
            df = normalize_timezone_columns(df)

            # Add source file information for debugging
            df["source_file"] = os.path.basename(file)

            dfs.append(df)
            logger.info(
                f"Successfully read {len(df)} rows from {os.path.basename(file)}"
            )
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            skipped_files.append(file)

    if not dfs:
        logger.error("No data frames were successfully read")
        return None

    # Combine all data frames
    logger.info("Combining all data frames...")
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined DataFrame has {len(combined_df)} rows before deduplication")

    # Deduplicate the combined data using title and link
    dedup_columns = []
    if "title" in combined_df.columns:
        dedup_columns.append("title")
    if "link" in combined_df.columns:
        dedup_columns.append("link")
    elif "url" in combined_df.columns:
        dedup_columns.append("url")

    if dedup_columns:
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=dedup_columns)
        after_dedup = len(combined_df)
        logger.info(
            f"Removed {before_dedup - after_dedup} duplicates. After deduplication: {after_dedup} rows"
        )
    else:
        logger.warning("Could not deduplicate - no title/link/url columns found")

    # Sort by published date if available
    if "published_date" in combined_df.columns:
        try:
            combined_df = combined_df.sort_values("published_date", ascending=False)
            logger.info("Sorted by published_date (newest first)")
        except Exception as e:
            logger.warning(f"Error sorting by published_date: {e}")

    # Save the combined file
    try:
        news_processed_dir = f"{processed_dir}/news"
        os.makedirs(news_processed_dir, exist_ok=True)

        output_path = save_to_parquet(combined_df, news_processed_dir, output_filename)

        # Also save a CSV for easier viewing if not too large
        if len(combined_df) < 100000:  # Only save as CSV if not too big
            csv_path = f"{news_processed_dir}/{output_filename}.csv"
            combined_df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV version to {csv_path}")
        else:
            logger.info(
                f"Dataset too large for CSV format ({len(combined_df)} rows). Saved as Parquet only."
            )

    except Exception as e:
        logger.error(f"Error saving combined file: {e}")
        return None

    return combined_df


def move_to_archive(raw_data_dir, archive_dir, files_to_keep=None):
    """
    Move intermediate files to archive directory to clean up.

    Parameters:
    -----------
    raw_data_dir : str
        Raw data directory
    archive_dir : str
        Archive directory
    files_to_keep : list, optional
        List of files to keep (not move to archive)
    """
    if files_to_keep is None:
        files_to_keep = []

    # Convert to basenames for comparison
    keep_basenames = [os.path.basename(f) for f in files_to_keep]

    news_raw_dir = f"{raw_data_dir}/news"

    # Find all intermediate files to move
    patterns = [
        f"{news_raw_dir}/*historical*.parquet",
        f"{news_raw_dir}/intermediate_*.parquet",
        f"{news_raw_dir}/collection_progress_*.pkl",
        f"{news_raw_dir}/*_articles_*.parquet",
        f"{news_raw_dir}/*gdelt*.parquet",
        f"{news_raw_dir}/*wayback*.parquet",
        f"{news_raw_dir}/*.csv",  # Also move any CSV files
    ]

    files_to_move = []
    for pattern in patterns:
        matching_files = glob.glob(pattern)
        # Filter out files we want to keep
        matching_files = [
            f for f in matching_files if os.path.basename(f) not in keep_basenames
        ]
        files_to_move.extend(matching_files)

    # Remove duplicates
    files_to_move = list(set(files_to_move))

    if not files_to_move:
        logger.info("No intermediate files found to archive")
        return

    logger.info(f"Moving {len(files_to_move)} files to archive")

    # Create archive directory if it doesn't exist
    os.makedirs(archive_dir, exist_ok=True)

    # Move files to archive
    moved_count = 0
    for file in tqdm(files_to_move, desc="Moving files to archive"):
        filename = os.path.basename(file)
        archive_path = os.path.join(archive_dir, filename)

        # Handle duplicate names in archive
        counter = 1
        original_archive_path = archive_path
        while os.path.exists(archive_path):
            name, ext = os.path.splitext(original_archive_path)
            archive_path = f"{name}_{counter}{ext}"
            counter += 1

        try:
            shutil.move(file, archive_path)
            logger.info(f"Moved {filename} to archive")
            moved_count += 1
        except Exception as e:
            logger.error(f"Error moving {filename} to archive: {e}")
            # Try copying instead
            try:
                shutil.copy2(file, archive_path)
                os.remove(file)
                logger.info(f"Copied {filename} to archive and deleted original")
                moved_count += 1
            except Exception as e2:
                logger.error(f"Error copying {filename} to archive: {e2}")

    logger.info(f"Successfully archived {moved_count} files")


def create_summary_report(df, output_filepath):
    """
    Create a summary report of the combined data.

    Parameters:
    -----------
    df : pd.DataFrame
        Combined DataFrame
    output_filepath : str
        Path to save the summary report
    """
    try:
        with open(output_filepath, "w") as f:
            f.write(f"SF Business News Data Combination Report\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(
                f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Total articles collected: {len(df):,}\n\n")

            # Date range analysis
            if "published_date" in df.columns:
                try:
                    min_date = df["published_date"].min()
                    max_date = df["published_date"].max()
                    f.write(f"Date Range:\n")
                    f.write(f"  Earliest article: {min_date}\n")
                    f.write(f"  Latest article: {max_date}\n")

                    # Calculate span
                    if pd.notna(min_date) and pd.notna(max_date):
                        span = (max_date - min_date).days
                        f.write(
                            f"  Time span: {span} days ({span/365.25:.1f} years)\n\n"
                        )

                except Exception as e:
                    f.write(f"Error calculating date range: {e}\n\n")

            # Year distribution
            if "year" in df.columns:
                try:
                    year_counts = df["year"].value_counts().sort_index()
                    f.write("Articles by Year:\n")
                    for year, count in year_counts.items():
                        if pd.notna(year):
                            f.write(f"  {int(year)}: {count:,} articles\n")
                    f.write("\n")
                except Exception as e:
                    f.write(f"Error calculating year distribution: {e}\n\n")

            # Source analysis
            if "source_file" in df.columns:
                f.write("Source File Distribution:\n")
                source_counts = df["source_file"].value_counts()
                for source, count in source_counts.items():
                    f.write(f"  {source}: {count:,} articles\n")
                f.write("\n")

            # Data source analysis
            if "data_source" in df.columns:
                f.write("Data Source Distribution:\n")
                source_counts = df["data_source"].value_counts()
                for source, count in source_counts.items():
                    f.write(f"  {source}: {count:,} articles\n")
                f.write("\n")

            # Column information
            f.write("Available Columns:\n")
            for col in sorted(df.columns):
                non_null_count = df[col].notna().sum()
                f.write(f"  {col}: {non_null_count:,} non-null values\n")

        logger.info(f"Summary report saved to {output_filepath}")

    except Exception as e:
        logger.error(f"Error creating summary report: {e}")


def main_file_combination(
    raw_data_dir,
    processed_dir,
    archive_dir,
    file_types=None,
    archive_intermediates=True,
):
    """
    Main function to combine files and clean up

    Parameters:
    -----------
    raw_data_dir : str
        Raw data directory
    processed_dir : str
        Processed data directory
    archive_dir : str
        Archive directory
    file_types : list, optional
        Types of files to combine
    archive_intermediates : bool
        Whether to move intermediate files to archive

    Returns:
    --------
    pd.DataFrame or None
        Combined DataFrame or None if failed
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define final output filename
    output_filename = f"sf_news_combined_final_{timestamp}"

    logger.info(f"Starting file combination process")
    logger.info(f"Output will be saved as: {output_filename}.parquet")

    # Combine all files
    combined_df = combine_parquet_files(
        raw_data_dir, processed_dir, archive_dir, output_filename, file_types
    )

    if combined_df is not None and not combined_df.empty:
        # Create summary report
        news_processed_dir = f"{processed_dir}/news"
        os.makedirs(news_processed_dir, exist_ok=True)
        summary_path = (
            f"{news_processed_dir}/sf_news_combination_summary_{timestamp}.txt"
        )
        create_summary_report(combined_df, summary_path)

        # Move intermediate files to archive if requested
        if archive_intermediates:
            final_output_path = f"{news_processed_dir}/{output_filename}.parquet"
            move_to_archive(raw_data_dir, archive_dir, [final_output_path])

        # Display summary
        print(f"\n" + "=" * 60)
        print(f"FILE COMBINATION COMPLETE")
        print(f"=" * 60)
        print(f"Total articles combined: {len(combined_df):,}")

        if "published_date" in combined_df.columns:
            try:
                min_date = combined_df["published_date"].min()
                max_date = combined_df["published_date"].max()
                print(f"Date range: {min_date} to {max_date}")
            except Exception as e:
                print(f"Error displaying date statistics: {e}")

        if "year" in combined_df.columns:
            try:
                year_counts = combined_df["year"].value_counts().sort_index()
                print(f"\nArticles by year:")
                for year, count in year_counts.items():
                    if pd.notna(year):
                        print(f"  {int(year)}: {count:,}")
            except Exception as e:
                print(f"Error displaying year statistics: {e}")

        print(f"\nFiles saved:")
        print(f"  - Combined data: {output_filename}.parquet")
        print(f"  - Summary report: sf_news_combination_summary_{timestamp}.txt")

        if archive_intermediates:
            print(f"  - Intermediate files moved to archive/")

        print(f"=" * 60)

        return combined_df
    else:
        logger.error("Failed to create combined file. Check the logs for details.")
        print("Error: Failed to create combined file. Check the logs for details.")
        return None


if __name__ == "__main__":
    # File Combination and Cleanup - Execution
    from logging_config_setup_02 import setup_logging, setup_directories

    logger = setup_logging()
    config = setup_directories()

    logger.info("Starting file combination and cleanup process...")

    # Combine all news-related files
    result = main_file_combination(
        config["raw_data_dir"],
        config["processed_dir"],
        config["archive_dir"],
        file_types=["rss", "gdelt", "wayback", "historical", "news", "articles"],
        archive_intermediates=True,
    )

    if result is not None and not result.empty:
        print(
            f"Successfully combined news files into final dataset with {len(result)} articles"
        )

        # Show data source breakdown
        if "data_source" in result.columns:
            print(f"\nData sources in final dataset:")
            print(result["data_source"].value_counts())

        # Show sample of final data
        print(f"\nSample of final combined data:")
        display_cols = ["title", "published_date", "data_source"]
        available_cols = [col for col in display_cols if col in result.columns]
        if available_cols:
            print(result[available_cols].head())
    else:
        print("File combination failed - no data to display")

    logger.info("File combination and cleanup process completed")
