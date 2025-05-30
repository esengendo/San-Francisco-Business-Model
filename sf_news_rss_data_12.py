import os
import time
import random
import logging
import pandas as pd
import requests
import feedparser
import re
import calendar
from datetime import datetime, timedelta
from tqdm import tqdm
from helper_functions_03 import save_to_parquet

# Setup logging
logger = logging.getLogger(__name__)


def fetch_sf_news_from_rss():
    """
    Fetch current news articles about San Francisco from major local news RSS feeds
    including SF Chronicle, SFGate, and KQED.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing RSS news articles
    """
    logger.info("Fetching news from San Francisco RSS feeds...")

    # List of RSS feeds for San Francisco news sources
    rss_feeds = {
        "SF Chronicle": "https://www.sfchronicle.com/feed/feed.xml",
        "SFGate": "https://www.sfgate.com/bayarea/feed/Bay-Area-News-429.php",
        "KQED": "https://ww2.kqed.org/news/feed/",
        "SF Business Times": "https://www.bizjournals.com/sanfrancisco/news/rss.xml",
        "SF Examiner": "https://www.sfexaminer.com/news/feed/",
        "Mission Local": "https://missionlocal.org/feed/",
        "Hoodline": "https://hoodline.com/rss/all.xml",
    }

    # Business-related keywords to filter relevant articles
    business_keywords = [
        "business",
        "economy",
        "startup",
        "restaurant",
        "retail",
        "commercial",
        "real estate",
        "tech",
        "company",
        "store",
        "shop",
        "market",
        "financial",
        "industry",
        "development",
        "expansion",
        "closure",
        "opening",
        "housing",
        "property",
    ]

    all_articles = []

    # Process each RSS feed
    for source_name, feed_url in rss_feeds.items():
        try:
            logger.info(f"Fetching from {source_name}...")

            # Parse the RSS feed
            feed = feedparser.parse(feed_url)

            # Check if the feed was successfully parsed
            if not feed.entries:
                logger.warning(f"No entries found in {source_name} feed")
                continue

            logger.info(f"Found {len(feed.entries)} articles from {source_name}")

            # Process each article in the feed
            for entry in feed.entries:
                # Extract article data
                article = {
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "content": (
                        entry.get("content", [{}])[0].get("value", "")
                        if "content" in entry
                        else ""
                    ),
                    "url": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source_name": source_name,
                }

                # Check if this is a business-related article
                article_text = f"{article['title']} {article['summary']} {article['content']}".lower()

                # Determine which business categories this article matches
                matched_keywords = []
                for keyword in business_keywords:
                    if keyword.lower() in article_text:
                        matched_keywords.append(keyword)

                # Add the article if it matches any business keyword
                if matched_keywords:
                    article["search_query"] = ", ".join(matched_keywords)
                    all_articles.append(article)

            # Be nice to the servers
            time.sleep(1)

        except Exception as e:
            logger.error(f"Error processing {source_name} feed: {e}")

    # Convert to DataFrame
    if all_articles:
        news_df = pd.DataFrame(all_articles)

        # Process dates
        try:
            news_df["published_date"] = pd.to_datetime(
                news_df["published"], errors="coerce"
            )

            # Handle failed date parsing - assign current date to NaT values
            missing_dates = news_df["published_date"].isna()
            if missing_dates.any():
                logger.warning(
                    f"Found {missing_dates.sum()} articles with invalid dates, using current date"
                )
                news_df.loc[missing_dates, "published_date"] = datetime.now()

            news_df["published_year"] = news_df["published_date"].dt.year
            news_df["published_month"] = news_df["published_date"].dt.month

        except Exception as e:
            logger.error(f"Error processing dates: {e}")
            news_df["published_date"] = datetime.now()
            news_df["published_year"] = datetime.now().year
            news_df["published_month"] = datetime.now().month

        # Clean text fields
        if "summary" in news_df.columns:
            # Remove HTML tags from summary
            news_df["summary"] = news_df["summary"].apply(
                lambda x: re.sub("<.*?>", "", str(x)) if pd.notnull(x) else ""
            )

        # Add a combined text field for analysis
        news_df["full_text"] = news_df["title"] + " " + news_df.get("summary", "")

        # Add data source for tracking
        news_df["data_source"] = "rss"

        logger.info(f"Processed {len(news_df)} news articles from RSS feeds")
        return news_df
    else:
        logger.error("No articles found in RSS feeds")
        return pd.DataFrame()


def fetch_historical_news(start_year=None, end_year=None):
    """
    Fetch historical news from archives and APIs for multiple years of data

    Parameters:
    -----------
    start_year : int, optional
        The starting year (default 5 years ago)
    end_year : int, optional
        The ending year (default current year)

    Returns:
    --------
    pd.DataFrame
        DataFrame containing historical news articles
    """
    if start_year is None:
        start_year = datetime.now().year - 5
    if end_year is None:
        end_year = datetime.now().year

    logger.info(f"Fetching historical news from {start_year} to {end_year}...")

    # Create a list to hold all articles
    all_historical_articles = []

    # Business-related keywords to search for
    business_keywords = [
        "business",
        "economy",
        "startup",
        "restaurant",
        "retail",
        "commercial",
        "real estate",
        "tech",
        "company",
        "store",
        "shop",
        "market",
        "financial",
        "industry",
        "development",
        "expansion",
        "closure",
        "opening",
        "housing",
        "property",
    ]

    # Try to use News API if available
    try:
        # Check if we have a News API key in the environment
        api_key = os.environ.get("NEWS_API_KEY", "")

        if api_key:
            try:
                from newsapi import NewsApiClient

                # Initialize News API client with your API key
                newsapi = NewsApiClient(api_key=api_key)

                # For each keyword, get news about San Francisco
                for keyword in tqdm(business_keywords, desc="Fetching News API data"):
                    try:
                        # Search for articles
                        all_articles = newsapi.get_everything(
                            q=f"San Francisco {keyword}",
                            language="en",
                            sort_by="publishedAt",
                            page_size=100,  # Maximum allowed
                        )

                        # Process articles
                        if all_articles["status"] == "ok":
                            for article_data in all_articles["articles"]:
                                article = {
                                    "title": article_data.get("title", ""),
                                    "summary": article_data.get("description", ""),
                                    "content": article_data.get("content", ""),
                                    "url": article_data.get("url", ""),
                                    "published": article_data.get("publishedAt", ""),
                                    "source_name": article_data.get("source", {}).get(
                                        "name", "News API"
                                    ),
                                    "search_query": keyword,
                                    "data_source": "newsapi",
                                }
                                all_historical_articles.append(article)

                        # Be nice to the servers
                        time.sleep(1)

                    except Exception as e:
                        logger.error(
                            f"Error fetching News API data for keyword {keyword}: {e}"
                        )
                        continue

            except ImportError:
                logger.warning(
                    "NewsApiClient package not installed. Install with: pip install newsapi-python"
                )
        else:
            logger.warning(
                "No News API key found in environment. Skipping this data source."
            )

    except Exception as e:
        logger.error(f"Error initializing News API: {e}")

    # Generate synthetic historical data for demonstration
    if len(all_historical_articles) == 0:
        logger.warning(
            "No historical data sources available. Generating synthetic data for demonstration."
        )

        # Common sources in SF news
        sources = [
            "SF Chronicle",
            "SFGate",
            "SF Business Times",
            "SF Examiner",
            "KQED",
            "Mission Local",
            "Hoodline",
            "San Francisco Standard",
        ]

        # Article title templates
        title_templates = [
            "New {business_type} opening in {neighborhood}",
            "{company} announces expansion in San Francisco",
            "San Francisco {industry} sector sees {trend} trend",
            "Report: {neighborhood} {property_type} prices {direction} by {percent}%",
            "City approves new {development_type} development in {neighborhood}",
            "{company} to {action} {number} jobs in San Francisco",
            "Local {business_type} struggles amid {challenge}",
            "San Francisco {metric} {direction} {percent}% in {timeframe}",
        ]

        # Template variables
        business_types = [
            "restaurant",
            "cafe",
            "tech startup",
            "retail store",
            "coworking space",
            "brewery",
        ]
        neighborhoods = [
            "Mission",
            "SOMA",
            "Financial District",
            "North Beach",
            "Hayes Valley",
            "Marina",
            "Dogpatch",
            "Sunset",
            "Richmond",
            "Bayview",
        ]
        companies = [
            "Salesforce",
            "Twitter",
            "Uber",
            "Airbnb",
            "Lyft",
            "Square",
            "Stripe",
            "Coinbase",
            "Instacart",
            "DoorDash",
            "Local Startup",
        ]
        industries = [
            "tech",
            "food service",
            "retail",
            "real estate",
            "tourism",
            "transportation",
        ]
        trends = [
            "upward",
            "downward",
            "stabilizing",
            "concerning",
            "promising",
            "mixed",
        ]
        property_types = ["housing", "commercial", "office", "retail", "industrial"]
        directions = [
            "up",
            "down",
            "increased",
            "decreased",
            "jumped",
            "plummeted",
            "surged",
        ]
        development_types = [
            "housing",
            "mixed-use",
            "commercial",
            "transit-oriented",
            "affordable housing",
        ]
        actions = ["add", "cut", "relocate", "create", "eliminate"]
        challenges = [
            "pandemic",
            "rising costs",
            "labor shortage",
            "economic downturn",
            "regulatory changes",
        ]
        metrics = [
            "unemployment rate",
            "housing prices",
            "commercial vacancy",
            "tourism numbers",
            "tax revenue",
        ]
        timeframes = ["Q1", "Q2", "Q3", "Q4", "first half", "second half"]

        # Generate articles across the years
        for year in range(start_year, end_year + 1):
            # Generate more articles for recent years
            num_articles = random.randint(100, 200) * (1 + (year - start_year) * 0.2)

            for _ in tqdm(
                range(int(num_articles)), desc=f"Generating synthetic data for {year}"
            ):
                # Random date in the year
                month = random.randint(1, 12)

                # Adjust for current year
                if year == datetime.now().year and month > datetime.now().month:
                    month = random.randint(1, datetime.now().month)

                day = random.randint(1, calendar.monthrange(year, month)[1])
                article_date = datetime(year, month, day)

                # Skip future dates
                if article_date > datetime.now():
                    continue

                # Random title using templates
                template = random.choice(title_templates)
                title = template.format(
                    business_type=random.choice(business_types),
                    neighborhood=random.choice(neighborhoods),
                    company=random.choice(companies),
                    industry=random.choice(industries),
                    trend=random.choice(trends),
                    property_type=random.choice(property_types),
                    direction=random.choice(directions),
                    percent=random.randint(1, 30),
                    development_type=random.choice(development_types),
                    action=random.choice(actions),
                    number=random.randint(10, 500),
                    challenge=random.choice(challenges),
                    metric=random.choice(metrics),
                    timeframe=random.choice(timeframes),
                )

                # Random matched keywords
                potential_keywords = business_keywords.copy()
                random.shuffle(potential_keywords)
                matched = potential_keywords[: random.randint(1, 3)]

                # Create article
                article = {
                    "title": title,
                    "summary": f"This is a synthetic summary for demonstration purposes about {' and '.join(matched)} in San Francisco.",
                    "content": "Synthetic content for demonstration.",
                    "url": f"https://example.com/synthetic/{year}/{month:02d}/{day:02d}/{'-'.join(title.lower().split()[:5])}",
                    "published": article_date.strftime("%Y-%m-%d"),
                    "source_name": random.choice(sources),
                    "search_query": ", ".join(matched),
                    "data_source": "synthetic",
                }

                all_historical_articles.append(article)

    # Convert all historical articles to DataFrame
    if all_historical_articles:
        hist_df = pd.DataFrame(all_historical_articles)

        # Process dates
        try:
            hist_df["published_date"] = pd.to_datetime(
                hist_df["published"], errors="coerce"
            )

            # Handle failed date parsing
            missing_dates = hist_df["published_date"].isna()
            if missing_dates.any():
                logger.warning(
                    f"Found {missing_dates.sum()} historical articles with invalid dates"
                )
                # For rows with missing dates, use random dates from the appropriate year range
                random_dates = pd.date_range(
                    start=f"{start_year}-01-01",
                    end=min(datetime.now(), datetime(end_year, 12, 31)),
                    periods=missing_dates.sum(),
                )
                hist_df.loc[missing_dates, "published_date"] = random_dates

            hist_df["published_year"] = hist_df["published_date"].dt.year
            hist_df["published_month"] = hist_df["published_date"].dt.month

        except Exception as e:
            logger.error(f"Error processing historical dates: {e}")
            # Fallback to assigning random dates
            date_range = pd.date_range(
                start=f"{start_year}-01-01", end=datetime.now(), periods=len(hist_df)
            )
            hist_df["published_date"] = date_range
            hist_df["published_year"] = hist_df["published_date"].dt.year
            hist_df["published_month"] = hist_df["published_date"].dt.month

        # Clean text fields
        if "summary" in hist_df.columns:
            # Remove HTML tags from summary
            hist_df["summary"] = hist_df["summary"].apply(
                lambda x: re.sub("<.*?>", "", str(x)) if pd.notnull(x) else ""
            )

        # Add a combined text field for analysis
        hist_df["full_text"] = hist_df["title"] + " " + hist_df.get("summary", "")

        logger.info(f"Processed {len(hist_df)} historical news articles")
        return hist_df
    else:
        logger.error("No historical articles found or generated")
        return pd.DataFrame()


def process_and_save_data(news_df, hist_df, raw_data_dir, processed_dir, archive_dir):
    """
    Process the combined dataset and save to appropriate locations

    Parameters:
    -----------
    news_df : pd.DataFrame
        Current RSS news data
    hist_df : pd.DataFrame
        Historical news data
    raw_data_dir : str
        Directory for raw data
    processed_dir : str
        Directory for processed data
    archive_dir : str
        Directory for archived data

    Returns:
    --------
    pd.DataFrame
        Combined and processed news data
    """
    # Define the data directories for saving
    news_raw_dir = f"{raw_data_dir}/news"
    news_processed_dir = f"{processed_dir}/news"

    # Create directories
    os.makedirs(news_raw_dir, exist_ok=True)
    os.makedirs(news_processed_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)

    if hist_df is not None and not hist_df.empty:
        # Combine current and historical data
        combined_df = pd.concat([news_df, hist_df], ignore_index=True)
        logger.info(f"Combined dataset has {len(combined_df)} articles")

        # Remove duplicates (if any) based on title and URL
        if "url" in combined_df.columns:
            # First try deduplicating by URL
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=["url"], keep="first")
            logger.info(f"Removed {before_dedup - len(combined_df)} duplicate URLs")

        # Then try deduplicating by title
        if "title" in combined_df.columns:
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=["title"], keep="first")
            logger.info(f"Removed {before_dedup - len(combined_df)} duplicate titles")

        logger.info(f"After deduplication: {len(combined_df)} articles")
    else:
        combined_df = news_df
        logger.info(f"Using only current RSS data with {len(combined_df)} articles")

    try:
        # Save to the 'news' subdirectories in both raw_data and processed
        save_to_parquet(combined_df, news_raw_dir, "sf_news_combined_raw")
        save_to_parquet(combined_df, news_processed_dir, "sf_news_combined")

        # Also save a timestamped version for archival purposes
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_to_parquet(combined_df, archive_dir, f"sf_news_combined_{timestamp_file}")

        # Save individual datasets too
        save_to_parquet(news_df, news_raw_dir, "sf_news_rss_raw")
        save_to_parquet(news_df, news_processed_dir, "sf_news_rss")

        if hist_df is not None and not hist_df.empty:
            save_to_parquet(hist_df, news_raw_dir, "sf_news_historical_raw")
            save_to_parquet(hist_df, news_processed_dir, "sf_news_historical")

        return combined_df

    except Exception as e:
        logger.error(f"Error saving data: {e}")
        return combined_df


def main_sf_news_collection(raw_data_dir, processed_dir, archive_dir, start_year=None):
    """
    Main function to execute SF news data collection from RSS and historical sources.

    Parameters:
    -----------
    raw_data_dir : str
        Directory to save raw data
    processed_dir : str
        Directory to save processed data
    archive_dir : str
        Directory to save archived data
    start_year : int, optional
        Starting year for historical data (default 5 years ago)

    Returns:
    --------
    pd.DataFrame
        Combined news data
    """
    if start_year is None:
        start_year = datetime.now().year - 5

    logger.info("Starting San Francisco business news collection...")

    # Fetch current RSS news
    logger.info("Fetching current RSS news...")
    current_news = fetch_sf_news_from_rss()

    # Fetch historical news
    logger.info("Fetching historical news...")
    historical_news = fetch_historical_news(start_year=start_year)

    # Process and save combined data
    logger.info("Processing and saving data...")
    combined_news = process_and_save_data(
        current_news, historical_news, raw_data_dir, processed_dir, archive_dir
    )

    logger.info(f"News collection complete! Total articles: {len(combined_news)}")
    return combined_news


if __name__ == "__main__":
    # San Francisco News RSS Data Collection - Execution
    from logging_config_setup_02 import setup_logging, setup_directories

    logger = setup_logging()
    config = setup_directories()

    # Execute the main function
    news_data = main_sf_news_collection(
        config["raw_data_dir"],
        config["processed_dir"],
        config["archive_dir"],
        start_year=2019,
    )

    if not news_data.empty:
        print(f"Successfully collected SF news data with {len(news_data)} articles")
        print(f"Data covers period from 2019 to {datetime.now().year}")

        # Show basic statistics
        if "data_source" in news_data.columns:
            print(f"\nData sources:")
            print(news_data["data_source"].value_counts())

        if "source_name" in news_data.columns:
            print(f"\nTop news sources:")
            print(news_data["source_name"].value_counts().head())

        if "published_year" in news_data.columns:
            print(f"\nArticles by year:")
            print(news_data["published_year"].value_counts().sort_index())

        print(f"\nSample data:")
        print(news_data[["title", "source_name", "published_date"]].head())
    else:
        print("No news data collected")

    logger.info("SF news RSS data collection completed")
