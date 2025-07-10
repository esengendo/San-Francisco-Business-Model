import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings("ignore")


def setup_logging_and_directories():
    """Configure logging and set up directory structure"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("SFBusinessPipeline")

    # base_dir = Path("/Users/baboo/Documents/San Francisco Business Model")
    base_dir = os.getenv("BASE_DIR", "/app/San_Francisco_Business_Model")
    processed_dir = os.path.join(base_dir, "processed")
    final_dir = os.path.join(processed_dir, "final")
    news_dir = os.path.join(processed_dir, "news")

    # Set pandas display options
    pd.set_option("display.max_columns", None)

    return logger, base_dir, final_dir, news_dir


def setup_nltk_dependencies(logger):
    """Download required NLTK data if not already downloaded"""
    logger.info("Setting up NLTK dependencies...")

    try:
        nltk.data.find("vader_lexicon")
    except LookupError:
        logger.info("Downloading NLTK VADER lexicon...")
        print("📥 Downloading NLTK VADER lexicon...")
        nltk.download("vader_lexicon")


def load_datasets(final_dir, news_dir, logger):
    """Load the business and news datasets"""
    logger.info("Loading datasets...")
    print("🔄 Loading datasets...")

    # Load the dataframes - keeping same input/output file names
    business_df = pd.read_parquet(
        os.path.join(final_dir, "sf_business_success_with_crime.parquet")
    )
    news_df = pd.read_parquet(os.path.join(news_dir, "sf_news_combined.parquet"))

    logger.info(f"Business data shape: {business_df.shape}")
    logger.info(f"News data shape: {news_df.shape}")

    print(f"✅ Business data shape: {business_df.shape}")
    print(f"✅ News data shape: {news_df.shape}")

    return business_df, news_df


def explore_business_data(business_df, logger):
    """Exploring Business Dataframe Key Features"""
    logger.info("Exploring business dataframe...")
    print("\n🔍 Exploring business dataframe...")

    # Examine key business columns related to our objectives
    key_business_cols = [
        "ttxid",
        "business_industry",
        "location",
        "success",
        "dba_start_date",
        "dba_end_date",
        "supervisor_district",
        "neighborhoods_analysis_boundaries",
        "year",
    ]

    # Display basic info about business success
    print("\n📊 Business success distribution:")
    print(business_df["success"].value_counts(normalize=True))

    # Check temporal coverage
    print("\n📅 Business data year distribution:")
    print(business_df["year"].value_counts().sort_index())

    # Examine location-related data
    print("\n🗺️ Location data samples:")
    if "supervisor_district" in business_df.columns:
        print(
            f"Supervisor districts: {business_df['supervisor_district'].nunique()} unique values"
        )

    if "neighborhoods_analysis_boundaries" in business_df.columns:
        print(
            f"Neighborhoods: {business_df['neighborhoods_analysis_boundaries'].nunique()} unique values"
        )


def explore_news_data(news_df, logger):
    """Exploring News Dataframe"""
    logger.info("Exploring news dataframe...")
    print("\n🔍 Exploring news dataframe...")

    # Examine news columns
    print("\n📋 News data info:")
    news_df.info()

    # Temporal distribution of news
    print("\n📅 News temporal coverage:")
    news_df["published_date"] = pd.to_datetime(news_df["published_date"])
    print("📅 News by year:")
    print(news_df["published_date"].dt.year.value_counts().sort_index())
    print("\n📅 News by month:")
    print(news_df["published_date"].dt.month.value_counts().sort_index())

    # Sample news titles
    print("\n📰 Sample news titles:")
    print(news_df["title"].sample(5).tolist())

    # Check news categories if available
    if "categories" in news_df.columns:
        # Extract all unique categories
        all_categories = []
        for cats in news_df["categories"].dropna():
            if isinstance(cats, list):
                all_categories.extend(cats)
            elif isinstance(cats, str):
                all_categories.append(cats)

        print("\n🏷️ Top news categories:")
        print(pd.Series(all_categories).value_counts().head(10))


def preprocess_news_data(news_df, logger):
    """News Data Preprocessing for Merging"""
    logger.info("Preprocessing news data for merging...")
    print("\n🔧 Preprocessing news data for merging...")

    # Create a copy for preprocessing
    news_processed = news_df.copy()

    # 1. Ensure datetime format is consistent
    news_processed["published_date"] = pd.to_datetime(news_processed["published_date"])

    # 2. Extract year and month for easier matching
    news_processed["news_year"] = news_processed["published_date"].dt.year
    news_processed["news_month"] = news_processed["published_date"].dt.month
    news_processed["news_quarter"] = news_processed["published_date"].dt.quarter

    return news_processed


def get_sentiment(text):
    """Get sentiment score for text"""
    sid = SentimentIntensityAnalyzer()
    if pd.isna(text) or not isinstance(text, str):
        return {"neg": 0, "neu": 0, "pos": 0, "compound": 0}
    return sid.polarity_scores(text)


def perform_sentiment_analysis(news_processed, logger):
    """Process the news content - Extract sentiment scores"""
    logger.info("Calculating sentiment scores for news articles...")
    print("🎭 Calculating sentiment scores for news articles...")

    # Check what text columns are available for sentiment analysis
    available_text_cols = []
    for col in ["title", "description", "summary", "content", "text"]:
        if col in news_processed.columns:
            available_text_cols.append(col)

    print(f"📝 Available text columns for sentiment analysis: {available_text_cols}")

    # Apply sentiment analysis to available text columns
    sentiment_scores = {}

    if "title" in news_processed.columns:
        print("  🔄 Analyzing title sentiment...")
        sentiment_scores["title_sentiment"] = (
            news_processed["title"].apply(get_sentiment).apply(lambda x: x["compound"])
        )

    if "description" in news_processed.columns:
        print("  🔄 Analyzing description sentiment...")
        sentiment_scores["desc_sentiment"] = (
            news_processed["description"]
            .apply(get_sentiment)
            .apply(lambda x: x["compound"])
        )
    elif "summary" in news_processed.columns:
        print("  🔄 Analyzing summary sentiment...")
        sentiment_scores["desc_sentiment"] = (
            news_processed["summary"]
            .apply(get_sentiment)
            .apply(lambda x: x["compound"])
        )
    elif "content" in news_processed.columns:
        print("  🔄 Analyzing content sentiment...")
        sentiment_scores["desc_sentiment"] = (
            news_processed["content"]
            .apply(get_sentiment)
            .apply(lambda x: x["compound"])
        )

    # Add sentiment scores to dataframe
    for col_name, scores in sentiment_scores.items():
        news_processed[col_name] = scores

    # Calculate overall sentiment based on available columns
    if len(sentiment_scores) == 2:
        # Both title and description/summary/content available
        news_processed["overall_sentiment"] = (
            sentiment_scores["title_sentiment"] + sentiment_scores["desc_sentiment"]
        ) / 2
    elif "title_sentiment" in sentiment_scores:
        # Only title available
        news_processed["overall_sentiment"] = sentiment_scores["title_sentiment"]
    elif "desc_sentiment" in sentiment_scores:
        # Only description/summary/content available
        news_processed["overall_sentiment"] = sentiment_scores["desc_sentiment"]
    else:
        # No text columns available - fill with neutral sentiment
        print("⚠️ No suitable text columns found for sentiment analysis")
        news_processed["overall_sentiment"] = 0
        news_processed["title_sentiment"] = 0
        news_processed["desc_sentiment"] = 0

    print("✅ Sentiment analysis complete!")

    return news_processed


def extract_location_mentions(news_processed, business_df, logger):
    """Extract key districts or neighborhoods mentioned in news"""
    logger.info("Extracting location mentions from news content...")
    print("🗺️ Extracting location mentions from news content...")

    try:
        # Get unique districts and neighborhoods
        districts = []
        neighborhoods = []

        try:
            districts = business_df["supervisor_district"].dropna().unique().tolist()
            # Convert numeric districts to strings if needed
            districts = [str(d) for d in districts if d is not None]
        except Exception as e:
            print(f"⚠️ Warning: Error getting districts: {str(e)}")

        try:
            neighborhoods = (
                business_df["neighborhoods_analysis_boundaries"]
                .dropna()
                .unique()
                .tolist()
            )
            neighborhoods = [
                n for n in neighborhoods if n is not None and isinstance(n, str)
            ]
        except Exception as e:
            print(f"⚠️ Warning: Error getting neighborhoods: {str(e)}")

        # Define extraction function that's robust to different data types
        def extract_locations(text, location_list):
            if pd.isna(text) or not isinstance(text, str):
                return []

            found_locations = []
            for location in location_list:
                if isinstance(location, str) and location.lower() in text.lower():
                    found_locations.append(location)
            return found_locations

        # Apply location extraction using a more robust approach
        mentioned_districts = []
        mentioned_neighborhoods = []

        # Use the best available text column for location extraction
        text_col = None
        if "description" in news_processed.columns:
            text_col = "description"
        elif "summary" in news_processed.columns:
            text_col = "summary"
        elif "content" in news_processed.columns:
            text_col = "content"
        elif "title" in news_processed.columns:
            text_col = "title"

        if text_col:
            print(f"  📍 Using '{text_col}' column for location extraction")
            for _, row in news_processed.iterrows():
                text_content = row.get(text_col, "")
                mentioned_districts.append(extract_locations(text_content, districts))
                mentioned_neighborhoods.append(
                    extract_locations(text_content, neighborhoods)
                )
        else:
            print("  ⚠️ No suitable text column found for location extraction")
            mentioned_districts = [[] for _ in range(len(news_processed))]
            mentioned_neighborhoods = [[] for _ in range(len(news_processed))]

        # Add to dataframe
        news_processed["mentioned_districts"] = mentioned_districts
        news_processed["mentioned_neighborhoods"] = mentioned_neighborhoods

        print(
            f"✅ Extracted mentions - Districts: {sum(len(d) for d in mentioned_districts)}, "
            f"Neighborhoods: {sum(len(n) for n in mentioned_neighborhoods)}"
        )

    except Exception as e:
        logger.error(f"Error in location extraction: {str(e)}")
        print(f"⚠️ Warning: Error in location extraction: {str(e)}")
        # Create empty columns as fallback
        news_processed["mentioned_districts"] = [[] for _ in range(len(news_processed))]
        news_processed["mentioned_neighborhoods"] = [
            [] for _ in range(len(news_processed))
        ]

    return news_processed


def create_time_based_aggregations(news_processed, logger):
    """Create time-based aggregations of news sentiment"""
    logger.info("Creating time-based sentiment aggregations...")
    print("📊 Creating time-based sentiment aggregations...")

    # Group by year, month, and calculate average sentiment
    sentiment_by_time = (
        news_processed.groupby(["news_year", "news_month"])[
            ["overall_sentiment", "title_sentiment", "desc_sentiment"]
        ]
        .agg(["mean", "count", "std"])
        .reset_index()
    )

    # Flatten the MultiIndex columns
    sentiment_by_time.columns = [
        "_".join(col).strip("_") for col in sentiment_by_time.columns.values
    ]

    print("\n📈 Sentiment by time period - sample:")
    print(sentiment_by_time.head())

    return sentiment_by_time


def perform_temporal_merging(business_df, sentiment_by_time, logger):
    """Temporal Merging - Link businesses to news sentiment by time period"""
    logger.info("Merging by time period...")
    print("📅 Merging by time period (LEFT MERGE to preserve all business records)...")

    # Extract year and month from business start dates safely
    try:
        business_df["business_year"] = pd.to_datetime(
            business_df["dba_start_date"]
        ).dt.year
        business_df["business_month"] = pd.to_datetime(
            business_df["dba_start_date"]
        ).dt.month
    except Exception as e:
        logger.warning(f"Error converting business dates: {str(e)}")
        print(f"⚠️ Warning: Error converting business dates: {str(e)}")
        # Use the 'year' column directly since it seems to already exist in the data
        business_df["business_year"] = business_df["year"]
        # Default month to January if we can't get it
        business_df["business_month"] = 1

    # Merge with temporally aggregated news data
    # Using left merge to keep all business records, adding news data where available
    merged_df_temporal = business_df.merge(
        sentiment_by_time,
        left_on=["business_year", "business_month"],
        right_on=["news_year", "news_month"],
        how="left",  # Left merge ensures we keep ALL business records
    )

    print(f"📊 Shape after temporal merging: {merged_df_temporal.shape}")

    # Check for NaN values in merged sentiment columns
    print("\n🔍 Missing values in sentiment columns after merge:")
    sentiment_cols = [col for col in merged_df_temporal.columns if "sentiment" in col]
    missing_counts = merged_df_temporal[sentiment_cols].isna().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} missing values")

    # Fill NAs with neutral sentiment (0)
    for col in sentiment_cols:
        merged_df_temporal[col] = merged_df_temporal[col].fillna(0)
    print("✅ Filled missing sentiment values with neutral (0)")

    return merged_df_temporal


def perform_spatial_merging(business_df, news_processed, logger):
    """District/Neighborhood based merging"""
    logger.info("Attempting spatial merging by district/neighborhood...")
    print("\n🗺️ Attempting spatial merging by district/neighborhood...")

    merged_df_spatial = business_df.copy()

    if "mentioned_districts" in news_processed.columns:
        try:
            # Create district sentiment mapping
            district_sentiment_dict = {}

            # Process each news article one by one
            for _, row in news_processed.iterrows():
                sentiment = row.get("overall_sentiment", 0)

                # Get list of mentioned districts
                districts = row.get("mentioned_districts", [])
                if not isinstance(districts, list):
                    continue

                # Update the sentiment dictionary for each district
                for district in districts:
                    if district not in district_sentiment_dict:
                        district_sentiment_dict[district] = {"sum": 0, "count": 0}

                    district_sentiment_dict[district]["sum"] += sentiment
                    district_sentiment_dict[district]["count"] += 1

            # Convert dictionary to dataframe
            district_data = []
            for district, stats in district_sentiment_dict.items():
                if stats["count"] > 0:
                    district_data.append(
                        {
                            "district": district,
                            "district_sentiment_mean": stats["sum"] / stats["count"],
                            "district_sentiment_count": stats["count"],
                        }
                    )

            # Create a pandas DataFrame
            district_sentiment_df = pd.DataFrame(district_data)

            if len(district_sentiment_df) > 0:
                print(
                    f"📊 Created district sentiment data for {len(district_sentiment_df)} districts"
                )
                print("\n📈 Sentiment by district - sample:")
                print(district_sentiment_df.head())

                # Merge with business data based on district
                # Using left merge to ensure we keep ALL business records
                merged_df_spatial = business_df.merge(
                    district_sentiment_df,
                    left_on="supervisor_district",
                    right_on="district",
                    how="left",  # Left merge preserves all business data rows
                )

                print(f"📊 Shape after spatial merging: {merged_df_spatial.shape}")
            else:
                print("⚠️ No district sentiment data available for spatial merging")
                merged_df_spatial = business_df.copy()
        except Exception as e:
            logger.error(f"Error in spatial merging: {str(e)}")
            print(f"❌ Error in spatial merging: {str(e)}")
            print("⏭️ Skipping spatial merging due to error")
            merged_df_spatial = business_df.copy()
    else:
        print("⚠️ No district mentions found in news data")
        merged_df_spatial = business_df.copy()

    return merged_df_spatial


def create_final_merged_dataset(
    business_df, merged_df_temporal, merged_df_spatial, logger
):
    """Final merged dataframe - combining both temporal and spatial features"""
    logger.info("Creating final merged dataset...")
    print("\n🔗 Creating final merged dataset...")

    try:
        # Start with temporal features (already merged)
        final_merged_df = merged_df_temporal.copy()

        # If spatial merging was successful, add those features too
        if "district_sentiment_mean" in merged_df_spatial.columns:
            # Select only the new sentiment columns from spatial merge
            spatial_cols = [
                col for col in merged_df_spatial.columns if "district_sentiment" in col
            ]

            if spatial_cols:
                # Get the indices and spatial columns
                spatial_data = merged_df_spatial[["ttxid"] + spatial_cols]

                # Merge spatial features into the final dataframe
                final_merged_df = final_merged_df.merge(
                    spatial_data, on="ttxid", how="left"
                )
                print(f"✅ Added spatial sentiment features: {spatial_cols}")

        # Check that we have the same number of rows as the original business dataframe
        if len(final_merged_df) != len(business_df):
            logger.warning(
                f"Row count mismatch - original: {len(business_df)}, merged: {len(final_merged_df)}"
            )
            print(
                f"⚠️ WARNING: Row count mismatch - original: {len(business_df)}, merged: {len(final_merged_df)}"
            )
            # Force correct row count by using original index
            final_merged_df = final_merged_df.set_index(business_df.index)
        else:
            print("✅ Row count preserved - successful merge")

    except Exception as e:
        logger.error(f"Error in final merging: {str(e)}")
        print(f"❌ Error in final merging: {str(e)}")
        print("📄 Using temporal merge only")
        final_merged_df = merged_df_temporal.copy()

    return final_merged_df


def create_additional_features(final_merged_df, logger):
    """Additional feature engineering for modeling"""
    logger.info("Creating additional news-based features...")
    print("\n🏗️ Creating additional news-based features...")

    # Create interaction features between news sentiment and business characteristics
    if (
        "overall_sentiment_mean" in final_merged_df.columns
        and "industry_success_rate" in final_merged_df.columns
    ):
        final_merged_df["sentiment_industry_interaction"] = (
            final_merged_df["overall_sentiment_mean"]
            * final_merged_df["industry_success_rate"]
        )
        print("✅ Created sentiment-industry interaction feature")

    return final_merged_df


def save_merged_dataset(final_merged_df, final_dir, base_dir, logger):
    """Save the merged dataframe"""
    logger.info("Saving merged dataset...")

    # Save the merged dataframe - keeping same output filename
    output_path = os.path.join(final_dir, "sf_business_with_news.parquet")
    print(f"\n💾 Saving merged dataset...")
    try:
        final_merged_df.to_parquet(output_path)
        logger.info(f"Saved merged dataframe to: {output_path}")
        print(f"✅ Saved merged dataframe to: {output_path}")
        print(f"📊 Final dataset shape: {final_merged_df.shape}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving to primary location: {e}")
        print(f"❌ Error saving to primary location: {e}")
        # Try alternate path
        alternate_path = os.path.join(base_dir, "sf_business_with_news.parquet")
        final_merged_df.to_parquet(alternate_path)
        logger.info(f"Saved to alternate location: {alternate_path}")
        print(f"✅ Saved to alternate location: {alternate_path}")
        return alternate_path


def analyze_news_relationships(final_merged_df, logger):
    """Analyze relationships without visualizations"""
    logger.info("Analyzing news sentiment relationships...")
    print("\n📊 Analyzing news sentiment relationships...")

    # Analyze relationship between news sentiment and business success
    if (
        "success" in final_merged_df.columns
        and "overall_sentiment_mean" in final_merged_df.columns
    ):
        # Calculate correlation
        correlation = final_merged_df["overall_sentiment_mean"].corr(
            final_merged_df["success"]
        )
        print(
            f"\n🔗 Correlation between overall sentiment and business success: {correlation:.4f}"
        )

        # Analyze by success groups
        success_sentiment = final_merged_df.groupby("success")[
            "overall_sentiment_mean"
        ].agg(["mean", "count", "std"])
        print("\n🏆 News Sentiment by Business Success:")
        for success_val, stats in success_sentiment.iterrows():
            success_label = "Successful" if success_val == 1 else "Unsuccessful"
            print(
                f"  {success_label}: Mean={stats['mean']:.4f}, Count={stats['count']}, Std={stats['std']:.4f}"
            )


def summarize_news_features(final_merged_df, output_path, logger):
    """Summary of News Feature Integration"""
    logger.info("Summarizing news feature integration...")
    print("\n📈 Summary of News Feature Integration")

    # Count new features added from news data
    news_features = [col for col in final_merged_df.columns if "sentiment" in col]
    print(f"\n🏗️ Added {len(news_features)} news-based features:")
    for feature in news_features:
        print(f"  • {feature}")

    # Basic correlation with success
    if "success" in final_merged_df.columns and news_features:
        print(f"\n📊 Correlation with business success:")
        corr_data = (
            final_merged_df[["success"] + news_features].corr()["success"].sort_values()
        )
        for feature, correlation in corr_data[1:].items():  # Skip self-correlation
            print(f"  • {feature}: {correlation:.4f}")


def main():
    """Main execution function for news integration"""
    # Setup logging and directories
    logger, base_dir, final_dir, news_dir = setup_logging_and_directories()

    logger.info("Starting News Data Integration Pipeline")
    print("=" * 80)
    print("NEWS DATA INTEGRATION PIPELINE")
    print("=" * 80)

    try:
        # Setup NLTK dependencies
        setup_nltk_dependencies(logger)

        # Load datasets
        business_df, news_df = load_datasets(final_dir, news_dir, logger)

        # Explore business data
        explore_business_data(business_df, logger)

        # Explore news data
        explore_news_data(news_df, logger)

        # Preprocess news data
        news_processed = preprocess_news_data(news_df, logger)

        # Perform sentiment analysis
        news_processed = perform_sentiment_analysis(news_processed, logger)

        # Extract location mentions
        news_processed = extract_location_mentions(news_processed, business_df, logger)

        # Create time-based aggregations
        sentiment_by_time = create_time_based_aggregations(news_processed, logger)

        # Perform temporal merging
        merged_df_temporal = perform_temporal_merging(
            business_df, sentiment_by_time, logger
        )

        # Perform spatial merging
        merged_df_spatial = perform_spatial_merging(business_df, news_processed, logger)

        # Create final merged dataset
        final_merged_df = create_final_merged_dataset(
            business_df, merged_df_temporal, merged_df_spatial, logger
        )

        # Create additional features
        final_merged_df = create_additional_features(final_merged_df, logger)

        # Save merged dataset
        output_path = save_merged_dataset(final_merged_df, final_dir, base_dir, logger)

        # Analyze news relationships
        analyze_news_relationships(final_merged_df, logger)

        # Summarize news features
        summarize_news_features(final_merged_df, output_path, logger)

        # Summary Report
        print("\n" + "=" * 80)
        print("NEWS INTEGRATION COMPLETE")
        print("=" * 80)

        print(f"\n📊 DATASET CREATED:")
        print(f"   • sf_business_with_news.parquet")

        print(f"\n📋 FINAL DATASET SUMMARY:")
        print(f"   • Records: {final_merged_df.shape[0]:,}")
        print(f"   • Features: {final_merged_df.shape[1]}")

        news_features = [col for col in final_merged_df.columns if "sentiment" in col]
        print(f"   • New News Features: {len(news_features)}")

        if "success" in final_merged_df.columns:
            print(f"   • Success Rate: {final_merged_df['success'].mean():.2%}")

        print(f"\n💾 OUTPUT FILE:")
        print(f"   • {output_path}")

        print(
            f"\n🎉 News integration complete! Dataset ready for business success modeling."
        )
        print(f"📁 Final dataset location: {output_path}")
        print(f"📊 Final dataset dimensions: {final_merged_df.shape}")

        logger.info("News Data Integration Pipeline completed successfully!")

        return final_merged_df

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"❌ Error in news integration pipeline: {e}")
        raise


# Execute if run as main script
if __name__ == "__main__":
    print("News Data Integration Pipeline Starting...")
    integrated_dataset = main()
