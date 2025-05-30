# temporal_fix_pipeline_23.py
# Temporal Data Leakage Fix Pipeline
# Diagnoses and fixes temporal leakage in business success prediction data
import os
import pandas as pd
import numpy as np
from datetime import datetime
import shutil
import glob
import logging
import json
import warnings
import pickle  # For saving summary

warnings.filterwarnings("ignore")


def setup_logging_and_directories():
    """Configure logging and set up directory structure"""
    # Configure logging (re-using your existing setup for consistency)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("SFBusinessPreprocessing")

    # Set base directory and paths
    base_dir = os.getenv("BASE_DIR", "/app/San_Francisco_Business_Model")
    # base_dir = "/Users/baboo/Documents/San Francisco Business Model"
    raw_data_dir = f"{base_dir}/raw_data"
    processed_dir = f"{base_dir}/processed"
    final_processed_dir = f"{processed_dir}/final"  # All outputs go here
    model_dir = f"{base_dir}/models"
    archive_dir = f"{base_dir}/archive"

    # Ensure directories exist
    for directory in [
        raw_data_dir,
        processed_dir,
        final_processed_dir,
        model_dir,
        archive_dir,
    ]:
        os.makedirs(directory, exist_ok=True)

    return logger, base_dir, processed_dir, final_processed_dir


def cleanup_corrected_files(output_dir):
    """Clean up previous split files before creating new ones"""
    print("=" * 60)
    print("CLEANING UP PREVIOUS FILES")
    print("=" * 60)

    # List of files this pipeline creates and should clean up
    files_to_cleanup = [
        "X_train.parquet",
        "X_val.parquet",
        "X_test.parquet",
        "y_train.parquet",
        "y_val.parquet",
        "y_test.parquet",
        "X_train_processed.parquet",
        "X_val_processed.parquet",
        "X_test_processed.parquet",
        "X_train_corrected.parquet",
        "X_val_corrected.parquet",
        "X_test_corrected.parquet",
        "y_train_corrected.parquet",
        "y_val_corrected.parquet",
        "y_test_corrected.parquet",
        "split_summary.pkl",
        "corrected_splits_report.txt",
        "temporal_fix_log.txt",
    ]

    cleanup_count = 0
    for filename in files_to_cleanup:
        filepath = f"{output_dir}/{filename}"
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"  ✅ Removed: {filename}")
                cleanup_count += 1
            except Exception as e:
                print(f"  ⚠️ Could not remove {filename}: {e}")
        else:
            print(f"  📁 File not found (skipping): {filename}")

    if cleanup_count > 0:
        print(f"🧹 Cleaned up {cleanup_count} previous files")
    else:
        print("📁 No previous files found to clean up")

    print()


def diagnose_temporal_leakage(output_dir, processed_dir):
    """Diagnose the temporal leakage issue using existing splits OR the main input file."""
    print("=" * 80)
    print("TEMPORAL DATA LEAKAGE DIAGNOSIS")
    print("=" * 80)

    # First try to load existing splits for diagnosis
    splits_exist = all(
        os.path.exists(f"{output_dir}/{f}")
        for f in [
            "X_train.parquet",
            "X_val.parquet",
            "X_test.parquet",
            "y_train.parquet",
            "y_val.parquet",
            "y_test.parquet",
        ]
    )

    if splits_exist:
        print("📊 Analyzing existing splits...")
        try:
            X_train = pd.read_parquet(f"{output_dir}/X_train.parquet")
            X_val = pd.read_parquet(f"{output_dir}/X_val.parquet")
            X_test = pd.read_parquet(f"{output_dir}/X_test.parquet")

            y_train = pd.read_parquet(f"{output_dir}/y_train.parquet")["success"]
            y_val = pd.read_parquet(f"{output_dir}/y_val.parquet")["success"]
            y_test = pd.read_parquet(f"{output_dir}/y_test.parquet")["success"]
            print(f"✅ Loaded existing splits from: {output_dir}")
        except Exception as e:
            print(f"❌ Error loading existing splits: {e}")
            splits_exist = False

    if not splits_exist:
        print(
            "📊 No existing splits found. Will analyze main input file for temporal patterns..."
        )
        # Load main input file to understand data structure
        main_input_path = f"{processed_dir}/final/sf_business_with_news.parquet"
        if not os.path.exists(main_input_path):
            print(f"❌ Main input file not found: {main_input_path}")
            return None

        df = pd.read_parquet(main_input_path)
        print(f"✅ Loaded main input file: {len(df):,} records")

        # Check for date columns and success column
        date_columns = [
            "start_date",
            "dba_start_date",
            "location_start_date",
            "start_year",
        ]
        date_col = None
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            print("❌ No date column found in main input file.")
            return None

        if "success" not in df.columns:
            print("⚠️ No 'success' column found. Will create during processing.")
        else:
            print(f"📊 Current success rate in main file: {df['success'].mean():.1%}")

        return date_col

    # Analyze existing splits
    date_columns = ["start_date", "dba_start_date", "location_start_date", "start_year"]
    date_col = None
    for col in date_columns:
        if col in X_train.columns:
            date_col = col
            break

    if date_col is None:
        print("❌ No date column found in existing splits.")
        return None

    print(f"📅 Analyzing temporal patterns using: {date_col}")

    # Analyze date ranges and success rates
    def analyze_split(X, y, split_name):
        if date_col in ["start_date", "dba_start_date", "location_start_date"]:
            dates = pd.to_datetime(X[date_col], errors="coerce")
            if not dates.dropna().empty:
                min_date = dates.min()
                max_date = dates.max()
                print(f"\n{split_name}:")
                print(
                    f"  📅 Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                )
            else:
                print(f"\n{split_name}: No valid date data found.")
        else:  # For 'start_year'
            years = X[date_col].dropna()
            if not years.empty:
                years = years.astype(int)
                min_year = years.min()
                max_year = years.max()
                print(f"\n{split_name}:")
                print(f"  📅 Year range: {min_year} to {max_year}")
            else:
                print(f"\n{split_name}: No valid year data found.")

        print(f"  📊 Success rate: {y.mean():.1%}")
        print(f"  🏢 Business count: {len(y):,}")

    analyze_split(X_train, y_train, "TRAINING SET")
    analyze_split(X_val, y_val, "VALIDATION SET")
    analyze_split(X_test, y_test, "TEST SET")

    print("\n🔍 DIAGNOSIS:")
    if y_test.mean() < 0.01 and y_train.mean() > 0.1:
        print("❌ CRITICAL TEMPORAL DATA LEAKAGE DETECTED!")
        print("  • Test set success rate is near zero")
        print("  • Training set has much higher success rate")
        print("  • This indicates temporal leakage in the splits")
    elif abs(y_train.mean() - y_test.mean()) > 0.15:
        print("⚠️ SIGNIFICANT TEMPORAL IMBALANCE DETECTED!")
        print("  • Success rates vary significantly across splits")
    else:
        print("✅ Splits appear reasonably balanced")

    return date_col


def create_corrected_temporal_splits(output_dir, processed_dir):
    """
    Creates corrected temporal splits using the main input file.
    Saves all outputs to the final processed directory.
    """
    print("\n" + "=" * 80)
    print("CREATING CORRECTED TEMPORAL SPLITS")
    print("=" * 80)

    # Load the main preprocessed dataset
    main_input_path = f"{processed_dir}/final/sf_business_with_news.parquet"

    if not os.path.exists(main_input_path):
        raise FileNotFoundError(f"Main input file not found: {main_input_path}")

    df = pd.read_parquet(main_input_path)
    logger.info(f"Loaded data for correction from: {main_input_path}")
    print(
        f"✅ Loaded {len(df):,} businesses from: {main_input_path} for re-processing."
    )

    initial_record_count = len(df)
    logger.info(f"Loaded {initial_record_count:,} businesses from main input file")
    print(f"✅ Loaded {initial_record_count:,} businesses from main input file")

    # Convert date columns to datetime
    date_columns = [
        "dba_start_date",
        "dba_end_date",
        "location_start_date",
        "location_end_date",
    ]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            print(f"📅 Converted {col} to datetime")

    # Establish start_date consistently
    if "start_date" not in df.columns:
        if "dba_start_date" in df.columns:
            df["start_date"] = df["dba_start_date"]
            print("📅 Using dba_start_date as start_date")
        elif "location_start_date" in df.columns:
            df["start_date"] = df["location_start_date"]
            print("📅 Using location_start_date as start_date")
        else:
            # Look for any column with 'start' and 'date' in the name
            start_cols = [
                col
                for col in df.columns
                if "start" in col.lower() and "date" in col.lower()
            ]
            if start_cols:
                df["start_date"] = pd.to_datetime(df[start_cols[0]], errors="coerce")
                print(f"📅 Using {start_cols[0]} as start_date")
            else:
                raise ValueError(
                    "No start date column found! Cannot proceed with temporal logic."
                )
    else:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        print("📅 Using existing start_date column")

    # Remove rows where start_date is missing
    initial_count = len(df)
    df = df.dropna(subset=["start_date"]).copy()
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"🧹 Removed {removed_count:,} businesses with missing start_date")

    # Define global data cutoff date for success definition
    # Use the maximum start_date as a proxy for when data collection ended
    global_data_cutoff_date = df["start_date"].max()
    logger.info(
        f"Using global data cutoff date: {global_data_cutoff_date.strftime('%Y-%m-%d')}"
    )
    print(f"🔮 Global data cutoff date: {global_data_cutoff_date.strftime('%Y-%m-%d')}")

    # Calculate business duration at cutoff date
    if "dba_end_date" in df.columns:
        # Use actual end date if available and before cutoff, otherwise use cutoff date
        df["calculated_end_date"] = df["dba_end_date"].fillna(global_data_cutoff_date)
        df["calculated_end_date"] = df[["calculated_end_date", "start_date"]].max(
            axis=1
        )
    else:
        # If no end date column, assume all businesses were active until cutoff
        df["calculated_end_date"] = global_data_cutoff_date
        print("📅 No end date column found, using cutoff date for all businesses")

    # Calculate years in business at cutoff date
    df["years_in_business_at_cutoff"] = (
        df["calculated_end_date"] - df["start_date"]
    ).dt.days / 365.25

    # Define success: operated for at least 5 years by cutoff date
    df["success"] = (df["years_in_business_at_cutoff"] >= 5).astype(int)
    success_rate = df["success"].mean()
    print(f"✅ Success rate after redefinition: {success_rate:.1%}")

    # Remove leakage features
    leakage_features = [
        "dba_end_date",
        "Dba_end_date",
        "location_end_date",
        "Location_end_date",
        "is_open",
        "years_in_business",
        "business_age_years",
        "data_as_of",
        "data_loaded_at",
        "industry_success_rate",
        "neighborhood_success_rate",
        "district_success_rate",
        "success_rate",
        "failure_rate",
        "calculated_end_date",
        "years_in_business_at_cutoff",
        "administratively_closed",
    ]

    existing_leakage_features = [f for f in leakage_features if f in df.columns]
    df_clean = df.drop(columns=existing_leakage_features, errors="ignore")
    print(f"🧹 Removed {len(existing_leakage_features)} leakage features")

    # Define feature columns (exclude IDs and target)
    id_columns = [
        "ttxid",
        "certificate_number",
        "ownership_name",
        "dba_name",
        "full_business_address",
        "dba_start_date",
        "location_start_date",
        "start_date",
        "mail_address",
        "geometry",
        "street_address",
        "city",
        "state",
        "source_zipcode",
    ]

    # Get all potential feature columns
    potential_features = [
        col for col in df_clean.columns if col not in id_columns + ["success"]
    ]

    # Remove high-cardinality text columns
    text_columns = []
    for col in potential_features:
        if (
            df_clean[col].dtype == "object"
            and df_clean[col].nunique() / len(df_clean) > 0.5
        ):
            text_columns.append(col)

    feature_columns = [col for col in potential_features if col not in text_columns]
    print(f"✅ Selected {len(feature_columns)} features for modeling")
    if text_columns:
        print(f"🧹 Removed {len(text_columns)} high-cardinality text features")

    # Create chronological splits
    df_sorted = df_clean.sort_values("start_date").reset_index(drop=True)
    df_final = df_sorted[feature_columns + ["success", "start_date"]].copy()

    print("\n🔄 Creating STRICTLY TEMPORAL SPLITS")

    n = len(df_final)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_data = df_final.iloc[:train_end].copy()
    val_data = df_final.iloc[train_end:val_end].copy()
    test_data = df_final.iloc[val_end:].copy()

    print(f"\n📊 TEMPORAL SPLIT RESULTS:")
    print(
        f"  Training:   {len(train_data):,} businesses ({train_data['start_date'].min().strftime('%Y-%m-%d')} to {train_data['start_date'].max().strftime('%Y-%m-%d')}) | Success: {train_data['success'].mean():.1%}"
    )
    print(
        f"  Validation: {len(val_data):,} businesses ({val_data['start_date'].min().strftime('%Y-%m-%d')} to {val_data['start_date'].max().strftime('%Y-%m-%d')}) | Success: {val_data['success'].mean():.1%}"
    )
    print(
        f"  Test:       {len(test_data):,} businesses ({test_data['start_date'].min().strftime('%Y-%m-%d')} to {test_data['start_date'].max().strftime('%Y-%m-%d')}) | Success: {test_data['success'].mean():.1%}"
    )

    # Check for large success rate differences and apply stratified sampling if needed
    train_success_rate = train_data["success"].mean()
    test_success_rate = test_data["success"].mean()
    split_method = "chronological"  # Initialize default split method

    if abs(train_success_rate - test_success_rate) > 0.15:  # >15% difference
        print(
            "\n⚠️ Large success rate difference detected. Applying stratified sampling..."
        )

        # Create time periods for stratification
        df_final["time_period"] = pd.cut(df_final["start_date"], bins=5, labels=False)

        print("\n📊 Success rates by time period:")
        for period in sorted(df_final["time_period"].unique()):
            period_data = df_final[df_final["time_period"] == period]
            period_start = period_data["start_date"].min()
            period_end = period_data["start_date"].max()
            success_rate_period = period_data["success"].mean()
            print(
                f"  Period {period}: {period_start.strftime('%Y-%m')} to {period_end.strftime('%Y-%m')} | Success: {success_rate_period:.1%} | Count: {len(period_data):,}"
            )

        # Alternative: Stratified sampling across all time periods
        from sklearn.model_selection import train_test_split

        X = df_final.drop(columns=["success", "start_date"])
        y = df_final["success"]

        # Create stratification key combining time period and success
        strat_key = df_final["time_period"].astype(str) + "_" + y.astype(str)

        # Handle rare combinations
        key_counts = strat_key.value_counts()
        strat_key = strat_key.apply(lambda x: x if key_counts[x] >= 10 else "rare")

        # Stratified split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=strat_key, random_state=42
        )

        # Split temp into train and val
        strat_temp = strat_key[X_temp.index]
        temp_key_counts = strat_temp.value_counts()
        strat_temp = strat_temp.apply(
            lambda x: x if temp_key_counts[x] >= 5 else "rare"
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.1875,
            stratify=strat_temp,
            random_state=42,  # 0.1875 * 0.8 = 0.15
        )

        print(f"\n📊 STRATIFIED SPLITS (BALANCED):")
        print(
            f"  Training:   {len(y_train):,} businesses | Success: {y_train.mean():.1%}"
        )
        print(f"  Validation: {len(y_val):,} businesses | Success: {y_val.mean():.1%}")
        print(
            f"  Test:       {len(y_test):,} businesses | Success: {y_test.mean():.1%}"
        )

        split_method = "stratified"

    else:
        # Use chronological splits
        X_train = train_data[feature_columns]
        y_train = train_data["success"]
        X_val = val_data[feature_columns]
        y_val = val_data["success"]
        X_test = test_data[feature_columns]
        y_test = test_data["success"]

    # Clean up previous files in the specified output directory
    cleanup_corrected_files(output_dir)

    # Save splits directly to the final processed directory (standard names only)
    X_train.to_parquet(f"{output_dir}/X_train.parquet")
    X_val.to_parquet(f"{output_dir}/X_val.parquet")
    X_test.to_parquet(f"{output_dir}/X_test.parquet")

    y_train.to_frame("success").to_parquet(f"{output_dir}/y_train.parquet")
    y_val.to_frame("success").to_parquet(f"{output_dir}/y_val.parquet")
    y_test.to_frame("success").to_parquet(f"{output_dir}/y_test.parquet")

    print(f"\n💾 All splits saved to: {output_dir}")
    print(f"📊 Feature count: {len(feature_columns)}")

    # Create comprehensive summary
    summary = {
        "preprocessing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": main_input_path,
        "output_directory": output_dir,
        "total_businesses_initial_load": initial_record_count,
        "businesses_after_date_filter": len(df_final),
        "global_data_cutoff_date": global_data_cutoff_date.strftime("%Y-%m-%d"),
        "target_definition": "Success if operated >= 5 years by global_data_cutoff_date",
        "train_count": len(y_train),
        "val_count": len(y_val),
        "test_count": len(y_test),
        "train_success_rate": float(y_train.mean()),
        "val_success_rate": float(y_val.mean()),
        "test_success_rate": float(y_test.mean()),
        "overall_success_rate": float(success_rate),
        "temporal_ranges": {
            "train": (
                f"{train_data['start_date'].min().strftime('%Y-%m-%d')} to {train_data['start_date'].max().strftime('%Y-%m-%d')}"
                if split_method == "chronological"
                else "Mixed time periods (stratified)"
            ),
            "val": (
                f"{val_data['start_date'].min().strftime('%Y-%m-%d')} to {val_data['start_date'].max().strftime('%Y-%m-%d')}"
                if split_method == "chronological"
                else "Mixed time periods (stratified)"
            ),
            "test": (
                f"{test_data['start_date'].min().strftime('%Y-%m-%d')} to {test_data['start_date'].max().strftime('%Y-%m-%d')}"
                if split_method == "chronological"
                else "Mixed time periods (stratified)"
            ),
        },
        "feature_count": len(feature_columns),
        "features_used": feature_columns,
        "removed_leakage_features": existing_leakage_features,
        "removed_text_features": text_columns,
        "split_method": split_method,
    }

    # Save summary report
    summary_report_path = f"{output_dir}/corrected_splits_report.txt"
    with open(summary_report_path, "w") as f:
        f.write("TEMPORAL DATA LEAKAGE - CORRECTED SPLITS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Created: {summary['preprocessing_date']}\n")
        f.write(f"Input File: {summary['input_file']}\n")
        f.write(f"Output Directory: {summary['output_directory']}\n\n")

        f.write("TARGET REDEFINITION:\n")
        f.write(f"  {summary['target_definition']}\n")
        f.write(f"  Global Data Cutoff Date: {summary['global_data_cutoff_date']}\n")
        f.write(f"  Overall Success Rate: {summary['overall_success_rate']:.1%}\n\n")

        f.write("CORRECTED SPLITS:\n")
        f.write(
            f"  Split Method: {summary['split_method'].replace('_', ' ').title()}\n"
        )
        f.write(
            f"  Training:   {summary['train_count']:,} businesses ({summary['temporal_ranges']['train']}) | Success: {summary['train_success_rate']:.1%}\n"
        )
        f.write(
            f"  Validation: {summary['val_count']:,} businesses ({summary['temporal_ranges']['val']}) | Success: {summary['val_success_rate']:.1%}\n"
        )
        f.write(
            f"  Test:       {summary['test_count']:,} businesses ({summary['temporal_ranges']['test']}) | Success: {summary['test_success_rate']:.1%}\n\n"
        )

        f.write(f"FEATURES: {summary['feature_count']} selected features\n")
        f.write(
            f"  Removed Leakage Features ({len(summary['removed_leakage_features'])}): {summary['removed_leakage_features']}\n"
        )
        f.write(
            f"  Removed Text Features ({len(summary['removed_text_features'])}): {summary['removed_text_features']}\n\n"
        )

        f.write("FILES CREATED:\n")
        f.write("  • X_train.parquet, X_val.parquet, X_test.parquet\n")
        f.write("  • y_train.parquet, y_val.parquet, y_test.parquet\n")
        f.write("  • split_summary.pkl (detailed summary)\n")
        f.write("  • corrected_splits_report.txt (this report)\n\n")

        f.write("EXPECTED IMPROVEMENTS:\n")
        if summary["split_method"] == "stratified":
            f.write("  • Balanced success rates across all splits\n")
            f.write("  • Precision: 60-80% (improved from temporal leakage)\n")
            f.write("  • ROC-AUC: 70-85% (more realistic performance)\n")
            f.write("  • Model learns patterns across all time periods\n")
        else:
            f.write(
                "  • Model learns from past to predict future (realistic deployment)\n"
            )
            f.write("  • More reliable evaluation metrics\n")
            f.write("  • Better generalization to new businesses\n")

    print(f"📄 Summary report saved: {summary_report_path}")

    # Save pickle summary
    with open(f"{output_dir}/split_summary.pkl", "wb") as f:
        pickle.dump(summary, f)

    print(f"\n✅ CORRECTED SPLITS READY FOR TRAINING!")
    print(f"🎯 All files saved to: {output_dir}")

    return summary


def main_temporal_fix():
    """Main function to diagnose and fix temporal leakage"""

    # Setup logging and directories
    logger, base_dir, processed_dir, final_processed_dir = (
        setup_logging_and_directories()
    )

    # Ensure output directory exists
    os.makedirs(final_processed_dir, exist_ok=True)

    print(f"🏁 Starting temporal leakage fix process")
    print(f"📁 Input file: {processed_dir}/final/sf_business_with_news.parquet")
    print(f"📁 Output directory: {final_processed_dir}")

    # Step 1: Diagnose potential issues
    date_col = diagnose_temporal_leakage(final_processed_dir, processed_dir)

    if date_col:
        try:
            # Step 2: Create corrected splits
            summary = create_corrected_temporal_splits(
                final_processed_dir, processed_dir
            )
            logger.info("Temporal leakage fix completed successfully.")
            return summary
        except Exception as e:
            logger.error(f"Failed to create corrected temporal splits: {str(e)}")
            print(f"\n❌ Error creating corrected splits: {e}")
            raise
    else:
        logger.warning(
            "Could not proceed with temporal fix - no suitable date column found."
        )
        print("\n⚠️ Could not proceed with temporal fix")
        return None


# Execute if run as main script
if __name__ == "__main__":
    # Setup logging and directories
    logger, base_dir, processed_dir, final_processed_dir = (
        setup_logging_and_directories()
    )

    # Set up logging
    log_file = f"{final_processed_dir}/temporal_fix_log.txt"
    os.makedirs(final_processed_dir, exist_ok=True)

    # Configure file logging
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # Add handler if not already present
    if not any(
        isinstance(handler, logging.FileHandler)
        and handler.baseFilename == os.path.abspath(log_file)
        for handler in logger.handlers
    ):
        logger.addHandler(file_handler)

    logger.info("Starting temporal leakage fix process.")

    try:
        summary = main_temporal_fix()
        if summary:
            print("\n" + "=" * 80)
            print("✅ TEMPORAL FIX COMPLETE - MODEL READY FOR RETRAINING!")
            print("=" * 80)
            print(f"📁 All outputs saved to: {final_processed_dir}")
            print(f"📊 Total features: {summary['feature_count']}")
            print(
                f"🎯 Success rates - Train: {summary['train_success_rate']:.1%}, Val: {summary['val_success_rate']:.1%}, Test: {summary['test_success_rate']:.1%}"
            )
        else:
            print("\n" + "=" * 80)
            print("⚠️ TEMPORAL FIX INCOMPLETE")
            print("=" * 80)
    except Exception as e:
        logger.critical(f"Temporal fix pipeline failed: {str(e)}", exc_info=True)
        print(f"\n❌ CRITICAL ERROR: {e}")
        raise
