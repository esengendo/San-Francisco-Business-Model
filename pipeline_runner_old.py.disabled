#!/usr/bin/env python3
"""
Pipeline runner to execute all scripts in correct order for SF Business Model
"""
import subprocess
import sys
import logging
import os
from pathlib import Path


def setup_logging():
    """Configure logging for the pipeline runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("PipelineRunner")


def run_script(script_name, logger):
    """Run a Python script and handle errors"""
    logger.info(f"Starting: {script_name}")

    try:
        result = subprocess.run(
            [sys.executable, script_name], capture_output=True, text=True, check=True
        )
        logger.info(f"Completed: {script_name}")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {script_name}")
        logger.error(f"Error output: {e.stderr}")
        logger.error(f"Return code: {e.returncode}")
        return False


def main():
    """Run the complete pipeline"""
    logger = setup_logging()
    logger.info("Starting SF Business Model Pipeline")

    # Define script execution order based on your numbering
    scripts = [
        "api_keys_validation_01.py",
        "logging_config_setup_02.py",
        "helper_functions_03.py",
        "sf_business_data_04.py",
        "fred_economic_data_05.py",
        "census_demographic_data_06.py",
        "sf_planning_data_07.py",
        "sf_crime_data_08.py",
        "sf311_data_09.py",
        "osm_business_data_10.py",
        "gdelt_news_data_11.py",
        "sf_news_rss_data_12.py",
        "wayback_historical_data_13.py",
        "file_combination_cleanup_14.py",
        "business_data_processing_15.py",
        "business_analysis_merge_16.py",
        "osm_enrichment_merge_17.py",
        "land_use_integration_merge_18.py",
        "permits_integration_merge_19.py",
        "sf311_integration_merge_20.py",
        "crime_integration_merge_21.py",
        "news_integration_merge_22.py",
        "premodeling_pipeline_23.py",
        "model_training_with_save_load_24.py",
    ]

    # Check if we're in "pipeline-only" mode or full mode
    pipeline_only = os.getenv("PIPELINE_ONLY", "false").lower() == "true"

    # Run each script in order
    for script in scripts:
        if not Path(script).exists():
            logger.warning(f"Script not found: {script} - skipping")
            continue

        if not run_script(script, logger):
            logger.error(f"Pipeline failed at: {script}")
            sys.exit(1)

    logger.info("Pipeline completed successfully!")

    # If pipeline-only mode, exit here
    if pipeline_only:
        logger.info("Pipeline-only mode: Exiting without starting Streamlit")
        return

    # Start Streamlit app
    logger.info("Starting Streamlit app...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.address",
                "0.0.0.0",
                "--server.port",
                "8501",
                "--server.headless",
                "true",
            ]
        )
    except KeyboardInterrupt:
        logger.info("Streamlit app stopped")


if __name__ == "__main__":
    main()
