import os
import sys
import subprocess
import time
import signal
import logging
from datetime import datetime
import json


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Script execution timed out")


class ResilientPipelineRunner:
    def __init__(self, timeout_minutes=120):  # 2 hour default timeout
        self.timeout_seconds = timeout_minutes * 60
        self.results = []
        self.start_time = datetime.now()
        
        # 🎯 NEW: Detect single run mode from environment
        self.single_run_mode = os.getenv("PIPELINE_MODE") == "single_run"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("pipeline_execution.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("ResilientPipelineRunner")
        
        # 🎯 NEW: Log the mode we're running in
        if self.single_run_mode:
            self.logger.info("🎯 Running in SINGLE RUN mode - will exit after completion")
        else:
            self.logger.info("🔄 Running in CONTINUOUS mode")

    def run_script_with_timeout(self, script_path, timeout_seconds=None):
        """Run a script with timeout and capture all output"""
        if timeout_seconds is None:
            timeout_seconds = self.timeout_seconds

        start_time = time.time()

        try:
            # Set up signal handler for timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)

            # Run the script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )

            # Clear the alarm
            signal.alarm(0)

            execution_time = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": execution_time,
                "timed_out": False,
            }

        except TimeoutError:
            execution_time = time.time() - start_time
            self.logger.warning(
                f"Script {script_path} timed out after {timeout_seconds} seconds"
            )
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Script timed out after {timeout_seconds} seconds",
                "execution_time": execution_time,
                "timed_out": True,
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "execution_time": execution_time,
                "timed_out": False,
            }
        finally:
            signal.alarm(0)  # Make sure to clear any pending alarm

    def retry_script(self, script_path, max_retries=2):
        """Retry a failed script up to max_retries times"""
        for attempt in range(max_retries + 1):
            if attempt > 0:
                self.logger.info(f"Retry attempt {attempt} for {script_path}")
                time.sleep(30)  # Wait 30 seconds between retries

            result = self.run_script_with_timeout(script_path)

            if result["success"]:
                if attempt > 0:
                    self.logger.info(
                        f"Script {script_path} succeeded on retry {attempt}"
                    )
                return result
            elif result["timed_out"]:
                # Don't retry timeouts, they'll likely timeout again
                self.logger.warning(f"Script {script_path} timed out, not retrying")
                break

        return result

    def run_pipeline(self, continue_on_failure=True, script_timeout_minutes=None):
        """Run the entire pipeline with resilience features"""

        # Custom timeouts for known slow scripts
        script_timeouts = {
            "wayback_historical_data_13.py": 45,  # 45 mins for wayback
            "sf_business_data_04.py": 30,  # 30 minutes for business data
            "census_demographic_data_06.py": 45,  # 45 minutes for census
            "sf_crime_data_08.py": 30,  # 30 minutes for crime data
            "sf311_data_09.py": 30,  # 30 minutes for 311 data
            "model_training_with_save_load_24.py": 60,  # 60 minutes for model training
        }

        # Get all Python scripts in current directory
        scripts = [
            # 1. Setup and validation
            "api_keys_validation_01.py",
            "logging_config_setup_02.py",
            "helper_functions_03.py",
            # 2. Raw data collection (must run before processing)
            "sf_business_data_04.py",  # Creates raw business data
            "fred_economic_data_05.py",
            "census_demographic_data_06.py",
            "sf_planning_data_07.py",
            "sf_crime_data_08.py",
            "sf311_data_09.py",
            "osm_business_data_10.py",
            "gdelt_news_data_11.py",
            "sf_news_rss_data_12.py",
            "wayback_historical_data_13.py",
            # 3. Data processing (must run after raw data collection)
            "file_combination_cleanup_14.py",
            "business_data_processing_15.py",  # Processes raw business data
            # 4. Data integration/merging (must run after processing)
            "business_analysis_merge_16.py",  # Needs processed business data
            "osm_enrichment_merge_17.py",
            "land_use_integration_merge_18.py",
            "permits_integration_merge_19.py",
            "sf311_integration_merge_20.py",
            "crime_integration_merge_21.py",
            "news_integration_merge_22.py",
            # 5. Pre-modeling (final step)
            "premodeling_pipeline_23.py",
            # 6. Model Training (FINAL STEP - trains the actual ML model)
            "model_training_with_save_load_24.py",
        ]

        # Only include scripts that actually exist
        scripts = [script for script in scripts if os.path.exists(script)]

        self.logger.info(
            f"Starting SF Business Model Pipeline with {len(scripts)} scripts"
        )
        self.logger.info(f"Continue on failure: {continue_on_failure}")
        
        # 🎯 NEW: Log single run mode status
        if self.single_run_mode:
            self.logger.info("🎯 Single run mode: Pipeline will exit after completion")

        total_scripts = len(scripts)
        successful_scripts = 0
        failed_scripts = 0

        for i, script in enumerate(scripts, 1):
            script_start_time = datetime.now()

            # Determine timeout for this script
            timeout_minutes = script_timeouts.get(script, script_timeout_minutes or 120)
            timeout_seconds = timeout_minutes * 60

            self.logger.info(
                f"Starting ({i}/{total_scripts}): {script} (timeout: {timeout_minutes}min)"
            )

            try:
                # Run script with retry logic (except for very slow scripts)
                if script in ["wayback_historical_data_13.py"]:
                    # Don't retry very slow scripts
                    result = self.run_script_with_timeout(script, timeout_seconds)
                else:
                    # Retry other scripts if they fail
                    result = self.retry_script(script)

                # Store result
                script_result = {
                    "script": script,
                    "start_time": script_start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "execution_time": result["execution_time"],
                    "success": result["success"],
                    "returncode": result["returncode"],
                    "timed_out": result["timed_out"],
                    "stdout_lines": (
                        len(result["stdout"].split("\n")) if result["stdout"] else 0
                    ),
                    "stderr_lines": (
                        len(result["stderr"].split("\n")) if result["stderr"] else 0
                    ),
                }

                self.results.append(script_result)

                if result["success"]:
                    successful_scripts += 1
                    self.logger.info(
                        f"✅ Completed: {script} ({result['execution_time']:.1f}s)"
                    )

                    # Extract and log key output
                    if result["stdout"]:
                        output_lines = result["stdout"].strip().split("\n")
                        key_output = [
                            line for line in output_lines[-10:] if line.strip()
                        ]
                        if key_output:
                            self.logger.info(f"Output: {' | '.join(key_output)}")

                else:
                    failed_scripts += 1
                    if result["timed_out"]:
                        self.logger.error(
                            f"❌ TIMEOUT: {script} (>{timeout_minutes}min)"
                        )
                    else:
                        self.logger.error(
                            f"❌ Failed: {script} (exit code: {result['returncode']})"
                        )

                    if result["stderr"]:
                        self.logger.error(f"Error output: {result['stderr'][:500]}")

                    if not continue_on_failure:
                        self.logger.error(
                            f"Pipeline stopped due to failure in {script}"
                        )
                        break
                    else:
                        self.logger.info(
                            f"Continuing with next script due to continue_on_failure=True"
                        )

            except KeyboardInterrupt:
                self.logger.warning(f"Pipeline interrupted by user during {script}")
                break
            except Exception as e:
                failed_scripts += 1
                self.logger.error(f"❌ Unexpected error in {script}: {e}")
                if not continue_on_failure:
                    break

        # Generate final report
        self.generate_final_report(total_scripts, successful_scripts, failed_scripts)
        return self.results

    def generate_final_report(self, total_scripts, successful_scripts, failed_scripts):
        """Generate a comprehensive final report"""
        end_time = datetime.now()
        total_time = end_time - self.start_time

        self.logger.info("=" * 60)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total execution time: {total_time}")
        self.logger.info(f"Scripts processed: {total_scripts}")
        self.logger.info(f"✅ Successful: {successful_scripts}")
        self.logger.info(f"❌ Failed: {failed_scripts}")
        self.logger.info(f"Success rate: {(successful_scripts/total_scripts)*100:.1f}%")

        # Detailed results
        self.logger.info("\nDETAILED RESULTS:")
        for result in self.results:
            status = "✅" if result["success"] else "❌"
            timing = f"({result['execution_time']:.1f}s)"
            extra = " [TIMEOUT]" if result.get("timed_out") else ""
            self.logger.info(f"{status} {result['script']} {timing}{extra}")

        # Save detailed results to JSON
        report_file = f"pipeline_report_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, "w") as f:
                json.dump(
                    {
                        "summary": {
                            "start_time": self.start_time.isoformat(),
                            "end_time": end_time.isoformat(),
                            "total_time_seconds": total_time.total_seconds(),
                            "total_scripts": total_scripts,
                            "successful_scripts": successful_scripts,
                            "failed_scripts": failed_scripts,
                            "success_rate": (successful_scripts / total_scripts) * 100,
                        },
                        "script_results": self.results,
                    },
                    f,
                    indent=2,
                )
            self.logger.info(f"Detailed report saved to: {report_file}")
        except Exception as e:
            self.logger.error(f"Could not save detailed report: {e}")

        # 🎯 NEW: Exit cleanly in single run mode
        if self.single_run_mode:
            self.logger.info("🎯 Single run mode: Pipeline completed, exiting cleanly")
            if successful_scripts == total_scripts:
                self.logger.info("🎉 All scripts completed successfully!")
                sys.exit(0)
            else:
                self.logger.warning(f"⚠️ Pipeline completed with {failed_scripts} failures")
                sys.exit(1)


if __name__ == "__main__":
    # Configuration
    CONTINUE_ON_FAILURE = True  # Set to False to stop on first failure
    DEFAULT_SCRIPT_TIMEOUT_MINUTES = 120  # 2 hours default

    runner = ResilientPipelineRunner()

    try:
        results = runner.run_pipeline(
            continue_on_failure=CONTINUE_ON_FAILURE,
            script_timeout_minutes=DEFAULT_SCRIPT_TIMEOUT_MINUTES,
        )

        # 🎯 NEW: This section only runs if NOT in single_run_mode
        # (because single_run_mode exits in generate_final_report)
        successful = sum(1 for r in results if r["success"])
        total = len(results)

        if successful == total:
            print(
                f"\n🎉 Pipeline completed successfully! All {total} scripts executed."
            )
            sys.exit(0)
        elif successful > 0:
            print(
                f"\n⚠️ Pipeline completed with partial success: {successful}/{total} scripts succeeded."
            )
            sys.exit(1)
        else:
            print(f"\n💥 Pipeline failed: 0/{total} scripts succeeded.")
            sys.exit(2)

    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user.")
        sys.exit(130)