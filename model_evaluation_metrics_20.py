import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import json
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging():
    """Initialize logging for model evaluation pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("SFBusinessModelEval")

def load_model_predictions(input_dir):
    """Load model predictions and actual values for evaluation
    
    Analyzes predictions for:
    - Business Success Probability
    - Revenue Growth Forecasts
    - Location Performance Scores
    - Market Opportunity Ratings
    
    Args:
        input_dir: Directory containing model output files
        
    Returns:
        Dictionary containing prediction DataFrames
    """
    logger = setup_logging()
    
    try:
        # Load success predictions
        success_df = pd.read_parquet(f"{input_dir}/predictions/business_success.parquet")
        logger.info(f"Loaded success predictions: {len(success_df):,} records")
        
        # Load revenue forecasts
        revenue_df = pd.read_parquet(f"{input_dir}/predictions/revenue_growth.parquet")
        logger.info(f"Loaded revenue forecasts: {len(revenue_df):,} records")
        
        # Load location scores
        location_df = pd.read_parquet(f"{input_dir}/predictions/location_scores.parquet")
        logger.info(f"Loaded location scores: {len(location_df):,} records")
        
        # Load market ratings
        market_df = pd.read_parquet(f"{input_dir}/predictions/market_ratings.parquet")
        logger.info(f"Loaded market ratings: {len(market_df):,} records")
        
    except FileNotFoundError as e:
        logger.error(f"Required prediction file not found: {e}")
        raise
    
    return {
        "success": success_df,
        "revenue": revenue_df,
        "location": location_df,
        "market": market_df
    }

def calculate_regression_metrics(y_true, y_pred):
    """Calculate performance metrics for regression models
    
    Evaluates:
    - Prediction Accuracy (R²)
    - Error Magnitude (RMSE, MAE)
    - Forecast Reliability
    - Model Consistency
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of regression metrics
    """
    metrics = {
        "r2_score": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mean_error": np.mean(y_true - y_pred),
        "error_std": np.std(y_true - y_pred)
    }
    return metrics

def calculate_classification_metrics(y_true, y_pred):
    """Calculate performance metrics for classification models
    
    Evaluates:
    - Prediction Accuracy
    - Success Detection Rate
    - Risk Assessment Precision
    - Overall Model Reliability
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of classification metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted")
    }
    return metrics

def evaluate_model_predictions(data_dict):
    """Evaluate model predictions across all business metrics
    
    Analyzes:
    - Business Success Prediction Accuracy
    - Revenue Forecast Precision
    - Location Score Reliability
    - Market Rating Validation
    
    Args:
        data_dict: Dictionary containing prediction DataFrames
        
    Returns:
        Dictionary containing evaluation results
    """
    evaluation = {}
    
    # Evaluate success predictions
    if not data_dict["success"].empty:
        success_metrics = calculate_classification_metrics(
            data_dict["success"]["actual"],
            data_dict["success"]["predicted"]
        )
        evaluation["success_metrics"] = success_metrics
    
    # Evaluate revenue forecasts
    if not data_dict["revenue"].empty:
        revenue_metrics = calculate_regression_metrics(
            data_dict["revenue"]["actual"],
            data_dict["revenue"]["predicted"]
        )
        evaluation["revenue_metrics"] = revenue_metrics
    
    # Evaluate location scores
    if not data_dict["location"].empty:
        location_metrics = calculate_regression_metrics(
            data_dict["location"]["actual"],
            data_dict["location"]["predicted"]
        )
        evaluation["location_metrics"] = location_metrics
    
    # Evaluate market ratings
    if not data_dict["market"].empty:
        market_metrics = calculate_regression_metrics(
            data_dict["market"]["actual"],
            data_dict["market"]["predicted"]
        )
        evaluation["market_metrics"] = market_metrics
    
    return evaluation

def generate_evaluation_plots(data_dict, evaluation, output_dir):
    """Create visualization of model performance metrics
    
    Generates:
    - Prediction Accuracy Charts
    - Error Distribution Plots
    - Performance Comparison Graphs
    - Reliability Analysis Plots
    
    Args:
        data_dict: Dictionary containing prediction DataFrames
        evaluation: Dictionary containing evaluation results
        output_dir: Directory for saving outputs
    """
    # Set visualization style
    plt.style.use("seaborn")
    
    # Success Prediction Analysis
    if "success_metrics" in evaluation:
        fig, ax = plt.subplots(figsize=(8, 6))
        metrics = pd.Series(evaluation["success_metrics"])
        metrics.plot(kind="bar")
        ax.set_title("Business Success Prediction Performance")
        ax.set_ylabel("Score")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/success_prediction_metrics.png")
        plt.close()
    
    # Revenue Forecast Analysis
    if "revenue_metrics" in evaluation:
        fig, ax = plt.subplots(figsize=(10, 6))
        actual = data_dict["revenue"]["actual"]
        predicted = data_dict["revenue"]["predicted"]
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
        ax.set_title("Revenue Growth Forecast Analysis")
        ax.set_xlabel("Actual Growth")
        ax.set_ylabel("Predicted Growth")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/revenue_forecast_analysis.png")
        plt.close()
    
    # Location Score Analysis
    if "location_metrics" in evaluation:
        fig, ax = plt.subplots(figsize=(10, 6))
        errors = data_dict["location"]["actual"] - data_dict["location"]["predicted"]
        sns.histplot(errors, kde=True)
        ax.set_title("Location Score Error Distribution")
        ax.set_xlabel("Prediction Error")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/location_score_errors.png")
        plt.close()
    
    # Market Rating Analysis
    if "market_metrics" in evaluation:
        fig, ax = plt.subplots(figsize=(10, 6))
        actual = data_dict["market"]["actual"]
        predicted = data_dict["market"]["predicted"]
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
        ax.set_title("Market Opportunity Rating Analysis")
        ax.set_xlabel("Actual Rating")
        ax.set_ylabel("Predicted Rating")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/market_rating_analysis.png")
        plt.close()

def save_evaluation_results(evaluation, output_dir):
    """Save model evaluation results and generate summary report
    
    Args:
        evaluation: Dictionary containing evaluation results
        output_dir: Directory for saving outputs
    """
    logger = setup_logging()
    
    # Save detailed metrics
    metrics_path = f"{output_dir}/model_evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(evaluation, f, indent=2)
    logger.info(f"Saved detailed metrics to {metrics_path}")
    
    # Generate summary report
    summary = {
        "business_success": {
            "accuracy": f"{evaluation['success_metrics']['accuracy']:.2%}",
            "precision": f"{evaluation['success_metrics']['precision']:.2%}",
            "recall": f"{evaluation['success_metrics']['recall']:.2%}"
        },
        "revenue_growth": {
            "r2_score": f"{evaluation['revenue_metrics']['r2_score']:.3f}",
            "rmse": f"{evaluation['revenue_metrics']['rmse']:.2f}",
            "mae": f"{evaluation['revenue_metrics']['mae']:.2f}"
        },
        "location_performance": {
            "r2_score": f"{evaluation['location_metrics']['r2_score']:.3f}",
            "rmse": f"{evaluation['location_metrics']['rmse']:.2f}",
            "mae": f"{evaluation['location_metrics']['mae']:.2f}"
        },
        "market_opportunity": {
            "r2_score": f"{evaluation['market_metrics']['r2_score']:.3f}",
            "rmse": f"{evaluation['market_metrics']['rmse']:.2f}",
            "mae": f"{evaluation['market_metrics']['mae']:.2f}"
        },
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = f"{output_dir}/model_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved evaluation summary to {summary_path}")
    return summary

def main():
    """Execute model evaluation pipeline"""
    base_dir = os.getenv("BASE_DIR", "/app/San_Francisco_Business_Model")
    input_dir = f"{base_dir}/models"
    output_dir = f"{base_dir}/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load and evaluate predictions
        prediction_data = load_model_predictions(input_dir)
        evaluation_results = evaluate_model_predictions(prediction_data)
        generate_evaluation_plots(prediction_data, evaluation_results, output_dir)
        summary = save_evaluation_results(evaluation_results, output_dir)
        
        # Display results
        print("\nModel Evaluation Complete")
        print("=" * 50)
        print("Business Success Prediction:")
        print(f"  - Accuracy: {summary['business_success']['accuracy']}")
        print(f"  - Precision: {summary['business_success']['precision']}")
        print(f"  - Recall: {summary['business_success']['recall']}")
        print("\nRevenue Growth Forecast:")
        print(f"  - R² Score: {summary['revenue_growth']['r2_score']}")
        print(f"  - RMSE: {summary['revenue_growth']['rmse']}")
        print(f"  - MAE: {summary['revenue_growth']['mae']}")
        print("\nLocation Performance Score:")
        print(f"  - R² Score: {summary['location_performance']['r2_score']}")
        print(f"  - RMSE: {summary['location_performance']['rmse']}")
        print(f"  - MAE: {summary['location_performance']['mae']}")
        print("\nMarket Opportunity Rating:")
        print(f"  - R² Score: {summary['market_opportunity']['r2_score']}")
        print(f"  - RMSE: {summary['market_opportunity']['rmse']}")
        print(f"  - MAE: {summary['market_opportunity']['mae']}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        raise

if __name__ == "__main__":
    main() 