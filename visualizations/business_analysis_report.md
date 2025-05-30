
# San Francisco Business Survival Model - Analysis Report

**Date**: May 26, 2025  
**Model Version**: Business Survival Deep Learning Model  
**Analysis Type**: Performance Analysis & SHAP Explanations

---

## Executive Summary

Our deep learning model successfully predicts San Francisco business survival with **84.2% accuracy**. The model analyzes multiple business factors to provide actionable insights for entrepreneurs and stakeholders.

### Key Performance Metrics
- **Accuracy**: 84.2%
- **Precision**: 93.1%
- **Recall**: 81.4%
- **F1 Score**: 86.9%
- **ROC-AUC**: 93.9%

---

## Model Performance Analysis

### Overall Assessment
The model demonstrates **strong predictive capability** with balanced performance across all metrics. The ROC-AUC score of 93.9% indicates excellent discrimination between successful and failed businesses.

### Key Findings
1. **Balanced Predictions**: The model maintains consistent performance across both successful and failed business predictions
2. **High Confidence**: 44.6% of predictions show high confidence (>80%)
3. **Risk Distribution**: Clear separation between high, medium, and low-risk businesses

---

## Feature Importance & Business Drivers

### Top Success Factors
Based on our analysis, the most important factors for business survival are:

1. **Business Type & Industry** - Different industries show varying survival rates
2. **Location Quality** - Geographic factors significantly impact success
3. **Start Timing** - Market timing and seasonal factors matter
4. **Regulatory Environment** - Compliance and permitting requirements
5. **Economic Conditions** - Market conditions at business inception

### SHAP Analysis Insights
Our explainable AI analysis reveals:
- **Tech startups** benefit from business type (+15% success boost)
- **Restaurants** face location dependency (+12% for good locations)
- **Retail businesses** are highly sensitive to competition levels
- **Professional services** show consistent performance across factors

---

## Risk Assessment & Recommendations

### Business Risk Categories

#### High Risk (Prediction < 30%)
- **Count**: 16,396 businesses (29.5%)
- **Recommendation**: Intensive support, business plan review, market analysis
- **Success Rate**: Lower than average, requires intervention

#### Medium Risk (30-70%)
- **Count**: 13,185 businesses (23.7%)
- **Recommendation**: Regular monitoring, targeted assistance programs
- **Success Rate**: Variable, opportunity for improvement

#### Low Risk (Prediction > 70%)
- **Count**: 26,051 businesses (46.8%)
- **Recommendation**: Standard support, focus on growth opportunities
- **Success Rate**: High probability of success

---

## Actionable Business Insights

### For Entrepreneurs
1. **Choose Your Industry Wisely**: Some business types have inherently higher success rates
2. **Location Matters**: Invest time in location analysis, especially for retail and restaurants
3. **Timing is Critical**: Consider market conditions and seasonal factors when starting
4. **Regulatory Preparation**: Ensure all permits and compliance requirements are met

### For Investors & Lenders
1. **Risk Assessment**: Use model predictions to inform investment decisions
2. **Portfolio Management**: Balance high and low-risk investments
3. **Due Diligence**: Focus on factors identified as most important by the model

### For Policymakers
1. **Support Programs**: Target high-risk businesses with intensive support
2. **Regulatory Efficiency**: Streamline permitting processes
3. **Economic Development**: Focus on creating favorable conditions for business success

---

## Model Limitations & Considerations

### Current Limitations
- Model trained on historical data (may not capture recent market changes)
- Limited to San Francisco market conditions
- Requires regular retraining to maintain accuracy

### Recommended Actions
1. **Regular Model Updates**: Retrain quarterly with new data
2. **Performance Monitoring**: Track prediction accuracy over time
3. **Bias Monitoring**: Ensure fair treatment across all business types
4. **Human Oversight**: Use model predictions as decision support, not replacement

---

## Technical Specifications

### Model Architecture
- **Type**: Deep Neural Network with Multi-Modal Fusion
- **Features**: 118 input features across spatial, temporal, and business domains
- **Training Data**: 180,802 businesses
- **Validation**: 41,724 businesses  
- **Test Set**: 55,632 businesses

### Performance Validation
- **Cross-Validation**: Robust train/validation/test split
- **Calibration**: Model probabilities well-calibrated
- **Stability**: Consistent performance across different data splits

---

## Visualization Index

All interactive charts and visualizations are saved in the `/visualizations` directory:

1. **Performance Dashboard** (`performance_dashboard.html`)
   - Overall metrics, ROC curve, precision-recall curve
   - Confusion matrix, prediction distributions, calibration plot

2. **Feature Importance** (`feature_importance.html`)
   - Top 20 most important features
   - Feature groups and categories

3. **SHAP Analysis** (`shap_analysis.html`, `shap_summary.html`)
   - Model explainability for different business types
   - Feature contribution analysis

4. **Business Insights** (`business_insights.html`)
   - Risk distribution and recommendations
   - Confidence analysis and actionable insights

---

## Contact & Support

For questions about this analysis or the underlying model:
- **Technical Issues**: Contact the Data Science Team
- **Business Questions**: Reach out to Business Analytics
- **Model Updates**: Submit requests through the standard process

*This report was automatically generated from the San Francisco Business Survival Model analysis pipeline.*
