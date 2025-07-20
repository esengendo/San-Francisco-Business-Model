# San Francisco Business Success Prediction Platform

[![CI/CD Pipeline](https://github.com/esengendo/San-Francisco-Business-Model/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/esengendo/San-Francisco-Business-Model/actions)
[![Docker Image](https://img.shields.io/docker/pulls/esengendo730/sanfrancisco_business_model.svg)](https://hub.docker.com/r/esengendo730/sanfrancisco_business_model)
[![Docker Image Size](https://img.shields.io/docker/image-size/esengendo730/sanfrancisco_business_model/latest)](https://hub.docker.com/r/esengendo730/sanfrancisco_business_model)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Platform](https://img.shields.io/badge/platform-linux%2Famd64%20%7C%20linux%2Farm64-lightgrey)](https://hub.docker.com/r/esengendo730/sanfrancisco_business_model)

**A production-ready deep learning solution for predicting 5-year business survival probability in San Francisco using multi-source data integration and PyTorch neural networks.**

## 🚀 **Quick Demo**

### Docker Deployment (Recommended)
```bash
# Quick start - latest stable version
docker run -p 8501:8501 esengendo730/sf-business-model:latest

# Multi-platform support (AMD64 + ARM64)
docker run --platform linux/amd64 -p 8501:8501 esengendo730/sf-business-model:latest
```

### Local Development
```bash
# Clone and setup
git clone https://github.com/esengendo/San-Francisco-Business-Model.git
cd San-Francisco-Business-Model
pip install -r requirements.txt

# Launch business intelligence dashboard
streamlit run app/business_dashboard.py
```

**🌐 Access**: http://localhost:8501

---

## **Business Problem & Solution**

**Challenge**: San Francisco entrepreneurs lack data-driven insights to assess business viability across different locations, timeframes, and economic conditions.

**Solution**: A comprehensive ML platform that analyzes **118 engineered features** from **13 API data sources** to predict business success probability, processing **250K+ business records** with actionable insights for strategic decision-making.

**Key Benefits**:
- **Risk Assessment**: Quantify location-specific business risks
- **Strategic Planning**: Optimize timing and location decisions  
- **Market Intelligence**: Understand economic trend impacts on survival rates

---

## **Technical Architecture**

### **Machine Learning Model**
- **Framework**: PyTorch Lightning with custom multi-branch neural architecture
- **Features**: 118 engineered features (spatial, temporal, business, economic)
- **Target**: Binary classification for 5-year business survival
- **Performance**: Production-optimized with robust error handling

### **Data Engineering Pipeline**
**13 Integrated Data Sources**:
- San Francisco Open Data (business registrations)
- FRED Economic API (GDP, unemployment, housing)
- Census Demographics & SF Planning Department
- Crime Data, SF 311 Services, OpenStreetMap
- GDELT News & RSS Feeds for sentiment analysis
- Historical data validation via Wayback Machine

### **Key Features**
- **Real-time Predictions**: Interactive business parameter input
- **Geospatial Visualization**: Location-based risk factor mapping
- **Model Interpretability**: Feature importance and confidence intervals
- **Trend Analysis**: Economic indicator impact assessment

---

## **Technology Stack**

**Core Technologies**:
- **ML**: PyTorch Lightning, Scikit-learn, NumPy/Pandas
- **Web App**: Streamlit with Plotly visualizations  
- **Data**: 13 REST APIs, Parquet storage format
- **Deployment**: Docker containerization with CI/CD

**Production Features**:
- Containerized deployment with health checks
- Automated testing and model validation
- Error handling and graceful degradation
- Model versioning and performance tracking

---

## **Key Insights**

The model reveals that SF business success is driven by:
1. **Location Intelligence**: Specific neighborhoods show 2-3x higher success rates
2. **Economic Timing**: Businesses launched during upturns have 40% higher survival
3. **Industry Clustering**: Co-location with complementary businesses increases success
4. **Infrastructure Quality**: Areas with better city services correlate with success

---

## **Getting Started**

### **Quick Deployment**
```bash
# Run the containerized application
docker run -p 8501:8501 esengendo730/sf-business-model:latest

# Access the interactive dashboard
open http://localhost:8501
```

### **Development Setup**
```bash
# Clone and setup
git clone https://github.com/esengendo/sf-business-model.git
cd sf-business-model
pip install -r requirements.txt

# Launch application
streamlit run app/app.py

# Run full data pipeline (optional)
python src/pipeline_runner.py
```

---

## 🚀 **CI/CD & Deployment**

### Automated Deployments
- **🔄 Continuous Integration**: Automated testing on every push
- **🐳 Docker Hub**: Multi-platform images (AMD64 + ARM64) 
- **📅 Monthly Maintenance**: Dependency updates and security scans (First Sunday of each month)
- **🔒 Security Scanning**: Vulnerability assessment with Trivy
- **⚡ Health Monitoring**: Automated endpoint testing

### Docker Hub Repository
- **Image**: [`esengendo730/sf-business-model`](https://hub.docker.com/r/esengendo730/sf-business-model)
- **Tags**: `latest`, `main-{sha}`, `{YYYYMMDD}` (monthly builds)
- **Platforms**: `linux/amd64`, `linux/arm64`
- **Automated Builds**: Triggered on push to main branch

### CI/CD Workflows
```bash
# Manual deployment trigger
gh workflow run "CI/CD Pipeline"

# Monthly maintenance (auto-scheduled)
gh workflow run "Monthly Maintenance"
```

---

## **Project Structure**
```
sf-business-model/
├── src/
│   ├── data_collection/     # 12 API integration scripts
│   ├── processing/          # ETL and feature engineering
│   └── utils/               # Configuration and helpers
├── app/                     # Streamlit web application
├── notebooks/               # Jupyter analysis notebooks
├── models/                  # Trained model artifacts
└── visualizations/          # Generated charts and reports
```

---

## **Business Applications**

**For Entrepreneurs**: Location optimization, timing analysis, risk mitigation
**For Investors**: Quantitative due diligence, portfolio risk assessment  
**For Policymakers**: Economic development planning, resource allocation

---

**Status**: ✅ Production-ready with automated CI/CD, Docker deployment, and comprehensive testing

---

## 🧪 **Testing & Quality Assurance**

**Automated Testing**: Complete test suite with 34+ unit tests covering pipeline components and dashboard functionality
**CI/CD Pipeline**: Automated testing, building, and deployment on every push to main branch
**Security Scanning**: Trivy vulnerability scanning and GitHub secret detection
**Health Monitoring**: Automated endpoint testing and container health checks 