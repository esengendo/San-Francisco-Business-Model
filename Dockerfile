# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    git \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force reinstall numpy<2 to avoid compatibility issues
RUN pip install "numpy<2" --force-reinstall

# Copy Python files
COPY *.py ./

# Create main directory structure
RUN mkdir -p "/app/San_Francisco_Business_Model/raw_data" \
    "/app/San_Francisco_Business_Model/processed" \
    "/app/San_Francisco_Business_Model/models" \
    "/app/San_Francisco_Business_Model/archive"

# Create all raw_data subdirectories explicitly
RUN mkdir -p \
    "/app/San_Francisco_Business_Model/raw_data/sf_business" \
    "/app/San_Francisco_Business_Model/raw_data/economic" \
    "/app/San_Francisco_Business_Model/raw_data/demographic" \
    "/app/San_Francisco_Business_Model/raw_data/planning" \
    "/app/San_Francisco_Business_Model/raw_data/crime" \
    "/app/San_Francisco_Business_Model/raw_data/sf311" \
    "/app/San_Francisco_Business_Model/raw_data/mobility" \
    "/app/San_Francisco_Business_Model/raw_data/yelp" \
    "/app/San_Francisco_Business_Model/raw_data/news" \
    "/app/San_Francisco_Business_Model/raw_data/historical" \
    "/app/San_Francisco_Business_Model/raw_data/final"

# Create all processed subdirectories explicitly  
RUN mkdir -p \
    "/app/San_Francisco_Business_Model/processed/sf_business" \
    "/app/San_Francisco_Business_Model/processed/economic" \
    "/app/San_Francisco_Business_Model/processed/demographic" \
    "/app/San_Francisco_Business_Model/processed/planning" \
    "/app/San_Francisco_Business_Model/processed/crime" \
    "/app/San_Francisco_Business_Model/processed/sf311" \
    "/app/San_Francisco_Business_Model/processed/mobility" \
    "/app/San_Francisco_Business_Model/processed/yelp" \
    "/app/San_Francisco_Business_Model/processed/news" \
    "/app/San_Francisco_Business_Model/processed/historical" \
    "/app/San_Francisco_Business_Model/processed/final"

# Set default command to run the pipeline
CMD ["python", "pipeline_runner.py"]