FROM python:3.11-slim

WORKDIR /app

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

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

# Create the directory structure that your scripts expect
ENV BASE_DIR=/app/San_Francisco_Business_Model
RUN mkdir -p "$BASE_DIR/raw_data" \
    "$BASE_DIR/processed" \
    "$BASE_DIR/models" \
    "$BASE_DIR/archive"

# Create subdirectories for all your data sources
RUN mkdir -p "$BASE_DIR/raw_data/"{sf_business,economic,demographic,planning,crime,sf311,mobility,yelp,news,historical,final} \
    "$BASE_DIR/processed/"{sf_business,economic,demographic,planning,crime,sf311,mobility,yelp,news,historical,final}

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for Streamlit
EXPOSE 8501


# Set environment variables