# Optimized Docker image for SF Business Model
FROM python:3.11-slim

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

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force reinstall numpy<2 to avoid compatibility issues
RUN pip install "numpy<2" --force-reinstall

# Copy source code and required files
COPY src/ ./src/
COPY app/ ./app/
COPY land_use_fallback.parquet ./

# Create streamlined directory structure
RUN mkdir -p data/{raw,processed,models} \
    logs results archive \
    San_Francisco_Business_Model/{raw_data,processed,models,archive}

# Create compatibility symlinks for legacy paths
RUN ln -s /app/data/raw /app/San_Francisco_Business_Model/raw_data && \
    ln -s /app/data/processed /app/San_Francisco_Business_Model/processed && \
    ln -s /app/data/models /app/San_Francisco_Business_Model/models

# Set environment variables
ENV PYTHONPATH=/app
ENV BASE_DIR=/app

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port for Streamlit
EXPOSE 8501

# Default command - Streamlit app (can be overridden for pipeline)
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]