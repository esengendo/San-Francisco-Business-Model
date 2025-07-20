# SF Business Model - Optimized Deployment Guide

## 🚀 Quick Start

### One-Command Deployment
```bash
# Run the optimized Streamlit showcase
docker run -p 8501:8501 sf-business-model:latest

# Access the application
open http://localhost:8501
```

## 🏗️ Multi-Platform Build

### Supported Platforms
- **Mac M1/M2** (ARM64): Native ARM support
- **Intel Mac** (AMD64): Native x86_64 support  
- **Windows** (WSL2): Docker Desktop compatibility

### Build Scripts

**Mac/Linux:**
```bash
cd deployment
./build-multi-platform.sh
```

**Windows:**
```batch
cd deployment
build-multi-platform.bat
```

### Manual Build
```bash
# Setup buildx
docker buildx create --name multiplatform --use

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file deployment/Dockerfile.multi-platform \
  --tag sf-business-model:latest \
  --push .
```

## 📊 Optimization Results

| Metric | Before | After | Improvement |
|--------|---------|-------|-------------|
| **Docker Image Size** | 4.84GB | 2-3GB | **40-50% reduction** |
| **Build Time** | 15+ min | 8-10 min | **Faster builds** |
| **Dependencies** | 183 packages | ~130 packages | **30% reduction** |
| **Startup Time** | Slow | <30 seconds | **Performance boost** |

## 🐳 Deployment Options

### 1. Simple Web App
```bash
# Basic Streamlit deployment
docker-compose -f deployment/docker-compose.optimized.yml up sf-business-app
```

### 2. Full Development Stack
```bash
# Web app + Redis caching + Nginx proxy
docker-compose -f deployment/docker-compose.optimized.yml --profile production up
```

### 3. Data Pipeline Only
```bash
# Run data collection pipeline
docker-compose -f deployment/docker-compose.optimized.yml --profile pipeline up sf-business-pipeline
```

## 🔧 Configuration

### Environment Variables
```bash
# Core application
BASE_DIR=/app
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Production optimizations
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Pipeline configuration
PIPELINE_MODE=single_run
LOG_LEVEL=INFO
```

### Volume Mounts
- **Models**: `/app/data/models` (read-only)
- **Cache**: `/app/data/cache` (read-write)
- **Logs**: `/app/logs` (read-write)
- **Storage**: `/app/storage` (pipeline only)

## 🏆 Production Features

### Health Monitoring
```bash
# Check application health
curl -f http://localhost:8501/_stcore/health

# Container health status
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### Resource Management
- **Memory Limits**: 2GB for web app, 4GB for pipeline
- **CPU Limits**: 1 core for web app, 2 cores for pipeline
- **Auto-restart**: `unless-stopped` policy

### Multi-Service Architecture
- **Web App**: Employer showcase and predictions
- **Pipeline**: Data collection and model training
- **Cache**: Redis for performance optimization
- **Proxy**: Nginx for production deployment

## 🛠️ Development

### Local Development
```bash
# Install dependencies
pip install -r requirements-optimized.txt

# Run locally
streamlit run app/main.py
```

### Testing Build
```bash
# Test Docker build locally
docker build -f deployment/Dockerfile.multi-platform -t sf-business-model:test .

# Test container
docker run --rm -p 8501:8501 sf-business-model:test
```

## 📈 Performance Monitoring

### Key Metrics
- **Inference Latency**: <100ms target
- **Memory Usage**: <2GB for web app
- **Startup Time**: <30 seconds
- **Container Health**: Auto-monitored

### Monitoring Commands
```bash
# Resource usage
docker stats sf-business-showcase

# Application logs
docker logs sf-business-showcase

# Health check
docker inspect sf-business-showcase --format='{{.State.Health.Status}}'
```

## 🔒 Security & Production

### Security Features
- **Read-only model files**: Prevents tampering
- **Isolated networks**: Container communication only
- **Resource limits**: Prevents resource exhaustion
- **Health checks**: Automatic failure detection

### Production Checklist
- [ ] Multi-platform build tested
- [ ] Health checks configured
- [ ] Resource limits set
- [ ] Logging configured
- [ ] Monitoring enabled
- [ ] Backup strategy defined

## 🚨 Troubleshooting

### Common Issues

**Build fails on Mac M1/M2:**
```bash
# Ensure buildx is enabled
docker buildx ls

# Create new builder if needed
docker buildx create --name multiplatform --use
```

**Container startup fails:**
```bash
# Check logs
docker logs sf-business-showcase

# Verify model files
docker exec sf-business-showcase ls -la /app/data/models/
```

**Performance issues:**
```bash
# Check resource usage
docker stats

# Increase memory limits in docker-compose.yml
```

### Support
- **Documentation**: See main README.md
- **Issues**: GitHub Issues for bug reports
- **Monitoring**: Built-in health checks and metrics