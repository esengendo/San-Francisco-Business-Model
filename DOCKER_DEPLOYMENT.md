# 🚀 Docker Hub Deployment Guide

## 📋 Overview

The SF Business Model app is automatically built and deployed to Docker Hub using GitHub Actions.

**Docker Hub Repository:** [`esengendo730/sf-business-model`](https://hub.docker.com/r/esengendo730/sf-business-model)

## 🔧 Required GitHub Secrets

Add these secrets to your GitHub repository (`Settings` → `Secrets and variables` → `Actions`):

- `DOCKERHUB_USERNAME`: `esengendo730`
- `DOCKERHUB_TOKEN`: Your Docker Hub access token

## 🐳 Running the Application

### Pull and Run Latest Version
```bash
docker pull esengendo730/sf-business-model:latest
docker run -p 8501:8501 esengendo730/sf-business-model:latest
```

### Access the App
- **Local:** http://localhost:8501
- **Network:** http://YOUR_IP:8501

## ⚡ Quick Start Commands

```bash
# Pull latest image
docker pull esengendo730/sf-business-model:latest

# Run with volume mounts (if you have local data)
docker run -p 8501:8501 \
  -v $(pwd)/processed:/app/San_Francisco_Business_Model/processed \
  -v $(pwd)/models:/app/San_Francisco_Business_Model/models \
  esengendo730/sf-business-model:latest

# Run in background
docker run -d -p 8501:8501 esengendo730/sf-business-model:latest

# View logs
docker logs $(docker ps -q --filter ancestor=esengendo730/sf-business-model:latest)
```

## 🔄 CI/CD Pipeline

The GitHub Actions workflow:
1. **Tests** - Validates Python dependencies
2. **Builds** - Creates multi-platform Docker image (amd64/arm64)  
3. **Pushes** - Uploads to Docker Hub on `main` branch
4. **Tags** - Auto-tags with branch name, latest, and commit SHA

## 🏷️ Available Tags

- `latest` - Latest stable version from main branch
- `main-<sha>` - Specific commit from main branch
- `develop` - Latest development version
- `v*` - Semantic version releases

## 🌟 Features

- ✅ **Multi-platform** support (Intel + ARM)
- ✅ **Automated builds** on every push
- ✅ **Health checks** included
- ✅ **Optimized caching** for faster builds
- ✅ **Production ready** configuration

## 🛠️ Development

For local development:
```bash
# Build locally
docker build -t sf-business-model .

# Run locally built image
docker run -p 8501:8501 sf-business-model
``` 