@echo off
REM Multi-platform Docker build script for Windows
REM Supports: Mac M1/M2 (ARM64), Intel Mac (AMD64), Windows (WSL2)

echo 🚀 Building SF Business Model - Multi-Platform Docker Image
echo ==================================================

REM Check if docker buildx is available
docker buildx version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: docker buildx is not available
    echo Please install Docker Buildx or use Docker Desktop
    exit /b 1
)

REM Create and use multi-platform builder
echo 🔧 Setting up multi-platform builder...
docker buildx create --name multiplatform --use 2>nul || docker buildx use multiplatform

REM Ensure builder is running
docker buildx inspect --bootstrap

echo 📦 Building for multiple platforms...
echo Platforms: linux/amd64 (Intel/Windows), linux/arm64 (Mac M1/M2)

REM Build multi-platform image
docker buildx build ^
    --platform linux/amd64,linux/arm64 ^
    --file deployment/Dockerfile.multi-platform ^
    --tag sf-business-model:latest ^
    --tag sf-business-model:v2.0 ^
    --push ^
    .

if errorlevel 1 (
    echo ❌ Build failed
    exit /b 1
)

echo ✅ Multi-platform build completed successfully!
echo.
echo 📊 Build Summary:
echo   • Platforms: linux/amd64, linux/arm64
echo   • Tags: sf-business-model:latest, sf-business-model:v2.0
echo   • Expected size: 2-3GB (down from 4.84GB)
echo.
echo 🚀 To run the application:
echo   docker run -p 8501:8501 sf-business-model:latest
echo.
echo 🌐 Access at: http://localhost:8501