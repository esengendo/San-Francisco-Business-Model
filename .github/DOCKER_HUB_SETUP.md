# Docker Hub CI/CD Setup Guide

## Prerequisites

### 1. Docker Hub Account Setup
- Ensure you have a Docker Hub account: https://hub.docker.com
- Repository: `esengendo730/sf-business-model` should exist
- Generate access token for CI/CD:
  1. Go to Docker Hub → Account Settings → Security
  2. Create "New Access Token" with name: "GitHub-Actions-SF-Business"
  3. Copy the token (you won't see it again)

### 2. GitHub Secrets Configuration
Add these secrets to your GitHub repository:

**Repository Settings → Secrets and Variables → Actions → New repository secret**

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `DOCKER_USERNAME` | `esengendo730` | Your Docker Hub username |
| `DOCKER_PASSWORD` | `{your-access-token}` | Docker Hub access token (NOT your password) |

### 3. Verify Repository Access
```bash
# Test Docker Hub repository access
curl -s "https://hub.docker.com/v2/repositories/esengendo730/sf-business-model/"

# Should return repository information
```

## Automated Workflows

### Main CI/CD Pipeline
**File**: `.github/workflows/ci-cd.yml`

**Triggers**:
- Push to `main` branch
- Pull requests to `main`
- Monthly schedule (first Sunday at 2 AM UTC)
- Manual dispatch

**Jobs**:
1. **Test**: Run pytest unit tests with coverage
2. **Build & Deploy**: Multi-platform Docker build and push
3. **Security Scan**: Trivy vulnerability scanning
4. **Health Check**: Container functionality verification

### Monthly Maintenance
**File**: `.github/workflows/monthly-maintenance.yml`

**Schedule**: First Sunday of each month at 6 AM UTC

**Jobs**:
1. **Dependency Update**: Check and create PR for package updates
2. **Security Audit**: Safety and bandit security scans
3. **Performance Test**: Container startup and response time testing
4. **Cleanup**: Resource management and maintenance summary

## Docker Image Tags

### Automatic Tagging Strategy
- `latest`: Latest stable build from main branch
- `main-{sha}`: Specific commit builds
- `{YYYYMMDD}`: Monthly scheduled builds
- `pr-{number}`: Pull request builds

### Image Information
- **Base**: Multi-stage optimized build
- **Size**: ~2-3GB (optimized from 4.84GB)
- **Platforms**: linux/amd64, linux/arm64
- **Health Check**: Built-in Streamlit health endpoint

## Usage Examples

### Pull Latest Image
```bash
# AMD64 (Intel/Windows)
docker pull esengendo730/sf-business-model:latest

# ARM64 (Mac M1/M2)
docker pull --platform linux/arm64 esengendo730/sf-business-model:latest
```

### Run Container
```bash
# Standard deployment
docker run -p 8501:8501 esengendo730/sf-business-model:latest

# With resource limits
docker run -p 8501:8501 --memory=2g --cpus=1 esengendo730/sf-business-model:latest

# Background deployment
docker run -d -p 8501:8501 --name sf-business esengendo730/sf-business-model:latest
```

### Health Monitoring
```bash
# Check container health
curl http://localhost:8501/_stcore/health

# View logs
docker logs sf-business

# Container stats
docker stats sf-business
```

## Troubleshooting

### Common Issues

1. **Docker Hub Authentication Failed**
   - Verify `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets
   - Ensure access token has write permissions
   - Check token hasn't expired

2. **Build Platform Issues**
   - GitHub Actions supports multi-platform builds
   - Use `docker/setup-buildx-action@v3` for advanced builds

3. **Image Size Concerns**
   - Current optimized size: ~2-3GB
   - Uses multi-stage builds and .dockerignore
   - Excludes data directories and model files

4. **Security Scan Failures**
   - Trivy scans for vulnerabilities
   - Update base images if critical issues found
   - Review security reports in GitHub Security tab

### Manual Deployment
```bash
# Build locally
docker build -f deployment/Dockerfile.multi-platform -t sf-business-local .

# Test locally
docker run -p 8501:8501 sf-business-local

# Tag for Docker Hub
docker tag sf-business-local esengendo730/sf-business-model:manual

# Push manually (requires authentication)
docker push esengendo730/sf-business-model:manual
```

## Monitoring & Alerts

### GitHub Actions Monitoring
- View workflow status: Repository → Actions tab
- Email notifications for failed builds
- Security alerts in Security tab

### Docker Hub Monitoring
- Image pull statistics
- Vulnerability scanning results
- Automated builds status

## Security Best Practices

1. **Never commit Docker Hub passwords**
2. **Use access tokens instead of passwords**
3. **Regularly rotate access tokens**
4. **Monitor security scan results**
5. **Keep base images updated**
6. **Review dependency updates monthly**

## Support

For deployment issues:
1. Check GitHub Actions logs
2. Review Docker Hub build status
3. Verify secret configuration
4. Test local Docker builds first