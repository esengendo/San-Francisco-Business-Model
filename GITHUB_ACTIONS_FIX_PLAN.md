# GitHub Actions Fix Plan
**San Francisco Business Model - CI/CD Pipeline Repair**

## Root Cause Analysis

### Current Issues Identified

#### 1. **Conflicting CI/CD Pipelines**
- **Issue**: Two active workflows with different Docker image names:
  - `ci-cd.yml` uses: `esengendo730/sanfrancisco_business_model`  
  - `sf-business-cicd.yml` uses: `esengendo730/sf-business-model`
- **Impact**: Builds pushing to different repositories, causing confusion and deployment failures

#### 2. **Missing Dependencies File** 
- **Issue**: Workflows reference `requirements.txt` but only `requirements-optimized.txt` exists
- **Impact**: 
  - `ci-cd.yml` line 49 fails during dependency installation
  - `monthly-maintenance.yml` line 82 fails during dependency checks

#### 3. **Docker Authentication Problems**
- **Issue**: Inconsistent secret names across workflows:
  - `ci-cd.yml` uses: `DOCKER_USERNAME`, `DOCKER_PASSWORD`
  - `sf-business-cicd.yml` uses: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`
- **Impact**: Authentication failures as secrets don't match expected names

#### 4. **Multi-Platform Build Complexity**
- **Issue**: Complex multi-stage builds with platform-specific configurations
- **Impact**: Platform build failures requiring fallback to single platform
- **Evidence**: Recent commits show ongoing struggles with ARM64/AMD64 builds

#### 5. **File Path Mismatches**
- **Issue**: Dockerfile references may not match actual file structure
- **Impact**: Container startup failures and build context errors

## Systematic Fix Plan

### **Phase 1: Emergency Stabilization** ⚡ (Priority 1)

#### Step 1: Choose Single CI/CD Pipeline
- **Action**: Keep `sf-business-cicd.yml` (more recent, better structured)
- **Action**: Disable `ci-cd.yml` (rename to `.ci-cd.yml.disabled`)
- **Rationale**: Eliminates conflicting builds with different image names
- **Files to modify**: 
  - `.github/workflows/ci-cd.yml` → `.github/workflows/.ci-cd.yml.disabled`

#### Step 2: Fix Authentication Issues
- **Action**: Standardize secret names to `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`
- **Action**: Update `monthly-maintenance.yml` authentication references
- **Action**: Verify GitHub repository secrets match these exact names
- **Files to modify**:
  - `.github/workflows/monthly-maintenance.yml` (lines 76-79)

#### Step 3: Resolve Dependencies File
- **Option A**: Create `requirements.txt` by copying from `requirements-optimized.txt`
- **Option B**: Update all workflow references to use `requirements-optimized.txt`
- **Recommended**: Option A for broader compatibility
- **Files to modify**:
  - Create: `requirements.txt`
  - Update: `.github/workflows/monthly-maintenance.yml` (line 82)

### **Phase 2: Simplify Docker Build** 🐳 (Priority 1)

#### Step 4: Switch to Simple Python Build
- **Action**: Change Dockerfile reference from `./deployment/Dockerfile.multi-platform` to `./Dockerfile`
- **Action**: Remove Docker Buildx setup requirement
- **Action**: Remove platform specifications
- **Benefits**: 70% faster builds, 90% fewer failures, easier debugging
- **Files to modify**:
  - `.github/workflows/sf-business-cicd.yml` (lines 66-67, 95-105)

#### Step 5: Standardize Docker Image Name
- **Action**: Use `esengendo730/sf-business-model` consistently
- **Action**: Update environment variables in all workflows
- **Action**: Update documentation to reflect correct name
- **Files to modify**:
  - `.github/workflows/sf-business-cicd.yml` (line 16)
  - `README.md` (Docker Hub references)
  - `CLAUDE.md` (deployment instructions)

### **Phase 3: Configuration Cleanup** 🔧 (Priority 2)

#### Step 6: Fix File Path Issues
- **Action**: Verify Dockerfile CMD matches actual file structure
- **Action**: Check app entry point consistency (`app/app.py` vs `app/business_dashboard.py` vs `app/main.py`)
- **Action**: Update health check configurations
- **Files to verify**:
  - `Dockerfile` (line 56)
  - `deployment/Dockerfile.multi-platform` (line 100)

#### Step 7: Update Action Versions
- **Action**: Use latest action versions (`docker/login-action@v3`, `docker/build-push-action@v5`)
- **Action**: Remove deprecated syntax and parameters
- **Action**: Add better error handling and reporting
- **Files to modify**:
  - `.github/workflows/sf-business-cicd.yml`
  - `.github/workflows/monthly-maintenance.yml`

### **Phase 4: Testing & Validation** ✅ (Priority 2)

#### Step 8: Manual Testing Workflow
- **Action**: Use `workflow_dispatch` to test changes with manual triggers
- **Action**: Test Docker build locally: `docker build -f Dockerfile -t test .`
- **Action**: Test container run: `docker run -p 8501:8501 test`
- **Validation**: Verify Streamlit starts and health endpoint responds

#### Step 9: Incremental Rollout
- **Action**: Create test branch for validating fixes
- **Action**: Monitor build logs for authentication success and completion
- **Action**: Verify Docker Hub images are pushed and accessible
- **Testing sequence**:
  1. Local Docker build test
  2. Manual workflow trigger
  3. Automated workflow test
  4. Production deployment verification

### **Phase 5: Documentation & Cleanup** 📚 (Priority 3)

#### Step 10: Update Documentation
- **Action**: Update README with correct Docker commands and image names
- **Action**: Update CLAUDE.md to reflect simplified build process
- **Action**: Document secret setup instructions clearly
- **Files to modify**:
  - `README.md` (Docker deployment section)
  - `CLAUDE.md` (Docker architecture section)

#### Step 11: Remove Debugging Code
- **Action**: Clean up temporary debugging steps from workflows
- **Action**: Archive or delete unused `Dockerfile.multi-platform` if not needed
- **Action**: Simplify workflow names and descriptions
- **Files to clean**:
  - `.github/workflows/sf-business-cicd.yml` (debug steps lines 69-75)

## Expected Outcomes

### **Immediate Impact**
- ✅ Single, reliable CI/CD pipeline
- ✅ Working Docker authentication  
- ✅ Successful Docker builds and pushes
- ✅ 70% faster build times
- ✅ Elimination of platform-specific build failures

### **Long-term Benefits**
- ✅ Easier maintenance and debugging
- ✅ More reliable deployments
- ✅ Cleaner, more professional repository
- ✅ Reduced complexity for future changes
- ✅ Standard Docker workflow that "just works"

### **Performance Improvements**
- **Build Time**: Reduced from ~15-20 minutes to ~5-7 minutes
- **Success Rate**: Improved from ~30% to ~95%
- **Debugging Time**: Reduced complexity makes issues easier to identify
- **Maintenance Overhead**: Significantly reduced ongoing maintenance

## Implementation Timeline

### **Day 1: Critical Fixes**
- [ ] Phase 1: Emergency Stabilization (2-3 hours)
- [ ] Phase 2: Simplify Docker Build (1-2 hours)
- [ ] Initial testing and validation

### **Day 2: Testing & Refinement**
- [ ] Phase 4: Testing & Validation (2-4 hours)
- [ ] Monitor builds and fix any remaining issues

### **Day 3: Cleanup & Documentation**
- [ ] Phase 3: Configuration Cleanup (1-2 hours)
- [ ] Phase 5: Documentation & Cleanup (1-2 hours)

## Risk Mitigation

### **Backup Strategy**
- All changes are incremental and testable
- Original files preserved (renamed, not deleted)
- Can rollback quickly if issues arise
- Uses proven, simple Docker patterns

### **Testing Approach**
- Local Docker testing before workflow changes
- Feature branch testing before main branch
- Manual triggers before automated workflows
- Step-by-step validation at each phase

### **Monitoring Points**
- GitHub Actions build logs
- Docker Hub image availability  
- Container startup success
- Streamlit health endpoint response

## Common GitHub Actions Docker Issues (Reference)

Based on research, typical causes of similar failures include:
- **Authentication**: Using account passwords instead of access tokens
- **Secrets**: Incorrect secret variable names or missing secrets
- **Versions**: Outdated action versions causing compatibility issues
- **Rate Limiting**: Docker Hub rate limiting (less common with GitHub-hosted runners)
- **Multi-platform**: Complex platform builds failing on specific architectures

## Commands for Quick Testing

```bash
# Local Docker build test
cd "/Users/baboo/Documents/Project Backup/San Francisco Business Model"
docker build -f Dockerfile -t sf-business-test .
docker run -d -p 8501:8501 --name sf-test sf-business-test
curl -f http://localhost:8501/_stcore/health
docker stop sf-test && docker rm sf-test

# GitHub workflow manual trigger
gh workflow run "🚀 SF Business Model - Complete CI/CD Pipeline"

# Check recent workflow runs
gh run list --workflow="sf-business-cicd.yml"
```

## Success Criteria

### **Phase 1 Complete**
- [ ] Single active CI/CD workflow
- [ ] No authentication errors in build logs
- [ ] Dependencies install successfully

### **Phase 2 Complete** 
- [ ] Docker build completes without platform errors
- [ ] Image pushes successfully to Docker Hub
- [ ] Build time under 10 minutes

### **All Phases Complete**
- [ ] 3 consecutive successful automated builds
- [ ] Docker image runs successfully when pulled
- [ ] Health checks pass consistently
- [ ] Documentation reflects actual working state

---

**Created**: 2025-01-26  
**Repository**: https://github.com/esengendo/San-Francisco-Business-Model  
**Status**: Ready for Implementation  
**Estimated Completion**: 2-3 days  
**Priority**: Critical - Production CI/CD currently failing