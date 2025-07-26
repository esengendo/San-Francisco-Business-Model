# Pipeline Analysis Report

## Overview
Analysis of the `pipeline_runner.py` script discovery patterns and identification of unreferenced/obsolete files in the San Francisco Business Model project.

## Pipeline Discovery Patterns

The `pipeline_runner.py` uses the following patterns to discover scripts:

1. **src/utils** - Scripts with numbers 01-03 (`*_0[1-3].py`)
2. **src/data_collection** - Scripts with numbers 04-13 (`*_0[4-9].py`, `*_1[0-3].py`)
3. **src/processing** - Scripts with numbers 14-23 (`*_1[4-9].py`, `*_2[0-3].py`)
4. **src/models** - Scripts with numbers 24+ (`*_2[4-9].py`)

## Scripts NOT Referenced by Pipeline Runner

### 1. Numbering Conflicts (Won't be picked up due to duplicates)

| File | Issue | Conflict With |
|------|-------|---------------|
| `src/data_collection/sf_business_data_07.py` | Duplicate number 07 | `sf_planning_data_07.py` |
| `src/processing/news_integration_merge_22.py` | Duplicate number 22 | `feature_engineering_22.py` |

**Impact:** The pipeline will only pick up one script per number, so duplicates will cause some scripts to be ignored.

### 2. Wrong Directory Placement

| File | Current Location | Issue | Recommended Action |
|------|------------------|-------|-------------------|
| `src/models/sf_business_trends_12.py` | models/ | Has number 12 but in models directory | Move to data_collection/ or rename to 25+ |
| `src/models/model_evaluation_metrics_20.py` | models/ | Has number 20 but in models directory | Rename to model_evaluation_metrics_25.py |

### 3. Unnumbered Scripts (Not in pipeline discovery patterns)

| File | Location | Status |
|------|----------|--------|
| `src/utils/config.py` | src/utils/ | Not numbered, won't be picked up |
| `src/models/model_loading_helper.py` | src/models/ | Not numbered, won't be picked up |

### 4. App Directory Scripts (Not part of pipeline)

These scripts are for the web application/dashboard, not the data pipeline:

- `app/app.py`
- `app/business_dashboard.py`
- `app/main.py`
- `app/components/business_analytics.py`
- `app/utils/config.py`
- `app/utils/model_loader.py`

### 5. Test Files (Not pipeline scripts)

- `tests/test_dashboard_components.py`
- `tests/test_integration.py`
- `tests/test_pipeline_components.py`

### 6. Other Files

- `notebooks/SF_Biz_Score.ipynb` - Jupyter notebook
- `scripts/` directory - Empty

## Current Pipeline Scripts (Properly Referenced)

### src/utils/ (01-03)
- ✅ `api_keys_validation_01.py`
- ✅ `logging_config_setup_02.py`
- ✅ `helper_functions_03.py`

### src/data_collection/ (04-13)
- ✅ `sf_business_data_04.py`
- ✅ `fred_economic_data_05.py`
- ✅ `census_demographic_data_06.py`
- ✅ `sf_planning_data_07.py`
- ✅ `sf_crime_data_08.py`
- ✅ `sf311_data_09.py`
- ✅ `osm_business_data_10.py`
- ✅ `gdelt_news_data_11.py`
- ✅ `sf_news_rss_data_12.py`
- ✅ `wayback_historical_data_13.py`

### src/processing/ (14-23)
- ✅ `file_combination_cleanup_14.py`
- ✅ `business_data_processing_15.py`
- ✅ `business_analysis_merge_16.py`
- ✅ `osm_enrichment_merge_17.py`
- ✅ `land_use_integration_merge_18.py`
- ✅ `permits_integration_merge_19.py`
- ✅ `sf311_integration_merge_20.py`
- ✅ `crime_integration_merge_21.py`
- ✅ `feature_engineering_22.py`
- ✅ `premodeling_pipeline_23.py`

### src/models/ (24+)
- ✅ `model_training_with_save_load_24.py`

## Potential Issues

### 1. Missing Model Scripts
The pipeline expects model training scripts numbered 24+ in the models directory, but there's only one (24). You might be missing scripts numbered 25+.

### 2. Numbering Conflicts
The pipeline will only pick up one script per number, so duplicates will cause some scripts to be ignored.

### 3. Directory Mismatches
Some scripts are in the wrong directories for their numbering scheme.

## Recommendations

### 1. Fix Numbering Conflicts
```bash
# Rename duplicate scripts
mv src/data_collection/sf_business_data_07.py src/data_collection/sf_business_data_07b.py
mv src/processing/news_integration_merge_22.py src/processing/news_integration_merge_22b.py
```

### 2. Fix Directory Placement
```bash
# Move or rename scripts in wrong directories
mv src/models/sf_business_trends_12.py src/data_collection/sf_business_trends_12.py
mv src/models/model_evaluation_metrics_20.py src/models/model_evaluation_metrics_25.py
```

### 3. Add Missing Model Scripts
Consider adding model evaluation scripts numbered 25+ if needed.

### 4. Review Unnumbered Scripts
Check if `src/utils/config.py` and `src/models/model_loading_helper.py` are actually needed. If they are utilities, they might be fine as unnumbered helper files.

## Pipeline Execution Order

Based on the discovery patterns, the pipeline will execute scripts in this order:

1. **Setup (01-03):** api_keys_validation → logging_config_setup → helper_functions
2. **Data Collection (04-13):** sf_business_data → fred_economic_data → census_demographic_data → sf_planning_data → sf_crime_data → sf311_data → osm_business_data → gdelt_news_data → sf_news_rss_data → wayback_historical_data
3. **Processing (14-23):** file_combination_cleanup → business_data_processing → business_analysis_merge → osm_enrichment_merge → land_use_integration_merge → permits_integration_merge → sf311_integration_merge → crime_integration_merge → feature_engineering → premodeling_pipeline
4. **Models (24+):** model_training_with_save_load

## Summary

- **Total scripts in pipeline:** 23 scripts
- **Scripts with conflicts:** 2 scripts
- **Scripts in wrong directories:** 2 scripts
- **Unnumbered utility scripts:** 2 scripts
- **App/dashboard scripts:** 6 scripts
- **Test scripts:** 3 scripts

The pipeline runner is designed to automatically discover scripts based on their numbering and directory placement, so any scripts that don't follow this pattern won't be executed as part of the pipeline. 