# Architecture Overview

## System Architecture Diagram

```mermaid
graph TB
    %% Data Sources Layer
    subgraph "📊 Data Sources (13 APIs)"
        FRED[FRED Economic Data]
        CENSUS[US Census Bureau]
        SFGOV[SF Gov Open Data]
        OSM[OpenStreetMap]
        GDELT[GDELT News Data]
        OTHERS[8+ Other APIs]
    end

    %% Data Collection Layer
    subgraph "🔄 Data Collection Layer"
        DC1[fred_economic_data_01.py]
        DC2[census_demographic_data_02.py]
        DC3[sf_business_registrations_03.py]
        DC4[sf_planning_permits_04.py]
        DC5[sf_crime_data_05.py]
        DC6[sf_transportation_data_06.py]
        DC7[openstreetmap_poi_07.py]
        DC8[gdelt_news_sentiment_08.py]
        DC9[+ 5 more collectors]
    end

    %% Data Processing Layer
    subgraph "⚙️ Data Processing Pipeline"
        MERGE1[business_merge_16.py]
        MERGE2[economic_merge_17.py]
        MERGE3[demographic_merge_18.py]
        MERGE4[crime_merge_19.py]
        MERGE5[transport_merge_20.py]
        MERGE6[planning_merge_21.py]
        MERGE7[news_merge_22.py]
        PREPROC[premodeling_pipeline_23.py]
    end

    %% Feature Engineering
    subgraph "🔧 Feature Engineering"
        SPATIAL[5 Spatial Features<br/>Lat/Lng/Neighborhoods]
        TEMPORAL[1 Temporal Feature<br/>Seasonality]
        BUSINESS[112 Business Features<br/>Industry/Economic/Crime]
    end

    %% ML Model Layer
    subgraph "🧠 ML Model (PyTorch Lightning)"
        MODEL[Multi-Branch Neural Network<br/>Spatial + Temporal + Business]
        TRAINING[model_training_24.py<br/>31 Functions]
        EVALUATION[model_evaluation_20.py<br/>23 Functions]
    end

    %% Application Layer
    subgraph "🌐 Web Application"
        LEGACY[app.py<br/>Legacy Monolithic<br/>1,315 lines]
        MODERN[main.py<br/>Modular Architecture]
        COMPONENTS[Components/<br/>Reusable UI]
        UTILS[Utils/<br/>Helper Functions]
    end

    %% Deployment Layer
    subgraph "🐳 Deployment & Infrastructure"
        DOCKER[Multi-Platform Docker<br/>ARM64 + AMD64]
        GITHUB[GitHub Actions CI/CD]
        DOCKERHUB[Docker Hub Registry]
        HEALTH[Health Monitoring]
    end

    %% Data Flow
    FRED --> DC1
    CENSUS --> DC2
    SFGOV --> DC3
    SFGOV --> DC4
    SFGOV --> DC5
    SFGOV --> DC6
    OSM --> DC7
    GDELT --> DC8
    OTHERS --> DC9

    DC1 --> MERGE1
    DC2 --> MERGE2
    DC3 --> MERGE1
    DC4 --> MERGE6
    DC5 --> MERGE4
    DC6 --> MERGE5
    DC7 --> MERGE1
    DC8 --> MERGE7

    MERGE1 --> PREPROC
    MERGE2 --> PREPROC
    MERGE3 --> PREPROC
    MERGE4 --> PREPROC
    MERGE5 --> PREPROC
    MERGE6 --> PREPROC
    MERGE7 --> PREPROC

    PREPROC --> SPATIAL
    PREPROC --> TEMPORAL
    PREPROC --> BUSINESS

    SPATIAL --> MODEL
    TEMPORAL --> MODEL
    BUSINESS --> MODEL

    MODEL --> TRAINING
    MODEL --> EVALUATION

    MODEL --> MODERN
    MODEL --> LEGACY

    MODERN --> COMPONENTS
    MODERN --> UTILS

    MODERN --> DOCKER
    LEGACY --> DOCKER

    DOCKER --> GITHUB
    GITHUB --> DOCKERHUB
    DOCKERHUB --> HEALTH

    %% Styling
    classDef dataSource fill:#e1f5fe
    classDef processing fill:#f3e5f5
    classDef model fill:#e8f5e8
    classDef app fill:#fff3e0
    classDef deploy fill:#fce4ec

    class FRED,CENSUS,SFGOV,OSM,GDELT,OTHERS dataSource
    class DC1,DC2,DC3,DC4,DC5,DC6,DC7,DC8,DC9,MERGE1,MERGE2,MERGE3,MERGE4,MERGE5,MERGE6,MERGE7,PREPROC processing
    class SPATIAL,TEMPORAL,BUSINESS,MODEL,TRAINING,EVALUATION model
    class LEGACY,MODERN,COMPONENTS,UTILS app
    class DOCKER,GITHUB,DOCKERHUB,HEALTH deploy
```

## Key Architecture Components

### 1. **Data Collection (13 API Sources)**
- **Economic**: FRED API for macroeconomic indicators
- **Demographic**: US Census Bureau data integration
- **Municipal**: SF Gov Open Data (business registrations, permits, crime)
- **Geographic**: OpenStreetMap points of interest
- **Sentiment**: GDELT news sentiment analysis
- **+ 8 Additional**: Transportation, weather, real estate, etc.

### 2. **Data Processing Pipeline**
- **File Combination**: `file_combination_cleanup_14.py`
- **Business Processing**: `business_data_processing_15.py` 
- **8 Merge Scripts**: Sequential integration (files 16-22)
- **Feature Engineering**: Creates 118 ML-ready features
- **Preprocessing**: `premodeling_pipeline_23.py`

### 3. **Multi-Branch Neural Network**
```
Input Features (118)
├── Spatial Branch (5) → 64 → 32 nodes
├── Temporal Branch (1) → 64 → 32 nodes  
└── Business Branch (112) → 128 → 64 → 32 nodes
    └── Fusion Layer → 128 → 64 → 1 (Binary Output)
```

### 4. **Application Architecture**
- **Legacy**: `app.py` (1,315 lines, monolithic)
- **Optimized**: `main.py` with modular components
- **Deployment**: Multi-platform Docker (ARM64 + AMD64)

### 5. **Production Infrastructure**
- **CI/CD**: GitHub Actions automated workflows
- **Containerization**: Optimized multi-stage Docker builds
- **Registry**: Docker Hub with multi-platform support
- **Monitoring**: Health checks and automated testing

## Data Flow Summary

```
13 API Sources → Raw Data → Processing Pipeline → 118 Features → ML Model → Web Dashboard
     ↓              ↓            ↓               ↓          ↓          ↓
 (Real-time)   (Parquet)   (8 merge scripts) (Engineered) (PyTorch) (Streamlit)
```

## Performance Metrics

- **Data Volume**: 250K+ business records
- **Feature Engineering**: 118 features from 13 sources
- **Model Training**: PyTorch Lightning with automated callbacks
- **Inference Speed**: <100ms per prediction
- **Container Size**: ~2-3GB (optimized from 4.84GB)
- **Platform Support**: Linux AMD64 + ARM64