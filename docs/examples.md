# GeoAI Examples

This document provides comprehensive, practical examples demonstrating how to combine Claude GeoAI Skills to solve real geospatial AI and scientific computing problems.

---

## 📋 Table of Contents

1. [Geospatial Data Analysis & Machine Learning](#geospatial-data-analysis--machine-learning)
2. [Large-Scale Spatial Data Processing](#large-scale-spatial-data-processing)
3. [Interactive Geospatial Dashboards](#interactive-geospatial-dashboards)
4. [Literature Review & Research Synthesis](#literature-review--research-synthesis)
5. [Spatial Predictive Modeling](#spatial-predictive-modeling)
6. [Statistical Analysis of Spatial Data](#statistical-analysis-of-spatial-data)
7. [Research Grant Writing for GeoAI](#research-grant-writing-for-geoai)
8. [Scientific Communication for Geospatial Research](#scientific-communication-for-geospatial-research)

---

## Geospatial Data Analysis & Machine Learning

### Example 1: Urban Land Use Classification Using Satellite Imagery

**Objective**: Classify land use types from satellite imagery and analyze spatial patterns across urban areas.

**Skills Used**:
- `geopandas` - Spatial data handling
- `scikit-learn` - Machine learning classification
- `matplotlib` - Visualization
- `seaborn` - Statistical plots
- `plotly` - Interactive maps
- `scientific-visualization` - Publication figures

**Workflow**:

```bash
# Always use available 'skills' when possible. Keep output organized.

Step 1: Load satellite imagery and spatial data
- Load multi-band satellite imagery (GeoTIFF format)
- Read vector boundaries (administrative units, study area)
- Reproject all data to common CRS (e.g., UTM zone)
- Extract spectral bands: visible (RGB), near-infrared, thermal

Step 2: Data preprocessing
- Mask clouds and cloud shadows using quality bands
- Calculate vegetation indices (NDVI, EVI)
- Calculate built-up indices (NDBI, UI)
- Normalize spectral values (0-1 scale)
- Extract pixel values for training polygons

Step 3: Feature engineering
- Create spectral features from bands
- Calculate texture features (GLCM entropy, contrast)
- Add spatial context features (distance to roads, water bodies)
- Aggregate features at multiple spatial scales

Step 4: Train land use classifier with scikit-learn
- Split data: 70% train, 30% test (stratified)
- Train Random Forest classifier with 200 trees
- Hyperparameter tuning with GridSearchCV:
  * n_estimators: [100, 200, 300]
  * max_depth: [10, 20, None]
  * min_samples_split: [2, 5, 10]
- Cross-validate with 5-fold CV
- Evaluate on test set

Step 5: Model evaluation
- Calculate accuracy, precision, recall, F1 per class
- Generate confusion matrix
- Compute overall accuracy and Kappa coefficient
- Identify most misclassified classes
- Extract feature importances

Step 6: Apply classifier to full study area
- Generate land use map for entire region
- Apply spatial smoothing (majority filter 3×3)
- Mask out clouds and invalid pixels
- Export classified raster to GeoTIFF

Step 7: Spatial analysis of land use
- Calculate land use area statistics by administrative unit
- Compute landscape metrics (fragmentation, edge density)
- Analyze spatial autocorrelation (Moran's I)
- Identify land use change hotspots (if temporal data)

Step 8: Create comprehensive visualizations
- Generate land use classification map with matplotlib
- Create interactive map with Plotly (hover tooltips)
- Plot feature importance bar chart
- Show confusion matrix with seaborn heatmap
- Create pie charts of land use proportions
- Visualize spatial patterns of misclassification

Step 9: Generate scientific report
- Methods: data sources, preprocessing, classification approach
- Results: overall accuracy, per-class metrics, feature importance
- Spatial analysis: land use patterns, fragmentation metrics
- Discussion: strengths, limitations, future improvements
- Publication-quality figures with proper legends and scales
- Export to PDF with citations

Expected Output:
- Land use classification map (raster and vector)
- Classification accuracy metrics (overall and per-class)
- Feature importance rankings
- Spatial analysis results (area statistics, fragmentation)
- Comprehensive scientific report with figures
```

---

### Example 2: Urban Heat Island Analysis

**Objective**: Analyze spatial patterns of urban heat islands using remote sensing and ground data.

**Skills Used**:
- `geopandas` - Spatial operations
- `polars` - High-performance data processing
- `statsmodels` - Spatial statistics
- `scikit-learn` - Clustering
- `matplotlib` - Visualization
- `plotly` - Interactive maps
- `scientific-writing` - Report generation

**Workflow**:

```bash
Step 1: Load thermal imagery and reference data
- Load Landsat thermal infrared band (LST calculation)
- Load land use/land cover data
- Load meteorological station data (temperature readings)
- Load urban boundary and administrative divisions
- Reproject all layers to common coordinate system

Step 2: Calculate land surface temperature
- Convert digital numbers to radiance
- Convert radiance to at-sensor brightness temperature
- Apply emissivity correction using land cover
- Calculate LST for entire study area
- Validate against ground station measurements

Step 3: Identify urban heat islands
- Define rural reference areas
- Calculate baseline rural temperature
- Compute temperature anomalies (ΔT = LST - baseline)
- Classify heat intensity categories:
  * Moderate: ΔT > 2°C
  * Strong: ΔT > 4°C
  * Extreme: ΔT > 6°C

Step 4: Correlate with land use patterns
- Extract land cover types within heat island zones
- Calculate mean LST by land use class
- Perform statistical tests (ANOVA) between categories
- Identify land use classes most associated with high temperatures

Step 5: Spatial clustering of hot spots
- Use scikit-learn DBSCAN to cluster hot pixels
- Identify distinct heat island clusters
- Calculate cluster properties (size, intensity, shape)
- Analyze cluster distribution relative to urban features

Step 6: Temporal analysis (if multi-temporal data)
- Analyze LST trends over time (diurnal, seasonal)
- Identify persistent vs transient heat islands
- Calculate heat island intensity by time of day/season
- Compare summer vs winter patterns

Step 7: Socio-economic correlation
- Load census/demographic data
- Join with spatial units
- Correlate heat intensity with:
  * Population density
  * Building density
  * Vegetation cover
  * Socio-economic indicators
- Use statsmodels for regression analysis

Step 8: Create comprehensive visualizations
- LST map with urban boundary overlay
- Heat island intensity classification map
- Interactive map with hover information
- Scatter plots: LST vs vegetation, population density
- Time series plots of temperature patterns
- Box plots of LST by land use type

Step 9: Statistical analysis
- Spatial autocorrelation (Moran's I) of LST
- Hotspot analysis (Getis-Ord Gi*)
- Spatial regression modeling
- Significance testing of patterns

Step 10: Generate urban heat island report
- Methods: data sources, LST calculation, analysis methods
- Results: heat island extent, intensity, spatial patterns
- Correlations with land use and socio-economic factors
- Mitigation recommendations:
  * Green infrastructure opportunities
  * Cool roof targets
  * Urban planning implications
- Publication-ready figures and tables
- Export PDF report

Expected Output:
- Urban heat island maps and intensity classification
- Statistical correlations with land use and demographics
- Cluster analysis of hot spots
- Mitigation recommendations
- Comprehensive scientific report
```

---

## Large-Scale Spatial Data Processing

### Example 3: Processing National-Scale Climate Data

**Objective**: Process and analyze terabytes of climate data across multiple variables and time periods.

**Skills Used**:
- `dask` - Parallel computing
- `zarr-python` - Chunked array storage
- `polars` - High-performance dataframes
- `geopandas` - Spatial operations
- `matplotlib` - Visualization
- `get-available-resources` - Resource detection

**Workflow**:

```bash
Step 1: Detect available computational resources
- Use get-available-resources to assess:
  * CPU cores and memory
  * GPU availability
  * Disk space for temporary files
  * Optimal chunk sizes for I/O
- Generate processing recommendations

Step 2: Load large climate datasets with Dask
- Load NetCDF files as Dask arrays
- Define optimal chunk sizes (e.g., time: 365, lat: 100, lon: 100)
- Create lazy computations (no immediate execution)
- Load metadata (coordinates, variable descriptions)

Step 3: Quality control and preprocessing
- Identify and flag missing values
- Perform temporal interpolation for small gaps
- Apply spatial interpolation for missing grid cells
- Remove outliers using statistical filters
- Document QC metrics (percentage of gaps, flags)

Step 4: Climate indices calculation
- Calculate derived climate indices:
  * Growing degree days
  * Frost days
  * Heat wave indices
  * Precipitation extremes (P95, P99)
  * Standardized precipitation index (SPI)
- Use Dask for parallel computation across time

Step 5: Spatial aggregation and statistics
- Calculate annual/seasonal means
- Compute climatological normals (30-year averages)
- Calculate trends using linear regression per grid cell
- Compute variability metrics (standard deviation, percentiles)
- Store intermediate results in Zarr format

Step 6: Regional analysis with GeoPandas
- Define administrative or ecological regions
- Extract time series for each region
- Compute regional statistics (mean, min, max, trends)
- Perform spatial interpolation if needed
- Calculate correlation between regions

Step 7: Extreme event analysis
- Identify extreme events:
  * Heat waves (3+ consecutive days > 90th percentile)
  * Drought periods (SPI < -2 for 3+ months)
  * Heavy precipitation events
- Extract event characteristics (duration, intensity, spatial extent)
- Create event catalog with temporal and spatial attributes

Step 8: Compute with Dask
- Execute lazy computations with Dask scheduler
- Use distributed processing if multiple workers available
- Monitor progress with Dask dashboards
- Optimize memory usage with chunking

Step 9: Data export and storage
- Save processed data to Zarr format
- Export summary statistics to Parquet with Polars
- Create NetCDF files for standard GIS compatibility
- Generate shapefiles of regional statistics
- Optimize file sizes for distribution

Step 10: Visualization of large datasets
- Use matplotlib with optimized rendering for large arrays
- Create decimated visualizations for quick preview
- Generate time series animations
- Create spatial distribution maps
- Export high-resolution figures for publication

Step 11: Generate processing report
- Summary statistics:
  * Data volume processed
  * Number of grid cells and time steps
  * Processing time and resource utilization
  * QC metrics and gaps
- Climate trends and patterns:
  * Regional warming trends
  * Changes in precipitation patterns
  * Extreme event frequencies
- Recommendations for further analysis
- Export PDF report

Expected Output:
- Processed climate datasets in Zarr format
- Regional climate statistics and trends
- Extreme event catalog
- Publication-ready visualizations
- Comprehensive processing report
```

---

### Example 4: Distributed Geospatial Machine Learning

**Objective**: Train machine learning models on large-scale spatial data using distributed computing.

**Skills Used**:
- `dask` - Distributed computing
- `pytorch-lightning` - Deep learning framework
- `scikit-learn` - Classical ML
- `modal` - Cloud execution
- `geopandas` - Spatial data
- `polars` - High-performance processing
- `shap` - Model interpretability

**Workflow**:

```bash
Step 1: Assess data scale and computational needs
- Calculate dataset size (GB/TB)
- Determine if distributed processing needed
- Use get-available-resources to plan execution
- Decide between local Dask vs cloud Modal

Step 2: Load large training dataset
- Use Dask to load data in chunks
- Partition spatial data into tiles/regions
- Create Dask DataFrames for tabular features
- Maintain spatial coordinates for each sample
- Balance classes across partitions

Step 3: Feature engineering at scale
- Calculate spatial features (distance to amenities, land use context)
- Aggregate features by spatial neighborhoods
- Normalize features with Dask operations
- Handle missing data with appropriate strategies
- Save feature cache for reproducibility

Step 4: Set up distributed training with Modal
- Define Modal function for distributed training
- Configure GPU requirements if deep learning
- Set up data loading pipeline:
  * Load partitions from cloud storage (S3, GCS)
  * Batch data appropriately
  * Augment data on the fly (if applicable)
- Scale out to multiple workers

Step 5: Train model with distributed computing
Option A: PyTorch Lightning for deep learning
- Set up LightningModule
- Configure distributed data parallel (DDP)
- Train across multiple GPUs or machines
- Monitor with TensorBoard
- Save checkpoints regularly

Option B: Scikit-learn with Dask-ML
- Use Dask-ML extensions for distributed training
- Train Random Forest or Gradient Boosting
- Hyperparameter tuning with Dask
- Combine models from workers

Step 6: Model evaluation on held-out set
- Load test dataset
- Evaluate distributed model predictions
- Calculate metrics (accuracy, precision, recall, AUC)
- Perform spatial cross-validation if needed
- Analyze performance by region

Step 7: Model interpretability with SHAP
- Calculate SHAP values on sample of data
- Aggregate SHAP values by feature importance
- Analyze spatial patterns of feature importance
- Identify key drivers of predictions
- Create SHAP summary plots

Step 8: Apply model to prediction dataset
- Load prediction area (could be larger than training area)
- Process in tiles to manage memory
- Generate predictions with trained model
- Apply spatial post-processing (smoothing, filtering)
- Export predictions to geospatial format

Step 9: Spatial analysis of predictions
- Analyze prediction confidence
- Identify regions of high uncertainty
- Correlate predictions with ancillary variables
- Validate predictions with ground truth (if available)
- Create accuracy assessment map

Step 10: Deploy model as service (optional)
- Package model for deployment
- Set up Modal web endpoint for predictions
- Create API documentation
- Implement batch prediction capability

Step 11: Generate comprehensive report
- Model architecture and training details
- Hyperparameters and optimization process
- Performance metrics and validation results
- SHAP interpretability analysis
- Spatial patterns in predictions
- Deployment recommendations
- Publication-ready figures
- Export PDF report

Expected Output:
- Trained machine learning model
- Performance metrics and validation
- SHAP interpretability analysis
- Prediction maps for study area
- Comprehensive ML report
```

---

## Interactive Geospatial Dashboards

### Example 5: Real-Time Environmental Monitoring Dashboard

**Objective**: Create an interactive dashboard for monitoring environmental conditions across multiple sensors.

**Skills Used**:
- `plotly` - Interactive visualizations
- `geopandas` - Spatial data
- `polars` - High-performance data processing
- `matplotlib` - Static figures
- `modal` - Cloud deployment
- `scientific-slides` - Presentation

**Workflow**:

```bash
Step 1: Load and process sensor data
- Load time series data from multiple sensors
- Process with Polars for high performance:
  * Handle missing values (interpolation, forward fill)
  * Remove outliers using statistical methods
  * Resample to consistent time intervals
- Load sensor locations as GeoDataFrame
- Assign spatial attributes (land use, proximity features)

Step 2: Aggregate and summarize data
- Calculate rolling averages and statistics
- Compute daily, weekly, monthly aggregates
- Calculate anomalies (deviations from baseline)
- Identify exceedances of thresholds
- Compute spatial interpolations (IDW, kriging)

Step 3: Create interactive visualizations
Time series components:
- Line charts for individual sensors over time
- Comparison plots (multiple sensors on same axis)
- Heat maps (time on x-axis, sensor on y-axis)
- Distribution plots (box plots, histograms)

Spatial components:
- Map of sensor locations with current readings
- Choropleth maps of interpolated values
- Time series of spatial averages
- Anomaly heat maps

Statistical components:
- Summary tables with key metrics
- Trend indicators (up/down arrows)
- Alert thresholds and exceedance counts
- Correlation matrices between sensors

Step 4: Design dashboard layout
- Create multi-panel layout with Plotly:
  * Top: Overview map with current conditions
  * Middle left: Time series selector
  * Middle right: Statistics and alerts
  * Bottom: Detailed views and comparisons
- Implement interactive controls:
  * Date/time range slider
  * Sensor selection dropdown
  * Parameter type selector
  * Time aggregation level

Step 5: Add interactivity features
- Hover tooltips showing detailed information
- Click to view sensor details
- Time series zoom and pan
- Dynamic filtering
- Animation of time series
- Export data buttons (CSV, PNG)

Step 6: Deploy dashboard with Modal
- Create Modal web endpoint
- Set up periodic data refresh
- Implement caching for performance
- Configure auto-scaling based on traffic
- Monitor with logs and metrics

Step 7: Add alert system
- Define alert thresholds for each parameter
- Monitor for threshold exceedances
- Send notifications (email, SMS) for alerts
- Log alert history
- Visualize alerts on dashboard

Step 8: Create presentation of dashboard
- Use scientific-slides to create presentation
- Include screenshots of dashboard
- Demonstrate key features
- Show use cases and benefits
- Provide technical architecture overview

Step 9: Testing and validation
- Test dashboard with different browsers
- Validate data accuracy and refresh rates
- Stress test with concurrent users
- Verify alert system functionality
- Get feedback from stakeholders

Step 10: Documentation and deployment
- Create user guide for dashboard
- Document API endpoints
- Create deployment guide
- Train end users
- Set up monitoring and maintenance

Expected Output:
- Interactive web-based dashboard
- Real-time environmental monitoring
- Alert system for threshold violations
- User documentation
- Presentation materials
```

---

## Literature Review & Research Synthesis

### Example 6: Systematic Review of GeoAI Applications

**Objective**: Conduct a comprehensive systematic review of geospatial AI applications in a specific domain.

**Skills Used**:
- `literature-review` - Systematic review tools
- `pubmed-database` - Biomedical literature search
- `openalex-database` - Comprehensive literature database
- `perplexity-search` - AI-powered web search
- `citation-management` - Bibliography management
- `scientific-writing` - Report generation
- `matplotlib` - Visualizations

**Workflow**:

```bash
Step 1: Define research questions and scope
- Define primary research questions
- Specify inclusion/exclusion criteria:
  * Time period (e.g., last 10 years)
  * Geographic scope (global, regional)
  * Application domains (specific to interest)
  * AI methods (deep learning, classical ML, etc.)
  * Data types (satellite, social media, sensor)
- Define outcomes of interest

Step 2: Search PubMed database
- Construct search query with Boolean operators:
  * Keywords: "geospatial AI" OR "spatial machine learning"
  * Specific terms for application domain
  * Methods terms: "deep learning", "CNN", "U-Net"
  * Data terms: "remote sensing", "satellite", "GIS"
- Use MeSH terms for indexing
- Filter by publication date, article type
- Export results (abstracts, metadata)
- Document search strategy

Step 3: Search OpenAlex database
- Query OpenAlex for geospatial AI papers
- Filter by publication year, fields of study
- Extract citation counts and impact metrics
- Identify highly cited papers
- Complement PubMed results with broader coverage

Step 4: Use Perplexity Search for recent work
- Search for preprints and recent publications
- Find conference papers and technical reports
- Search for specific techniques and applications
- Identify emerging trends not yet indexed
- Gather supplementary materials and code repositories

Step 5: Screen and select studies
- Screen titles and abstracts for relevance
- Apply inclusion/exclusion criteria
- Create inclusion/exclusion log with reasons
- Retrieve full texts for eligible studies
- Perform full-text screening

Step 6: Extract data from included studies
- Use standardized extraction form for each study:
  * Study characteristics (year, location, data sources)
  * AI methods (algorithms, architectures, frameworks)
  * Performance metrics (accuracy, F1, MAE, etc.)
  * Application details (specific problem, scale)
  * Limitations and challenges reported
- Maintain data extraction in structured format

Step 7: Quality assessment
- Assess methodological quality of each study
- Evaluate:
  * Data quality and preprocessing
  * Validation methods (cross-validation, temporal split)
  * Performance reporting (multiple metrics, statistical tests)
  * Reproducibility (code availability, data access)
- Document quality scores

Step 8: Synthesize findings
- Categorize studies by application domain
- Group by AI method/approach
- Identify trends over time
- Compare performance across methods
- Extract best practices and common challenges
- Identify research gaps

Step 9: Manage citations
- Use citation-management to organize references
- Create bibliography in required format
- Check for duplicates
- Ensure consistent formatting
- Export to BibTeX for report generation

Step 10: Create visualizations
- Timeline of publications by year
- Bar chart of applications by domain
- Network visualization of citation relationships
- Performance comparison tables/plots
- Word cloud of techniques used

Step 11: Generate systematic review report
Use scientific-writing to create comprehensive report:
- Abstract and keywords
- Introduction and background
- Methods: search strategy, screening, data extraction
- Results: study selection, synthesis, trends
- Discussion: implications, limitations, future directions
- Conclusion and recommendations
- References in proper format
- Tables and figures
- Export to PDF or manuscript format

Step 12: PRISMA diagram
- Create PRISMA flow diagram showing:
  * Identification (records identified, duplicates removed)
  * Screening (excluded with reasons)
  * Eligibility (assessed for eligibility)
  * Included (studies in review)
- Use matplotlib for diagram

Expected Output:
- Systematic review report (20-50 pages)
- Database of extracted studies with metadata
- Quality assessment scores
- Visualizations of trends and patterns
- PRISMA diagram
- Reference bibliography
- Publication-ready manuscript
```

---

## Spatial Predictive Modeling

### Example 7: Predicting Urban Sprawl Using Machine Learning

**Objective**: Build predictive models to forecast urban expansion and land use change.

**Skills Used**:
- `geopandas` - Spatial operations
- `scikit-learn` - Machine learning
- `torch_geometric` - Graph neural networks
- `pytorch-lightning` - Deep learning training
- `polars` - High-performance data processing
- `statsmodels` - Statistical modeling
- `matplotlib` - Visualization
- `plotly` - Interactive maps
- `shap` - Model interpretability

**Workflow**:

```bash
Step 1: Collect historical land use data
- Load land use maps for multiple time points (e.g., 2000, 2010, 2020)
- Create change detection:
  * Identify areas that converted to urban
  * Identify non-conversion areas
- Define target variable: urbanized (1) vs not urbanized (0)

Step 2: Compile predictor variables
Proximity variables (with GeoPandas):
- Distance to existing urban areas
- Distance to roads, highways
- Distance to city center
- Distance to public transport

Socio-economic variables:
- Population density (census data)
- Income levels
- Employment rates
- Land use restrictions/zoning

Environmental variables:
- Slope and elevation (DEM)
- Soil type
- Flood risk zones
- Protected areas

Infrastructure variables:
- Utility availability
- School locations
- Hospital locations
- Commercial centers

Step 3: Feature engineering
- Calculate distance variables for each time point
- Aggregate socio-economic variables by spatial units
- Create interaction features
- Normalize/scale variables
- Handle missing data

Step 4: Prepare training dataset
- Extract features for historical time points
- Label based on subsequent urbanization
- Create balanced training set (if class imbalance)
- Split data: 70% train, 15% validation, 15% test
- Ensure spatial independence (no spatial autocorrelation in splits)

Step 5: Train classical ML models with scikit-learn
- Train Random Forest for baseline
- Train Gradient Boosting (XGBoost or LightGBM)
- Train Logistic Regression for interpretability
- Use GridSearchCV for hyperparameter tuning
- Evaluate using spatial cross-validation

Step 6: Train graph neural network with Torch Geometric
- Build spatial graph:
  * Nodes: spatial units (grid cells, parcels)
  * Edges: spatial adjacency (neighbors within distance threshold)
- Create node features from predictor variables
- Use Graph Convolutional Network (GCN) or GraphSAGE
- Train with PyTorch Lightning:
  * Binary cross-entropy loss
  * Adam optimizer
  * Early stopping on validation loss
- Compare performance with classical ML

Step 7: Model evaluation
- Calculate accuracy, precision, recall, F1
- ROC curves and AUC
- Spatial cross-validation (block-based to avoid leakage)
- Calibration plots
- Compare models:
  * Classical ML vs GNN
  * Training periods vs test periods
  * Different spatial scales

Step 8: Model interpretability with SHAP
- Calculate SHAP values for Random Forest/GNN
- Identify most important features
- Create SHAP summary plots (beeswarm, bar)
- Analyze spatial patterns of SHAP values
- Understand drivers of urbanization

Step 9: Generate future predictions
- Update predictor variables to current/future values
- Apply trained model to predict urbanization probability
- Generate predictions for multiple scenarios:
  * Business as usual
  * Aggressive growth
  * Conservation-focused
- Threshold probabilities to get binary predictions

Step 10: Scenario analysis and policy implications
- Calculate urban expansion under each scenario
- Identify areas at highest risk of conversion
- Assess environmental impacts:
  * Loss of agricultural land
  * Impact on ecosystems
  * Infrastructure needs
- Evaluate policy levers:
  * How much does restricting expansion affect growth?
  * Impact of transit-oriented development
  * Effectiveness of conservation zones

Step 11: Create comprehensive visualizations
- Maps of predicted urban expansion by scenario
- Probability heat maps
- Change maps comparing scenarios
- Feature importance plots (SHAP)
- Time series of urban area
- Interactive maps with Plotly for exploration

Step 12: Generate planning report
- Executive summary of findings
- Model description and validation
- Scenario results and implications
- Policy recommendations:
  * Where to focus conservation efforts
  * Infrastructure planning priorities
  * Zoning recommendations
- Limitations and uncertainties
- Publication-quality figures
- Export PDF report

Expected Output:
- Trained predictive models (ML and GNN)
- Urban expansion forecasts for multiple scenarios
- Feature importance analysis (SHAP)
- Scenario comparison maps
- Policy recommendations report
- Interactive visualization dashboard
```

---

## Statistical Analysis of Spatial Data

### Example 8: Spatial Regression Analysis of Housing Prices

**Objective**: Analyze spatial patterns in housing prices and identify key determinants using spatial statistical methods.

**Skills Used**:
- `geopandas` - Spatial data handling
- `statsmodels` - Statistical modeling
- `scikit-learn` - Machine learning
- `polars` - Data processing
- `matplotlib` - Visualization
- `seaborn` - Statistical plots
- `shap` - Interpretability

**Workflow**:

```bash
Step 1: Load and prepare housing data
- Load property transactions data
- Load spatial data (parcels, building footprints)
- Load ancillary variables:
  * Demographics (census)
  * Amenities (schools, parks, transport)
  * Environmental factors (noise, pollution)
- Join all datasets using spatial joins
- Clean and preprocess data

Step 2: Exploratory data analysis
- Calculate summary statistics
- Visualize price distributions
- Identify outliers and anomalies
- Check missing data patterns
- Examine correlations between variables

Step 3: Spatial data exploration
- Visualize spatial distribution of prices
- Calculate spatial autocorrelation (Moran's I)
- Identify spatial clusters (hotspots, coldspots)
- Assess need for spatial modeling
- Visualize residuals of non-spatial model

Step 4: Feature engineering
- Calculate distance-based features:
  * Distance to CBD, schools, parks, transit
  * Distance to amenities
- Create neighborhood features:
  * Average prices within buffer
  * Density metrics
- Time-based features:
  * Seasonality, trends
- Interaction terms between variables

Step 5: Train baseline OLS model
- Use statsmodels for ordinary least squares
- Include all relevant features
- Check for multicollinearity (VIF)
- Examine residuals:
  * Normality (Q-Q plot)
  * Homoscedasticity
  * Spatial autocorrelation (Moran's I)

Step 6: Spatial econometric models
If spatial autocorrelation detected:
Option A: Spatial Lag Model (SAR)
- Model includes spatially lagged dependent variable
- Captures spatial spillover effects

Option B: Spatial Error Model (SEM)
- Models spatial autocorrelation in error term
- Captures unobserved spatial dependence

Option C: Spatial Durbin Model (SDM)
- Includes both lagged dependent variable and lagged independent variables
- Most general form of spatial model

Step 7: Compare models
- Fit OLS, SAR, SEM, SDM models
- Compare using:
  * AIC/BIC
  * Log-likelihood
  * R-squared
  * Spatial autocorrelation of residuals
- Use likelihood ratio tests for nested models
- Select best-fitting model

Step 8: Machine learning comparison
- Train Random Forest and Gradient Boosting
- Use spatial cross-validation (block-based)
- Compare performance with spatial models
- Evaluate feature importance
- Use SHAP for interpretability

Step 9: Model interpretation
- Extract coefficients from best model
- Interpret spatial lag/autocorrelation terms
- Identify key determinants of housing prices
- Quantify magnitude of effects
- Create partial dependence plots

Step 10: Spatial heterogeneity analysis
- Geographically Weighted Regression (GWR):
  * Estimate local coefficients across space
  * Identify spatially varying relationships
- Visualize coefficient surfaces
- Identify regions with different drivers

Step 11: Create comprehensive visualizations
- Maps of housing prices and predictions
- Maps of model residuals
- Maps of local GWR coefficients
- Scatter plots of predicted vs actual
- Spatial autocorrelation plots (Moran's I)
- Feature importance plots (SHAP, Gini importance)

Step 12: Generate statistical analysis report
- Methods: data sources, modeling approach
- Results:
  * Model comparison table
  * Coefficient estimates with significance
  * Spatial autocorrelation metrics
  * Performance metrics
- Interpretation:
  * Key drivers of housing prices
  * Spatial patterns and heterogeneity
  * Policy implications
- Limitations and future work
- Publication-ready figures
- Export PDF report

Expected Output:
- Spatial regression model estimates
- Model comparison and validation
- Maps of predictions and residuals
- GWR coefficient surfaces
- Statistical analysis report
- Interpretation of housing price determinants
```

---

## Research Grant Writing for GeoAI

### Example 9: Writing an NSF Proposal for GeoAI Research

**Objective**: Prepare a competitive research grant proposal for GeoAI project funding.

**Skills Used**:
- `research-grants` - Grant writing framework
- `literature-review` - Background research
- `openalex-database` - Citation analysis
- `pubmed-database` - Relevant literature
- `perplexity-search` - Recent developments
- `scientific-writing` - Proposal writing
- `scientific-slides` - Presentation preparation
- `matplotlib` - Budget figures

**Workflow**:

```bash
Step 1: Define research project
- Clarify research objectives and hypotheses
- Identify scientific questions to address
- Define deliverables and expected outcomes
- Assess significance and broader impacts

Step 2: Conduct literature review
- Use openalex-database to search:
  * Recent publications in GeoAI
  * Citation trends and gaps
  * Leading researchers and institutions
- Use pubmed-database for related fields
- Use perplexity-search for recent preprints
- Identify:
  * State of the art
  * Methodological gaps
  * Novelty of proposed work
- Document review findings

Step 3: Research methodology design
- Design technical approach:
  * Data sources and collection methods
  * AI/ML methods and algorithms
  * Validation strategies
  * Experimental design
- Create timeline and milestones
- Identify risks and mitigation strategies

Step 4: Prepare proposal components using research-grants
1. Project Summary (1 page)
   - Concise description of research
   - Intellectual merit and broader impacts
   - Statement of suitability

2. Project Description (15 pages)
   a. Introduction and Motivation (2-3 pages)
      - Problem statement
      - Importance and significance
      - Current state of knowledge
      - Gap to be addressed

   b. Literature Review and Background (2-3 pages)
      - Summary of relevant work
      - Identification of gaps
      - Position of proposed work

   c. Research Objectives and Hypotheses (1 page)
      - Clear, testable objectives
      - Specific hypotheses to test
      - Expected outcomes

   d. Preliminary Results (2-3 pages)
      - Prior work by PI
      - Feasibility demonstration
      - Supporting data and figures

   e. Research Methods (4-5 pages)
      - Detailed methodology
      - Experimental design
      - Data collection and processing
      - AI/ML approaches
      - Validation and evaluation
      - Timeline

   f. Expected Results and Impact (1 page)
      - Anticipated outcomes
      - Scientific contributions
      - Broader impacts (societal, educational)
      - Dissemination plan

3. Budget and Justification (2-3 pages)
   - Personnel (PI, Co-PI, students, postdocs)
   - Equipment and computing resources
   - Travel (conferences, field work)
   - Supplies and materials
   - Publication costs
   - Indirect costs

4. Facilities and Resources (1 page)
   - Institutional support
   - Available computing resources
   - Laboratory space
   - Collaborative arrangements

5. Biographical Sketches (2 pages each)
   - PI and key personnel
   - Education and training
   - Research experience and funding
   - Publications and presentations
   - Synergistic activities

6. Current and Pending Support (1 page)
   - Active grants
   - Overlap justification

7. Collaborators and Consultants (if applicable)
   - Letters of commitment
   - Roles and contributions

Step 5: Budget preparation
- Create detailed line-item budget
- Justify each category
- Include personnel effort (person-months)
- Budget for computational resources:
  * Cloud computing (Modal credits)
  * Storage for large datasets
  * Software and licenses
- Create budget tables and charts with matplotlib

Step 6: Review criteria alignment
NSF review criteria:
- Intellectual Merit:
  * Importance of research
  * Novelty and innovation
  * Methodological soundness
- Broader Impacts:
  * Societal benefits
  * Educational components
  * Diversity and inclusion
  * Technology transfer

Align proposal sections with criteria

Step 7: Internal review and revision
- Get feedback from colleagues
- Revise based on suggestions
- Check length requirements
- Proofread carefully
- Ensure compliance with formatting guidelines

Step 8: Create presentation materials
- Use scientific-slides to create presentation
- 15-20 slides for panel interview
- Include:
  * Background and motivation
  * Research objectives
  * Methods and innovation
  * Expected impacts
  * Timeline and budget
- Practice presentation

Step 9: Submission preparation
- Gather required supplementary materials:
  * Letters of collaboration
  * Institutional commitment
  * Data management plan
  * Conflict of interest statements
- Complete online submission forms
- Upload proposal components
- Verify submission before deadline

Step 10: Follow-up and response planning
- Prepare for reviewer questions
- Plan resubmission if needed
- Document reviewer feedback
- Address weaknesses in resubmission

Expected Output:
- Complete grant proposal (15-20 pages)
- Detailed budget and justification
- Supplementary materials
- Presentation slides
- Internal review feedback
- Submitted proposal to funding agency
```

---

## Scientific Communication for Geospatial Research

### Example 10: Creating Publication-Ready Geospatial Visualizations

**Objective**: Create comprehensive, publication-quality visualizations for geospatial research paper.

**Skills Used**:
- `matplotlib` - Core plotting
- `seaborn` - Statistical plotting
- `plotly` - Interactive plots
- `geopandas` - Spatial data
- `scientific-visualization` - Best practices
- `generate-image` - AI-assisted diagrams
- `scientific-slides` - Presentation figures

**Workflow**:

```bash
Step 1: Understand journal/venue requirements
- Identify target journal/conference
- Check figure specifications:
  * Preferred formats (TIFF, PNG, PDF)
  * Resolution (300 DPI for print, 150 DPI for web)
  * Size limits (single/double column)
  * Color requirements (color vs grayscale)
  * Font sizes and styles

Step 2: Plan figure layouts
- Create list of required figures
- Sketch conceptual layouts
- Group related subplots into composite figures
- Label figures (Figure 1, 2, 3, etc.)
- Plan figure captions

Step 3: Create spatial distribution maps
Use matplotlib and geopandas:
- Load vector or raster data
- Choose appropriate projection
- Create choropleth maps:
  * Color schemes (sequential, diverging, qualitative)
  * Colorblind-safe palettes
  * Appropriate classification (quantiles, natural breaks)
- Add map elements:
  * North arrow
  * Scale bar
  * Legend
  * Inset maps for context
- Ensure clear labels and annotations

Step 4: Create statistical plots
Use seaborn and matplotlib:
- Scatter plots with regression lines
- Box plots, violin plots for distributions
- Bar plots with error bars
- Heatmaps for correlations
- Time series plots
- Add statistical annotations:
  * P-values
  * Confidence intervals
  * Significance stars
- Use consistent styling across figures

Step 5: Create interactive visualizations with Plotly
- Create interactive maps:
  * Hover information
  * Zoom and pan
  * Layer controls
  * Time slider for temporal data
- Create interactive charts:
  * Linked brushing
  * Dynamic filtering
  * Hover tooltips with details
- Export static versions for paper

Step 6: Generate supplementary diagrams
Use generate-image for:
- Methodological workflow diagrams
- Conceptual framework figures
- System architecture diagrams
- Process flowcharts
- Iterate on prompts to refine quality

Step 7: Apply journal-specific styling
- Use journal's template or style guide
- Set consistent font sizes:
  * Title: 12-14 pt
  * Labels: 10-12 pt
  * Text: 8-10 pt
- Use consistent line weights
- Ensure accessibility (colorblind-safe)
- Check contrast ratios

Step 8: Multi-panel figures
- Create composite figures using subplots
- Align axes and labels
- Share legends when appropriate
- Label subpanels (A, B, C, etc.)
- Maintain consistent scale
- Ensure readability at reduced size

Step 9: Color and accessibility
- Use colorblind-safe palettes
- Test with color vision simulators
- Provide alternative representations:
  * Patterns in addition to colors
  * Text labels for clarity
- Document color choices for reproducibility

Step 10: Quality control
- Check resolution and file sizes
- Verify text is legible at print size
- Ensure no clipping or overlap
- Test in color and grayscale
- Review with colleagues
- Make final revisions

Step 11: Export and organize figures
- Export in required formats:
  * TIFF (300 DPI, LZW compression)
  * PDF (vector for crisp text)
  * PNG (150 DPI for web)
- Organize files by figure number
- Create version control (v1, v2, final)
- Generate figure list for manuscript

Step 12: Create presentation figures
- Adapt figures for slides using scientific-slides:
  * Larger fonts (18-24 pt)
  * Simpler designs
  * Less clutter
  * Animated transitions for time series
  * Interactive elements if possible
- Create supplementary online materials

Step 13: Figure captions and legends
- Write clear, descriptive captions
- Include enough information to understand figure
- Explain symbols, abbreviations
- Note statistical significance where relevant
- Follow journal style guidelines

Step 14: Create supplementary materials
- High-resolution versions
- Interactive web versions (Plotly HTML)
- Additional analysis figures
- Code and data repositories
- Color versions and grayscale versions

Expected Output:
- Complete set of publication-ready figures
- Interactive visualizations
- Figure captions and legends
- Supplementary materials
- Presentation versions of key figures
- Compliance with journal requirements
```

---

## Best Practices for GeoAI Workflows

### 1. Data Management
- Always document data sources and processing steps
- Use version control for code and workflows
- Maintain data lineage and provenance
- Store intermediate results for reproducibility

### 2. Spatial Considerations
- Always check spatial autocorrelation
- Use appropriate coordinate reference systems
- Be aware of scale effects
- Consider edge effects and boundary issues

### 3. Model Evaluation
- Use spatial cross-validation to avoid leakage
- Evaluate both accuracy and spatial patterns
- Validate with independent test regions
- Assess model interpretability

### 4. Visualization
- Use colorblind-safe palettes
- Include map elements (scale, legend, north arrow)
- Ensure readability at intended size
- Provide both static and interactive versions

### 5. Reproducibility
- Document random seeds and parameters
- Share code and data where possible
- Use containerization for environment consistency
- Provide detailed methods descriptions

### 6. Computational Efficiency
- Assess resource requirements before starting
- Use appropriate chunking strategies
- Leverage distributed computing for large datasets
- Optimize I/O operations

---

This examples document provides practical, end-to-end workflows that combine multiple GeoAI skills to solve real-world problems. Each example demonstrates:
- Clear objectives and expected outcomes
- Specific skills used and their integration
- Detailed step-by-step workflows
- Comprehensive outputs and deliverables
