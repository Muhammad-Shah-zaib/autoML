# AutoML Classification System

An end-to-end automated machine learning system for classification tasks built with Streamlit. This system simplifies the entire ML workflow from data upload to model evaluation and reporting.

## Table of Contents

- [Overview](#overview)
- [System Features](#system-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Application Workflow](#application-workflow)
- [Technical Details](#technical-details)
- [Project Structure](#project-structure)
- [Models and Algorithms](#models-and-algorithms)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Contributors](#contributors)
- [License](#license)

## Overview

We developed this AutoML system to automate and streamline classification tasks for machine learning practitioners. The system handles the complete workflow including exploratory data analysis, data quality issue detection, preprocessing, model training, and performance comparison.

The application provides an interactive interface where users can upload their datasets, configure preprocessing steps, train multiple classification models simultaneously, and compare results through comprehensive visualizations and reports.

## System Features

## System Features

### Dataset Management
We implemented comprehensive dataset handling capabilities including:
- CSV file upload with automatic encoding detection (UTF-8 and Latin-1 support)
- Built-in sample datasets for testing (Iris, Wine Quality)
- Dataset statistics and metadata display
- Target column selection interface
- Class distribution visualization and analysis

### Exploratory Data Analysis
Our EDA module performs automated analysis including:
- Missing value detection with per-feature and global statistics
- Outlier identification using IQR and Z-score methods
- Correlation analysis with interactive heatmaps
- Distribution analysis through histograms and Q-Q plots
- Box plots for numerical features
- Categorical feature analysis with bar charts
- Configurable train-test split ratios with stratification

### Data Quality Issue Detection
The system automatically detects six categories of data quality issues:
- Missing values in features
- Outliers in numerical columns
- Class imbalance in target variable
- High cardinality categorical features
- Constant or near-constant features
- Duplicate rows in the dataset

For each detected issue, we provide detailed warnings, suggest appropriate fixes, and request user confirmation before applying any transformations.

### Preprocessing Pipeline
We built a comprehensive preprocessing system that handles:
- Missing value treatment with multiple imputation strategies (mean, median, mode, constant)
- Outlier handling through removal, capping (Winsorization), or retention
- Feature scaling using StandardScaler or MinMaxScaler
- Categorical encoding via One-Hot Encoding or Label Encoding
- Class balancing through SMOTE oversampling, class weights adjustment, or random undersampling
- Automated train-test split with configurable ratios and stratification

### Model Training and Optimization
Our system trains seven different classification algorithms:
1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Naive Bayes
5. Random Forest
6. Support Vector Machine
7. Rule-Based Classifier (custom implementation)

We implemented two hyperparameter optimization strategies:
- Grid Search Cross-Validation for exhaustive parameter search
- Randomized Search Cross-Validation for faster optimization
- Configurable cross-validation fold numbers

### Model Evaluation
Each trained model provides comprehensive evaluation metrics:
- Accuracy score
- Precision, Recall, and F1-Score (macro and weighted averages)
- Confusion Matrix (raw counts and normalized)
- ROC-AUC score for binary classification
- Matthews Correlation Coefficient
- Training time measurement
- Detailed classification report

### Model Comparison
We developed interactive comparison dashboards featuring:
- Side-by-side metric comparison bar charts
- Radar charts for multi-metric visualization
- Training time analysis plots
- Confusion matrix comparisons
- ROC curves with AUC scores
- Sortable comparison tables
- CSV export functionality

### Report Generation
The system generates comprehensive reports including:
- Dataset overview and statistical summary
- EDA findings and visualizations
- Detected issues and applied resolutions
- Preprocessing decisions and transformations
- Model configurations and hyperparameters
- Performance comparison tables
- Best model identification with justification
- Recommendations for model improvement

Reports can be exported in multiple formats:
- Markdown (.md)
- HTML (styled and formatted)
- CSV (tabular results only)
- PDF (planned feature)

## Installation

### Prerequisites

Before installation, ensure you have the following installed on your system:
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Setup Instructions

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Muhammad-Shah-zaib/autoML.git
   cd autoML
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Verify installation by running the application:
   ```bash
   streamlit run app.py
   ```

The application will launch in your default web browser at `http://localhost:8501`

### Dependencies

Our system requires the following main packages:
- streamlit >= 1.39.0
- pandas >= 2.2.0
- numpy >= 2.1.0
- scikit-learn >= 1.6.0, < 1.8.0
- imbalanced-learn >= 0.14.0
- matplotlib >= 3.8.1
- seaborn >= 0.13.2
- plotly >= 5.21.0
- scipy >= 1.11.2
- reportlab >= 4.0.0

All dependencies are listed in `requirements.txt` and will be installed automatically.

## Getting Started

### Quick Start Guide

1. Launch the application using `streamlit run app.py`
2. The interface will open in your browser
3. Follow the guided seven-step workflow:
   - Upload Dataset
   - Perform EDA
   - Detect and Fix Issues
   - Configure Preprocessing
   - Train Models
   - Compare Results
   - Generate Report

### Using Sample Datasets

If you want to test the system without your own data, we provide sample datasets:
- Iris Dataset: Classic classification problem with 3 classes
- Wine Quality Dataset: Multi-class wine quality classification

Select these from the Dataset Upload page to get started immediately.

## Application Workflow

## Application Workflow

Our system implements a guided seven-step workflow to ensure proper sequence and data validation:

### Step 1: Dataset Upload
- Upload your CSV file through the file uploader
- Alternatively, select from built-in sample datasets
- System performs automatic encoding detection
- Select the target column for classification
- View initial dataset statistics and class distribution
- Progress to EDA once dataset is validated

### Step 2: Exploratory Data Analysis
- Review missing value statistics and visualizations
- Examine outlier detection results using IQR and Z-score methods
- Analyze feature correlations through heatmaps
- Inspect distribution plots for numerical features
- View categorical feature distributions
- Configure train-test split ratio (default 80-20)
- Move to issue detection after completing analysis

### Step 3: Issue Detection and Fixing
- System automatically scans for six types of data quality issues
- Review each detected issue with detailed statistics
- Select appropriate fixes for each issue type:
  - Missing values: Choose imputation method
  - Outliers: Select handling strategy
  - Class imbalance: Pick balancing technique
  - High cardinality: Decide on feature treatment
  - Constant features: Confirm removal
  - Duplicate rows: Approve deletion
- Confirm all decisions before proceeding
- Continue to preprocessing step

### Step 4: Data Preprocessing
- Configure preprocessing parameters based on detected issues
- Select imputation strategy for missing values
- Choose outlier handling method
- Pick feature scaling technique
- Select encoding method for categorical variables
- Configure class balancing if needed
- Review preprocessing summary
- System applies all transformations
- Proceed to model training

### Step 5: Model Training
- Choose hyperparameter optimization method (Grid Search or Randomized Search)
- Configure number of cross-validation folds
- Select which models to train from the seven available algorithms
- Initiate training process
- Monitor progress with real-time updates
- View individual model results as training completes
- Review training time and performance metrics
- Continue to model comparison

### Step 6: Model Comparison
- Access comprehensive comparison dashboard
- View side-by-side metric comparisons
- Examine confusion matrices for all models
- Analyze ROC curves for binary classification
- Study radar charts for multi-metric evaluation
- Compare training times across models
- Export comparison results to CSV
- Proceed to report generation

### Step 7: Final Report
- Generate comprehensive project report
- Review all workflow steps and decisions
- Examine model performance summaries
- Read recommendations for improvement
- Download report in preferred format (HTML, Markdown, or CSV)
- Option to start new analysis or return to previous steps

## Technical Details

### Architecture

We designed the system using a modular architecture with clear separation of concerns:
- Main application orchestrator (app.py)
- Specialized utility modules for each workflow component
- Session state management for data persistence
- Cached computations for performance optimization

### Data Processing Pipeline

Our data processing follows these stages:
1. Input validation and encoding detection
2. Exploratory analysis with statistical computations
3. Issue detection using rule-based and statistical methods
4. User-guided preprocessing configuration
5. Transformation application with error handling
6. Train-test split with stratification
7. Feature and target separation

### Model Training Process

We implemented the following training workflow:
1. User selects models and optimization parameters
2. System initializes selected algorithms with default hyperparameters
3. Grid Search or Randomized Search performs hyperparameter optimization
4. Cross-validation evaluates each parameter combination
5. Best parameters selected based on validation performance
6. Final model trained on complete training set
7. Predictions generated on test set
8. Comprehensive metrics calculated and stored

### Performance Optimizations

To ensure responsive user experience, we implemented:
- Streamlit caching for expensive computations
- Lazy loading of visualization components
- Efficient data structures for large datasets
- Progress indicators for long-running operations
- Background computation where possible

### Error Handling

The system includes comprehensive error handling:
- Input validation at each workflow step
- Graceful degradation for missing or invalid data
- User-friendly error messages with correction guidance
- Automatic recovery from non-critical errors
- Logging of issues for debugging

## Project Structure

```
autoML/
├── app.py                          Main Streamlit application
├── requirements.txt                Python package dependencies
├── README.md                       Project documentation
├── QUICKSTART.md                   Quick start guide
├── generate_sample_data.py         Sample dataset generator
├── generate_test_dataset.py        Test dataset generator
└── utils/                          Utility modules directory
    ├── __init__.py                 Package initialization
    ├── data_loader.py              Dataset loading and display
    ├── eda.py                      Exploratory data analysis
    ├── issue_detector.py           Data quality issue detection
    ├── preprocessor.py             Data preprocessing pipeline
    ├── model_trainer.py            Model training and optimization
    ├── model_comparison.py         Model comparison and visualization
    └── report_generator.py         Report generation and export
```

### Module Descriptions

**app.py**: Main application file that orchestrates the workflow, manages session state, handles navigation, and coordinates between different modules.

**data_loader.py**: Handles CSV file loading with automatic encoding detection, displays dataset information, and provides basic statistics.

**eda.py**: Performs comprehensive exploratory data analysis including missing value analysis, outlier detection, correlation analysis, and distribution visualization.

**issue_detector.py**: Implements rule-based detection for six types of data quality issues, provides fix suggestions, and manages user confirmation workflow.

**preprocessor.py**: Applies user-selected preprocessing transformations including imputation, scaling, encoding, outlier handling, and class balancing.

**model_trainer.py**: Trains classification models with hyperparameter optimization using Grid Search or Randomized Search, calculates performance metrics, and stores results.

**model_comparison.py**: Creates interactive visualizations for model comparison including bar charts, radar charts, confusion matrices, and ROC curves.

**report_generator.py**: Generates comprehensive reports in multiple formats, summarizing the entire workflow from data upload to model evaluation.

## Models and Algorithms

```
AutoML/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── generate_sample_data.py     # Sample data generator
├── data/                       # Sample datasets
│   └── sample_titanic.csv
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and info display
│   ├── eda.py                 # Exploratory data analysis
│   ├── issue_detector.py      # Issue detection and handling
│   ├── preprocessor.py        # Data preprocessing
│   ├── model_trainer.py       # Model training and optimization
│   ├── model_comparison.py    # Model comparison and visualization
│   └── report_generator.py    # Report generation
└── screenshots/                # Application screenshots (to be added)
```

## Models and Algorithms

We implemented seven classification algorithms with appropriate hyperparameter grids:

### 1. Logistic Regression
Linear model suitable for binary and multiclass classification problems.

**Hyperparameters we optimize:**
- C: Inverse of regularization strength (0.01, 0.1, 1, 10, 100)
- penalty: Regularization type (l1, l2)
- solver: Optimization algorithm (liblinear, saga)

**Strengths:**
- Fast training and prediction
- Interpretable coefficients
- Works well with linearly separable data
- Probabilistic predictions

**Limitations:**
- Limited to linear decision boundaries
- May underfit complex patterns
- Sensitive to feature scaling

### 2. K-Nearest Neighbors
Instance-based learning algorithm that classifies based on nearest training examples.

**Hyperparameters we optimize:**
- n_neighbors: Number of neighbors to consider (3, 5, 7, 9, 11)
- weights: Weight function (uniform, distance)
- metric: Distance metric (euclidean, manhattan)

**Strengths:**
- No training phase
- Works well with small datasets
- Naturally handles multi-class problems
- Non-parametric approach

**Limitations:**
- Slow prediction on large datasets
- Sensitive to feature scaling
- Suffers from curse of dimensionality
- Memory intensive

### 3. Decision Tree
Tree-based model that learns decision rules from features.

**Hyperparameters we optimize:**
- max_depth: Maximum tree depth (3, 5, 7, 10, None)
- min_samples_split: Minimum samples to split node (2, 5, 10)
- criterion: Split quality measure (gini, entropy)

**Strengths:**
- Highly interpretable
- Handles non-linear relationships
- No feature scaling needed
- Handles mixed data types

**Limitations:**
- Prone to overfitting
- Unstable with small data changes
- Biased toward dominant classes
- Cannot extrapolate

### 4. Naive Bayes
Probabilistic classifier based on Bayes theorem with feature independence assumption.

**Hyperparameters we optimize:**
- var_smoothing: Portion of largest variance added for stability (1e-9 to 1e-7)

**Strengths:**
- Fast training and prediction
- Works well with small datasets
- Handles high-dimensional data
- Probabilistic predictions

**Limitations:**
- Assumes feature independence
- Sensitive to feature distributions
- May underperform with correlated features
- Requires sufficient training data per class

### 5. Random Forest
Ensemble of decision trees that reduces overfitting through bagging.

**Hyperparameters we optimize:**
- n_estimators: Number of trees (50, 100, 200)
- max_depth: Maximum tree depth (5, 10, 20, None)
- max_features: Features per split (sqrt, log2)
- min_samples_split: Minimum samples to split (2, 5)

**Strengths:**
- High accuracy on many problems
- Reduces overfitting compared to single tree
- Provides feature importance
- Handles non-linear relationships

**Limitations:**
- Less interpretable than single tree
- Slower training than simple models
- Can be memory intensive
- May overfit noisy datasets

### 6. Support Vector Machine
Maximum margin classifier that finds optimal separating hyperplane.

**Hyperparameters we optimize:**
- C: Regularization parameter (0.1, 1, 10)
- kernel: Kernel function (linear, rbf, poly)
- gamma: Kernel coefficient (scale, auto)

**Strengths:**
- Effective in high-dimensional spaces
- Memory efficient using support vectors
- Versatile through different kernels
- Works well with clear margins

**Limitations:**
- Slow training on large datasets
- Sensitive to hyperparameter selection
- No direct probability estimates
- Difficult to interpret

### 7. Rule-Based Classifier
Custom implementation using simple decision rules based on feature statistics.

**Hyperparameters:**
- None (uses deterministic rules)

**Strengths:**
- Extremely interpretable
- Fast predictions
- No hyperparameter tuning needed
- Good baseline model

**Limitations:**
- Limited accuracy
- Cannot capture complex patterns
- Fixed rule structure
- No learning of optimal rules

## Configuration

## Configuration

### Preprocessing Configuration

Users can configure preprocessing through the interactive interface:

**Missing Value Imputation:**
- Mean imputation for numerical features
- Median imputation for skewed distributions
- Mode imputation for categorical features
- Constant value imputation with user-defined value

**Outlier Handling:**
- Remove outliers completely
- Cap outliers using Winsorization
- Keep outliers without modification

**Feature Scaling:**
- StandardScaler (mean=0, std=1)
- MinMaxScaler (range 0-1)
- No scaling

**Categorical Encoding:**
- One-Hot Encoding for nominal features
- Label Encoding for ordinal features

**Class Balancing:**
- SMOTE oversampling of minority classes
- Random undersampling of majority classes
- Class weight adjustment in algorithms
- No balancing

### Model Training Configuration

**Optimization Method:**
- Grid Search: Exhaustive search over parameter grid
- Randomized Search: Random sampling of parameters (faster)

**Cross-Validation:**
- Configurable number of folds (3, 5, or 10)
- Stratified splitting to preserve class distributions

**Model Selection:**
- Select any combination of the seven available models
- Option to train all models simultaneously

### Application Settings

The application uses Streamlit configuration:
- Page title: "AutoML Classification System"
- Layout: Wide mode for better visualization
- Sidebar: Expanded by default for easy navigation

## Deployment

### Local Deployment

For local deployment on your machine:

1. Ensure all dependencies are installed via `pip install -r requirements.txt`
2. Run `streamlit run app.py` from the project directory
3. Access the application at `http://localhost:8501`
4. For network access, use `streamlit run app.py --server.address=0.0.0.0`

### Cloud Deployment on Streamlit Cloud

We can deploy this application to Streamlit Cloud for public access:

1. Push the repository to GitHub:
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

3. Click "New app" and configure:
   - Repository: Select your AutoML repository
   - Branch: main
   - Main file path: app.py

4. Click "Deploy" and wait for build completion

5. Application will be available at `https://[your-app-name].streamlit.app`

### Deployment Requirements

Ensure your repository includes:
- requirements.txt with all dependencies
- Python 3.8+ compatibility
- All utility modules in utils/ directory
- No hardcoded file paths
- No local file system dependencies

### Environment Variables

If needed, configure environment variables in Streamlit Cloud:
- Go to app settings
- Add secrets in TOML format
- Access via `st.secrets` in code

### Performance Considerations

For production deployment:
- Consider dataset size limits (Streamlit has memory constraints)
- Implement data caching for frequently accessed datasets
- Optimize model training for timeouts
- Consider using pre-trained models for large datasets
- Monitor application metrics through Streamlit dashboard

## Best Practices

### Data Preparation

We recommend the following before uploading datasets:
- Clean column names (remove special characters)
- Ensure consistent data types in columns
- Remove or handle extreme outliers manually if known
- Verify target column has appropriate class labels
- Check for data leakage features

### Model Selection

Choose models based on your requirements:
- Small datasets: Naive Bayes, KNN, Decision Tree
- Large datasets: Logistic Regression, Random Forest
- High accuracy needed: Random Forest, SVM
- Interpretability required: Decision Tree, Logistic Regression, Rule-Based
- Fast predictions needed: Logistic Regression, Naive Bayes

### Hyperparameter Optimization

Select optimization method based on constraints:
- Grid Search: When parameter space is small and accuracy is critical
- Randomized Search: When parameter space is large or time is limited
- More CV folds: Better generalization but slower training
- Fewer CV folds: Faster training but potentially less reliable

### Performance Tips

To optimize system performance:
- Use Randomized Search for initial exploration
- Reduce CV folds for very large datasets
- Select fewer models initially to reduce training time
- Consider class balancing carefully (SMOTE can be slow)
- Export and save best models for reuse

## Troubleshooting

### Common Issues and Solutions

**Issue: Out of memory errors during training**
- Solution: Reduce dataset size or use sampling
- Solution: Select fewer models to train
- Solution: Use Randomized Search with fewer iterations

**Issue: Slow application performance**
- Solution: Clear browser cache and restart application
- Solution: Check dataset size (recommend under 10MB)
- Solution: Reduce number of CV folds

**Issue: Models not training**
- Solution: Verify preprocessing completed successfully
- Solution: Check for NaN values in processed data
- Solution: Ensure target column has valid labels

**Issue: Poor model performance**
- Solution: Review EDA for data quality issues
- Solution: Try different preprocessing strategies
- Solution: Ensure sufficient data for each class
- Solution: Consider feature engineering

**Issue: Application crashes on certain datasets**
- Solution: Check for special characters in column names
- Solution: Verify CSV encoding (UTF-8 recommended)
- Solution: Remove completely empty columns
- Solution: Check for mixed data types in columns

## Contributors

This project was developed as a collaborative effort by our team. We worked together on:
- System architecture and design
- Feature implementation and testing
- Documentation and user guides
- Bug fixes and optimizations

We welcome contributions from the community. If you would like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request with a clear description

Please ensure your code follows the existing style and includes appropriate documentation.

## Future Enhancements

We are considering the following enhancements for future versions:

**Model Improvements:**
- Add deep learning models (neural networks)
- Implement ensemble methods (stacking, blending)
- Add gradient boosting algorithms (XGBoost, LightGBM, CatBoost)
- Support for regression tasks

**Feature Engineering:**
- Automated feature selection
- Polynomial feature generation
- Feature interaction detection
- Dimensionality reduction (PCA, t-SNE)

**Advanced Functionality:**
- Model persistence and loading
- Batch prediction on new data
- API endpoint for predictions
- Real-time model monitoring
- AutoML pipeline export as Python script

**User Experience:**
- Enhanced visualizations
- Custom theme support
- Progress saving and loading
- Comparison across multiple runs
- Interactive feature importance plots

**Performance:**
- Parallel model training
- Incremental learning for large datasets
- Model compression techniques
- GPU acceleration support

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software in accordance with the license terms.

## Acknowledgments

We would like to acknowledge the following open-source projects that made this work possible:

- Streamlit for providing an excellent framework for building data applications
- Scikit-learn for comprehensive machine learning algorithms and utilities
- Pandas and NumPy for efficient data manipulation
- Matplotlib, Seaborn, and Plotly for visualization capabilities
- Imbalanced-learn for handling imbalanced datasets
- The entire open-source community for continuous support and development

