
# 🤖 AutoML Classification System

An intelligent, end-to-end automated machine learning system for classification tasks built with Streamlit.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models & Algorithms](#models--algorithms)
- [Screenshots](#screenshots)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This AutoML system automates the entire machine learning workflow for classification problems, from data upload to model deployment. It features:

- **Intelligent Data Analysis**: Automatic EDA with comprehensive visualizations
- **Smart Issue Detection**: Identifies data quality problems and suggests fixes
- **Automated Preprocessing**: Handles missing values, outliers, encoding, and scaling
- **Multi-Model Training**: Trains 7 different classification algorithms
- **Hyperparameter Optimization**: Grid Search and Randomized Search support
- **Interactive Dashboards**: Beautiful visualizations for model comparison
- **Report Generation**: Downloadable reports in HTML, Markdown, and CSV formats

## ✨ Features

### 📁 Dataset Management
- CSV file upload
- Support for built-in sample datasets (Iris, Wine, Titanic)
- Comprehensive dataset statistics and metadata
- Class distribution analysis

### 📊 Exploratory Data Analysis
- **Missing Value Analysis**: Per-feature and global missing data detection
- **Outlier Detection**: IQR and Z-score methods
- **Correlation Analysis**: Heatmaps and correlation matrices
- **Distribution Plots**: Histograms, Q-Q plots, and box plots
- **Categorical Analysis**: Bar charts and cross-tabulation
- **Train/Test Split**: Configurable split ratios

### ⚠️ Intelligent Issue Detection
The system automatically detects and flags:
- Missing values
- Outliers
- Class imbalance
- High cardinality features
- Constant or near-constant features
- Duplicate rows

For each issue, the system:
1. Shows a detailed warning message
2. Suggests appropriate fixes
3. Asks for user confirmation before applying changes

### ⚙️ Preprocessing Pipeline
- **Missing Value Treatment**: Mean, median, mode, or constant imputation
- **Outlier Handling**: Removal, capping (Winsorization), or retention
- **Feature Scaling**: StandardScaler or MinMaxScaler
- **Encoding**: One-Hot or Label Encoding
- **Class Balancing**: SMOTE, class weights, or undersampling
- **Train-Test Split**: User-configurable with stratification

### 🎯 Model Training

The system trains and optimizes 7 classification algorithms:

1. **Logistic Regression** - Linear model for binary and multiclass
2. **K-Nearest Neighbors** - Instance-based learning
3. **Decision Tree** - Rule-based tree classifier
4. **Naive Bayes** - Probabilistic classifier
5. **Random Forest** - Ensemble of decision trees
6. **Support Vector Machine** - Maximum margin classifier
7. **Rule-Based Classifier** - Custom rule-based approach

**Optimization Methods:**
- Grid Search CV
- Randomized Search CV
- Configurable cross-validation folds

### 📈 Model Evaluation

Each model reports:
- ✅ Accuracy
- 📊 Precision, Recall, F1-Score
- 🎯 Confusion Matrix
- 📈 ROC-AUC (binary classification)
- ⏱️ Training Time
- 📑 Detailed classification report

### 📊 Model Comparison Dashboard

Interactive visualizations include:
- Metrics comparison bar charts
- Radar charts for multi-metric comparison
- Training time analysis
- Confusion matrices (raw and normalized)
- ROC curves with AUC scores
- Downloadable comparison table (CSV)

### 📄 Auto-Generated Reports

Comprehensive reports include:
1. Dataset overview and statistics
2. EDA findings and insights
3. Detected issues and resolutions
4. Preprocessing decisions and steps
5. Model configurations and hyperparameters
6. Performance comparison tables
7. Best model summary with justification
8. Recommendations for improvement

**Export Formats:**
- 📝 Markdown (.md)
- 🌐 HTML (styled)
- 📊 CSV (results only)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd AutoML
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate sample data** (optional)
   ```bash
   python generate_sample_data.py
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The application will open in your default browser at `http://localhost:8501`

## 📖 Usage

### Step-by-Step Guide

1. **Upload Dataset**
   - Navigate to "📁 Dataset Upload"
   - Upload your CSV file or select a sample dataset
   - Choose your target column

2. **Explore Data**
   - Go to "📊 Exploratory Data Analysis"
   - Review missing values, outliers, correlations, and distributions
   - Configure train/test split ratio

3. **Detect & Fix Issues**
   - Visit "⚠️ Issue Detection & Fixing"
   - Review detected issues
   - Select fixes for each issue
   - Confirm your decisions

4. **Preprocess Data**
   - Navigate to "⚙️ Preprocessing"
   - Configure preprocessing options
   - Review preprocessing summary

5. **Train Models**
   - Go to "🎯 Model Training"
   - Select optimization method
   - Choose models to train
   - Click "Start Training"

6. **Compare Results**
   - Visit "📈 Model Comparison"
   - Explore interactive visualizations
   - Download comparison results

7. **Generate Report**
   - Navigate to "📄 Final Report"
   - Review the comprehensive report
   - Download in your preferred format

## 📁 Project Structure

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

## 🤖 Models & Algorithms

### Logistic Regression
- **Use Case**: Binary and multiclass classification
- **Hyperparameters**: C, penalty, solver
- **Pros**: Fast, interpretable, works well for linearly separable data
- **Cons**: Limited to linear decision boundaries

### K-Nearest Neighbors
- **Use Case**: Non-parametric classification
- **Hyperparameters**: n_neighbors, weights, metric
- **Pros**: No training phase, works well with small datasets
- **Cons**: Slow prediction, sensitive to feature scaling

### Decision Tree
- **Use Case**: Non-linear classification, feature importance
- **Hyperparameters**: max_depth, min_samples_split, criterion
- **Pros**: Interpretable, handles non-linear relationships
- **Cons**: Prone to overfitting

### Naive Bayes
- **Use Case**: Text classification, probabilistic predictions
- **Hyperparameters**: var_smoothing
- **Pros**: Fast, works well with small datasets
- **Cons**: Assumes feature independence

### Random Forest
- **Use Case**: Complex patterns, reduces overfitting
- **Hyperparameters**: n_estimators, max_depth, max_features
- **Pros**: High accuracy, handles non-linearity
- **Cons**: Less interpretable, slower training

### Support Vector Machine
- **Use Case**: High-dimensional data, complex boundaries
- **Hyperparameters**: C, kernel, gamma
- **Pros**: Effective in high dimensions, memory efficient
- **Cons**: Slow on large datasets

### Rule-Based Classifier
- **Use Case**: Baseline model, interpretable rules
- **Hyperparameters**: None (custom implementation)
- **Pros**: Extremely interpretable
- **Cons**: Limited accuracy

## 📸 Screenshots

*(Add screenshots of your application here)*

### Dataset Upload
![Dataset Upload Screenshot]

### EDA Dashboard
![EDA Screenshot]

### Model Comparison
![Model Comparison Screenshot]

### Report Generation
![Report Screenshot]

## 🌐 Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Click "Deploy"

3. **Configuration**
   - Ensure `requirements.txt` is in the root directory
   - Python version: 3.8+
   - No additional packages needed

### Hosted App Link

🔗 **Live Demo**: [Your Streamlit Cloud URL]

## 🛠️ Development

### Adding New Models

1. Add model class to `utils/model_trainer.py`
2. Define hyperparameter grid
3. Update model selection in UI
4. Test with sample data

### Customizing Preprocessing

Edit `utils/preprocessor.py` to add:
- Custom imputation strategies
- Additional encoding methods
- Feature engineering steps

## 📊 Performance Tips

1. **Large Datasets**: Use Randomized Search instead of Grid Search
2. **Memory Issues**: Reduce cross-validation folds
3. **Speed**: Disable SMOTE for very large datasets
4. **Accuracy**: Try ensemble methods and feature engineering

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Your Name** - Initial work

## 🙏 Acknowledgments

- Scikit-learn for machine learning algorithms
- Streamlit for the amazing web framework
- Plotly and Seaborn for visualizations
- The open-source community

## 📧 Contact

For questions or feedback, please reach out:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

**Built with ❤️ using Streamlit**
