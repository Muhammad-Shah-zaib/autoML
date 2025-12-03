"""
Report Generation Module
Generate comprehensive final reports
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

def generate_report(dataset, target_column, detected_issues, preprocessing_decisions, model_results):
    """
    Generate auto-generated final report
    
    Args:
        dataset: Original dataset
        target_column: Target column name
        detected_issues: Detected issues dictionary
        preprocessing_decisions: Preprocessing decisions
        model_results: Model training results
    """
    st.subheader("📄 Auto-Generated Final Report")
    
    # Report format selection
    report_format = st.radio(
        "Select report format:",
        ["HTML", "Markdown", "PDF (via HTML)"],
        horizontal=True
    )
    
    # Generate report content
    report_content = generate_report_content(
        dataset, target_column, detected_issues, 
        preprocessing_decisions, model_results
    )
    
    # Display report preview
    st.markdown("---")
    st.subheader("📋 Report Preview")
    
    with st.expander("View Full Report", expanded=True):
        st.markdown(report_content)
    
    # Download options
    st.markdown("---")
    st.subheader("💾 Download Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Markdown download
        st.download_button(
            label="📥 Download as Markdown",
            data=report_content,
            file_name=f"automl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    with col2:
        # HTML download
        html_content = markdown_to_html(report_content)
        st.download_button(
            label="📥 Download as HTML",
            data=html_content,
            file_name=f"automl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )
    
    with col3:
        # CSV download (results only)
        csv_content = generate_results_csv(model_results)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_content,
            file_name=f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def generate_report_content(dataset, target_column, detected_issues, 
                           preprocessing_decisions, model_results):
    """Generate the report content in Markdown format"""
    
    report = f"""
# AutoML Classification Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Executive Summary

This report presents the results of an automated machine learning pipeline for classification tasks. 
The system analyzed the dataset, detected data quality issues, applied preprocessing steps, 
trained multiple classification models, and identified the best-performing model.

### Key Findings

"""
    
    # Best model summary
    if model_results:
        best_model = max(model_results.items(), key=lambda x: x[1]['f1_score'])
        best_model_name = best_model[0]
        best_model_metrics = best_model[1]
        
        report += f"""
**Best Model:** {best_model_name}
- **Accuracy:** {best_model_metrics['accuracy']:.4f}
- **Precision:** {best_model_metrics['precision']:.4f}
- **Recall:** {best_model_metrics['recall']:.4f}
- **F1-Score:** {best_model_metrics['f1_score']:.4f}
"""
        
        if best_model_metrics['roc_auc']:
            report += f"- **ROC-AUC:** {best_model_metrics['roc_auc']:.4f}\n"
        
        report += f"- **Training Time:** {best_model_metrics['training_time']:.2f} seconds\n"
    
    # Dataset overview
    report += f"""

---

## 2. Dataset Overview

### Basic Information

- **Total Samples:** {len(dataset):,}
- **Total Features:** {len(dataset.columns) - 1}
- **Target Variable:** {target_column}
- **Number of Classes:** {dataset[target_column].nunique()}

### Feature Types

"""
    
    numerical_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    report += f"- **Numerical Features:** {len(numerical_cols)}\n"
    report += f"- **Categorical Features:** {len(categorical_cols)}\n\n"
    
    # Class distribution
    report += "### Class Distribution\n\n"
    class_dist = dataset[target_column].value_counts()
    
    report += "| Class | Count | Percentage |\n"
    report += "|-------|-------|------------|\n"
    
    for cls, count in class_dist.items():
        pct = (count / len(dataset)) * 100
        report += f"| {cls} | {count} | {pct:.2f}% |\n"
    
    # EDA findings
    report += f"""

---

## 3. Exploratory Data Analysis

### Missing Values

"""
    
    missing_values = detected_issues.get('missing_values', [])
    if missing_values:
        report += f"Found missing values in {len(missing_values)} feature(s):\n\n"
        report += "| Feature | Missing Count | Missing % |\n"
        report += "|---------|---------------|------------|\n"
        
        for mv in missing_values:
            report += f"| {mv['column']} | {mv['missing_count']} | {mv['missing_percentage']:.2f}% |\n"
    else:
        report += "✅ No missing values detected.\n"
    
    report += "\n### Outliers\n\n"
    
    outliers = detected_issues.get('outliers', [])
    if outliers:
        report += f"Detected outliers in {len(outliers)} feature(s) using IQR method:\n\n"
        report += "| Feature | Outlier Count | Outlier % |\n"
        report += "|---------|---------------|------------|\n"
        
        for out in outliers:
            report += f"| {out['column']} | {out['outlier_count']} | {out['outlier_percentage']:.2f}% |\n"
    else:
        report += "✅ No significant outliers detected.\n"
    
    report += "\n### Class Imbalance\n\n"
    
    imbalance = detected_issues.get('class_imbalance')
    if imbalance and imbalance.get('is_imbalanced'):
        report += f"⚠️ Class imbalance detected with ratio {imbalance['imbalance_ratio']:.2f}:1\n"
    else:
        report += "✅ Classes are relatively balanced.\n"
    
    # Detected issues
    report += f"""

---

## 4. Detected Issues & Resolutions

The following data quality issues were identified and addressed:

"""
    
    if preprocessing_decisions:
        for decision, action in preprocessing_decisions.items():
            report += f"- **{decision.replace('_', ' ').title()}:** {action}\n"
    else:
        report += "No major issues required intervention.\n"
    
    # Preprocessing steps
    report += f"""

---

## 5. Preprocessing Pipeline

The following preprocessing steps were applied:

"""
    
    preprocessing_steps = [
        "1. **Data Cleaning:** Removed duplicates and constant features",
        "2. **Missing Value Treatment:** Applied imputation strategies",
        "3. **Outlier Handling:** Capped or removed outliers as specified",
        "4. **Feature Encoding:** Encoded categorical variables",
        "5. **Feature Scaling:** Standardized numerical features",
        "6. **Train-Test Split:** Split dataset for model validation"
    ]
    
    for  step in preprocessing_steps:
        report += f"{step}\n"
    
    # Model configurations
    report += f"""

---

## 6. Model Training & Hyperparameter Optimization

### Trained Models

The following classification models were trained with hyperparameter optimization:

"""
    
    if model_results:
        for i, (model_name, results) in enumerate(model_results.items(), 1):
            report += f"\n#### {i}. {model_name}\n\n"
            
            report += "**Best Hyperparameters:**\n\n"
            for param, value in results['best_params'].items():
                report += f"- {param}: {value}\n"
            
            report += f"\n**Performance Metrics:**\n\n"
            report += f"- Accuracy: {results['accuracy']:.4f}\n"
            report += f"- Precision: {results['precision']:.4f}\n"
            report += f"- Recall: {results['recall']:.4f}\n"
            report += f"- F1-Score: {results['f1_score']:.4f}\n"
            
            if results['roc_auc']:
                report += f"- ROC-AUC: {results['roc_auc']:.4f}\n"
            
            report += f"- Training Time: {results['training_time']:.2f}s\n"
    
    # Model comparison
    report += f"""

---

## 7. Model Comparison

### Performance Summary

"""
    
    if model_results:
        report += "| Model | Accuracy | Precision | Recall | F1-Score | Training Time |\n"
        report += "|-------|----------|-----------|--------|----------|---------------|\n"
        
        # Sort by F1-score
        sorted_models = sorted(model_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        for model_name, results in sorted_models:
            report += f"| {model_name} | {results['accuracy']:.4f} | "
            report += f"{results['precision']:.4f} | {results['recall']:.4f} | "
            report += f"{results['f1_score']:.4f} | {results['training_time']:.2f}s |\n"
    
    # Best model summary
    report += f"""

---

## 8. Best Model Summary & Justification

"""
    
    if model_results:
        best_model = max(model_results.items(), key=lambda x: x[1]['f1_score'])
        best_model_name = best_model[0]
        best_model_metrics = best_model[1]
        
        report += f"### Recommended Model: **{best_model_name}**\n\n"
        
        report += "#### Justification\n\n"
        report += f"The {best_model_name} was selected as the best model based on the following criteria:\n\n"
        report += f"1. **Highest F1-Score:** {best_model_metrics['f1_score']:.4f} - "
        report += "indicating the best balance between precision and recall\n"
        report += f"2. **Accuracy:** {best_model_metrics['accuracy']:.4f} - "
        report += "demonstrating strong overall performance\n"
        report += f"3. **Training Efficiency:** {best_model_metrics['training_time']:.2f}s - "
        report += "reasonable training time\n\n"
        
        report += "#### Confusion Matrix\n\n"
        cm = best_model_metrics['confusion_matrix']
        report += "```\n"
        report += str(cm)
        report += "\n```\n\n"
        
        report += "#### Recommendations\n\n"
        report += "- Use this model for production predictions\n"
        report += "- Monitor model performance regularly\n"
        report += "- Retrain with new data when available\n"
        report += "- Consider ensemble methods for further improvement\n"
    
    # Conclusions
    report += f"""

---

## 9. Conclusions & Next Steps

### Key Takeaways

1. Successfully trained and evaluated {len(model_results) if model_results else 0} classification models
2. Identified {best_model_name if model_results else 'N/A'} as the best-performing model
3. Achieved {best_model_metrics['f1_score']:.2%} F1-score on the test set

### Recommendations for Improvement

- Collect more training data to improve model generalization
- Explore feature engineering techniques
- Try ensemble methods (e.g., stacking, voting)
- Perform cross-validation for more robust evaluation
- Monitor model performance in production
- Consider domain-specific features

### Next Steps

1. Deploy the best model to production
2. Set up monitoring and logging
3. Create a feedback loop for continuous improvement
4. Document model assumptions and limitations

---

**Report End**

*This report was automatically generated by the AutoML Classification System.*
"""
    
    return report

def markdown_to_html(markdown_content):
    """Convert Markdown to HTML with styling"""
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AutoML Classification Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .metric {{
            background-color: #e8f4f8;
            padding: 10px;
            border-left: 4px solid #3498db;
            margin: 10px 0;
        }}
        hr {{
            border: none;
            border-top: 2px solid #eee;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        {markdown_content.replace('**', '<strong>').replace('**', '</strong>')}
    </div>
</body>
</html>
"""
    
    return html

def generate_results_csv(model_results):
    """Generate CSV with model results"""
    
    if not model_results:
        return ""
    
    data = []
    for model_name, results in model_results.items():
        data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1_Score': results['f1_score'],
            'ROC_AUC': results['roc_auc'] if results['roc_auc'] else 'N/A',
            'Training_Time_Seconds': results['training_time']
        })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)
