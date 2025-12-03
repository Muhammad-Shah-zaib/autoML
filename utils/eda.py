"""
Exploratory Data Analysis Module
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def perform_eda(df, target_column):
    """
    Perform comprehensive exploratory data analysis
    
    Args:
        df: pandas DataFrame
        target_column: name of the target column
    """
    st.markdown("### 🔍 Comprehensive Data Analysis")
    
    # Create tabs for different analyses
    tabs = st.tabs([
        "📊 Missing Values",
        "📉 Outlier Detection",
        "🔗 Correlation Analysis",
        "📈 Distribution Analysis",
        "📊 Categorical Analysis",
        "✂️ Train/Test Split"
    ])
    
    # Tab 1: Missing Values Analysis
    with tabs[0]:
        missing_values_analysis(df)
    
    # Tab 2: Outlier Detection
    with tabs[1]:
        outlier_detection(df, target_column)
    
    # Tab 3: Correlation Analysis
    with tabs[2]:
        correlation_analysis(df, target_column)
    
    # Tab 4: Distribution Analysis
    with tabs[3]:
        distribution_analysis(df, target_column)
    
    # Tab 5: Categorical Analysis
    with tabs[4]:
        categorical_analysis(df, target_column)
    
    # Tab 6: Train/Test Split Info
    with tabs[5]:
        train_test_split_info(df)

def missing_values_analysis(df):
    """Analyze missing values in the dataset"""
    st.subheader("📊 Missing Values Analysis")
    
    # Calculate missing values
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(
        'Missing Percentage', ascending=False
    )
    
    if len(missing_data) == 0:
        st.success("✅ No missing values found in the dataset!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(missing_data, use_container_width=True)
            
            # Global missing percentage
            total_cells = df.shape[0] * df.shape[1]
            total_missing = df.isnull().sum().sum()
            global_missing_pct = (total_missing / total_cells * 100).round(2)
            st.metric("Global Missing Percentage", f"{global_missing_pct}%")
        
        with col2:
            # Visualization
            fig = px.bar(
                missing_data,
                x='Column',
                y='Missing Percentage',
                title='Missing Values by Column',
                labels={'Missing Percentage': 'Missing (%)'},
                color='Missing Percentage',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap of missing values
        st.subheader("🗺️ Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=True, cmap='YlOrRd', ax=ax)
        plt.title('Missing Values Heatmap')
        st.pyplot(fig)
        plt.close()

def outlier_detection(df, target_column):
    """Detect outliers using IQR and Z-score methods"""
    st.subheader("📉 Outlier Detection")
    
    # Get numerical columns (excluding target)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    
    if len(numerical_cols) == 0:
        st.warning("No numerical features found for outlier detection.")
        return
    
    # Method selection
    method = st.radio(
        "Select outlier detection method:",
        ["IQR Method", "Z-Score Method", "Both"],
        horizontal=True
    )
    
    outlier_summary = []
    
    for col in numerical_cols:
        data = df[col].dropna()
        
        # IQR Method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        
        # Z-Score Method
        z_scores = np.abs(stats.zscore(data))
        zscore_outliers = (z_scores > 3).sum()
        
        outlier_summary.append({
            'Column': col,
            'IQR Outliers': iqr_outliers,
            'IQR %': f"{(iqr_outliers / len(data) * 100):.2f}%",
            'Z-Score Outliers': zscore_outliers,
            'Z-Score %': f"{(zscore_outliers / len(data) * 100):.2f}%"
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    st.dataframe(outlier_df, use_container_width=True)
    
    # Visualize outliers for selected column
    st.subheader("📊 Outlier Visualization")
    selected_col = st.selectbox("Select column to visualize:", numerical_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot
        fig = px.box(df, y=selected_col, title=f'Box Plot: {selected_col}')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Histogram with outliers highlighted
        data = df[selected_col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, name='Normal', marker_color='lightblue'))
        fig.add_vline(x=lower_bound, line_dash="dash", line_color="red", 
                     annotation_text="Lower Bound")
        fig.add_vline(x=upper_bound, line_dash="dash", line_color="red", 
                     annotation_text="Upper Bound")
        fig.update_layout(title=f'Distribution with IQR Bounds: {selected_col}')
        st.plotly_chart(fig, use_container_width=True)

def correlation_analysis(df, target_column):
    """Analyze correlations between features"""
    st.subheader("🔗 Correlation Analysis")
    
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) < 2:
        st.warning("Need at least 2 numerical features for correlation analysis.")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
    st.pyplot(fig)
    plt.close()
    
    # Correlation with target
    if target_column in numerical_cols:
        st.subheader(f"📊 Correlation with Target ({target_column})")
        target_corr = corr_matrix[target_column].drop(target_column).sort_values(
            key=abs, ascending=False
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(
                pd.DataFrame({
                    'Feature': target_corr.index,
                    'Correlation': target_corr.values.round(3)
                }),
                use_container_width=True
            )
        
        with col2:
            fig = px.bar(
                x=target_corr.values,
                y=target_corr.index,
                orientation='h',
                title=f'Feature Correlation with {target_column}',
                labels={'x': 'Correlation', 'y': 'Feature'},
                color=target_corr.values,
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # High correlation pairs
    st.subheader("⚠️ Highly Correlated Feature Pairs")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j].round(3)
                })
    
    if len(high_corr_pairs) > 0:
        st.warning(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8)")
        st.dataframe(pd.DataFrame(high_corr_pairs), use_container_width=True)
    else:
        st.success("No highly correlated feature pairs found.")

def distribution_analysis(df, target_column):
    """Analyze distributions of numerical features"""
    st.subheader("📈 Distribution Analysis")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    
    if len(numerical_cols) == 0:
        st.warning("No numerical features found.")
        return
    
    # Select feature to visualize
    selected_feature = st.selectbox("Select feature:", numerical_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = px.histogram(
            df,
            x=selected_feature,
            marginal='box',
            title=f'Distribution: {selected_feature}',
            nbins=30
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Q-Q plot for normality check
        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(df[selected_feature].dropna(), dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: {selected_feature}')
        st.pyplot(fig)
        plt.close()
    
    # Statistical tests
    st.subheader("📊 Statistical Summary")
    col1, col2, col3 = st.columns(3)
    
    data = df[selected_feature].dropna()
    
    with col1:
        st.metric("Mean", f"{data.mean():.3f}")
        st.metric("Median", f"{data.median():.3f}")
    
    with col2:
        st.metric("Std Dev", f"{data.std():.3f}")
        st.metric("Variance", f"{data.var():.3f}")
    
    with col3:
        st.metric("Skewness", f"{data.skew():.3f}")
        st.metric("Kurtosis", f"{data.kurtosis():.3f}")
    
    # Distribution by target (if categorical target)
    if target_column and df[target_column].dtype == 'object' or df[target_column].nunique() < 10:
        st.subheader(f"📊 Distribution by {target_column}")
        fig = px.box(df, x=target_column, y=selected_feature, 
                    title=f'{selected_feature} by {target_column}')
        st.plotly_chart(fig, use_container_width=True)

def categorical_analysis(df, target_column):
    """Analyze categorical features"""
    st.subheader("📊 Categorical Features Analysis")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    if len(categorical_cols) == 0:
        st.warning("No categorical features found.")
        return
    
    # Select feature to visualize
    selected_feature = st.selectbox("Select categorical feature:", categorical_cols)
    
    # Value counts
    value_counts = df[selected_feature].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Value Counts:**")
        st.dataframe(
            pd.DataFrame({
                'Category': value_counts.index,
                'Count': value_counts.values,
                'Percentage': (value_counts.values / len(df) * 100).round(2)
            }),
            use_container_width=True
        )
    
    with col2:
        # Bar plot
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f'Distribution: {selected_feature}',
            labels={'x': selected_feature, 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Relationship with target
    if target_column:
        st.subheader(f"🔗 Relationship with {target_column}")
        
        # Cross-tabulation
        crosstab = pd.crosstab(df[selected_feature], df[target_column], normalize='index') * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        crosstab.plot(kind='bar', stacked=True, ax=ax)
        plt.title(f'{selected_feature} vs {target_column}')
        plt.xlabel(selected_feature)
        plt.ylabel('Percentage (%)')
        plt.legend(title=target_column, bbox_to_anchor=(1.05, 1))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

def train_test_split_info(df):
    """Display information about train/test split"""
    st.subheader("✂️ Train/Test Split Configuration")
    
    st.info("""
    The dataset will be split into training and testing sets during the preprocessing phase.
    This ensures proper model evaluation on unseen data.
    """)
    
    # Allow user to configure split ratio
    test_size = st.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data to use for testing"
    )
    
    train_size = 100 - test_size
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    
    with col2:
        train_samples = int(len(df) * train_size / 100)
        st.metric("Training Samples", f"{train_samples:,}")
    
    with col3:
        test_samples = len(df) - train_samples
        st.metric("Testing Samples", f"{test_samples:,}")
    
    # Visualization
    fig = go.Figure(data=[go.Pie(
        labels=['Training Set', 'Test Set'],
        values=[train_size, test_size],
        hole=.3
    )])
    fig.update_layout(title='Train/Test Split Ratio')
    st.plotly_chart(fig, use_container_width=True)
    
    # Store in session state
    st.session_state.test_size = test_size / 100
