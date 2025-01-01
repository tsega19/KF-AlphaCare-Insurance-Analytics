import streamlit as st
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.append(os.path.abspath('..'))

from EDA import (
    descriptive_statistics,
    plot_countplot,
    bivariate_multivariate_analysis,
    data_comparison,
    outlier_detection,
    creative_visualizations,
)
from load_data import data_structure
from model import (
    handle_missing_data,
    feature_engineering,
    prepare_data,
    evaluate_model,
    analyze_feature_importance_lime,
    analyze_feature_importance_shap
)
from hypothesis_test import (
    perform_ttest,
    perform_ztest,
    hypothesis_test_and_visualize,
    display_conclusion
)

# Configure the Streamlit page
st.set_page_config(page_title='Data Dashboard', layout='wide')
st.sidebar.title('Navigation')

# Sidebar navigation
page = st.sidebar.radio('Go to', ["EDA", "Visualizations", "Hypothesis Testing", "Model Evaluation"])

# Paths for the input and output data files
data = pd.read_csv('scripts\MachineLearningRating_v3.csv')
# EDA page
if page == 'EDA':
    st.title('Exploratory Data Analysis')
    st.write("### First 10 rows of the data")
    st.write(data.head(100))
    
    st.write("### Descriptive Statistical Description")
    st.write(descriptive_statistics(data))
    
    st.write("### Data Structure")
    st.write(data_structure(data))

# Visualizations page
elif page == 'Visualizations':
    st.title('Data Visualizations')
    
    visualization = st.sidebar.selectbox(
        "Select Visualization Type",
        [
            "Histogram",
            "Correlation Heatmap",
            "Categorical Data Analysis",
            "Bivariate/Multivariate Analysis",
            "Data Comparison",
            "Outlier Detection",
            "Creative Visualizations",
        ]
    )
    
    if visualization == "Histogram":
        st.write("### Histograms of Numerical Columns")
        fig, ax = plt.subplots(figsize=(10, 10))
        data[['TotalClaims', 'TotalPremium', 'CalculatedPremiumPerTerm', 'SumInsured']].hist(bins=20, ax=ax)
        st.pyplot(fig)

    elif visualization == "Correlation Heatmap":
        st.write("### Correlation Heatmap of Numerical Columns")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data[['TotalClaims', 'TotalPremium', 'CalculatedPremiumPerTerm', 'SumInsured']].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    elif visualization == "Categorical Data Analysis":
        st.write("### Count Plots for Categorical Columns")
        fig, ax = plt.subplots()
        plot_countplot(data)
        st.pyplot(fig)

    elif visualization == "Bivariate/Multivariate Analysis":
        st.write("### Bivariate/Multivariate Analysis")
        bivariate_multivariate_analysis(data)

    elif visualization == "Data Comparison":
        st.write("### Data Comparison")
        data_comparison(data)

    elif visualization == "Outlier Detection":
        st.write("### Outlier Detection")
        outlier_detection(data)

    elif visualization == "Creative Visualizations":
        st.write("### Creative and Insightful Visualizations")
        creative_visualizations(data)
# Hypothesis Testing page
elif page == 'Hypothesis Testing':
    st.title('Hypothesis Testing')
    
    # Perform hypothesis tests
    p_value_provinces, fig_provinces, text_provinces = hypothesis_test_and_visualize(
        data, 'Province', 'TotalClaims',
        data['Province'].unique()[0], data['Province'].unique()[1],
        perform_ttest,
        "Hypothesis 1: Risk differences across provinces"
    )
    st.write(text_provinces)
    st.pyplot(fig_provinces)
    
    p_value_postal, fig_postal, text_postal = hypothesis_test_and_visualize(
        data, 'PostalCode', 'TotalClaims',
        data['PostalCode'].unique()[0], data['PostalCode'].unique()[1],
        perform_ttest,
        "Hypothesis 2: Risk differences between postal codes"
    )
    st.write(text_postal)
    st.pyplot(fig_postal)
    
    p_value_margin, fig_margin, text_margin = hypothesis_test_and_visualize(
        data, 'PostalCode', 'TotalPremium',
        data['PostalCode'].unique()[0], data['PostalCode'].unique()[1],
        perform_ttest,
        "Hypothesis 3: Margin differences between postal codes"
    )
    st.write(text_margin)
    st.pyplot(fig_margin)
    
    p_value_gender, fig_gender, text_gender = hypothesis_test_and_visualize(
        data, 'Gender', 'TotalClaims',
        'Female', 'Male',
        perform_ttest,
        "Hypothesis 4: Risk differences between Women and Men"
    )
    st.write(text_gender)
    st.pyplot(fig_gender)
    
    p_value_z = perform_ztest(
        data[data['Province'] == 'A']['TotalClaims'],
        data[data['Province'] == 'B']['TotalClaims']
    )
    st.write(f"Z-Test p-value: {p_value_z:.4f}")
    
    # Display conclusion
    conclusion_text = display_conclusion(p_value_provinces, p_value_postal, p_value_margin, p_value_gender, p_value_z)
    st.write(conclusion_text)
# Model Evaluation page
elif page == 'Model Evaluation':
    st.title('Model Evaluation')
    
    # Prepare data for modeling
    data = handle_missing_data(data)
    data = feature_engineering(data)
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor()
    }
    
    results = {}
    for model_name, model in models.items():
        mse, r2, accuracy, precision, recall, f1, fig = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
        results[model_name] = {
            "MSE": mse,
            "R2": r2,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
        st.write(f"{model_name} - MSE: {mse}, R2: {r2}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        st.pyplot(fig)
    
    # Display results in a DataFrame
    results_df = pd.DataFrame(results).T
    st.write(results_df)
    
    # Feature Importance Analysis
    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_model = models[best_model_name]
    st.write(f"Analyzing feature importance for the best model: {best_model_name}")
    
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]
    analyze_feature_importance_lime(best_model, X_train, X_test, feature_names)
    analyze_feature_importance_shap(best_model, X_train)