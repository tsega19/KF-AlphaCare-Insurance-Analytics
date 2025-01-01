import streamlit as st
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the Python path for module imports
sys.path.append(os.path.abspath('..'))

# Import functions from EDA.py and load_data.py
from scripts.EDA import (
    descriptive_statistics,
    plot_countplot,
    bivariate_multivariate_analysis,
    data_comparison,
    outlier_detection,
    creative_visualizations,
)
from scripts.load_data import load_data_from_txt, data_structure

# Configure the Streamlit page
st.set_page_config(page_title='Data Dashboard', layout='wide')
st.sidebar.title('Navigation')

# Sidebar navigation
page = st.sidebar.radio('Go to', ["EDA", "Visualizations"])

# Paths for the input and output data files
input_filepath = '../MachineLearningRating_v3.txt'
output_filepath = '../MachineLearningRating_v3.csv'

# Load data
if not os.path.exists(output_filepath):
    data = load_data_from_txt(input_filepath, output_filepath)
else:
    data = pd.read_csv(output_filepath)

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
