# EDA for Insurance Data Analysis

This repository contains Python scripts and resources for performing Exploratory Data Analysis (EDA) on historical insurance data. The goal of this project is to provide insights into insurance claim trends, detect anomalies, and identify actionable metrics for business decisions. Additionally, it includes a Streamlit dashboard for interactive visualization and analysis.

## Project Structure

```
├── EDA.py                # Core script containing EDA functions
├── load_data.py          # Utility script for loading and preprocessing data
├── dashboard.py          # Streamlit dashboard script for interactive analysis
├── data/                 # Folder for storing datasets
├── output/               # Folder for saving results and processed files
├── README.md             # Documentation file
```

## Features

1. **Data Loading:**
   - Load data from text files and save as CSV for easier analysis.
   - Handle missing values effectively with appropriate strategies.

2. **Descriptive Statistics:**
   - Generate statistics such as mean, median, variance, skewness, and more.

3. **Visualizations:**
   - Histograms for numerical distributions.
   - Correlation heatmaps to understand relationships between variables.
   - Count plots for categorical variables.

4. **Advanced Analysis:**
   - Bivariate and multivariate analysis to explore correlations and trends.
   - Geographic trends analysis (e.g., premiums across provinces).
   - Outlier detection using box plots.
   - Creative and insightful visualizations.

5. **Interactive Dashboard:**
   - A user-friendly Streamlit dashboard for interactive data exploration and visualization.

## Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn streamlit
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/insurance-eda.git
cd insurance-eda
```

2. Run the main EDA script:

```bash
python EDA.py
```

3. Run the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

4. Modify paths in the scripts to point to your dataset if necessary.

## Data

The data used in this project includes features related to:

- **Client Details:** Citizenship, marital status, gender, etc.
- **Vehicle Details:** Make, model, year, type, etc.
- **Insurance Plans:** Premium amounts, claims, coverage type, etc.

Ensure you have the dataset in the correct format before running the scripts or launching the dashboard.

