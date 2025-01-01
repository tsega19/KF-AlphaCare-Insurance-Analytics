import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Load Data
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path, low_memory=False)

# Handle Missing Data
def handle_missing_data(data):
    """
    Fill missing values in the given DataFrame.
    This function fills missing values in both categorical and numerical columns.
    For categorical columns, it fills missing values with the mode (most frequent value).
    For numerical columns, it fills missing values with the mean of the column.
    """
    # Dynamically identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    numerical_cols = data.select_dtypes(include=['number']).columns

    # Fill missing values for categorical columns
    for col in categorical_cols:
        if data[col].mode().empty:
            data[col].fillna('Unknown', inplace=True)
        else:
            data[col].fillna(data[col].mode()[0], inplace=True)

    # Fill missing values for numerical columns
    for col in numerical_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert non-numeric to NaN
        data[col].fillna(data[col].mean(), inplace=True)  # Fill with mean

    return data

# Feature Engineering
def feature_engineering(data):
    """
    Create features relevant to TotalPremium and TotalClaims.
    """
    data['ClaimsToPremiumRatio'] = data['TotalClaims'] / (data['TotalPremium'] + 1e-5)
    return data

# Data Preparation
def prepare_data(data):
    """
    Prepare data for modeling.
    """
    data = handle_missing_data(data)
    data = feature_engineering(data)

    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    numerical_cols = data.select_dtypes(include=['number']).columns.drop('TotalClaims')

    # Impute missing values and encode categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    X = data.drop(['TotalClaims'], axis=1) 
    y = data['TotalClaims']

    X = preprocessor.fit_transform(X)

    return train_test_split(X, y, test_size=0.3, random_state=42)

# Model Building and Evaluation
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train and evaluate a model.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # For classification metrics, we need to convert regression predictions to binary outcomes
    y_test_binary = (y_test > y_test.mean()).astype(int)
    predictions_binary = (predictions > predictions.mean()).astype(int)
    
    accuracy = accuracy_score(y_test_binary, predictions_binary)
    precision = precision_score(y_test_binary, predictions_binary)
    recall = recall_score(y_test_binary, predictions_binary)
    f1 = f1_score(y_test_binary, predictions_binary)

    # Graphical representation
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5, label=f"{model_name} Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name} Performance")
    plt.legend()
    plt.grid()
    plt.show()

    return mse, r2, accuracy, precision, recall, f1

# Feature Importance Analysis using SHAP
def analyze_feature_importance_shap(model, X_train):
    """
    Analyze feature importance using SHAP.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train)

# Feature Importance Analysis using LIME
def analyze_feature_importance_lime(model, X_train, X_test, feature_names):
    """
    Analyze feature importance using LIME.
    """
    # Ensure X_train and X_test are NumPy arrays
    if not isinstance(X_train, np.ndarray):
        X_train = X_train.toarray() if hasattr(X_train, 'toarray') else np.array(X_train)
    if not isinstance(X_test, np.ndarray):
        X_test = X_test.toarray() if hasattr(X_test, 'toarray') else np.array(X_test)

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode='regression', feature_names=feature_names)
    i = np.random.randint(0, X_test.shape[0])
    exp = explainer.explain_instance(X_test[i], model.predict, num_features=10)
    exp.show_in_notebook(show_table=True)