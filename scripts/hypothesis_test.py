import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.weightstats import ztest

# Load Data Function
def load_data(file_path):
    """
    Load data from a CSV file.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pandas.DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)

# Perform T-Test Function
def perform_ttest(group_a, group_b):
    """
    Perform a T-test for two independent samples.
    Args:
        group_a (array-like): Data for group A.
        group_b (array-like): Data for group B.
    Returns:
        float: p-value of the test.
    """
    t_stat, p = ttest_ind(group_a, group_b, equal_var=False, nan_policy='omit')
    return p

# Perform Z-Test Function
def perform_ztest(group_a, group_b):
    """
    Performs a two-sample Z-test.

    Args:
        group_a (array-like): Data for group A.
        group_b (array-like): Data for group B.

    Returns:
        float: p-value of the test.
    """
    # Check if either group is empty or contains only NaNs
    if len(group_a) < 2 or len(group_b) < 2 or np.isnan(group_a).all() or np.isnan(group_b).all():
        # Handle the case where Z-test is not appropriate
        # You can choose to return NaN, raise an error, or provide a default value
        # Here, we return NaN to indicate that the test couldn't be performed
        return np.nan

    z_stat, p = ztest(group_a, group_b, alternative='two-sided')
    return p

# Visualization Function
def plot_comparison(group_a_mean, group_b_mean, group_a_std, group_b_std, labels, title, ylabel):
    """
    Create a bar plot with error bars for two groups.
    Args:
        group_a_mean (float): Mean value for group A.
        group_b_mean (float): Mean value for group B.
        group_a_std (float): Standard deviation for group A.
        group_b_std (float): Standard deviation for group B.
        labels (list): Labels for the two groups.
        title (str): Plot title.
        ylabel (str): Y-axis label.
    """
    means = [group_a_mean, group_b_mean]
    stds = [group_a_std, group_b_std]
    plt.bar(labels, means, yerr=stds, capsize=5, color=['blue', 'orange'])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()

# Hypothesis Testing and Visualization Function
def hypothesis_test_and_visualize(data, group_column, metric_column, group_a_label, group_b_label, test_func, hypothesis_title):
    """
    Perform hypothesis testing and visualize the results.
    Args:
        data (pd.DataFrame): Dataset containing the data.
        group_column (str): Column to segment groups.
        metric_column (str): Column to test.
        group_a_label (str): Label for group A.
        group_b_label (str): Label for group B.
        test_func (function): Statistical test function to use.
        hypothesis_title (str): Title for the hypothesis.
    """
    group_a = data[data[group_column] == group_a_label][metric_column]
    group_b = data[data[group_column] == group_b_label][metric_column]

    p_value = test_func(group_a, group_b)

    group_a_mean, group_a_std = group_a.mean(), group_a.std()
    group_b_mean, group_b_std = group_b.mean(), group_b.std()

    print(f"{hypothesis_title}")
    print(f"Group A Mean ({group_a_label}): {group_a_mean:.2f}, Std: {group_a_std:.2f}")
    print(f"Group B Mean ({group_b_label}): {group_b_mean:.2f}, Std: {group_b_std:.2f}")
    print(f"P-value: {p_value:.4f}\n")

    plot_comparison(
        group_a_mean, group_b_mean, group_a_std, group_b_std,
        [group_a_label, group_b_label],
        hypothesis_title,
        metric_column
    )
    return p_value

# Conclusion Display Function
def display_conclusion(p_value_provinces, p_value_postal, p_value_margin, p_value_gender, p_value_z):
    print("=== Conclusion ===\n")
    
    print("1. Risk differences across provinces:")
    if interpret_p_value(p_value_provinces) == "Reject Null Hypothesis":
        print("There is a significant difference in risk across provinces.")
    else:
        print("No significant difference in risk across provinces was found.")
    
    print("\n2. Risk differences between postal codes:")
    if interpret_p_value(p_value_postal) == "Reject Null Hypothesis":
        print("There is a significant difference in risk between postal codes.")
    else:
        print("No significant difference in risk between postal codes was found.")
    
    print("\n3. Margin differences between postal codes:")
    if interpret_p_value(p_value_margin) == "Reject Null Hypothesis":
        print("There is a significant difference in margin between postal codes.")
    else:
        print("No significant difference in margin between postal codes was found.")
    
    print("\n4. Risk differences between Women and Men:")
    if interpret_p_value(p_value_gender) == "Reject Null Hypothesis":
        print("There is a significant difference in risk between Women and Men.")
    else:
        print("No significant difference in risk between Women and Men was found.")
    
    print("\n5. Z-test result for risk difference between groups A and B:")
    if interpret_p_value(p_value_z) == "Reject Null Hypothesis":
        print("There is a significant difference in risk between groups A and B.")
    else:
        print("No significant difference in risk between groups A and B was found.")
    
    print("\n=== Business Implications ===")
    print("Based on the results, consider focusing on postal codes or demographic factors where significant differences were observed. "
          "Further exploration of provinces might also be valuable if the p-values were marginally above the threshold.")

# Interpret the p-value
def interpret_p_value(p_value):
    if p_value < 0.05:
        return f"Reject Null Hypothesis\n( {p_value} is less than 0.05 )"
    else:
        return f"Fail to Reject Null Hypothesis\n({p_value} is not less than 0.05)"
