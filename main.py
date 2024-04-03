from datetime import datetime
import calendar
import random
import time
from time import perf_counter, sleep
from functools import wraps
from typing import Callable, Any
import warnings
from tqdm import tqdm
import re

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report, \
    roc_curve, auc, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import mode

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import balanced_accuracy_score, accuracy_score
import optuna
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import optuna
import shap
from xgboost import XGBClassifier

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# %%
# set random seeds for reproducibility
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# %% md
## Let's load the dataset
# %%
df = pd.read_csv(r'C:\Users\jmask\Desktop\ING\in_time.csv')

columns_drop = ['External_term_loan_balance', 'External_mortgage_balance', 'External_credit_card_balance']
df = df.drop(columns=columns_drop)

# new 2 variables
# List of time points
time_points = ['_H' + str(i) for i in range(0, 13)]  # Adjust the range if necessary

# Iterate over each time point and create the TOTAL_amount_balance variable
for time_point in time_points:
    current_col = f'Current_amount_balance{time_point}'
    savings_col = f'Savings_amount_balance{time_point}'
    total_col = f'TOTAL_amount_balance{time_point}'

    # Sum the two columns and assign to the new total column
    df[total_col] = df[current_col] + df[savings_col]

# Iterate over each time point and create the TOTAL_amount_balance variable
for time_point in time_points:
    inc_col = f'inc_transactions_amt{time_point}'
    out_col = f'out_transactions_amt{time_point}'
    total_col = f'Diff_transactions{time_point}'

    # Sum the two columns and assign to the new total column
    df[total_col] = df[inc_col] - df[out_col]


# df = df.iloc[:100]

#%%
# change into linear model for trend

time_attention_variables = {'Current_amount_balance',
 'DPD_credit_card',
 'DPD_mortgage',
 'DPD_term_loan',
 'Default_flag',
 'Income',
 'Os_credit_card',
 'Os_mortgage',
 'Os_term_loan',
 'Overdue_credit_card',
 'Overdue_mortgage',
 'Overdue_term_loan',
 'Payments_credit_card',
 'Payments_mortgage',
 'Payments_term_loan',
 'Savings_amount_balance',
 'inc_transactions',
 'inc_transactions_amt',
 'limit_in_revolving_loans',
 'out_transactions',
 'out_transactions_amt',
 'utilized_limit_in_revolving_loans'
}

time_linear_variables = {
 'TOTAL_amount_balance',
 'Income',
 'Diff_transactions',
}

from sklearn.linear_model import LinearRegression

def slope_to_angle(slope):
    """Convert slope to angle in degrees."""
    return np.degrees(np.arctan(slope))

def calculate_trend(series):
    time = np.arange(len(series)).reshape(-1, 1)
    model = LinearRegression().fit(time, series)
    return model.coef_[0]

def calculate_trend_and_variance(df, variable_names):
    for base_var in variable_names:
        trend_col_name = f"{base_var}_trend_angle"
        variance_col_name = f"{base_var}_variance_pct"

        # Initialize columns
        df[trend_col_name] = np.nan
        df[variance_col_name] = np.nan

        for i in tqdm(range(len(df))):
            series_values = df.loc[i, [f"{base_var}_H{j}" for j in range(12, -1, -1)]].values.astype(float)
            series_values = series_values/series_values[0] if series_values[0] != 0 else series_values
            if not np.any(np.isnan(series_values)):
                slope = calculate_trend(series_values.reshape(-1, 1))
                angle = slope_to_angle(slope)
                variance_pct = np.var(series_values) / np.mean(series_values) * 100 if np.mean(
                    series_values) != 0 else 0

                # Store results
                df.at[i, trend_col_name] = angle
                df.at[i, variance_col_name] = variance_pct

    return df

# Assuming df is your DataFrame and time_linear_variables is your set of variables
variable_names = time_linear_variables
results_df = calculate_trend_and_variance(df, variable_names)

print(results_df)

#%%
# remove all H.. name columns
import re

columns_to_keep = [col for col in df.columns if not re.search(r'H\d+', col)]

df_filtered = df[columns_to_keep]

columns = ["Birth_date", "Contract_origination_date", "Contract_end_date", "Oldest_account_date"]

ref = pd.to_datetime(df_filtered['Ref_month'], format='%m-%Y')
for col in columns:
    d = pd.to_datetime(df_filtered[col], format='%d-%m-%Y')
    df_filtered[col + '_diff'] = abs(ref - d).dt.days
    print(df_filtered[col + '_diff'])

columns_to_replace = ['Active_accounts', 'Active_loans', 'Active_mortgages']

# Replace -9999 with 0 in the specified columns
df_filtered[columns_to_replace] = df_filtered[columns_to_replace].replace(-9999, 0)


#%%
X = df_filtered.drop('Target', axis=1)
y = df_filtered['Target']
X, y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train_tuning, X_val, y_train_tuning, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Initialize a dictionary to hold the best AUC score for each column
best_auc_scores = {}

# Loop over each column in X_train_tuning
for column in X_train_tuning.columns:
    best_auc = -np.inf
    best_transformation = ''

    # Define transformations
    transformations = {
    'original': X_train_tuning[column],
    'log': np.log(X_train_tuning[column] + np.abs(X_train_tuning[column].min()) + 1),
    'poly': X_train_tuning[column] + np.power(X_train_tuning[column], 2),
    'inverse': 1 / (X_train_tuning[column] + np.abs(X_train_tuning[column].min()) + 1),
    'sqrt': np.sqrt(X_train_tuning[column] + np.abs(X_train_tuning[column].min())),
    # 'exp': np.exp(X_train_tuning[column] - np.abs(X_train_tuning[column].min())),
    # 'cubic': np.power(X_train_tuning[column], 3),
    # 'reciprocal_sqrt': 1 / np.sqrt(X_train_tuning[column] + np.abs(X_train_tuning[column].min())),
    'sine': np.sin(X_train_tuning[column]),
    'cosine': np.cos(X_train_tuning[column])
}

    for trans_name, trans_data in transformations.items():
        # Standardize the transformed data
        scaler = StandardScaler()
        X_train_trans = scaler.fit_transform(trans_data.values.reshape(-1, 1))
        X_val_trans = scaler.transform(X_val[column].values.reshape(-1, 1))

        # Fit logistic regression model
        model = LogisticRegression(random_state=1)
        model.fit(X_train_trans, y_train_tuning)

        # Predict on validation set and calculate AUC
        y_pred_prob = model.predict_proba(X_val_trans)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_prob)

        # Check if this is the best AUC for the column
        if auc_score > best_auc:
            best_auc = auc_score
            best_transformation = trans_name

    # Save the best AUC score and transformation for the column
    best_auc_scores[column] = (best_transformation, best_auc)

# Print the best transformation and AUC score for each column
for column, (transformation, auc) in best_auc_scores.items():
    print(f"Column: {column}, Best Transformation: {transformation}, AUC: {auc}")

# Initialize a list to hold the results
results = []

# Loop over each pair of columns in X_train_tuning
for col1, col2 in combinations(X_train_tuning.columns, 2):
    # Create a new feature by dividing col1 by col2, adding a small constant to avoid division by zero
    X_train_ratio = X_train_tuning[col1] / (X_train_tuning[col2] + 1e-8)
    X_val_ratio = X_val[col1] / (X_val[col2] + 1e-8)

    # Standardize the new feature
    scaler = StandardScaler()
    X_train_ratio_scaled = scaler.fit_transform(X_train_ratio.values.reshape(-1, 1))
    X_val_ratio_scaled = scaler.transform(X_val_ratio.values.reshape(-1, 1))

    # Fit logistic regression model
    model = LogisticRegression(random_state=1)
    model.fit(X_train_ratio_scaled, y_train_tuning)

    # Predict on validation set and calculate AUC
    y_pred_prob = model.predict_proba(X_val_ratio_scaled)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_prob)

    # Save the results
    results.append(((col1, col2), auc_score))

# Sort the results by AUC score in descending order
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

# Print the sorted results
for (col1, col2), auc in results_sorted:
    print(f"Column Pair: {col1} / {col2}, AUC: {auc}")

# Initialize a list to hold the results
results = []

# Loop over each pair of columns in X_train_tuning
for col1, col2 in combinations(X_train_tuning.columns, 2):
    # Create a new feature by dividing col1 by col2, adding a small constant to avoid division by zero
    X_train_ratio = X_train_tuning[col1] * (X_train_tuning[col2])
    X_val_ratio = X_val[col1] * (X_val[col2])

    # Standardize the new feature
    scaler = StandardScaler()
    X_train_ratio_scaled = scaler.fit_transform(X_train_ratio.values.reshape(-1, 1))
    X_val_ratio_scaled = scaler.transform(X_val_ratio.values.reshape(-1, 1))

    # Fit logistic regression model
    model = LogisticRegression(random_state=1)
    model.fit(X_train_ratio_scaled, y_train_tuning)

    # Predict on validation set and calculate AUC
    y_pred_prob = model.predict_proba(X_val_ratio_scaled)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_prob)

    # Save the results
    results.append(((col1, col2), auc_score))

# Sort the results by AUC score in descending order
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

# Print the sorted results
for (col1, col2), auc in results_sorted:
    print(f"Column Pair: {col1} * {col2}, AUC: {auc}")