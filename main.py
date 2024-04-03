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
results_df.to_csv(r'C:\Users\jmask\Desktop\ING\train_added.csv', index=False)

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
df_filtered = df_filtered.drop(columns=['Ref_month', 'Birth_date', 'Contract_origination_date', 'Contract_end_date', 'Oldest_account_date', 'Customer_id'])
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

#%%

def plot_roc_curve(fpr, tpr, title='ROC Curve'):
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def evaluate_predictions(y_true, y_pred):
    # Compute metrics
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    # Display metrics
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    # Confusion Matrix calculation and display
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    print("\nConfusion Matrix:")
    print(cm)

    # Display Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
#%%
# XGBoost model

# XGB - same dataset as before
X_train_XGB = X_train_tuning
X_val_XGB = X_val
X_test_XGB = X_test
y_train_XGB = y_train_tuning
y_val_XGB = y_val
y_test_XGB = y_test

# Apply SMOTE to the training dataset after feature selection
smote = SMOTE(random_state=1234)
X_train_XGB_resampled, y_train_XGB_resampled = smote.fit_resample(X_train_tuning, y_train_tuning)

#%%

def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'eval_metric': 'auc',
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.000001, 0.05),
        'max_depth': trial.suggest_int('max_depth', 3, 200),
        'min_child_weight': trial.suggest_int('min_child_weight', 20, 2000),
        'subsample': trial.suggest_float('subsample', 0.2, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'lambda': trial.suggest_float('lambda', 1e-2, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-2, 10.0, log=True),
    }

    model = XGBClassifier(**{k: v for k, v in param.items() if k != 'early_stopping_rounds'})
    model.set_params(**{'early_stopping_rounds': 100})
    # Make sure to use the resampled and RFE-transformed dataset
    model.fit(X_train_XGB_resampled, y_train_XGB_resampled, eval_set=[(X_val, y_val)], verbose=False)

    preds_proba = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, preds_proba)

    return auc_score

# %%
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
# %%
print('Best trial:', study.best_trial.params)
# %%

model_XGB = XGBClassifier(**study.best_trial.params)
model_XGB.fit(X_train_XGB_resampled, y_train_XGB_resampled, eval_set=[(X_val, y_val)], verbose=False)
#%%
def k_fold_validation(X, y, best_params, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    auc_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        # Apply SMOTE
        smote = SMOTE(random_state=1234)
        X_train_res, y_train_res = smote.fit_resample(X_train_fold, y_train_fold)

        model = XGBClassifier(**best_params)
        model.fit(X_train_res, y_train_res)

        y_pred_proba = model.predict_proba(X_test_fold)[:, 1]
        auc_score = roc_auc_score(y_test_fold, y_pred_proba)
        auc_scores.append(auc_score)

    return np.mean(auc_scores), np.std(auc_scores)

# Perform K-Fold Cross-Validation
mean_auc, std_auc = k_fold_validation(X_train, y_train.reset_index(drop=True), study.best_trial.params)
print(f"Mean AUC: {mean_auc}, Standard Deviation: {std_auc}")


# %%
# Evaluate the model on the test set
predictions = model_XGB.predict(X_test)
print(f"Balanced accuracy: {balanced_accuracy_score(y_test_XGB, predictions)}")
print(f"Accuracy         : {accuracy_score(y_test_XGB, predictions)}")

# Evaluate the model using the simple evaluation function
print("Train set XGB:")
evaluate_predictions(y_train_XGB, model_XGB.predict(X_train_tuning))
print("Validation set XGB:")
evaluate_predictions(y_val_XGB, model_XGB.predict(X_val))
print("Test set XGB:")
evaluate_predictions(y_test_XGB, predictions)

# %%
y_train_pred_prob = model_XGB.predict_proba(X_train_tuning)[:, 1]
fpr_train, tpr_train, _ = roc_curve(y_train_tuning.astype(int).values, y_train_pred_prob)
plot_roc_curve(fpr_train, tpr_train, 'Training Set ROC Curve XGB')

y_val_pred_prob = model_XGB.predict_proba(X_val)[:, 1]
fpr_val, tpr_val, _ = roc_curve(y_val.astype(int).values, y_val_pred_prob)
plot_roc_curve(fpr_val, tpr_val, 'Validation Set ROC Curve XGB')

y_test_pred_prob = model_XGB.predict_proba(X_test)[:, 1]
fpr_test, tpr_test, _ = roc_curve(y_test.astype(int).values, y_test_pred_prob)
plot_roc_curve(fpr_test, tpr_test, 'Test Set ROC Curve XGB')

#%%
from lime import lime_tabular

# Initialize the explainer
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['Negative', 'Positive'],
    mode='classification'
)

# Choose an instance to explain
idx_to_explain = 1
instance = X_test.iloc[idx_to_explain].values

# Generate explanation for the chosen instance
exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=model_XGB.predict_proba,  # model prediction probability function
    num_features=30
)

plt.figure(figsize=(12, 6))

# Visualize the explanation
fig = exp.as_pyplot_figure()
fig.subplots_adjust(left=0.4)  # Adjust this value as needed to prevent label cut-off

# Show the plot
plt.show()

#%%
import shap

explainer = shap.TreeExplainer(model_XGB)

# Calculate SHAP values for the validation set
shap_values = explainer.shap_values(X_val)

# Get the expected value from the explainer, handling both binary and multi-class cases
expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

# Visualize the force plot for a single prediction
instance_idx = 1
selected_instance = X_val.iloc[instance_idx]
selected_shap_values = shap_values[instance_idx]

# Plotting force plot for one instance
shap.initjs()
force_plot = shap.force_plot(
    expected_value,
    selected_shap_values,
    selected_instance,
    feature_names=X_val.columns.tolist(),
    matplotlib=True
)
plt.show()

# Generating summary plot for all features across all data
shap.summary_plot(shap_values, X_val, plot_type="bar")
plt.show()

# Generating detailed summary plot showing the impact of each feature on model output
shap.summary_plot(shap_values, X_val)
plt.show()

# Generate an Explanation object for the waterfall plot
shap_explanation = shap.Explanation(
    values=selected_shap_values,
    base_values=expected_value,
    data=selected_instance.values,
    feature_names=X_val.columns.tolist()
)

# Plotting waterfall plot for one instance
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_explanation, max_display=14)
plt.tight_layout()
plt.show()


#%%
from sklearn.calibration import CalibratedClassifierCV
# Calibrate the model on the validation set
calibrated_model = CalibratedClassifierCV(model_XGB, method='isotonic', cv='prefit')
calibrated_model.fit(X_val, y_val)

# Evaluation functions should remain the same as before

# Evaluate the calibrated model on the test set using your existing metrics
predictions_calibrated = calibrated_model.predict(X_test)
print(f"Balanced accuracy (Calibrated): {balanced_accuracy_score(y_test, predictions_calibrated)}")
print(f"Accuracy (Calibrated): {accuracy_score(y_test, predictions_calibrated)}")

# You may want to compare ROC curves and other metrics before and after calibration
y_test_pred_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
fpr_test_calibrated, tpr_test_calibrated, _ = roc_curve(y_test.astype(int).values, y_test_pred_prob_calibrated)
# Use your existing plot_roc_curve function to plot the ROC curve for the calibrated model
plot_roc_curve(fpr_test_calibrated, tpr_test_calibrated, 'Test Set ROC Curve XGB (Calibrated)')