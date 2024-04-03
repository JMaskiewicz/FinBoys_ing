from datetime import datetime
import calendar
import random
import time
from time import perf_counter, sleep
from functools import wraps
from typing import Callable, Any
import warnings
from tqdm import tqdm

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
df = pd.read_csv(r'C:\Users\jmask\Desktop\ING\train_added.csv')

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


columns_to_replace = ['Active_accounts', 'Active_loans', 'Active_mortgages']

# Replace -9999 with 0 in the specified columns
df_filtered[columns_to_replace] = df_filtered[columns_to_replace].replace(-9999, 0)
error
#%%
df_filtered = df_filtered.drop(columns=['Ref_month', 'Birth_date', 'Contract_origination_date', 'Contract_end_date', 'Oldest_account_date', 'Customer_id'])

#%%
df_filtered = df_filtered[['No_dependants', 'Time_in_current_job', 'Current_installment', 'Diff_transactions_variance_pct',
    'TOTAL_amount_balance_trend_angle', 'TOTAL_amount_balance_variance_pct', 'Num_borrowers', 'Active_loans', 'Target']]

# Num_borrowers / Current_installment
# Column: Current_installment, Best Transformation: original, AUC: 0.5947317816593466
# Column: Diff_transactions_variance_pct, Best Transformation: sqrt, AUC: 0.5764569704530964
# Column: TOTAL_amount_balance_trend_angle, Best Transformation: log, AUC: 0.5614210648582394
# Column: TOTAL_amount_balance_variance_pct, Best Transformation: sqrt, AUC: 0.5414900490988291
# Column Pair: Active_loans * Current_installment, AUC: 0.6007384326820855
# Column Pair: Current_installment * TOTAL_amount_balance_trend_angle, AUC: 0.5971654777271912

df_filtered['Diff_transactions_variance_pct'] = np.sqrt(df_filtered['Diff_transactions_variance_pct'])
df_filtered['TOTAL_amount_balance_variance_pct'] = np.sqrt(df_filtered['TOTAL_amount_balance_variance_pct'])
df_filtered['TOTAL_amount_balance_trend_angle'] = np.log(df_filtered['TOTAL_amount_balance_trend_angle'] + 1)

# Assuming Active_loans column exists in your dataset for the next operation
df_filtered['Active_loans_Current_installment'] = df_filtered['Active_loans'] * df_filtered['Current_installment']

# Apply transformation for the pair (after ensuring 'Active_loans' exists in your DataFrame)
df_filtered['Current_installment_TOTAL_amount_balance_trend_angle'] = df_filtered['Current_installment'] * np.log(df_filtered['TOTAL_amount_balance_trend_angle'] + 1)

'''df_filtered = df_filtered[['Current_installment', 'Diff_transactions_variance_pct', 'Target']]
#Column: Current_installment, Best Transformation: original, AUC: 0.5947317816593466
#Column: Diff_transactions_trend_angle, Best Transformation: log, AUC: 0.5093402310516277
#Column: Diff_transactions_variance_pct, Best Transformation: sqrt, AUC: 0.5764569704530964'''

# Calculating the correlation matrix
correlation_matrix = df_filtered.corr()

print(correlation_matrix)


X = df_filtered.drop('Target', axis=1)
y = df_filtered['Target']
#%%
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X, y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train_tuning, X_val, y_train_tuning, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training set and transform it to fill in the missing values
X_train_tuning_imputed = imputer.fit_transform(X_train_tuning)

# Transform the validation and test sets using the same imputer
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# Standardize the data: Fit on X_train_tuning and transform all sets
scaler = StandardScaler()

'''X_train_logit = scaler.fit_transform(X_train_tuning_imputed)
X_val_logit = scaler.transform(X_val_imputed)
X_test_logit = scaler.transform(X_test_imputed)'''

X_train_logit = X_train_tuning_imputed
X_val_logit = X_val_imputed
X_test_logit = X_test_imputed
#%%
model_Logit = sm.Logit(y_train_tuning, X_train_logit).fit(disp=0, maxiter=10000)  # if fail restart kernel as in some python versions there is a bug
model_Logit.summary()

#%%
# Evaluation function
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
#%% md
## LOGIT MODEL PREDICTIONS

#%%
y_val_pred_prob = model_Logit.predict(X_val_logit)

# Calculate ROC curve from validation set
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_prob)
gmeans = np.sqrt(tpr * (1 - fpr))

# Locate the index of the largest G-mean
ix = np.argmax(gmeans)

# The optimal cutoff is the threshold with the largest G-mean
optimal_cutoff_Logit = thresholds[ix]
print('Optimal Cut-off:', optimal_cutoff_Logit)

# Convert probabilities to binary outcome using the optimal cutoff
y_train_pred_optimal = [1 if prob > optimal_cutoff_Logit else 0 for prob in model_Logit.predict(X_train_logit)]
y_val_pred_optimal = [1 if prob > optimal_cutoff_Logit else 0 for prob in y_val_pred_prob]
y_test_pred_optimal = [1 if prob > optimal_cutoff_Logit else 0 for prob in model_Logit.predict(X_test_logit)]

# Evaluate the model using the simple evaluation function
print("Train set Logit:")
evaluate_predictions(y_train_tuning, y_train_pred_optimal)

#%%
# Evaluate on validation set
print("Validation set Logit:")
evaluate_predictions(y_val, y_val_pred_optimal)
#%%
# Evaluate on test set
print("Test set Logit:")
evaluate_predictions(y_test, y_test_pred_optimal)
#%%
# function to plot ROC curve
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

y_train_pred_prob = model_Logit.predict(X_train_logit)
fpr_train, tpr_train, _ = roc_curve(y_train_tuning.astype(int).values, y_train_pred_prob)
plot_roc_curve(fpr_train, tpr_train, 'Training Set ROC Curve Logit')

y_val_pred_prob = model_Logit.predict(X_val_logit)
fpr_val, tpr_val, _ = roc_curve(y_val.astype(int).values, y_val_pred_prob)
plot_roc_curve(fpr_val, tpr_val, 'Validation Set ROC Curve Logit')

y_test_pred_prob = model_Logit.predict(X_test_logit)
fpr_test, tpr_test, _ = roc_curve(y_test.astype(int).values, y_test_pred_prob)
plot_roc_curve(fpr_test, tpr_test, 'Test Set ROC Curve Logit')


