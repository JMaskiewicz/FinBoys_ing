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
description = pd.read_excel('https://challengerocket.com/files/lions-den-ing-2024/variables_description.xlsx', header=0)
train = pd.read_csv('https://files.challengerocket.com/files/lions-den-ing-2024/development_sample.csv')

# %% md
## 1.1.1 Describe the dataset
# %%
print(train.shape)
print(train[train['Application_status'] == 'Rejected'].shape[0] / train.shape[0])

# %%
df = train[train['Application_status'] == 'Approved']
# %%
df['target'].value_counts(normalize=True)

# %%
not_features = ['ID', 'customer_id', 'application_date', 'Application_status', 'Var13', '_r_']

target = 'target'

categorical_features = ['Var2', 'Var3', 'Var11', 'Var12', 'Var14', 'Var18', 'Var19']

binary_features = ['Var27', 'Var28']

days_from_employment = 'Var13_b'

numerical_features = ['Var1', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 'Var9', 'Var10', 'Var15', 'Var16', 'Var17',
                      'Var20', 'Var21', 'Var22', 'Var23', 'Var24', 'Var25', 'Var26', 'Var29', 'Var30',
                      days_from_employment]

# %% md
### Get days from employment
# %%
d1 = pd.to_datetime(df['application_date'].copy(), format='%d%b%Y %H:%M:%S', errors='coerce')
d2 = pd.to_datetime(df['Var13'], format='%d%b%Y', errors='coerce')
df[days_from_employment] = (d1 - d2).dt.days
# %% md
## Descriptive analysis
# %% md
### Descirption of numerical values

# %%
df[numerical_features].describe()
# %% md
### Description of categorical values
# %%
categorical_features_desc = pd.DataFrame(columns=['value', 'proportion', 'column'])

for feature in categorical_features + binary_features:
    x = df[feature].value_counts(normalize=True).reset_index().rename(columns={feature: 'value'})
    x['column'] = feature
    categorical_features_desc = pd.concat([categorical_features_desc, x], axis=0, ignore_index=True)

categorical_features_desc
# %% md
## Missing value analysis
# %% md
list_of_nans = df.isna().sum()
columns_w_nans = list_of_nans[list_of_nans > 0]

# Set up an empty dict for results
nan_values_correlation = {}

for column in list(columns_w_nans.index):
    temp_df = df[[column, 'target']].copy()
    # Divde column values into NaN and not-NaN values
    temp_df[column + '_na'] = temp_df[column].isna().apply(lambda x: 'none' if x else 'not_none')
    # Get a crosstab
    crosstab = pd.crosstab(temp_df[column + '_na'], temp_df['target'])
    # Get p-value
    chi2, p, dof, expected = chi2_contingency(crosstab)
    nan_values_correlation[column] = p
# %%
missing_values_table = pd.concat([columns_w_nans / df.shape[0], pd.Series(nan_values_correlation)
                                  ], axis=1, keys=['missing_rate', 'p-value']).reset_index(names='column')
missing_values_table['data type'] = missing_values_table['column'].apply(
    lambda x: 'categorical' if x in categorical_features else ('numerical' if x in numerical_features else 'other'))
missing_values_table.merge(right=description, left_on='column', right_on='Column')
missing_values_table

# %%
backupt = df.copy()
# %%
fill_with_zero = ['Var8', 'Var25', 'Var26', days_from_employment]
add_other_category = ['Var18', 'Var19', 'Var2', 'Var3']

df['Var17'].fillna(df['Var17'].median(), inplace=True)

for var in fill_with_zero:
    df[var].fillna(0, inplace=True)

for var in add_other_category:
    df[var].fillna('other', inplace=True)

columns_to_drop = ['Var10', 'Var12']

df.drop(columns=columns_to_drop, inplace=True)

# %%
need_dummies = set(categorical_features) - set(columns_to_drop)

for feature in need_dummies:
    one_hot = pd.get_dummies(df[feature], prefix=feature, drop_first=True).astype(int)
    df = df.drop(feature, axis=1)
    df = df.join(one_hot)

# %% md
### Correlation matrix
# %%
get_correlation = list(set(numerical_features + [days_from_employment]) - set(columns_to_drop))

# Get correlation matrix
corr_matrix = df[get_correlation].corr()

# Flatten correlation matrix for visibility
corr_flattened = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
corr_flattened
pairwise_corr = corr_flattened.stack().reset_index()
pairwise_corr.columns = ['Feature_1', 'Feature_2', 'Correlation']

# Get absolute values (for negativly and positively correlated)
pairwise_corr['Correlation_abs'] = pairwise_corr['Correlation'].abs()

correlation_results = pairwise_corr.sort_values(by='Correlation_abs', ascending=False)
correlation_results[correlation_results['Correlation_abs'] > 0.7]

# %%
# Get features and label
X_vip = df[get_correlation]
Y_vip = df['target']

# Create and fit PLS Regression model
pls = PLSRegression(n_components=2)
pls.fit(X_vip, Y_vip)

# Extract model parameters
weights = pls.x_weights_
variance = pls.x_scores_.var(axis=0)
total_variance = variance.sum()
n_components = pls.n_components
n_predictors = X_vip.shape[1]

# Calculare VIP scores
vip_scores = np.sqrt(n_predictors * (weights ** 2 * variance / total_variance).sum(axis=1) / n_components)

# Get column names
variable_names = X_vip.columns
vip_dict = dict(zip(variable_names, vip_scores))

vip_points = pd.Series(vip_dict, name='vip').sort_values(ascending=False).reset_index()
vip_points

# %%
vif_data = pd.DataFrame({
    'Feature': X_vip.columns,
    'VIF': [variance_inflation_factor(X_vip.values, i) for i in range(X_vip.shape[1])]
})
vif_data
# %%
# Join both metrics
vip_points = vip_points.merge(right=vif_data, left_on='index', right_on='Feature').drop(columns=['Feature'])
# %%
pd.merge(
    pd.merge(
        correlation_results[correlation_results['Correlation_abs'] > 0.7], vip_points, left_on='Feature_1',
        right_on='index'),
    vip_points, left_on='Feature_2', right_on='index').drop(columns=['index_x', 'index_y'])

# %%
other_to_drop = ['Var22', 'Var23', 'Var21', 'Var16', 'Var4', 'Var30']
# %%
df.drop(columns=other_to_drop, inplace=True)
# %%
vip_points.sort_values(by='vip', ascending=False)

# %%
# Drop not feature columns
df.drop(columns=not_features, inplace=True)
# %%
X = df.drop('target', axis=1)
y = df['target']
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score

# Normalize the data
scaler_norm = MinMaxScaler()
X_train_norm = scaler_norm.fit_transform(X_train)
X_val_norm = scaler_norm.transform(X_val)
X_test_norm = scaler_norm.transform(X_test)

# Standardize the data
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_val_std = scaler_std.transform(X_val)
X_test_std = scaler_std.transform(X_test)

# Initialize performance tracking
performance_metrics = {
    'Normalization': {'Best K': None, 'Validation AUC': 0, 'Test AUC': 0},
    'Standardization': {'Best K': None, 'Validation AUC': 0, 'Test AUC': 0}
}

# Range of Ks to evaluate
k_values = range(1, 500)

# Evaluate models with varying K and preprocessing methods
for preprocess_method, X_train_preprocessed, X_val_preprocessed in [
    ('Normalization', X_train_norm, X_val_norm),
    ('Standardization', X_train_std, X_val_std)
]:
    for k in tqdm(k_values):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_preprocessed, y_train)
        val_pred_probs = knn.predict_proba(X_val_preprocessed)[:, 1]  # Probability estimates for the positive class
        val_auc = roc_auc_score(y_val, val_pred_probs)

        if val_auc > performance_metrics[preprocess_method]['Validation AUC']:
            performance_metrics[preprocess_method]['Best K'] = k
            performance_metrics[preprocess_method]['Validation AUC'] = val_auc

# Evaluate best models on test set
for preprocess_method, X_train_preprocessed, X_test_preprocessed in [
    ('Normalization', X_train_norm, X_test_norm),
    ('Standardization', X_train_std, X_test_std)
]:
    best_k = performance_metrics[preprocess_method]['Best K']
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_preprocessed, y_train)
    test_pred_probs = knn.predict_proba(X_test_preprocessed)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred_probs)
    performance_metrics[preprocess_method]['Test AUC'] = test_auc

# Output the performance metrics
print(performance_metrics)