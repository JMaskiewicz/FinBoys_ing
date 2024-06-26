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
other_to_drop = ['Var22', 'Var23', 'Var21', 'Var16', 'Var4']
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
'''from sklearn.feature_selection import RFE

# Initialize an XGBoost classifier for use with RFE
xgb_for_rfe = XGBClassifier()

# Initialize RFE with the XGBoost model, selecting the top 10 features
rfe = RFE(estimator=xgb_for_rfe, n_features_to_select=20)

# Fit RFE on the training data used for tuning (before applying SMOTE)
rfe.fit(X_train_tuning, y_train_tuning)

# Transform the datasets to include only the selected features
X_train_tuning_rfe = rfe.transform(X_train_tuning)
X_val_rfe = rfe.transform(X_val)
X_test_rfe = rfe.transform(X_test)'''
# %%

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

# Assuming X, y are defined, and X_train_tuning, X_val, y_train_tuning, y_val are already split as per the instructions.

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
    #'cubic': np.power(X_train_tuning[column], 3),
    #'reciprocal_sqrt': 1 / np.sqrt(X_train_tuning[column] + np.abs(X_train_tuning[column].min())),
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