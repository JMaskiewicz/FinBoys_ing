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


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_tree_and_logistic_models(X_train, y_train, max_leaf_nodes):
    tree_model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=1, min_samples_split=50)
    tree_model.fit(X_train, y_train)

    leaf_ids = tree_model.apply(X_train)

    models = []
    scalers = []
    leaf_info = []

    # Train logistic regression models on each leaf node
    for leaf_id in np.unique(leaf_ids):
        leaf_X_train = X_train[leaf_ids == leaf_id]
        leaf_y_train = y_train[leaf_ids == leaf_id]

        # Check if the leaf node contains samples from more than one class
        if len(np.unique(leaf_y_train)) > 1:
            scaler = StandardScaler().fit(leaf_X_train)
            leaf_X_train_scaled = scaler.transform(leaf_X_train)

            model = LogisticRegression(random_state=1)
            model.fit(leaf_X_train_scaled, leaf_y_train)

            models.append(model)
            scalers.append(scaler)
            leaf_info.append((leaf_id, 'LR Model Trained'))
        else:
            # For leaf nodes with only one class
            models.append(None)
            scalers.append(None)
            leaf_info.append((leaf_id, f'Single class: {np.unique(leaf_y_train)[0]}'))

    return tree_model, models, scalers, leaf_info


def predict_with_tree_splits(X, tree_model, models, scalers, leaf_info):
    # Predict leaf node IDs for X
    leaf_ids = tree_model.apply(X)

    # Initialize an empty list to store predictions
    probabilities = []

    # Iterate over each sample in X
    for i in tqdm(range(len(X))):
        leaf_id = leaf_ids[i]

        # Find the model and scaler for the current leaf_id
        if leaf_id in [info[0] for info in leaf_info if 'LR Model Trained' in info[1]]:
            model_idx = [info[0] for info in leaf_info].index(leaf_id)
            model = models[model_idx]
            scaler = scalers[model_idx]

            # Scale the features for the current sample
            X_scaled = scaler.transform(X.iloc[[i]])

            # Predict probabilities using the logistic regression model
            pred_probs = model.predict_proba(X_scaled)

            prob = pred_probs[:, 1]
        else:
            class_label = [info[1].split(':')[1].strip() for info in leaf_info if info[0] == leaf_id][0]
            prob = np.array([1.0 if class_label == '1' else 0.0])

        probabilities.append(prob[0])

    return np.array(probabilities)


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

# 0.77 to beat
# %%
tree_model, models, scalers, leaf_info = train_tree_and_logistic_models(X_train, y_train, max_leaf_nodes=8)

y_train_pred_prob = predict_with_tree_splits(X_train, tree_model, models, scalers, leaf_info)
y_val_pred_prob = predict_with_tree_splits(X_val, tree_model, models, scalers, leaf_info)
y_test_pred_prob = predict_with_tree_splits(X_test, tree_model, models, scalers, leaf_info)

fpr_train, tpr_train, _ = roc_curve(y_train.astype(int), y_train_pred_prob)
plot_roc_curve(fpr_train, tpr_train, 'Training Set ROC Curve Tree - Logit')

fpr_val, tpr_val, _ = roc_curve(y_val.astype(int), y_val_pred_prob)
plot_roc_curve(fpr_val, tpr_val, 'Val Set ROC Curve Tree - Logit')

fpr_test, tpr_test, _ = roc_curve(y_test.astype(int), y_test_pred_prob)
plot_roc_curve(fpr_test, tpr_test, 'Test Set ROC Curve Tree - Logit')

# test set