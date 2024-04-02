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
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
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
import shap

explainer = shap.TreeExplainer(model_XGB,
    feature_perturbation="interventional",
    model_output="probability",
    data=X_val
                               )

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