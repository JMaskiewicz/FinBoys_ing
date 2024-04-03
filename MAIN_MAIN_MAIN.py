import random
from tqdm import tqdm
import re

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report, \
    roc_curve, auc, roc_auc_score

import matplotlib.pyplot as plt

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
# %% md
## Let's load the dataset
df = pd.read_csv(r'C:\Users\jmask\Desktop\ING\in_time.csv')

columns_drop = ['External_term_loan_balance', 'External_mortgage_balance', 'External_credit_card_balance']
df = df.drop(columns=columns_drop)
df['Last_Income'] = df['Income_H0']
#df = df[:10000]
#%%
# new 3 variables
# List of time points
time_points = ['_H' + str(i) for i in range(0, 13)]

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


time_linear_variables = {
 'TOTAL_amount_balance',
 'Income',
 'Diff_transactions',
}
#%%
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
            #series_values = series_values/series_values[0] if series_values[0] != 0 else series_values
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
# already in the csv

import re

columns_to_keep = [col for col in df.columns if not re.search(r'H\d+', col)]

df_filtered = df[columns_to_keep]

columns = ["Birth_date", "Contract_origination_date", "Contract_end_date", "Oldest_account_date"]

ref = pd.to_datetime(df_filtered['Ref_month'], format='%m-%Y')
for col in columns:
    d = pd.to_datetime(df_filtered[col], format='%d-%m-%Y')
    df_filtered[col + '_diff'] = abs(ref - d).dt.days
    # print(df_filtered[col + '_diff'])

columns_to_replace = ['Active_accounts', 'Active_loans', 'Active_mortgages']

# Replace -9999 with 0 in the specified columns
df_filtered[columns_to_replace] = df_filtered[columns_to_replace].replace(-9999, 0)

#%%
df_filtered = df_filtered.drop(columns=['Ref_month', 'Birth_date', 'Contract_origination_date', 'Contract_end_date', 'Oldest_account_date', 'Customer_id'])

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
# data split
X = df_filtered.drop('Target', axis=1)
y = df_filtered['Target']
X, y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train_tuning, X_val, y_train_tuning, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


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
    model.set_params(**{'early_stopping_rounds': 25})
    # Make sure to use the resampled and RFE-transformed dataset
    model.fit(X_train_XGB_resampled, y_train_XGB_resampled, eval_set=[(X_val, y_val)], verbose=False)

    preds_proba = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, preds_proba)

    return auc_score

# %%
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
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
# 4 variables related to trends
df['Diff_transactions_variance_pct'] = np.sqrt(df['Diff_transactions_variance_pct'])
df['TOTAL_amount_balance_variance_pct'] = np.sqrt(df['TOTAL_amount_balance_variance_pct'])
df['TOTAL_amount_balance_trend_angle'] = np.sin(df['TOTAL_amount_balance_trend_angle'])

# Assuming Active_loans column exists in your dataset for the next operation
df['Active_loans_Current_installment'] = df['Active_loans'] * df['Current_installment']
df['Ratio'] = df['Current_installment'] / df['Last_Income']

DF_FINAL = df[['Diff_transactions_variance_pct', 'TOTAL_amount_balance_variance_pct',
               'TOTAL_amount_balance_trend_angle', 'Active_loans_Current_installment',
                        'Num_borrowers', 'Ratio', 'Target']]

# as function
def transform_dataframe(df):
    df_copy = df.copy()

    # Apply transformations
    df_copy['Diff_transactions_variance_pct'] = np.sqrt(df_copy['Diff_transactions_variance_pct'])
    df_copy['TOTAL_amount_balance_variance_pct'] = np.sqrt(df_copy['TOTAL_amount_balance_variance_pct'])
    df_copy['TOTAL_amount_balance_trend_angle'] = np.sin(np.radians(df_copy[
                                                                        'TOTAL_amount_balance_trend_angle']))

    # New features based on existing ones
    df_copy['Active_loans_Current_installment'] = df_copy['Active_loans'] * df_copy['Current_installment']
    df_copy['Ratio'] = df_copy['Current_installment'] / df_copy['Last_Income']

    # Select specific features for the final DataFrame
    DF_FINAL = df_copy[['Diff_transactions_variance_pct', 'TOTAL_amount_balance_variance_pct',
                        'TOTAL_amount_balance_trend_angle', 'Active_loans_Current_installment',
                        'Num_borrowers', 'Ratio', 'Target']]
    DF_FINAL.fillna(DF_FINAL.mean(), inplace=True)
    column_means = DF_FINAL.mean(skipna=True)

    return DF_FINAL, column_means

DF_FINAL, imputation_means = transform_dataframe(df)



#%%
X2 = df.drop('Target', axis=1)
y2 = df['Target']
X2, y2

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=1)
X_train_tuning, X_val, y_train_tuning, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# FL - financial liabilities
X_train_FL = X_train_tuning
X_val_FL = X_val
X_test_FL = X_test
y_train_FL = y_train_tuning
y_val_FL = y_val
y_test_FL = y_test

import statsmodels.api as sm
# %%
def get_financial_liabilities(data: pd.DataFrame, y: pd.Series) -> pd.Series:
    columns_financial_liabilities = ["Credit_cards",
                                     "Active_credit_card_lines",
                                     "Debit_cards",
                                     "Active_loans",
                                     "Active_mortgages",
                                     "Active_accounts"]

    X = data[columns_financial_liabilities].replace(-9999, 0)  # we replaced -9999 with zero
    # Add a constant to the features (important for statsmodels)
    X = sm.add_constant(X)

    # Fit the logistic regression model
    model = sm.Logit(y, X).fit()

    # Apply the coefficients as weights to sum all columns in each row
    # Note: We skip the first coefficient as it's the intercept
    coefficients = model.params[1:]  # Exclude intercept
    weighted_sum = (X.iloc[:, 1:] * coefficients.values).sum(axis=1)
    return weighted_sum


get_financial_liabilities(X_train_FL, y_train_FL)

# %%
import statsmodels.api as sm

# %%

X_train_ATT = X_train_tuning
X_val_ATT = X_val
X_test_ATT = X_test
y_train_ATT = y_train_tuning
y_val_ATT = y_val
y_test_ATT = y_test

# %%

def max_by_mean(data: pd.DataFrame, prefix: str) -> pd.Series:
    data_for_prefix = data[data.columns[data.columns.str.startswith(prefix)]]
    data_for_prefix_adj = data_for_prefix + 1

    return data_for_prefix_adj.max(axis=1) / data_for_prefix_adj.mean(axis=1)


def variance(data: pd.DataFrame, prefix: str) -> pd.Series:
    data_for_prefix = data[data.columns[data.columns.str.startswith(prefix)]]

    return data_for_prefix.var(axis=1)


# %%

X_train_ATT.isna().any().any()
# %%

from sklearn.tree import DecisionTreeClassifier


# %%

def get_data_attention(data: pd.DataFrame):
    time_series_data = ['DPD_credit_card',
                        'DPD_mortgage',
                        'DPD_term_loan',
                        'Default_flag',
                        'Os_credit_card',
                        'Os_mortgage',
                        'Overdue_credit_card',
                        'Overdue_mortgage',
                        'Overdue_term_loan', ]

    df = data.copy()
    data_attention = pd.DataFrame()

    for columns in time_series_data:
        data_attention[f'var_{columns}'] = variance(data, columns)
        data_attention[f'max_by_mean_{columns}'] = max_by_mean(data, columns)

    return data_attention


get_data_attention(df)


# %%
def attention_tree_model(data: pd.DataFrame, y: pd.Series) -> pd.Series:
    X = get_data_attention(data)
    print(data.columns)
    # Initialize and train classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X, y)

    return dt_classifier

attention_tree_model = attention_tree_model(get_data_attention(X_train_ATT), y_train_ATT)

a = attention_tree_model.predict_proba(get_data_attention(X_test_ATT))[:, 1]


# %%
# add predictions of 3 models
# add XGBoost results
full_predictions = model_XGB.predict_proba(X)[:, 1]

financial_liabilities_scores = get_financial_liabilities(X2, y2)

# Step 2: Generate Attention Model Predictions for the whole dataset
attention_model_predictions = attention_tree_model.predict_proba(get_data_attention(X2))[:, 1]

# Add the predictions and scores as new columns to DF_FINAL
DF_FINAL['Financial_Liabilities_Score'] = financial_liabilities_scores
DF_FINAL['Attention_Model_Prediction'] = attention_model_predictions
DF_FINAL['XGB_Predictions'] = full_predictions

train_indices = X_train_tuning.index
val_indices = X_val.index
test_indices = X_test.index

# Now, create the splits using these indices
DF_TRAIN_FINAL = DF_FINAL.loc[train_indices]
DF_VALIDATION_FINAL = DF_FINAL.loc[val_indices]
DF_TEST_FINAL = DF_FINAL.loc[test_indices]


#%%
# final input
# 1 XGBoost model
# 2 - active cards MODEL
# 3 attention model
# 'Diff_transactions_variance_pct',
# 'TOTAL_amount_balance_variance_pct',
# 'TOTAL_amount_balance_trend_angle',
# 'Active_loans_Current_installment',
# 'Num_borrowers',
# 'Ratio'

from sklearn.preprocessing import MinMaxScaler

# logit on final 9 variables
DF_TRAIN_FINAL = DF_FINAL.loc[train_indices]
DF_VALIDATION_FINAL = DF_FINAL.loc[val_indices]
DF_TEST_FINAL = DF_FINAL.loc[test_indices]


X_train_FINAL = DF_TRAIN_FINAL.drop('Target', axis=1)
y_train_FINAL = DF_TRAIN_FINAL['Target']
X_val_FINAL = DF_VALIDATION_FINAL.drop('Target', axis=1)
y_val_FINAL = DF_VALIDATION_FINAL['Target']
X_test_FINAL = DF_TEST_FINAL.drop('Target', axis=1)
y_test_FINAL = DF_TEST_FINAL['Target']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the training feature data and transform it
X_train_FINAL_scaled = scaler.fit_transform(X_train_FINAL)

# Transform the validation and test feature data based on the fitted scaler
X_val_FINAL_scaled = scaler.transform(X_val_FINAL)
X_test_FINAL_scaled = scaler.transform(X_test_FINAL)

# Convert the scaled arrays back to DataFrames for ease of use later
X_train_FINAL_scaled = pd.DataFrame(X_train_FINAL_scaled, columns=X_train_FINAL.columns, index=X_train_FINAL.index)
X_val_FINAL_scaled = pd.DataFrame(X_val_FINAL_scaled, columns=X_val_FINAL.columns, index=X_val_FINAL.index)
X_test_FINAL_scaled = pd.DataFrame(X_test_FINAL_scaled, columns=X_test_FINAL.columns, index=X_test_FINAL.index)
#%%
logit_model = LogisticRegression(max_iter=1000)
logit_model.fit(X_train_FINAL_scaled, y_train_FINAL)

predictions = logit_model.predict(X_test_FINAL_scaled)
print(f"Balanced accuracy: {balanced_accuracy_score(y_test_FINAL, predictions)}")
print(f"Accuracy         : {accuracy_score(y_test_FINAL, predictions)}")

# Evaluate the model using the simple evaluation function
print("Train set ENSEMBLE:")
evaluate_predictions(y_train_FINAL, logit_model.predict(X_train_FINAL_scaled))
print("Validation set ENSEMBLE:")
evaluate_predictions(y_val_FINAL, logit_model.predict(X_val_FINAL_scaled))
print("Test set ENSEMBLE:")
evaluate_predictions(y_test_FINAL, predictions)

# %%
y_train_pred_prob = logit_model.predict_proba(X_train_FINAL_scaled)[:, 1]
fpr_train, tpr_train, _ = roc_curve(y_train_FINAL.astype(int).values, y_train_pred_prob)
plot_roc_curve(fpr_train, tpr_train, 'Training Set ROC Curve ENSEMBLE')

y_val_pred_prob = logit_model.predict_proba(X_val_FINAL_scaled)[:, 1]
fpr_val, tpr_val, _ = roc_curve(y_val_FINAL.astype(int).values, y_val_pred_prob)
plot_roc_curve(fpr_val, tpr_val, 'Validation Set ROC Curve ENSEMBLE')

y_test_pred_prob = logit_model.predict_proba(X_test_FINAL_scaled)[:, 1]
fpr_test, tpr_test, _ = roc_curve(y_test_FINAL.astype(int).values, y_test_pred_prob)
plot_roc_curve(fpr_test, tpr_test, 'Test Set ROC Curve ENSEMBLE')
