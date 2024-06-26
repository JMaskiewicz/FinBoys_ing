# FinBoys_ing

This repository is crafted for the ING Hackathon for team FinBoys.



### time managment plan (8h)
0.5 - task reading 
2.5h – data preparation
1h – modeling 
2h – description
1h – reserve time 


## Folders:

### 1. Data
- `pklt/csv`: Raw data files

### 2. Data Edit
- **a. Funkcje do edytowania danych**: Functions for data editing
- **b. Analysis of nans**: Analyzing missing values
- **c. Drop useless variables/cols**: Removing unnecessary variables
- **d. Outlier**: Identifying and handling outliers
- **e. Standarize/normalize data**: Standardizing or normalizing data
- **f. One-hot encoding**: Applying one-hot encoding to categorical variables (worth to remember that some variables could not be present in test)
- **g. PCA?**: Principal Component Analysis (optional)
- **h. VIF and VIP**: Checking Variance Inflation Factor and Variable Importance in Projection
- **i. Korelacje**: Correlation analysis
- **j. Some random time functions?**: Miscellaneous time-related functions
- **h. REF** - https://machinelearningmastery.com/rfe-feature-selection-in-python/
- **l. KNN filling**
- 

### 3. Models
- **a. XGboost**: XGBoost algorithm
- **b. Logit**: Logistic regression
- **c. SVM/KNN**: Support Vector Machines and K-Nearest Neighbors
- **d. Tree + logit**: Decision Trees combined with Logistic Regression
- **e. NN**: Neural Networks

### 4. Evaluate/Test
- **a. K-fold**: K-fold cross-validation
- **b. Statistics**: Metrics like accuracy, balanced accuracy, F1, etc.
- **c. PFI** https://scikit-learn.org/stable/modules/permutation_importance.html

### 5. Plots
- **a. ROC**: Receiver Operating Characteristic curves
- **b. SHAP**: SHAP values for model explanation
- **c. LIME**: Local Interpretable Model-agnostic Explanations
- **d. Confusion matrix**: Confusion matrices for classification tasks
- **e. For specific models?**: Additional model-specific plots

### 7. MAIN – Jupyter Notebook
- **a. Ensemble predictions?**
  - **i. Majority voting**: Combining model predictions by voting
  - **ii. Weights**: Weighted combination of model predictions
