"""This module contains different functions for missing values analysis."""

import pandas as pd 

def get_missing_values_columns(df: pd.DataFrame) -> pd.Series: 
    return df.columns[df.isna().any()]

def get_missing_values_report(df: pd.DataFrame) -> pd.DataFrame: 
    df_mva = df.isna().sum().reset_index(name='test')
    return df_mva

def get_missing_values_importance(df: pd.DataFrame, target: str = "target") -> pd.DataFrame: 
    missing_columns = get_missing_values_columns(df=df)
    
    for column in missing_columns: 
        print(column)

    return 0 
# description = pd.read_excel('https://challengerocket.com/files/lions-den-ing-2024/variables_description.xlsx', header=0)
train = pd.read_csv('https://files.challengerocket.com/files/lions-den-ing-2024/development_sample.csv')
train=train.dropna(subset=['target'])


# print(get_missing_values_columns(train))


print(get_missing_values_importance(df=train))

# list_of_nans = df.isna().sum()
# columns_w_nans = list_of_nans[list_of_nans>0]

# # Set up an empty dict for results
# nan_values_correlation = {}

# for column in list(columns_w_nans.index):
#   temp_df = df[[column, 'target']].copy()
#   # Divde column values into NaN and not-NaN values
#   temp_df[column + '_na'] = temp_df[column].isna().apply(lambda x: 'none' if x else 'not_none')
#   # Get a crosstab
#   crosstab = pd.crosstab(temp_df[column + '_na'], temp_df['target'])
#   # Get p-value
#   chi2, p, dof, expected = chi2_contingency(crosstab)
#   nan_values_correlation[column]=p