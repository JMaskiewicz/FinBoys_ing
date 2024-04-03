# %%
from PythonAPI import *
# %%
user = "team01"
password = "rg66u8gG}%7n"
check_connection()

x_variables = ["Dependents", "Education", "Self_Employed", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"]
y_variables = "target"

# %% Load
load_data(user, password, "trainData.csv",
         x_variables,
         y_variables)
check_status(user, password)
# %%
params = get_parameters(user, password)
train(user, password, 100, 100, 0.1, "Adam", "SU2U4")
check_status(user, password)
# %%
validate(user, password, "testData.csv",
         x_variables,
         y_variables, "SU2U4", params)
