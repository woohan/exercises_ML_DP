import pandas as pd
import numpy as np


selected_columns = ['age','experience','income','family','ccavg','education','mortgage','securities_account','cd_account','online','creditcard','personal_loan']
df = pd.read_csv("./sourceData/bank_Loan.csv", usecols=selected_columns)

test_df = df[df['personal_loan']==1]
test_df.to_csv('./outputs/bankloan_positive.csv',index=False)
