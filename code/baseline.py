'''
    This code generates baseline submission
'''

import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tools import *


# read data
competition_name = 'santander-value-prediction-challenge'
train_path = './data/train.csv'
test_path = './data/test.csv'

trn = pd.read_csv(train_path)
tst = pd.read_csv(test_path)


# split trn/val
drop_cols = ['ID', 'target']
X = trn.drop(drop_cols, axis=1)
y = trn.target.values.astype(np.int32)
x_trn, x_val, y_trn, y_val = train_test_split(X, y, test_size=0.2, random_state=2018)


# No Feature Engineering


# Fit model 
model = RandomForestRegressor(n_estimators=20, oob_score=True, 
                              max_depth=10, min_samples_leaf=10, 
                              random_state=2018, n_jobs=-1)
model.fit(x_trn, y_trn)


# Evaluate model
evaluate(model, x_trn, y_trn, x_val, y_val, rmsle)
'''
# trn loss : 1.86302689495
# val loss : 1.99915397023
# trn R2 : 0.349567050331
# val R2 : 0.292105707347
# OOB_score : 0.178236166967
'''

# Make prediction
x_tst = tst.drop(['ID'], axis=1)
tst_pred = model.predict(x_tst)
tst_subm = pd.DataFrame({'ID': tst.ID, 'target': tst_pred})
baseline_fname = 'baseline.20181017'
tst_subm.to_csv(baseline_fname, index=False)


# Submit to Kaggle
msg = 'baseline.rf.noFE'
submit_kaggle_result(competition_name, baseline_fname, msg)

# Check result
check_kaggle_result(competition_name, baseline_fname)
'''
fileName           date                 description       status    publicScore  privateScore
-----------------  -------------------  ----------------  --------  -----------  ------------
baseline.20181017  2018-10-17 08:52:46  baseline.rf.noFE  complete  1.93257      1.87086
'''
