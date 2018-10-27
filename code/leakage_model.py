'''
    Python implementation of Jack's solution
    - link : https://www.kaggle.com/rsakata/21st-place-solution-bug-fixed-private-0-52785
'''

import pandas as pd
import numpy as np
import xgboost as xgb
from tools import *

trn = pd.read_csv('../data/train.csv')
tst = pd.read_csv('../data/test.csv')

trn['set'] = 'train'
tst['set'] = 'test'
tst['target'] = np.nan

data = pd.concat([trn, tst], axis=0, sort=False).reset_index(drop=True)

# 40 time-series columns (source of leakage)
target_cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
               '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
               'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b',
               '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992',
               'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd',
               '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
               '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2',
               '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']

# align time-series column to lag
leaky_cols = pd.DataFrame({'col': target_cols})
leaky_cols['lag'] = np.arange(len(leaky_cols))
leaky_cols['latest'] = target_cols[0]

# get leak data
for it in range(1,3):
    print('# iter : {}'.format(it))
    leaky_rows = pd.DataFrame()

    ## get pairwise lag data ('ID' : current ID, 'ID_prev' : lag ID, 'lag' : amount of time diff) ##
    # map each 'ID' to its potential leaking 'ID_prev'
    for i in range(1, 40):
        print('# i = {}'.format(i))

        # align colnames by lag value
        # key_x : lag column name
        # key_y : lag - 1 column name (more recent, later ID_prev)
        tmp = leaky_cols.copy()
        tmp['lag'] += i
        tmp = leaky_cols.merge(tmp, on=['latest','lag'], how='inner')
        key_x = tmp.col_x
        key_y = tmp.col_y

        if it == 1:
            # extract rows with unique values > 2 for train, > 3 for test
            nuniq = data[key_x].nunique(axis=1)
            trn_tmp = (nuniq > 2) & (data['set'] == 'train')
            tst_tmp = (nuniq > 3) & (data['set'] == 'test')
            tmp = trn_tmp | tst_tmp
        else:
            tmp = (data[key_x] > 0).sum(axis=1) > 0

        # dataframe 'right' has colnames shifted by lag value
        # save 'ID_prev' in leaky_row_temp which are potential leaky rows
        # leaky row is defined by time-series columns having same value after lag-shift
        left = data.loc[tmp, list(key_x) + ['set', 'ID', 'target']]
        right = data[list(key_y) + ['set', 'ID']]
        leaky_row_temp = left.merge(right, left_on=(list(key_x) + ['set']), right_on=(list(key_y) + ['set']),
                                    how='inner', suffixes=('', '_prev'))[list(key_x) + ['set', 'ID', 'ID_prev', 'target']]

        # add to leaky_row if both 'ID' and 'ID_prev' is unique
        if leaky_row_temp.shape[0] > 0:
            leaky_row_temp = leaky_row_temp[['ID', 'ID_prev', 'target']]
            leaky_row_temp['lag'] = i

            if leaky_rows.shape[0] == 0:
                leaky_rows = leaky_row_temp
            else:
                leaky_rows_id_list = leaky_rows.ID.unique().tolist()
                leaky_rows_prev_id_list = leaky_rows.ID_prev.unique().tolist()
                idx = (leaky_row_temp.ID.apply(lambda x : x not in leaky_rows_id_list)) & (leaky_row_temp.ID_prev.apply(lambda x : x not in leaky_rows_prev_id_list))
                leaky_row_temp = leaky_row_temp[idx].reset_index(drop=True)
                leaky_rows = pd.concat([leaky_rows, leaky_row_temp], axis=0, sort=False).reset_index(drop=True)

    # omit confusing leaks
    leaky_rows['ID_lag_concat'] = leaky_rows.ID + '_' + leaky_rows.lag.map(str)
    leaky_rows['ID_prev_lag_concat'] = leaky_rows.ID_prev + '_' + leaky_rows.lag.map(str)
    omit_id = [l.split('_')[0] for l in leaky_rows.ID_lag_concat.value_counts()[leaky_rows.ID_lag_concat.value_counts() > 1].index.tolist()]
    omit_id_prev = [l.split('_')[0] for l in leaky_rows.ID_prev_lag_concat.value_counts()[leaky_rows.ID_prev_lag_concat.value_counts() > 1].index.tolist()]

    leaky_rows = leaky_rows[leaky_rows.ID.apply(lambda x: x not in omit_id)]
    leaky_rows = leaky_rows[leaky_rows.ID_prev.apply(lambda x: x not in omit_id_prev)]
    leaky_rows = leaky_rows[['ID', 'ID_prev', 'target', 'lag']].reset_index(drop=True)

    ## find the first row for each customer ##
    leaky_rows['ID_first'] = leaky_rows.ID_prev
    while True:
        # by merging on 'ID_first' and 'ID', we recursively reach up to the most recent 'ID_first'
        # equivalent to the first row of each customer
        tmp = leaky_rows.merge(leaky_rows[['ID', 'ID_first', 'lag']],
                           left_on='ID_first', right_on='ID', how='left')[['ID_first_x', 'ID_first_y', 'lag_y']]
        tmp.columns = ['ID_first', 'ID_first_y', 'lag']

        if (~tmp.lag.isnull()).sum() > 0:
            # update 'ID_first' and add lag values
            leaky_rows.ID_first[~tmp.lag.isnull()] = tmp.ID_first_y[~tmp.lag.isnull()]
            leaky_rows.lag[~tmp.lag.isnull()] = leaky_rows.lag[~tmp.lag.isnull()] + tmp.lag[~tmp.lag.isnull()]
            leaky_rows[leaky_rows.lag > 1e6] = np.nan
        else:
            break
    leaky_rows = leaky_rows[~leaky_rows.lag.isnull()][['ID', 'target', 'ID_first', 'lag']]

    # add all rows with ID in 'ID_first' from leaky_rows (their lag is zero, because they are the most recent rows)
    leaky_rows_id_list = leaky_rows.ID_first.unique().tolist()
    tmp = data[data.ID.apply(lambda x: x in leaky_rows_id_list)][['ID', 'target']]
    tmp['ID_first'] = tmp.ID
    tmp['lag'] = 0
    leaky_rows = pd.concat([leaky_rows, tmp], axis=0, sort=False).sort_values(by=['ID_first', 'lag'])


    ## get the target values by leaks ##
    leaky_rows['target_leak'] = np.nan
    leaky_rows = leaky_rows.merge(data[['ID'] + target_cols], on='ID', how='inner')
    for i in range(1, 41):
        tmp = leaky_rows[['ID_first', 'lag']]
        tmp['lag'] += i + 1

        # select lag values
        temp = tmp.merge(leaky_rows[['ID_first', 'lag'] + [target_cols[i - 1]]], on=['ID_first', 'lag'], how='left')
        idx = leaky_rows.target_leak.isnull()
        # get lag values!
        leaky_rows.loc[idx, 'target_leak'] = temp[target_cols[i - 1]][idx]

    temp = leaky_rows[~leaky_rows.target.isnull()]
    temp['error'] = np.abs(temp.target - temp.target_leak)
    print('# train : {} leaks are found, {} leaks are correct'.format((~temp.target_leak.isnull()).sum(), (temp.error == 0).sum()))

    temp = leaky_rows[leaky_rows.target.isnull()]
    print('# test : {} leaks are found'.format((~temp.target_leak.isnull()).sum()))

    ##### find more leaky columns by leaky rows #####
    # pairs of ['ID' ~ 'ID_prev'] with lag-diff of (-1) is chosen. hence "before"
    before = data.drop(['set'], axis=1).merge(leaky_rows[['ID_first', 'lag', 'ID']], on='ID')
    tmp = leaky_rows[['ID_first', 'lag']]
    tmp['lag'] -= 1
    before = before.merge(tmp, on=['ID_first', 'lag']).sort_values(by=['ID_first', 'lag'])

    # pairs of ['ID' ~ 'ID_prev'] with lag-diff of (+1) is chosen. hence "after"
    tmp = leaky_rows[['ID_first', 'lag', 'ID']]
    tmp['lag'] -= 1
    after = data.drop(['set'], axis=1).merge(tmp, on='ID')
    after = after.merge(leaky_rows[['ID_first', 'lag']], on=['ID_first', 'lag']).sort_values(by=['ID_first', 'lag'])

    nonzero_before = (before > 0).sum(axis=0).drop(['ID_first', 'lag', 'ID', 'target'], axis=0)
    nonzero_after = (after > 0).sum(axis=0).drop(['ID_first', 'lag', 'ID', 'target'], axis=0)

    leaky_cols = pd.DataFrame({'col': nonzero_before.index.tolist(),
                           'nonzero' : nonzero_before,
                           'latest' : '',
                           'lag' : 0}).reset_index(drop=True)
    valid_cols = leaky_cols[(nonzero_before > 10).values].col.values.tolist()

    for col1 in valid_cols:
        if leaky_cols[leaky_cols['col'] == col1].latest.values[0] != '':
            continue

        # find the latest column of col1
        latest_col1 = col1
        updated = True

        while(updated):
            updated = False
            # get candidate variables where sum of nonzero values are same
            cand_vars = [nonzero_before.index.tolist()[j] for j in np.where(nonzero_before == nonzero_after[latest_col1])[0]]
            temp = []

            # get latest column
            for col2 in cand_vars:
                if (before[col2].values != after[latest_col1].values).sum() == 0:
                    temp.append(col2)
            if len(temp) == 1:
                updated = True
                latest_col1 = temp[0]

        # update latest col
        leaky_cols.loc[leaky_cols.col == latest_col1, 'latest'] = latest_col1


        # find group columns of col1
        group_col1 = latest_col1
        lag = 0
        updated = True

        while(updated):
            updated = False
            cand_vars = [nonzero_after.index.tolist()[j] for j in np.where(nonzero_before[group_col1] == nonzero_after)[0]]
            temp = []

            for col2 in cand_vars:
                if (before[group_col1].values != after[col2].values).sum() == 0:
                    temp.append(col2)
            if len(temp) == 1:
                updated = True
                group_col1 = temp[0]
                lag += 1
                leaky_cols.loc[leaky_cols.col == group_col1, 'latest'] = latest_col1
                leaky_cols.loc[leaky_cols.col == group_col1, 'lag'] = lag

    # drop leaky cols with lag = 0
    latest_uniq = leaky_cols[leaky_cols.lag > 0].latest.unique().tolist()
    idx = leaky_cols.latest.apply(lambda x: x in latest_uniq)
    leaky_cols = leaky_cols[idx].sort_values(by=['latest', 'lag'])
    print('# Found {} column groups (total {} cols)'.format(leaky_cols.latest.nunique(), leaky_cols.shape[0]))

'''
iter 1:
          train:  3288 leaks are found, 3288 leaks are correct
          test:   6545 leaks are found
          Found 101 column groups (total 4040 columns)
iter 2:
          train:  3896 leaks are found, 3888 leaks are correct
          test:   8078 leaks are found
'''


## Feature Engineering ##
data[data == 0] = np.nan
trn = data[data.set == 'train'].merge(leaky_rows[['ID', 'target_leak']], on='ID', how='left')
tst = data[data.set == 'test'].merge(leaky_rows[['ID', 'target_leak']], on='ID', how='left')
del data

# add leaky rows of test data to training data
tst.loc[(tst['target_leak'] < 30000) | (tst['target_leak'] > 4e7), 'target_leak'] = np.nan
tst['target'] = tst['target_leak']
trn = pd.concat([trn, tst[~tst.target.isnull()]], axis=0, sort=False)
trn.target = np.log1p(trn.target)

temp = leaky_cols.sort_values(by='nonzero', ascending=False).drop_duplicates('latest', keep='first')[['latest', 'nonzero']].reset_index(drop=True)
for i in range(50):
    cols = leaky_cols[leaky_cols.latest.apply(lambda x: x in temp.latest[i])].col
    trn['mean_{}'.format(i)] = trn[cols].mean(axis=1)
    tst['mean_{}'.format(i)] = tst[cols].mean(axis=1)
    trn['logmean_{}'.format(i)] = np.log(trn[cols].mean(axis=1))
    tst['logmean_{}'.format(i)] = np.log(tst[cols].mean(axis=1))


## Model Training ##
exp_vars = target_cols + ['mean_{}'.format(i) for i in range(50)] + ['logmean_{}'.format(1) for i in range(50)]
params = {
    'eta' : 0.01,
    'gamma' : 0,
    'max_depth': 12,
    'min_child_weight': 8,
    'max_delta_step': 0,
    'subsample': 0.8,
    'colsample_bytree': 1,
    'colsample_bylevel': 0.4,
    'lambda' : 1,
    'alpha': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': np.mean(trn.target),
    'nthread': 10
}

bag = 10
for seed in range(bag):
    print('# iter : {}'.format(seed))
    
    x_trn = trn[exp_vars]
    y_trn = trn.target
    dtrain = xgb.DMatrix(data=np.asmatrix(x_trn), label=y_trn)
    del x_trn
    x_tst = tst[exp_vars]
    dtest = xgb.DMatrix(data=np.asmatrix(x_tst))
    
    model = xgb.train(params=params,
                dtrain=dtrain,
                num_boost_round=450)
    
    preds = model.predict(dtest)
    result_temp = tst[['ID', 'target_leak']]
    result_temp['target'] = preds
    
    if seed == 0:
        result = result_temp
    else:
        result['target'] += result_temp.target
        
result.loc[:, 'target'] /= (1. * bag)
result.loc[:, 'target'] = np.expm1(result['target'])
result.loc[~result['target_leak'].isnull(), 'target'] = result.loc[~result['target_leak'].isnull(), 'target_leak']
fname = 'leakage_solution.csv'
result[['ID', 'target']].to_csv(fname, index=False)

# Submit to Kaggle
competition_name = 'santander-value-prediction-challenge'
submit_kaggle_result(competition_name, fname, msg='')

# Check result
check_kaggle_result(competition_name, fname)
'''
fileName           date                 description       status    publicScore  privateScore
-----------------  -------------------  ----------------  --------  -----------  ------------
leakage_solution   2018-10-17 08:52:46                    complete  1.93257      1.87086
'''
