import pandas as pd
import os
import numpy as np
# import dask
# import dask.dataframe as dd
# import ray.dataframe as pd
import gc
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm

target = 'is_attributed'

train_df = pd.read_hdf('train_df.hdf', 'data')
valid_df = pd.read_hdf('valid_df.hdf', 'data')
test_df = pd.read_hdf('test_df.hdf', 'data')

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.2,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 198,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 8,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        'scale_pos_weight': 99,
    }

predictors = [
    #     'ip_rate', 'app_rate', 'device_rate', 'os_rate', 'channel_rate', 'hour_rate',
    'ip','app', 'device', 'os', 'channel', 'hour','minute', 'quarter', 'user_clicks_this_quarter_count',
    'user_clicks_quarter_count_1', 'user_clicks_quarter_count_2'
    #               'mean_diff_click_seconds',
    #'ip_day_hour_count', 'ip_app_count', 'ip_app_os_count'
]

categorical = ['app', 'device', 'os', 'channel', 'hour', 'minute', 'quarter']

xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )
xgvalid = lgb.Dataset(valid_df[predictors].values, label=valid_df[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )
del train_df
del valid_df
gc.collect()

evals_results = {}
bst = lgb.train(lgb_params,
                xgtrain,
                valid_sets=[xgtrain, xgvalid],
                valid_names=['train', 'valid'],
                evals_result=evals_results,
                num_boost_round=300,
                early_stopping_rounds=10,
                verbose_eval=1
                )

n_estimators = bst.best_iteration
print("\nModel Report")
print("n_estimators : ", n_estimators)
print('AUC: ', evals_results['valid']['auc'][n_estimators-1])

fig, ax = plt.subplots(figsize=(12, 18))
lgb.plot_importance(bst, height=0.5, ax=ax)
plt.savefig('feature_importance_lgb.png', bbox_inches='tight')

sub['is_attributed'] = bst.predict(test_df[predictors])
sub.to_csv('sub.csv', index=False)