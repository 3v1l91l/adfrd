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

root_path = '.'
data_path = os.path.join(root_path, 'input')
train_data_path = os.path.join(data_path, 'train.csv')
test_data_path = os.path.join(data_path, 'test.csv')

# train_rows_num = sum(1 for l in open(train_data_path))
train_file_rows_num = 184903891

train_rows_num = int(train_file_rows_num * 0.5)
valid_rows_num = int(train_rows_num * 0.1)

dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

target = 'is_attributed'
common_cols = ['ip', 'app','device', 'os', 'channel', 'click_time']
train_cols = common_cols + [target]
test_cols = common_cols + ['click_id']
train_df = pd.read_csv(train_data_path, dtype=dtypes, usecols=train_cols, nrows=train_rows_num)
valid_df = pd.read_csv(train_data_path, dtype=dtypes, usecols=train_cols,
                       skiprows=range(1, train_file_rows_num - valid_rows_num - 1))
test_df = pd.read_csv(test_data_path, dtype=dtypes, usecols=test_cols)
train_df.to_pickle('train_df.pickle')
valid_df.to_pickle('valid_df.pickle')
test_df.to_pickle('test_df.pickle')

train_df = pd.read_pickle('train_df.pickle')
valid_df = pd.read_pickle('valid_df.pickle')
test_df = pd.read_pickle('test_df.pickle')

len_train = len(train_df)
len_valid = len(valid_df)
train_df = train_df.append(valid_df)
train_df = train_df.append(test_df)
del test_df
del valid_df
gc.collect()

train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
gc.collect()

gp = train_df[['ip', 'day', 'hour', 'channel']].groupby(
    by=['ip', 'day', 'hour'])[['channel']].count().reset_index().rename(index=str,
                                                                        columns={'channel': 'ip_day_hour_count'})
train_df = train_df.merge(gp, on=['ip', 'day', 'hour'], how='left')
del gp
gc.collect()

gp = train_df[['ip', 'app', 'channel']].groupby(
    by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip', 'app'], how='left')
# valid_df = valid_df.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()

gp = train_df[['ip', 'app', 'os', 'channel']].groupby(
    by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip', 'app', 'os'], how='left')
# valid_df = valid_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()

train_df['ip_day_hour_count'] = train_df['ip_day_hour_count'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

valid_df = train_df[len_train:len_train + len_valid]
test_df = train_df[len_train + len_valid:]
train_df = train_df[:len_train]

# # predictors = []
# used_in_rate_columns = ['ip', 'app','device', 'os', 'channel', 'hour']
# # rate_columns = ['channel']
# for column in tqdm(used_in_rate_columns):
#     rate_feature_column = column + '_rate'
# #     predictors += [rate_feature_column]

#     ratio = train_df.groupby(column)['is_attributed'].agg(lambda x: x.sum() / len(x)).to_frame()
#     ratio.rename(columns={'is_attributed': rate_feature_column}, inplace=True)
#     mean_attributed = ratio[rate_feature_column].mean()
#     train_df = pd.merge(train_df, ratio, how='left', left_on=column, right_index=True)
#     train_df[rate_feature_column] = train_df[rate_feature_column].fillna(mean_attributed)
#     valid_df = pd.merge(valid_df, ratio, how='left', left_on=column, right_index=True)
#     valid_df[rate_feature_column] = valid_df[rate_feature_column].fillna(mean_attributed)
#     test_df = pd.merge(test_df, ratio, how='left', left_on=column, right_index=True)
#     test_df[rate_feature_column] = test_df[rate_feature_column].fillna(mean_attributed)

# diff_click_seconds_by_channel = train_df.groupby('channel')['click_time'].agg(
#     lambda x: (pd.to_datetime(x[1:]).values - pd.to_datetime(x[:-1]).values).mean()).dt.total_seconds().to_frame()


# diff_click_seconds_by_channel.rename(columns={'click_time': 'mean_diff_click_seconds'}, inplace=True)
# mean_diff_click_seconds_by_channel = diff_click_seconds_by_channel['mean_diff_click_seconds'].mean()
# diff_click_seconds_by_channel.fillna(mean_diff_click_seconds_by_channel, inplace=True)
# train_df = pd.merge(train_df, diff_click_seconds_by_channel, how='left', left_on='channel', right_index=True)
# valid_df = pd.merge(valid_df, diff_click_seconds_by_channel, how='left', left_on='channel', right_index=True)

# gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
# train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')

# predictors = ['app','device','os', 'channel', 'hour', 'qty']
sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
    'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  # -1 means no limit
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
    'app', 'device', 'os', 'channel', 'hour',
    #               'mean_diff_click_seconds',
    'ip_day_hour_count', 'ip_app_count', 'ip_app_os_count']

categorical = ['app', 'device', 'os', 'channel', 'hour']


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
                verbose_eval=10
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