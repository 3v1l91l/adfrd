import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc

config = {
    'save_df': True
}

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.01,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params,
                     xgtrain,
                     valid_sets=[xgtrain, xgvalid],
                     valid_names=['train','valid'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10,
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1

path = 'input/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

# len_train = 10000
len_train = 40000000
len_valid = int(len_train*0.1)
print('load train...')
# train_df = pd.read_csv(path+"train.csv", skiprows=range(1,139903891), nrows=len_train,dtype=dtypes,
#                        usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'],
#                        parse_dates=['click_time'])
train_df = pd.read_csv(path+"train.csv", nrows=len_train,dtype=dtypes,
                       usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'],
                       parse_dates=['click_time'])
print('load test...')
# test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
test_df = pd.DataFrame(columns=np.concatenate((train_df.columns,['click_id'])))


train_df=train_df.append(test_df)

del test_df
gc.collect()

print('data prep...')
train_df['day'] = train_df['click_time'].dt.day.astype('uint8')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
gc.collect()
if config['save_df']:
    train_df.to_pickle('train_df.pickle')

# train_df = pd.read_pickle('train_df.pickle')
print('Prev and next click sec...')
train_df['prev_click_sec'] = train_df[['ip','device','os','day','click_time']].groupby(
    by=['ip','device','os','day'])[['click_time']].diff()['click_time'].dt.total_seconds().astype('uint16')
print(f'Mean prev_click_sec {train_df.prev_click_sec.mean()}')
print(f'Mean prev_click_sec ratio null {train_df.prev_click_sec.isnull().sum()/len(train_df)}')
if config['save_df']:
    train_df.to_pickle('train_df.pickle')

train_df['next_click_sec'] = train_df[['ip','device','os','day','click_time']].groupby(
    by=['ip','device','os','day'])[['click_time']].diff(-1)['click_time'].dt.total_seconds().abs().astype('uint16')
print(f'Mean next_click_sec {train_df.next_click_sec.mean()}')
print(f'Mean next_click_sec ratio null {train_df.next_click_sec.isnull().sum()/len(train_df)}')
gc.collect()
if config['save_df']:
    train_df.to_pickle('train_df.pickle')

gp = train_df[['ip', 'app', 'device', 'os', 'hour', 'day', 'channel']].groupby(
    by=['ip', 'app', 'device', 'os', 'hour', 'day'])[['channel']].count().reset_index().rename(index=str, columns={
    'channel': 'user_clicks_this_hour_count'})
gp['user_clicks_this_hour_count'] = gp['user_clicks_this_hour_count'].astype('uint16')
train_df = train_df.merge(gp, on=['ip', 'app', 'device', 'os', 'hour', 'day'], how='left')
print(f'Mean next_click_sec {train_df.user_clicks_this_hour_count.mean()}')
del gp
gc.collect()
if config['save_df']:
    train_df.to_pickle('train_df.pickle')

print('group by : unique_apps_hour')
gp = train_df[['ip','device', 'os', 'channel', 'day', 'hour', 'app']].groupby(by=['ip','device', 'os', 'channel', 'day', 'hour'])[['app']].nunique().reset_index().rename(index=str, columns={'app': 'unique_apps_hour'})
train_df = train_df.merge(gp, on=['ip','device', 'os', 'channel', 'day', 'hour'], how='left')
print(f'Mean unique_apps_hour ${train_df.unique_apps_hour.mean()}')
del gp
gc.collect()
if config['save_df']:
    train_df.to_pickle('train_df.pickle')

print('group by : unique_apps')
gp = train_df[['ip','device', 'os', 'channel', 'app']].groupby(by=['ip','device', 'os', 'channel'])[['app']].nunique().reset_index().rename(index=str, columns={'app': 'unique_apps'})
train_df = train_df.merge(gp, on=['ip','device', 'os', 'channel'], how='left')
gp['unique_apps_hour'] = gp['unique_apps'].astype('uint16')
print(f'Mean unique_apps ${train_df.unique_apps.mean()}')
del gp
gc.collect()
if config['save_df']:
    train_df.to_pickle('train_df.pickle')

print('group by : unique_channels_hour')
gp = train_df[['ip','device', 'os', 'day', 'hour', 'channel']].groupby(by=['ip','device', 'os', 'day', 'hour'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'unique_channels_hour'})
train_df = train_df.merge(gp, on=['ip','device', 'os', 'day', 'hour'], how='left')
gp['unique_channels_hour'] = gp['unique_channels_hour'].astype('uint16')
print(f'Mean unique_channels_hour ${train_df.unique_channels_hour.mean()}')
del gp
gc.collect()
if config['save_df']:
    train_df.to_pickle('train_df.pickle')

print('group by : unique_channels')
gp = train_df[['ip','device', 'os', 'channel']].groupby(by=['ip','device', 'os'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'unique_channels'})
gp['unique_channels'] = gp['unique_channels'].astype('uint16')
train_df = train_df.merge(gp, on=['ip','device', 'os'], how='left')
print(f'Mean unique_channels ${train_df.unique_channels.mean()}')
del gp
gc.collect()
if config['save_df']:
    train_df.to_pickle('train_df.pickle')

print("vars and data type: ")
train_df.info()

test_df = train_df[len_train:]
val_df = train_df[(len_train-len_valid):len_train]
train_df = train_df[:(len_train-len_valid)]

print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

target = 'is_attributed'
predictors = [
    'app','device','os', 'channel',
    'hour',
    'unique_apps_hour', 'unique_apps', 'unique_channels_hour', 'unique_channels',
    'prev_click_sec', 'next_click_sec'
              ]
categorical = [
    'app','device','os','channel'
    'hour'
               ]


sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

gc.collect()

print("Training...")
params = {
    'learning_rate': 0.2,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 13,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 4,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.89,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99 # because training data is extremely unbalanced
}
bst = lgb_modelfit_nocv(params,
                        train_df,
                        val_df,
                        predictors,
                        target,
                        objective='binary',
                        metrics='auc',
                        early_stopping_rounds=5,
                        verbose_eval=True,
                        num_boost_round=100,
                        categorical_features=categorical)

del train_df
del val_df
gc.collect()

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv('sub_lgb_balanced99.csv',index=False)
print("done...")
print(sub.info())