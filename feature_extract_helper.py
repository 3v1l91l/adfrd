import pandas as pd
import datetime as dt
import os
import numpy as np
from tqdm import tqdm
import time

def split_train_to_hdf():
    print('Splitting train and save to hdf')
    start_time = time.time()
    root_path = '.'
    data_path = os.path.join(root_path, 'input')
    train_data_path = os.path.join(data_path, 'train.csv')
    test_data_path = os.path.join(data_path, 'test.csv')

    target = 'is_attributed'
    common_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
    train_cols = common_cols + [target]
    test_cols = common_cols + ['click_id']

    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32'
    }
    test_df = pd.read_csv(test_data_path, dtype=dtypes, usecols=test_cols)

    test_start_datetime = pd.to_datetime(test_df['click_time'].min()).time()
    test_end_datetime = pd.to_datetime(test_df['click_time'].max()).time()

    valid_date = dt.datetime(2017, 11, 9).date()
    train_dates = np.array([dt.datetime(2017, 11, 7).date(), dt.datetime(2017, 11, 8).date()])

    train_df = pd.DataFrame(columns=train_cols)
    for chunk in pd.read_csv(train_data_path, dtype=dtypes, usecols=train_cols, chunksize=10000000):
        click_time = pd.to_datetime(chunk['click_time']).dt.time

        train_df = train_df.append(chunk[(click_time > test_start_datetime) & (click_time < test_end_datetime)],
                                   ignore_index=True)
    click_dates = pd.to_datetime(train_df['click_time']).dt.date
    valid_df = train_df[click_dates == valid_date]
    train_df = train_df[np.isin(click_dates, train_dates)]

    train_df.to_hdf('train_df.hdf', 'data', mode='w')
    valid_df.to_hdf('valid_df.hdf', 'data', mode='w')
    test_df.to_hdf('test_df.hdf', 'data', mode='w')

    end_time = time.time()
    print(f'Finished in {end_time - start_time} seconds')


def set_clicks_per_hour_features():
    print('Set clicks per time features')
    start_time = time.time()

    train_df = pd.read_hdf('train_df.hdf', 'data')
    valid_df = pd.read_hdf('valid_df.hdf', 'data')
    test_df = pd.read_hdf('test_df.hdf', 'data')

    gp = train_df[['ip', 'app', 'device', 'os', 'hour', 'day', 'channel']].groupby(
        by=['ip', 'app', 'device', 'os', 'hour', 'day'])[['channel']].count().reset_index().rename(index=str, columns={
        'channel': 'user_clicks_this_hour_count'})
    mean_user_clicks_this_hour_count = int(gp['user_clicks_this_hour_count'].mean())
    train_df = train_df.merge(gp, on=['ip', 'app', 'device', 'os', 'hour', 'day'], how='left')
    train_df.fillna({'user_clicks_this_hour_count': mean_user_clicks_this_hour_count}, inplace=True)

    gp = valid_df[['ip', 'app', 'device', 'os', 'hour', 'day', 'channel']].groupby(
        by=['ip', 'app', 'device', 'os', 'hour', 'day'])[['channel']].count().reset_index().rename(index=str, columns={
        'channel': 'user_clicks_this_hour_count'})
    mean_user_clicks_this_hour_count = int(gp['user_clicks_this_hour_count'].mean())
    valid_df = valid_df.merge(gp, on=['ip', 'app', 'device', 'os', 'hour', 'day'], how='left')
    valid_df.fillna({'user_clicks_this_hour_count': mean_user_clicks_this_hour_count}, inplace=True)

    gp = test_df[['ip', 'app', 'device', 'os', 'hour', 'day', 'channel']].groupby(
        by=['ip', 'app', 'device', 'os', 'hour', 'day'])[['channel']].count().reset_index().rename(index=str, columns={
        'channel': 'user_clicks_this_hour_count'})
    mean_user_clicks_this_hour_count = int(gp['user_clicks_this_hour_count'].mean())
    test_df = test_df.merge(gp, on=['ip', 'app', 'device', 'os', 'hour', 'day'], how='left')
    test_df.fillna({'user_clicks_this_hour_count': mean_user_clicks_this_hour_count}, inplace=True)

    train_df.to_hdf('train_df.hdf', 'data', mode='w')
    valid_df.to_hdf('valid_df.hdf', 'data', mode='w')
    test_df.to_hdf('test_df.hdf', 'data', mode='w')

    end_time = time.time()
    print(f'Finished in {end_time - start_time} seconds')


def set_clicks_per_quarter_features():
    print('Set clicks per quarter features')
    start_time = time.time()

    train_df = pd.read_hdf('train_df.hdf', 'data')
    valid_df = pd.read_hdf('valid_df.hdf', 'data')
    test_df = pd.read_hdf('test_df.hdf', 'data')

    gp = train_df[['ip', 'app', 'device', 'os', 'quarter', 'day', 'channel']].groupby(
        by=['ip', 'app', 'device', 'os', 'quarter', 'day'])[['channel']].count().reset_index().rename(index=str, columns={
        'channel': 'user_clicks_this_quarter_count'})
    mean_user_clicks_this_quarter_count = int(gp['user_clicks_this_quarter_count'].mean())
    train_df = train_df.merge(gp, on=['ip', 'app', 'device', 'os', 'quarter', 'day'], how='left')
    train_df.fillna({'user_clicks_this_quarter_count': mean_user_clicks_this_quarter_count}, inplace=True)
    gp['quarter'] += 1
    gp.rename(columns={
        'user_clicks_this_quarter_count': 'user_clicks_quarter_count_1'}, inplace=True)
    train_df = train_df.merge(gp, on=['ip', 'app', 'device', 'os', 'quarter', 'day'], how='left')
    train_df.fillna({'user_clicks_quarter_count_1': mean_user_clicks_this_quarter_count}, inplace=True)
    gp['quarter'] += 1
    gp.rename(columns={
        'user_clicks_quarter_count_1': 'user_clicks_quarter_count_2'}, inplace=True)
    train_df = train_df.merge(gp, on=['ip', 'app', 'device', 'os', 'quarter', 'day'], how='left')
    train_df.fillna({'user_clicks_quarter_count_2': mean_user_clicks_this_quarter_count}, inplace=True)
    train_df.to_hdf('train_df.hdf', 'data', mode='w')


    gp = valid_df[['ip', 'app', 'device', 'os', 'quarter', 'day', 'channel']].groupby(
        by=['ip', 'app', 'device', 'os', 'quarter', 'day'])[['channel']].count().reset_index().rename(index=str, columns={
        'channel': 'user_clicks_this_quarter_count'})
    mean_user_clicks_this_quarter_count = int(gp['user_clicks_this_quarter_count'].mean())
    valid_df = valid_df.merge(gp, on=['ip', 'app', 'device', 'os', 'quarter', 'day'], how='left')
    valid_df.fillna({'user_clicks_this_quarter_count': mean_user_clicks_this_quarter_count}, inplace=True)
    gp['quarter'] += 1
    gp.rename(columns={
        'user_clicks_this_quarter_count': 'user_clicks_quarter_count_1'}, inplace=True)
    valid_df = valid_df.merge(gp, on=['ip', 'app', 'device', 'os', 'quarter', 'day'], how='left')
    valid_df.fillna({'user_clicks_quarter_count_1': mean_user_clicks_this_quarter_count}, inplace=True)
    gp['quarter'] += 1
    gp.rename(columns={
        'user_clicks_quarter_count_1': 'user_clicks_quarter_count_2'}, inplace=True)
    valid_df = valid_df.merge(gp, on=['ip', 'app', 'device', 'os', 'quarter', 'day'], how='left')
    valid_df.fillna({'user_clicks_quarter_count_2': mean_user_clicks_this_quarter_count}, inplace=True)
    valid_df.to_hdf('valid_df.hdf', 'data', mode='w')


    gp = test_df[['ip', 'app', 'device', 'os', 'quarter', 'day', 'channel']].groupby(
        by=['ip', 'app', 'device', 'os', 'quarter', 'day'])[['channel']].count().reset_index().rename(index=str, columns={
        'channel': 'user_clicks_this_quarter_count'})
    mean_user_clicks_this_quarter_count = int(gp['user_clicks_this_quarter_count'].mean())
    test_df = test_df.merge(gp, on=['ip', 'app', 'device', 'os', 'quarter', 'day'], how='left')
    test_df.fillna({'user_clicks_this_quarter_count': mean_user_clicks_this_quarter_count}, inplace=True)
    gp['quarter'] += 1
    gp.rename(columns={
        'user_clicks_this_quarter_count': 'user_clicks_quarter_count_1'}, inplace=True)
    test_df = test_df.merge(gp, on=['ip', 'app', 'device', 'os', 'quarter', 'day'], how='left')
    test_df.fillna({'user_clicks_quarter_count_1': mean_user_clicks_this_quarter_count}, inplace=True)
    gp['quarter'] += 1
    gp.rename(columns={
        'user_clicks_quarter_count_1': 'user_clicks_quarter_count_2'}, inplace=True)
    test_df = test_df.merge(gp, on=['ip', 'app', 'device', 'os', 'quarter', 'day'], how='left')
    test_df.fillna({'user_clicks_quarter_count_2': mean_user_clicks_this_quarter_count}, inplace=True)
    test_df.to_hdf('test_df.hdf', 'data', mode='w')

    end_time = time.time()
    print(f'Finished in {end_time - start_time} seconds')



def set_rates_features():
    print('Set rates features')
    start_time = time.time()
    train_df = pd.read_hdf('train_df.hdf', 'data')
    valid_df = pd.read_hdf('valid_df.hdf', 'data')
    test_df = pd.read_hdf('test_df.hdf', 'data')

    used_in_rate_columns = ['app','device', 'os', 'channel', 'hour']
    for column in tqdm(used_in_rate_columns):
        rate_feature_column = column + '_rate'

        ratio = train_df.groupby(column)['is_attributed'].agg(lambda x: x.sum() / len(x)).to_frame()
        ratio.rename(columns={'is_attributed': rate_feature_column}, inplace=True)
        mean_attributed = ratio[rate_feature_column].mean()
        train_df = pd.merge(train_df, ratio, how='left', left_on=column, right_index=True)
        train_df[rate_feature_column] = train_df[rate_feature_column].fillna(mean_attributed)
        valid_df = pd.merge(valid_df, ratio, how='left', left_on=column, right_index=True)
        valid_df[rate_feature_column] = valid_df[rate_feature_column].fillna(mean_attributed)
        test_df = pd.merge(test_df, ratio, how='left', left_on=column, right_index=True)
        test_df[rate_feature_column] = test_df[rate_feature_column].fillna(mean_attributed)

    train_df.to_hdf('train_df.hdf', 'data', mode='w')
    valid_df.to_hdf('valid_df.hdf', 'data', mode='w')
    test_df.to_hdf('test_df.hdf', 'data', mode='w')

    end_time = time.time()
    print(f'Finished in {end_time - start_time} seconds')

def set_time_features():
    print('Set time features')
    start_time = time.time()
    train_df = pd.read_hdf('train_df.hdf', 'data')
    valid_df = pd.read_hdf('valid_df.hdf', 'data')
    test_df = pd.read_hdf('test_df.hdf', 'data')

    train_df['day'] = pd.to_datetime(train_df['click_time']).dt.day.astype('uint8')
    valid_df['day'] = pd.to_datetime(valid_df['click_time']).dt.day.astype('uint8')
    test_df['day'] = pd.to_datetime(test_df['click_time']).dt.day.astype('uint8')
    train_df['hour'] = pd.to_datetime(train_df['click_time']).dt.hour.astype('uint8')
    valid_df['hour'] = pd.to_datetime(valid_df['click_time']).dt.hour.astype('uint8')
    test_df['hour'] = pd.to_datetime(test_df['click_time']).dt.hour.astype('uint8')
    train_df['minute'] = pd.to_datetime(train_df['click_time']).dt.minute.astype('uint8')
    valid_df['minute'] = pd.to_datetime(valid_df['click_time']).dt.minute.astype('uint8')
    test_df['minute'] = pd.to_datetime(test_df['click_time']).dt.minute.astype('uint8')
    train_df['quarter'] = ((4 * train_df['hour']) + (train_df['minute'] // 15)).astype('uint8')
    valid_df['quarter'] = ((4 * valid_df['hour']) + (valid_df['minute'] // 15)).astype('uint8')
    test_df['quarter'] = ((4 * test_df['hour']) + (test_df['minute'] // 15)).astype('uint8')

    train_df.drop(['click_time'], inplace=True, axis=1)
    valid_df.drop(['click_time'], inplace=True, axis=1)
    test_df.drop(['click_time'], inplace=True, axis=1)

    train_df.to_hdf('train_df.hdf', 'data', mode='w')
    valid_df.to_hdf('valid_df.hdf', 'data', mode='w')
    test_df.to_hdf('test_df.hdf', 'data', mode='w')

    end_time = time.time()
    print(f'Finished in {end_time - start_time} seconds')