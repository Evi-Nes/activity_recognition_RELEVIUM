import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

accel_folder = ('wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/accel')
gyro_folder = ('wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/gyro')

merged_files = []

for file in os.listdir(accel_folder):
    if file.startswith('data'):

        file_path = os.path.join(accel_folder, file)
        accel_data = pd.read_csv(file_path, header=None, encoding="ISO-8859-1")

        accel_data.columns = ['user_id', 'activity', 'timestamp', 'accel_x', 'accel_y', 'accel_z']
        accel_data['accel_z'] = accel_data['accel_z'].str.rstrip(';')

        user_id = accel_data['user_id'][0]

        # 20Hz -> 25Hz
        n_rows = len(accel_data)
        yesterday = datetime.now() - timedelta(days=1)
        frequency = timedelta(milliseconds=50)
        timestamps = [yesterday + i * frequency for i in range(n_rows)]

        accel_data['timestamp'] = timestamps

        accel_data = accel_data[['timestamp', 'user_id', 'activity', 'accel_x', 'accel_y', 'accel_z']]
        accel_data.set_index('timestamp', inplace=True)
        accel_data.dropna()
        # print(accel_data.head)

        numeric_cols = ['accel_x', 'accel_y', 'accel_z']
        categorical_cols = ['user_id', 'activity']
        numeric_resampled = accel_data[numeric_cols].resample('40ms').apply(lambda x: np.nan if x.empty else x.interpolate(method='linear', limit_direction='both'))
        categorical_resampled = accel_data[categorical_cols].resample('40ms').ffill()

        accel_data_resampled = pd.concat([numeric_resampled, categorical_resampled], axis=1).reset_index()
        accel_data_resampled = accel_data_resampled.fillna(method='ffill').fillna(method='bfill')
        accel_data_resampled = accel_data_resampled[['timestamp', 'user_id', 'activity', 'accel_x', 'accel_y', 'accel_z']]
        print(accel_data_resampled.head)
        # print(accel_data_resampled.isna().sum())

        merged_file_path = os.path.join(accel_folder, f'merged_user_{user_id}.csv')
        accel_data_resampled.to_csv(merged_file_path, index=False)

        merged_files.append(merged_file_path)

all_data = []
for file in merged_files:
    df = pd.read_csv(file)
    all_data.append(df)

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv('wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/accel_data.csv', index=False)

print('Accelerometer data done')

# Gyro data
merged_files = []
for file in os.listdir(gyro_folder):
    if file.startswith('data'):

        file_path = os.path.join(gyro_folder, file)
        gyro_data = pd.read_csv(file_path, header=None, encoding="ISO-8859-1")
        gyro_data.columns = ['user_id', 'activity', 'timestamp', 'gyro_x', 'gyro_y', 'gyro_z']
        gyro_data['gyro_z'] = gyro_data['gyro_z'].str.rstrip(';')

        gyro_data['timestamp'] = gyro_data['timestamp'].astype(int)
        gyro_data['user_id'] = gyro_data['user_id']
        user_id = gyro_data['user_id'][0]

        # 20Hz -> 25Hz
        n_rows = len(gyro_data)
        yesterday = datetime.now() - timedelta(days=1)
        frequency = timedelta(milliseconds=50)
        timestamps = [yesterday + i * frequency for i in range(n_rows)]

        gyro_data['timestamp'] = timestamps

        gyro_data = gyro_data[['timestamp', 'user_id', 'activity', 'gyro_x', 'gyro_y', 'gyro_z']]
        gyro_data.set_index('timestamp', inplace=True)
        gyro_data.dropna()
        # print(gyro_data.head)

        numeric_cols = ['gyro_x', 'gyro_y', 'gyro_z']
        categorical_cols = ['user_id', 'activity']
        numeric_resampled = gyro_data[numeric_cols].resample('40ms').apply(lambda x: np.nan if x.empty else x.interpolate(method='linear', limit_direction='both'))
        categorical_resampled = gyro_data[categorical_cols].resample('40ms').ffill()

        gyro_data_resampled = pd.concat([numeric_resampled, categorical_resampled], axis=1).reset_index()
        gyro_data_resampled = gyro_data_resampled.fillna(method='ffill').fillna(method='bfill')
        gyro_data_resampled = gyro_data_resampled[['timestamp', 'user_id', 'activity','gyro_x', 'gyro_y', 'gyro_z']]
        print(gyro_data_resampled.head)
        # print(gyro_data_resampled.isna().sum())

        merged_file_path = os.path.join(gyro_folder, f'merged_user_{user_id}.csv')
        gyro_data_resampled.to_csv(merged_file_path, index=False)

        merged_files.append(merged_file_path)

all_data = []
for file in merged_files:
    df = pd.read_csv(file)
    all_data.append(df)

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv('wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/gyro_data.csv', index=False)

print('Gyroscope data done')

# Final Merge
smartwatch_acc = pd.read_csv('wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/accel_data.csv')
smartwatch_gyro = pd.read_csv('wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/gyro_data.csv')
merged_data = pd.merge(smartwatch_acc, smartwatch_gyro, on=['user_id', 'activity', 'timestamp'])
merged_data = merged_data[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]

merged_data.to_csv('data_wisdm.csv', index=False)
print("Saved merged data")

