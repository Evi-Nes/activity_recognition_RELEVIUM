import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# 20Hz frequency
# A Walking
# B Jogging
# C Stairs
# D Sitting
# E Standing
# F Typing
# G Brushing Teeth
# H Eating Soup
# I Eating Chips
# J Eating Pasta
# K Drinking from Cup
# L Eating Sandwich
# M Kicking (Soccer Ball)
# O Playing Catch w/Tennis Ball
# P Dribblinlg (Basketball)
# Q Writing
# R Clapping
# S Folding Clothes

accel_folder = 'wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/accel'
gyro_folder = 'wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/gyro'
root_folder = 'wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch'
merged_files = []
numeric_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
categorical_cols = ['user_id', 'activity']

for file in os.listdir(accel_folder):
    if file.startswith('data'):

        file_path = os.path.join(accel_folder, file)
        accel_data = pd.read_csv(file_path, header=None, encoding="ISO-8859-1")
        accel_data.columns = ['user_id', 'activity', 'timestamp', 'accel_x', 'accel_y', 'accel_z']
        accel_data['accel_z'] = accel_data['accel_z'].str.rstrip(';')
        user_id = accel_data['user_id'][0]

        gyro_path = os.path.join(gyro_folder, f'data_{user_id}_gyro_watch.txt')
        gyro_data = pd.read_csv(gyro_path, header=None, encoding="ISO-8859-1")
        gyro_data.columns = ['user_id', 'activity', 'timestamp', 'gyro_x', 'gyro_y', 'gyro_z']
        gyro_data['gyro_z'] = gyro_data['gyro_z'].str.rstrip(';')

        merged_data = pd.merge(accel_data, gyro_data, on=['user_id', 'activity', 'timestamp'])
        merged_data = merged_data[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]

        # 20Hz -> 25Hz
        n_rows = len(merged_data)
        yesterday = datetime.now() - timedelta(days=1)
        frequency = timedelta(milliseconds=50)
        timestamps = [yesterday + i * frequency for i in range(n_rows)]

        merged_data['timestamp'] = timestamps
        merged_data.set_index('timestamp', inplace=True)
        merged_data.dropna()

        numeric_resampled = merged_data[numeric_cols].resample('40ms').apply(lambda x: np.nan if x.empty else x.interpolate(method='linear', limit_direction='both'))
        categorical_resampled = merged_data[categorical_cols].resample('40ms').ffill()

        merged_data_resampled = pd.concat([numeric_resampled, categorical_resampled], axis=1).reset_index()
        merged_data_resampled = merged_data_resampled.ffill()
        merged_data_resampled = merged_data_resampled[['timestamp', 'user_id', 'activity', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]
        print(merged_data_resampled.head)

        merged_file_path = os.path.join(root_folder, f'merged_user_{user_id}.csv')
        merged_data_resampled.to_csv(merged_file_path, index=False)
        merged_files.append(merged_file_path)


all_data = []
for file in merged_files:
    df = pd.read_csv(file)
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
combined_df['user_id'] = combined_df['user_id'].astype(int)
combined_df['user_id'] = combined_df['user_id'].astype(str) + 'W'

# remove activities
desired_activities = ['A', 'B', 'D', 'E']
combined_df = combined_df[combined_df['activity'].isin(desired_activities)]

# rename activities
activity_mapping = {
    'A': 'walking', 'B': 'running', 'D': 'sitting', 'E': 'standing'}
combined_df['activity'] = combined_df['activity'].replace(activity_mapping)
combined_df = combined_df[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]

missing_values = combined_df.isnull().sum()
print("Missing values in each column:\n", missing_values)
combined_df = combined_df.dropna()

for column in numeric_cols:
    combined_df[column] = combined_df[column].apply(lambda x: round(x, 3))

print('Activity values after up sampling \n', combined_df['activity'].value_counts())
print('Final data \n', combined_df.head())
combined_df.to_csv('final_wisdm.csv', index=False)
