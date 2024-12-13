import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

##### The DOMINO dataset #####
# 140Hz frequency, folder per user, different files for accel, gyro and activity labels
# Overall DOMINO includes data about a TRANSITION activity + 14 activities:
# - BRUSHING TEETH
# - CYCLING
# - ELEVATOR DOWN
# - ELEVATOR UP
# - LYING
# - MOVING BY CAR
# - RUNNING
# - SITTING
# - SITTING ON TRANSPORT
# - STAIRS DOWN
# - STAIRS UP
# - STANDING
# - STANDING ON TRANSPORT
# - WALKING

domino_folder = 'DOMINO'
merged_files = []
numeric_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
categorical_cols = ['user_id', 'activity']

for folder in tqdm(os.listdir(domino_folder)):
    folder_path = os.path.join(domino_folder, folder)

    if os.path.isdir(folder_path) and folder.startswith('user-'):
        user_id = re.search(r'\d+', folder).group()

        activity_labels_path = os.path.join(folder_path, 'activity_labels.csv')
        accel_path = os.path.join(folder_path, 'smartwatch_acc.csv')
        gyro_path = os.path.join(folder_path, 'smartwatch_gyr.csv')

        activity_labels = pd.read_csv(activity_labels_path)
        smartwatch_acc = pd.read_csv(accel_path)
        smartwatch_gyro = pd.read_csv(gyro_path)

        smartwatch_acc.rename(columns={'x': 'accel_x', 'y': 'accel_y', 'z': 'accel_z'}, inplace=True)
        smartwatch_gyro.rename(columns={'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'}, inplace=True)

        smartwatch_acc['key'] = 1
        activity_labels['key'] = 1
        cross_joined = pd.merge(smartwatch_acc, activity_labels, on='key').drop('key', axis=1)

        # Filter to keep only the rows where ts is within the range [ts_start, ts_end]
        result = cross_joined[(cross_joined['ts'] >= cross_joined['ts_start']) & (cross_joined['ts'] <= cross_joined['ts_end'])]

        merged_data = pd.merge_asof(result, smartwatch_gyro, on='ts', direction='nearest', tolerance=20)

        merged_data['user_id'] = user_id
        merged_data.rename(columns={'ts': 'timestamp'}, inplace=True)
        merged_data.rename(columns={'label': 'activity'}, inplace=True)

        merged_data = merged_data.drop('ts_start', axis=1)
        merged_data = merged_data.drop('ts_end', axis=1)
        merged_data = merged_data[['timestamp', 'user_id', 'activity', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]

        merged_file_path = os.path.join(domino_folder, f'merged_data_user_{user_id}.csv')
        merged_data.to_csv(merged_file_path, index=False)
        print(f"Saved merged data for user {user_id}")

        merged_files.append(merged_file_path)

# Combine all individual merged files into one final CSV file
if not merged_files:
    print("No merged files found.")
all_data = []

for file in merged_files:
    df = pd.read_csv(file)
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

missing_values = combined_df.isnull().sum()
print("Missing values in each column:\n", missing_values)
combined_df = combined_df.dropna()

combined_df = combined_df[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]

# remove activities
desired_activities = ['CYCLING', 'LYING', 'RUNNING', 'SITTING', 'SITTING_ON_TRANSPORT', 'STAIRS_DOWN', 'STAIRS_UP', 'STANDING', 'STANDING_ON_TRANSPORT', 'WALKING']
combined_df = combined_df[combined_df['activity'].isin(desired_activities)]

# rename and merge activities
activity_mapping = {
    'CYCLING': 'cycling', 'LYING': 'lying', 'RUNNING': 'running', 'SITTING': 'sitting', 'SITTING_ON_TRANSPORT': 'sitting',
    'STAIRS_DOWN': 'walking', 'STAIRS_UP': 'walking', 'STANDING': 'standing', 'STANDING_ON_TRANSPORT': 'standing', 'WALKING': 'walking'}
combined_df['activity'] = combined_df['activity'].replace(activity_mapping)
combined_df['user_id'] = combined_df['user_id'].astype(str) + 'D'
print('Activity values before down sampling \n', combined_df['activity'].value_counts())

for column in numeric_cols:
    combined_df[column] = combined_df[column].apply(lambda x: round(x, 3))

reduction_factor = 5
# Create a mask to drop rows in approximately 5.6 steps
indices = np.arange(len(combined_df))
mask = (indices % round(reduction_factor)) == 0
downsampled_data = combined_df[mask]

# # Downsample 140Hz -> 28Hz
# downsampled_rows = []
# step = 4
#
# for i in range(0, len(combined_df), step):
#     group = combined_df.iloc[i:i + step]
#
#     aggregated_row = {
#         'timestamp': group['timestamp'].iloc[0],
#         'activity': group['activity'].iloc[0],
#         'user_id': group['user_id'].iloc[0],
#         'accel_x': group['accel_x'].mean(),
#         'accel_y': group['accel_y'].mean(),
#         'accel_z': group['accel_z'].mean(),
#         'gyro_x': group['gyro_x'].mean(),
#         'gyro_y': group['gyro_y'].mean(),
#         'gyro_z': group['gyro_z'].mean(),
#     }
#     downsampled_rows.append(aggregated_row)
#
# downsampled_df = pd.DataFrame(downsampled_rows)
# print('First down sampling completed')
#
# downsampled_df.to_csv('data_domino.csv', index=False)
# # Downsample 28Hz -> 25Hz
# downsampled_df['timestamp'] = pd.to_datetime(downsampled_df['timestamp'], unit='ms')
# downsampled_df = downsampled_df.set_index('timestamp')
# print(downsampled_df)
#
# numeric_resampled = downsampled_df[numeric_cols].resample('40ms').apply(
#     lambda x: np.nan if x.empty else x.interpolate(method='linear', limit_direction='both'))
# categorical_resampled = downsampled_df[categorical_cols].resample('40ms').ffill()
#
# merged_data_resampled = pd.concat([numeric_resampled, categorical_resampled], axis=1).reset_index()
# merged_data_resampled = merged_data_resampled.ffill()
downsampled_data = downsampled_data[
    ['timestamp', 'user_id', 'activity', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]
print(downsampled_data.head)

print('Activity values after down sampling \n', downsampled_data['activity'].value_counts())
print('Final data \n', downsampled_data.head())
print(f"Final size of dataframe: {len(downsampled_data)}")
downsampled_data.to_csv('final_domino.csv', index=False)
