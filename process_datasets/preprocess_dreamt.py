import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

main_folder = 'dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-1.0.1/data'
desired_stages = ['W', 'N1', 'N2', 'N3', 'R']
numeric_cols = ['accel_x', 'accel_y', 'accel_z']
categorical_cols = ['user_id', 'activity', 'hr']

# Define a low-pass filter for gravity estimation
def low_pass_filter(data, cutoff=0.3, fs=25, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

#
# for filename in os.listdir(main_folder):
#     if not filename.endswith('_whole_df.csv'):
#         continue
#     print(filename)
#     user_id = filename.replace('S0', '').replace('_whole_df.csv', '')
#     df = pd.read_csv(os.path.join(main_folder, filename))
#     df = df[df['Sleep_Stage'].isin(desired_stages)]
#     df.rename(columns={'TIMESTAMP': 'timestamp', 'ACC_X': 'accel_x', 'ACC_Y': 'accel_y', 'ACC_Z': 'accel_z', 'HR': 'hr', 'Sleep_Stage': 'activity'}, inplace=True)
#
#     df['user_id'] = user_id + 'DR'
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
#     df = df[['timestamp', 'user_id', 'activity', 'accel_x', 'accel_y', 'accel_z', 'hr']]
#
#     # Timestamp at 64Hz, accelerometer at 32Hz
#     df = df.iloc[::2].reset_index(drop=True)
#     df = df.set_index('timestamp')
#     # print('Initial data', df['activity'].value_counts())
#     # print(df.head())
#     df.to_csv(os.path.join(main_folder, filename.replace('_whole_df.csv', '_processed_32Hz.csv')))
#
#     # Resample numeric columns and interpolate
#     numeric_resampled = df[numeric_cols].resample('40ms').apply(
#         lambda x: x.interpolate(method='linear', limit_direction='both') if not x.empty else np.nan)
#     categorical_resampled = df[categorical_cols].resample('40ms').ffill()
#
#     data_resampled = pd.concat([numeric_resampled, categorical_resampled], axis=1)
#     data_resampled = data_resampled.ffill()
#
#     original_timestamps = df.index
#     mask = ~data_resampled.index.isin(original_timestamps)
#     data_resampled_filtered = data_resampled[mask]
#
#     data_resampled_filtered.reset_index(inplace=True)
#     data_resampled_filtered.dropna(inplace=True)
#     data_resampled_filtered = data_resampled_filtered[['timestamp', 'user_id', 'activity', 'accel_x', 'accel_y', 'accel_z', 'hr']]
#     print('Resampled data filtered', data_resampled_filtered['activity'].value_counts())
#     print(data_resampled_filtered.head())
#
#     lying_data = data_resampled_filtered[data_resampled_filtered['activity'] == 'W']
#     sleeping_data = data_resampled_filtered[data_resampled_filtered['activity'].isin(['N1', 'N2', 'N3', 'R'])]
#
#     retain_fraction = 0.7  #keep those
#     downsampled_sleeping = (sleeping_data.groupby('activity', group_keys=False)
#                             .apply(lambda x: x.sample(frac=retain_fraction, random_state=42)))
#     reduced_data = pd.concat([lying_data, downsampled_sleeping])
#     print('Reduced data', reduced_data['activity'].value_counts())
#
#     reduced_data.to_csv(os.path.join(main_folder, filename.replace('_whole_df.csv', '_processed_25Hz.csv')))
#     print(f'Merged {filename}')

# Combine 25Hz data with classes N1, N2, N3, R, W
all_data = []
for filename in os.listdir(main_folder):
    if not filename.endswith('_processed_25Hz.csv'):
        continue
    data = pd.read_csv(os.path.join(main_folder, filename))
    all_data.append(data)

combined_df = pd.concat(all_data, ignore_index=True)
fs = 25
# Estimate gravity using a low-pass filter
combined_df['gravity_x'] = low_pass_filter(combined_df['accel_x'], cutoff=0.3, fs=fs)
combined_df['gravity_y'] = low_pass_filter(combined_df['accel_y'], cutoff=0.3, fs=fs)
combined_df['gravity_z'] = low_pass_filter(combined_df['accel_z'], cutoff=0.3, fs=fs)

# Subtract gravity to get dynamic acceleration
combined_df['accel_x'] = combined_df['accel_x'] - combined_df['gravity_x']
combined_df['accel_y'] = combined_df['accel_y'] - combined_df['gravity_y']
combined_df['accel_z'] = combined_df['accel_z'] - combined_df['gravity_z']

missing_values = combined_df.isnull().sum()
print("Missing values in each column:\n", missing_values)
combined_df = combined_df.dropna()

print('Activity values after down sampling \n', combined_df['activity'].value_counts())
print('Final data \n', combined_df.head())
print(f"Final size of dataframe: {len(combined_df)}")
combined_df.to_csv('combined_dreamt_25Hz.csv', index=False)


# # Combine 32Hz data with classes N1, N2, N3, R, W
# all_data = []
# for filename in os.listdir(main_folder):
#     if not filename.endswith('_processed_32Hz.csv'):
#         continue
#     data = pd.read_csv(os.path.join(main_folder, filename))
#     all_data.append(data)
#
# combined_df = pd.concat(all_data, ignore_index=True)
#
# missing_values = combined_df.isnull().sum()
# print("Missing values in each column:\n", missing_values)
# combined_df = combined_df.dropna()
#
# print('Activity values after down sampling \n', combined_df['activity'].value_counts())
# print('Final data \n', combined_df.head())
# print(f"Final size of dataframe: {len(combined_df)}")
# combined_df.to_csv('combined_dreamt_32Hz.csv', index=False)