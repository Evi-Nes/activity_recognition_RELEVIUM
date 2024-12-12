import os
import pandas as pd
import numpy as np

main_folder = 'dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-1.0.1/data'
desired_stages = ['W', 'N1', 'N2', 'N3', 'R']
merged_files = []

for filename in os.listdir(main_folder):
    user_id = filename.replace('S0', '').replace('_whole_df.csv', '')
    df = pd.read_csv(os.path.join(main_folder, filename))
    df = df[df['Sleep_Stage'].isin(desired_stages)]
    df.rename(columns={'TIMESTAMP': 'timestamp', 'ACC_X': 'accel_x', 'ACC_Y': 'accel_y', 'ACC_Z': 'accel_z', 'Sleep_Stage': 'activity'}, inplace=True)
    mapping = {'W': 'lying', 'N1': 'sleeping', 'N2': 'sleeping', 'N3': 'sleeping', 'R': 'sleeping'}

    df['activity'] = df['activity'].replace(mapping)
    df['user_id'] = user_id + 'DR'
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df[['timestamp', 'user_id', 'activity', 'accel_x', 'accel_y', 'accel_z']]

    numeric_cols = ['accel_x', 'accel_y', 'accel_z']
    categorical_cols = ['user_id', 'activity']
    print(df['activity'].value_counts())

    df = df.iloc[::2].reset_index(drop=True)
    df = df.set_index('timestamp')
    print(df['activity'].value_counts())

    # Resample numeric columns and interpolate
    numeric_resampled = df[numeric_cols].resample('40ms').apply(
        lambda x: x.interpolate(method='linear', limit_direction='both') if not x.empty else np.nan)
    categorical_resampled = df[categorical_cols].resample('40ms').ffill()

    data_resampled = pd.concat([numeric_resampled, categorical_resampled], axis=1)
    data_resampled = data_resampled.fillna(method='ffill').fillna(method='bfill')

    # print(data_resampled)

    original_timestamps = df.index

    mask = ~data_resampled.index.isin(original_timestamps)
    data_resampled_filtered = data_resampled[mask]

    data_resampled_filtered.reset_index(inplace=True)
    data_resampled_filtered.dropna(inplace=True)
    data_resampled_filtered = data_resampled_filtered[['timestamp', 'user_id', 'activity', 'accel_x', 'accel_y', 'accel_z']]
    print(data_resampled_filtered)
    print(data_resampled_filtered['activity'].value_counts())

    merged_files.append(data_resampled_filtered)
    print(f'Merged {filename}')


combined_df = pd.concat(merged_files, ignore_index=True)
combined_df.to_csv('final_dreamt.csv', index=False)




