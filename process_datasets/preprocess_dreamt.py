import os
import pandas as pd
import numpy as np

main_folder = 'dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-1.0.1/data'
desired_stages = ['W', 'N1', 'N2', 'N3', 'R']
merged_files = []
numeric_cols = ['accel_x', 'accel_y', 'accel_z']
categorical_cols = ['user_id', 'activity', 'hr']

for filename in os.listdir(main_folder):
    user_id = filename.replace('S0', '').replace('_whole_df.csv', '')
    df = pd.read_csv(os.path.join(main_folder, filename))
    df = df[df['Sleep_Stage'].isin(desired_stages)]
    df.rename(columns={'TIMESTAMP': 'timestamp', 'ACC_X': 'accel_x', 'ACC_Y': 'accel_y', 'ACC_Z': 'accel_z', 'HR': 'hr', 'Sleep_Stage': 'activity'}, inplace=True)

    df['user_id'] = user_id + 'DR'
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df[['timestamp', 'user_id', 'activity', 'accel_x', 'accel_y', 'accel_z', 'hr']]

    # Timestamp at 64Hz, accelerometer at 32Hz
    df = df.iloc[::2].reset_index(drop=True)
    df = df.set_index('timestamp')
    print(df['activity'].value_counts())

    # Resample numeric columns and interpolate
    numeric_resampled = df[numeric_cols].resample('40ms').apply(
        lambda x: x.interpolate(method='linear', limit_direction='both') if not x.empty else np.nan)
    categorical_resampled = df[categorical_cols].resample('40ms').ffill()

    data_resampled = pd.concat([numeric_resampled, categorical_resampled], axis=1)
    data_resampled = data_resampled.ffill()

    original_timestamps = df.index
    mask = ~data_resampled.index.isin(original_timestamps)
    data_resampled_filtered = data_resampled[mask]

    data_resampled_filtered.reset_index(inplace=True)
    data_resampled_filtered.dropna(inplace=True)
    data_resampled_filtered = data_resampled_filtered[['timestamp', 'user_id', 'activity', 'accel_x', 'accel_y', 'accel_z', 'hr']]

    # new_desired_stages = ['W', 'N1', 'N2', 'R']
    # data_resampled_filtered = data_resampled_filtered[data_resampled_filtered['activity'].isin(new_desired_stages)]
    #
    # mapping = {'W': 'lying', 'N1': 'sleeping', 'N2': 'sleeping', 'R': 'sleeping'}
    # data_resampled_filtered['activity'] = data_resampled_filtered['activity'].replace(mapping)
    # print(data_resampled_filtered)

    lying_data = data_resampled_filtered[data_resampled_filtered['activity'] == 'lying']
    sleeping_data = data_resampled_filtered[data_resampled_filtered['activity'].isin(['N1', 'N2', 'N3', 'R'])]
    retain_fraction = 0.6
    downsampled_sleeping = (sleeping_data.groupby('activity', group_keys=False)
                            .apply(lambda x: x.sample(frac=retain_fraction, random_state=42)))
    reduced_data = pd.concat([lying_data, downsampled_sleeping])

    merged_files.append(reduced_data)
    print(f'Merged {filename}')


combined_df = pd.concat(merged_files, ignore_index=True)
print('Activity values after down sampling \n', combined_df['activity'].value_counts())
print('Final data \n', combined_df.head())
print(f"Final size of dataframe: {len(combined_df)}")
combined_df.to_csv('final_dreamt.csv', index=False)

