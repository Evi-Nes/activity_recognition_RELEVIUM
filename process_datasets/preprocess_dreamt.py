import os
import pandas as pd
import numpy as np
from numpy.ma import count

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

    df = df[['timestamp', 'user_id', 'activity', 'accel_x', 'accel_y', 'accel_z']]
    unique_activities = df['activity'].unique()
    for activity in unique_activities:
        print(activity)
        print(count(df[df['activity'] == activity]))

    df = df.set_index('timestamp')
    new_timestamps = np.arange(df.index.min(), df.index.max(), 1 / 25)

    resampled_data = pd.DataFrame(index=new_timestamps)
    resampled_data = resampled_data.join(df)

    resampled_data[['accel_x', 'accel_y', 'accel_z']] = resampled_data[['accel_x', 'accel_y', 'accel_z']].interpolate()
    resampled_data[['user_id', 'activity']] = resampled_data[['user_id', 'activity']].fillna(method='ffill')
    unique_activities = resampled_data['activity'].unique()
    for activity in unique_activities:
        print(activity)
        print(count(resampled_data[resampled_data['activity'] == activity]))

    resampled_data.reset_index(inplace=True)
    resampled_data.rename(columns={'index': 'timestamp'}, inplace=True)

    resampled_data = resampled_data[['timestamp', 'user_id', 'activity', 'accel_x', 'accel_y', 'accel_z']]

    missing_values = resampled_data.isnull().sum()
    print("Missing values in each column:\n", missing_values)
    print("Original DataFrame shape:", resampled_data.shape)
    resampled_data = resampled_data.dropna()
    print("Cleaned DataFrame shape:", resampled_data.shape)

    merged_files.append(resampled_data)
    print(f'Merged {filename}')

    print(resampled_data)


combined_df = pd.concat(merged_files, ignore_index=True)
combined_df.to_csv('final_dreamt.csv', index=False)




