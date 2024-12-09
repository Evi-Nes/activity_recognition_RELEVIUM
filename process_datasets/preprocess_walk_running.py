import os
from datetime import date, timedelta, datetime
import pandas as pd

folder = 'single_activity_walk_running_data_csv'
merged_files = []
i = 1
columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

for file in os.listdir(folder):
    data = pd.read_csv(os.path.join(folder, file))
    data.columns = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    user_id = '1E'

    if file.startswith('Running'):
        activity = 'running'
    else:
        activity = 'walking'

    data['activity'] = activity
    data['user_id'] = user_id
    for column in columns:
        data[column] = data[column].apply(lambda x: round(x, 3))

    merged_file_path = os.path.join(folder, f'data_batch_{i}.csv')
    merged_files.append(merged_file_path)
    data.to_csv(merged_file_path, index=False)
    print(f'Writing {merged_file_path}')
    i = i + 1


if merged_files:
    all_data = []

    for file in merged_files:
        df = pd.read_csv(file)
        all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        missing_values = combined_df.isnull().sum()
        print("Missing values in each column:\n", missing_values)
        combined_df_cleaned = combined_df.dropna()
        print("Original DataFrame shape:", combined_df.shape)
        print("Cleaned DataFrame shape:", combined_df_cleaned.shape)

        combined_df_cleaned = combined_df_cleaned[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]
        combined_file_path = os.path.join(folder, 'final_walk_running.csv')
        combined_df_cleaned.to_csv('final_walk_running.csv', index=False)
        print(f"Saved final combined data to {combined_file_path}")

    else:
        print("No dataframes found to concatenate.")

else:
    print("No merged files found.")