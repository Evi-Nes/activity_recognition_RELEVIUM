import os
import pandas as pd

folder = 'single_activity_walk_running_data_csv'
merged_files = []
i = 1
columns_to_round = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
user_id = '1E'

for file in os.listdir(folder):
    data = pd.read_csv(os.path.join(folder, file))
    data.columns = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

    if file.startswith('Running'):
        activity = 'running'
    else:
        activity = 'walking'

    data['activity'] = activity
    data['user_id'] = user_id
    for column in columns_to_round:
        data[column] = data[column].apply(lambda x: round(x, 3))

    merged_file_path = os.path.join(folder, f'data_batch_{i}.csv')
    merged_files.append(merged_file_path)
    data.to_csv(merged_file_path, index=False)
    print(f'Writing {merged_file_path}')
    i = i + 1

if not merged_files:
    print("No merged files found.")
all_data = []

for file in merged_files:
    df = pd.read_csv(file)
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
print('Activity values before down sampling \n', combined_df['activity'].value_counts())

# Downsample 50Hz -> 25Hz
downsampled_rows = []
step = 2

for i in range(0, len(combined_df), step):
    group = combined_df.iloc[i:i + step]

    # Aggregate specific columns
    aggregated_row = {
        'timestamp': group['timestamp'].iloc[0],
        'activity': group['activity'].iloc[0],
        'user_id': group['user_id'].iloc[0],
        'accel_x': group['accel_x'].mean(),
        'accel_y': group['accel_y'].mean(),
        'accel_z': group['accel_z'].mean(),
        'gyro_x': group['gyro_x'].mean(),
        'gyro_y': group['gyro_y'].mean(),
        'gyro_z': group['gyro_z'].mean(),
    }
    downsampled_rows.append(aggregated_row)

downsampled_df = pd.DataFrame(downsampled_rows)

missing_values = downsampled_df.isnull().sum()
print("Missing values in each column:\n", missing_values)
downsampled_df_cleaned = downsampled_df.dropna()

for column in columns_to_round:
    downsampled_df_cleaned[column] = downsampled_df_cleaned[column].apply(lambda x: round(x, 3))

downsampled_df_cleaned = downsampled_df_cleaned[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]
downsampled_df_cleaned.to_csv('final_walking_running.csv', index=False)

print(downsampled_df_cleaned)
print('Activity values after down sampling\n', downsampled_df_cleaned['activity'].value_counts())
print("Saved final combined data")
