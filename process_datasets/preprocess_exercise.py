import os
from datetime import date, timedelta, datetime
import pandas as pd

columns = ['elapsed_seconds', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
columns_round = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
hour = 0
merged_files = []

for filename in os.listdir('exercise_data'):
    df = pd.read_csv(f'exercise_data/{filename}', header=None, names=columns, skiprows=1)
    filename_split = filename.split('_')
    df['user_id'] = filename_split[0].replace('subject', '')
    df['user_id'] = df['user_id'].astype(str) + 'E'

    for column in columns_round:
        df[column] = df[column].apply(lambda x: round(x, 3))

    start_time = datetime.now() +timedelta(hours=hour)
    df['timestamp'] = df['elapsed_seconds'].apply(lambda x: start_time + timedelta(seconds=x))
    hour += 1

    df['activity'] = 'exercise'
    df = df[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    # print(df.head())

    merged_files.append(df)

combined_df = pd.concat(merged_files, ignore_index=True)
combined_df.to_csv('final_exercise.csv', index=False)
print(combined_df.head())