import numpy as np
import pandas as pd
from numpy.ma import count

# 1 lying
# 2 sitting
# 3 standing
# 4 walking
# 5 running
# 6 cycling
# 7 Nordic walking
# 9 watching TV
# 10 computer work
# 11 car driving
# 12 ascending stairs
# 13 descending stairs
# 16 vacuum cleaning
# 17 ironing
# 18 folding laundry
# 19 house cleaning
# 20 playing soccer
# 24 rope jumping
# 0 other (transient activities)

df = pd.read_csv('data_pamap2.csv')
df['user_id'] = df['user_id'].astype(str) + 'P'
print(df.head())

# remove activities
desired_activities = [1, 2, 3, 4, 5, 6, 7, 24]
df = df[df['activity'].isin(desired_activities)]
unique_activities = df['activity'].unique()

# rename activities
activity_mapping = {1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking', 5: 'running', 6: 'cycling', 7: 'walking', 24: 'exercise'}
df['activity'] = df['activity'].replace(activity_mapping)

unique_activities = df['activity'].unique()
for activity in unique_activities:
    print(activity)
    print(count(df[df['activity'] == activity]))

df['timestamp'] = df['timestamp'] * 1000
df['datetime_ms'] = pd.to_datetime(df['timestamp'], unit='ms')
print(df.head())

# Downsample 100Hz -> 25Hz
downsampled_rows = []
step = 4

for i in range(0, len(df), step):
    group = df.iloc[i:i + step]

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
columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
for column in columns:
    downsampled_df[column] = downsampled_df[column].apply(lambda x: round(x, 3))

print(f"New size of dataframe: {len(downsampled_df)}")

downsampled_df.to_csv('final_pamap2.csv', index=False)



