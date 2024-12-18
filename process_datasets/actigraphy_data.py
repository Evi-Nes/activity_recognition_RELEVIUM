import pandas as pd

df = pd.read_csv('combined_dreamt_32Hz.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Downsample to 1-second intervals
grouped = df.groupby('user_id')
resampled_groups = []
for user, group in grouped:
    resampled = group.resample('1s').agg({
        'accel_x': 'mean',
        'accel_y': 'mean',
        'accel_z': 'mean',
        'hr': 'mean',
        'activity': 'first',
        'user_id': 'first'
    }).reset_index()
    resampled_groups.append(resampled)

# Combine all resampled groups back into a single DataFrame
resampled_data = pd.concat(resampled_groups)
resampled_data.reset_index(inplace=True)

resampled_data = resampled_data.ffill()
columns_to_round = ['accel_x', 'accel_y', 'accel_z']
for column in columns_to_round:
    resampled_data[column] = resampled_data[column].apply(lambda x: round(x, 3))

resampled_data['hr'] = resampled_data['hr'].apply(lambda x: int(x))
resampled_data = resampled_data[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z', 'hr']]
print(resampled_data.head())
print(resampled_data['activity'].value_counts())
resampled_data.to_csv('final_dreamt_1Hz.csv', index=False)