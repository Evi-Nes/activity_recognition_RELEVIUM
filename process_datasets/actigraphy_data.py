import pandas as pd

df = pd.read_csv('combined_dreamt_32Hz.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
print(df.head())

# Downsample to 1-second intervals
resampled_data = df.resample('1s').agg({
    'accel_x': 'mean',
    'accel_y': 'mean',
    'accel_z': 'mean',
    'hr': 'mean',
    'activity': 'first',
    'user_id': 'first'
}).reset_index()

columns_to_round = ['accel_x', 'accel_y', 'accel_z']
for column in columns_to_round:
    resampled_data[column] = resampled_data[column].apply(lambda x: round(x, 3))

resampled_data['hr'] = resampled_data['hr'].apply(lambda x: int(x))
resampled_data = resampled_data[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z', 'hr']]
print(resampled_data.head())
resampled_data.to_csv('final_dreamt_1Hz.csv', index=False)