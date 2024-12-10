import pandas as pd
from numpy.ma import count

# Overall DOMINO includes data about
# a TRANSITION activity + 14 activities:
# - BRUSHING TEETH
# - CYCLING
# - ELEVATOR DOWN
# - ELEVATOR UP
# - LYING
# - MOVING BY CAR
# - RUNNING
# - SITTING
# - SITTING ON TRANSPORT
# - STAIRS DOWN
# - STAIRS UP
# - STANDING
# - STANDING ON TRANSPORT
# - WALKING

df = pd.read_csv('data_domino.csv')
print(f"Initial size of dataframe: {len(df)}")

# remove activities
desired_activities = ['CYCLING', 'LYING', 'RUNNING', 'SITTING', 'SITTING_ON_TRANSPORT', 'STAIRS_DOWN', 'STAIRS_UP', 'STANDING', 'STANDING_ON_TRANSPORT', 'WALKING']
df = df[df['activity'].isin(desired_activities)]

# rename and merge activities
activity_mapping = {
    'CYCLING': 'cycling', 'LYING': 'lying', 'RUNNING': 'running', 'SITTING': 'sitting', 'SITTING_ON_TRANSPORT': 'sitting',
    'STAIRS_DOWN': 'walking', 'STAIRS_UP': 'walking', 'STANDING': 'standing', 'STANDING_ON_TRANSPORT': 'standing', 'WALKING': 'walking'}
df['activity'] = df['activity'].replace(activity_mapping)

unique_activities = df['activity'].unique()
for activity in unique_activities:
    print(activity)
    print(count(df[df['activity'] == activity]))
print(f"New size of dataframe: {len(df)}")

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
downsampled_df['user_id'] = downsampled_df['user_id'].astype(str) + 'D'
columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
for column in columns:
    downsampled_df[column] = downsampled_df[column].apply(lambda x: round(x, 3))
print(downsampled_df.head())

print(f"Final size of dataframe: {len(downsampled_df)}")
downsampled_df.to_csv('final_domino.csv', index=False)
