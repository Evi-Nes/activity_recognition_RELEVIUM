import os
import pandas as pd

##### The PAMAP2 dataset #####
# 100Hz frequency, all files in one folder, all data in one file per user
# Overall PAMAP2 includes data about a TRANSITION activity + 24 activities:
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


dat_directory = 'PAMAP2_Dataset/Protocol'
dat_files = ['subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat', 'subject105.dat', 'subject106.dat', 'subject107.dat', 'subject108.dat']

merged_files = []
items_per_row = 54


def split_data_into_rows(data, items_per_row):
    return [data[i:i + items_per_row] for i in range(0, len(data), items_per_row)]


for file in dat_files:
    filename = os.path.basename(file)
    user_id = filename.replace('subject', '').replace('.dat', '')

    full_path = os.path.join(dat_directory, file)
    with open(full_path, 'r') as f:
        flat_data = f.read().split()

    rows = split_data_into_rows(flat_data, items_per_row)
    df = pd.DataFrame(rows)

    selected_columns = df.iloc[:, [0, 1, 4, 5, 6, 10, 11, 12]]
    selected_columns['user_id'] = user_id + 'P'

    merged_files.append(selected_columns)
    print(f'Merged {file}')

# Concatenate all the DataFrames in the list into a single DataFrame
combined_df = pd.concat(merged_files, ignore_index=True)
combined_df.rename(columns={0: 'timestamp'}, inplace=True)
combined_df.rename(columns={1: 'activity'}, inplace=True)
combined_df.rename(columns={4: 'accel_x'}, inplace=True)
combined_df.rename(columns={5: 'accel_y'}, inplace=True)
combined_df.rename(columns={6: 'accel_z'}, inplace=True)
combined_df.rename(columns={10: 'gyro_x'}, inplace=True)
combined_df.rename(columns={11: 'gyro_y'}, inplace=True)
combined_df.rename(columns={12: 'gyro_z'}, inplace=True)

combined_df = combined_df[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]
missing_values = combined_df.isnull().sum()
print("Missing values in each column:\n", missing_values)
combined_df = combined_df.dropna()
print("Combined all data")

columns_to_int = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
for column in columns_to_int:
    combined_df[column] = combined_df[column].astype(float)
combined_df['activity'] = combined_df['activity'].astype(int)

# remove activities
desired_activities = [1, 2, 3, 4, 5, 6, 7, 12]
combined_df = combined_df[combined_df['activity'].isin(desired_activities)]

# rename activities
activity_mapping = {1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking', 5: 'running', 6: 'cycling', 7: 'walking', 12: 'walking'}
combined_df['activity'] = combined_df['activity'].replace(activity_mapping)
print('Activity values before down sampling \n', combined_df['activity'].value_counts())

# Downsample 100Hz -> 25Hz
downsampled_rows = []
step = 4

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
columns_to_round = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
for column in columns_to_round:
    downsampled_df[column] = downsampled_df[column].apply(lambda x: round(x, 3))

print('Activity values after down sampling \n', downsampled_df['activity'].value_counts())

print('Final data \n', downsampled_df.head())
print(f"Final size of dataframe: {len(downsampled_df)}")
downsampled_df.to_csv('final_pamap2.csv', index=False)
