import os
import pandas as pd
from tqdm import tqdm
import re

# Define the path to the DOMINO folder
domino_folder = 'DOMINO'
merged_files = []


# Iterate through all user folders in the DOMINO folder
for folder in tqdm(os.listdir(domino_folder)):
    folder_path = os.path.join(domino_folder, folder)

    if os.path.isdir(folder_path) and folder.startswith('user-'):
        # Extract the numeric part of the user folder name
        user_id = re.search(r'\d+', folder).group()

        activity_labels_path = os.path.join(folder_path, 'activity_labels.csv')
        acc_path = os.path.join(folder_path, 'smartwatch_acc.csv')
        gyro_path = os.path.join(folder_path, 'smartwatch_gyr.csv')

        if os.path.exists(activity_labels_path):
            activity_labels = pd.read_csv(activity_labels_path)
            smartwatch_acc = pd.read_csv(acc_path)
            smartwatch_gyro = pd.read_csv(gyro_path)

            smartwatch_acc.rename(columns={'x': 'accel_x', 'y': 'accel_y', 'z': 'accel_z'}, inplace=True)
            smartwatch_gyro.rename(columns={'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'}, inplace=True)

            smartwatch_acc['key'] = 1
            activity_labels['key'] = 1
            cross_joined = pd.merge(smartwatch_acc, activity_labels, on='key').drop('key', axis=1)

            # Filter to keep only the rows where ts is within the range [ts_start, ts_end]
            result = cross_joined[(cross_joined['ts'] >= cross_joined['ts_start']) & (cross_joined['ts'] <= cross_joined['ts_end'])]

            merged_data = pd.merge_asof(result, smartwatch_gyro, on='ts', direction='nearest', tolerance=20)

            merged_data['user_id'] = user_id
            merged_data.rename(columns={'ts': 'timestamp'}, inplace=True)
            merged_data.rename(columns={'label': 'activity'}, inplace=True)

            merged_data = merged_data.drop('ts_start', axis=1)
            merged_data = merged_data.drop('ts_end', axis=1)

            merged_file_path = os.path.join(domino_folder, f'merged_user_{user_id}.csv')
            merged_data.to_csv(merged_file_path, index=False)
            print(f"Saved merged data for user {user_id} to {merged_file_path}")

            merged_files.append(merged_file_path)

# Step 4: Combine all individual merged files into one final CSV file
if merged_files:
    all_data = []

    for file in merged_files:
        df = pd.read_csv(file)
        all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Check and remove Nan values
        missing_values = combined_df.isnull().sum()
        print("Missing values in each column:\n", missing_values)
        combined_df_cleaned = combined_df.dropna()
        print("Original DataFrame shape:", combined_df.shape)
        print("Cleaned DataFrame shape:", combined_df_cleaned.shape)

        combined_file_path = os.path.join(domino_folder, 'data_domino.csv')
        combined_df_cleaned.to_csv('data_domino.csv', index=False)
        print(f"Saved final combined data to {combined_file_path}")

    else:
        print("No dataframes found to concatenate.")

else:
    print("No merged files found.")
