import pandas as pd
import os
from tqdm import tqdm
import re

main_folder = 'multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/MMASH/DataPaper'
reference_date = pd.Timestamp('2024-12-02')
merged_files = []

for folder in tqdm(os.listdir(main_folder)):
    folder_path = os.path.join(main_folder, folder)

    if os.path.isdir(folder_path) and folder.startswith('user_'):
        user_id = re.search(r'\d+', folder).group()

        activity_labels_path = os.path.join(folder_path, 'Activity.csv')
        acc_path = os.path.join(folder_path, 'Actigraph.csv')

        if os.path.exists(activity_labels_path):
            activity_labels = pd.read_csv(activity_labels_path)
            smartwatch_acc = pd.read_csv(acc_path)

            # Convert 'day' to actual dates
            activity_labels['Date'] = reference_date + pd.to_timedelta(activity_labels['Day'] - 1, unit='D')
            smartwatch_acc['date'] = reference_date + pd.to_timedelta(smartwatch_acc['day'] - 1, unit='D')

            activity_labels = activity_labels.dropna(subset=['Start', 'End'])

            def fix_invalid_times(row):
                if row['Start'] == '24:00':
                    row['Start'] = '00:00'
                    row['Day'] += 1
                if row['End'] == '24:00':
                    row['End'] = '00:00'
                    row['Day'] += 1
                return row

            activity_labels = activity_labels.apply(fix_invalid_times, axis=1)

            activity_labels['time_start'] = pd.to_datetime(activity_labels['Date'].astype(str) + ' ' + activity_labels['Start'])
            activity_labels['time_end'] = pd.to_datetime(activity_labels['Date'].astype(str) + ' ' + activity_labels['End'])
            smartwatch_acc['timestamp'] = pd.to_datetime(smartwatch_acc['date'].astype(str) + ' ' + smartwatch_acc['time'])

            smartwatch_acc.rename(columns={'Axis1': 'accel_x', 'Axis2': 'accel_y', 'Axis3': 'accel_z'}, inplace=True)

            smartwatch_acc['key'] = 1
            activity_labels['key'] = 1
            cross_joined = pd.merge(smartwatch_acc, activity_labels, on='key').drop('key', axis=1)

            result = cross_joined[(cross_joined['timestamp'] >= cross_joined['time_start']) & (cross_joined['timestamp'] <= cross_joined['time_end'])]
            result['user_id'] = result['user_id'] = user_id.astype(str) + 'M'
            result.rename(columns={'Activity': 'activity'}, inplace=True)
            result = result[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z']]

            desired_activities = [2, 3]
            result = result[result['activity'].isin(desired_activities)]
            mapping = {2: 'lying', 3: 'sitting'}
            result['activity'] = result['activity'].replace(mapping)

            merged_file_path = os.path.join(main_folder, f'data_user_{user_id}.csv')
            result.to_csv(merged_file_path, index=False)
            print(f"Saved data for user {user_id} to {merged_file_path}")

            merged_files.append(merged_file_path)


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

        combined_df_cleaned = combined_df_cleaned[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z']]
        combined_file_path = os.path.join(main_folder, 'final_multilevel_1-2.csv')
        combined_df_cleaned.to_csv('final_multilevel_1-2.csv', index=False)
        print(f"Saved final combined data to {combined_file_path}")

    else:
        print("No dataframes found to concatenate.")

else:
    print("No merged files found.")