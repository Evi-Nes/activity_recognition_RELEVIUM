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

        sleep_label_path = os.path.join(folder_path, 'sleep.csv')
        acc_path = os.path.join(folder_path, 'Actigraph.csv')

        if os.path.exists(sleep_label_path):
            sleep_label = pd.read_csv(sleep_label_path)
            smartwatch_acc = pd.read_csv(acc_path)

            # Convert 'day' to actual dates
            sleep_label['InDate'] = reference_date + pd.to_timedelta(sleep_label['In Bed Date'] - 1, unit='D')
            sleep_label['OutDate'] = reference_date + pd.to_timedelta(sleep_label['Out Bed Date'] - 1, unit='D')

            smartwatch_acc['date'] = reference_date + pd.to_timedelta(smartwatch_acc['day'] - 1, unit='D')

            sleep_label = sleep_label.dropna(subset=['In Bed Time', 'Out Bed Time'])

            def fix_invalid_times(row):
                if row['In Bed Time'] == '24:00':
                    row['In Bed Time'] = '00:00'
                    row['In Bed Day'] += 1
                if row['Out Bed Time'] == '24:00':
                    row['Out Bed Time'] = '00:00'
                    row['Out Bed Day'] += 1
                return row

            sleep_label = sleep_label.apply(fix_invalid_times, axis=1)

            sleep_label['InBedTime'] = pd.to_datetime(sleep_label['InDate'].astype(str) + ' ' + sleep_label['In Bed Time'])
            sleep_label['OutBedTime'] = pd.to_datetime(sleep_label['OutDate'].astype(str) + ' ' + sleep_label['Out Bed Time'])
            smartwatch_acc['timestamp'] = pd.to_datetime(smartwatch_acc['date'].astype(str) + ' ' + smartwatch_acc['time'])

            smartwatch_acc.rename(columns={'Axis1': 'accel_x', 'Axis2': 'accel_y', 'Axis3': 'accel_z'}, inplace=True)

            smartwatch_acc['key'] = 1
            sleep_label['key'] = 1
            cross_joined = pd.merge(smartwatch_acc, sleep_label, on='key').drop('key', axis=1)

            result = cross_joined[(cross_joined['timestamp'] >= cross_joined['InBedTime']) & (cross_joined['timestamp'] <= cross_joined['OutBedTime'])]
            result['user_id'] = user_id.astype(str) + 'M'
            result['activity'] = 'sleeping'
            result = result[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z']]

            merged_file_path = os.path.join(main_folder, f'sleep_data_user_{user_id}.csv')
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
        combined_file_path = os.path.join(main_folder, 'final_multilevel_sleep.csv')
        combined_df_cleaned.to_csv('final_multilevel_sleep.csv', index=False)
        print(f"Saved final combined data to {combined_file_path}")

    else:
        print("No dataframes found to concatenate.")

else:
    print("No merged files found.")