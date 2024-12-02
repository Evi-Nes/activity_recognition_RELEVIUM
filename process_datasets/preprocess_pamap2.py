import os
import pandas as pd
# from tqdm import tqdm
# import re

# Define the path to the DOMINO folder
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
    selected_columns['user_id'] = user_id

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
# combined_df['user_id'] = '101'

combined_df.to_csv('data_pamap2.csv', index=False)
print(combined_df.head())
