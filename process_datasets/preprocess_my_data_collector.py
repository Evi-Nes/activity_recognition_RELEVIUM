import os
import pandas as pd

folder = 'my_data_collector'
merged_files = []
i = 1
columns_to_round = ['accel_x', 'accel_y', 'accel_z']
# user_id = 'ME'

for file in os.listdir(folder):
    data = pd.read_csv(os.path.join(folder, file))
    data.columns = ['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z']
    # data['user_id'] = user_id

    data = data[10:]
    size = len(data)
    data = data[:size - 20]

    for column in columns_to_round:
        data[column] = data[column].apply(lambda x: round(x, 3))

    merged_file_path = os.path.join(folder, f'data_batch_{i}.csv')
    merged_files.append(merged_file_path)
    data.to_csv(merged_file_path, index=False)
    print(f'Writing {merged_file_path}')
    i = i + 1

if not merged_files:
    print("No merged files found.")
all_data = []

for file in merged_files:
    df = pd.read_csv(file)
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

missing_values = combined_df.isnull().sum()
print("Missing values in each column:\n", missing_values)
df_cleaned = combined_df.dropna()

df_cleaned = df_cleaned[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z']]
df_cleaned.to_csv('final_my_data_collector.csv', index=False)

print(df_cleaned)
print("Saved final combined data")
print('Activity values \n', df_cleaned['activity'].value_counts())

