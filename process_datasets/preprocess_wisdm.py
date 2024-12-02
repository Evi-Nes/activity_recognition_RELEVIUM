import os
import pandas as pd

accel_folder = ('wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/accel')
gyro_folder = ('wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/gyro')

merged_files = []

for file in os.listdir(accel_folder):
    if file.startswith('data'):

        file_path = os.path.join(accel_folder, file)
        accel_data = pd.read_csv(file_path, header=None, encoding="ISO-8859-1")

        accel_data.columns = ['user_id', 'activity', 'timestamp', 'accel_x', 'accel_y', 'accel_z']
        accel_data['accel_z'] = accel_data['accel_z'].str.rstrip(';')

        accel_data['timestamp'] = accel_data['timestamp'].astype(int)
        accel_data['user_id'] = accel_data['user_id']
        user_id = accel_data['user_id'][0]

        merged_file_path = os.path.join(accel_folder, f'merged_user_{user_id}.csv')
        accel_data.to_csv(merged_file_path, index=False)

        merged_files.append(merged_file_path)

all_data = []
for file in merged_files:
    df = pd.read_csv(file)
    all_data.append(df)

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv('wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/accel_data.csv', index=False)

print('Accelerometer data done')

# Gyro data
merged_files = []
for file in os.listdir(gyro_folder):
    if file.startswith('data'):

        file_path = os.path.join(gyro_folder, file)
        gyro_data = pd.read_csv(file_path, header=None, encoding="ISO-8859-1")
        gyro_data.columns = ['user_id', 'activity', 'timestamp', 'gyro_x', 'gyro_y', 'gyro_z']
        gyro_data['gyro_z'] = gyro_data['gyro_z'].str.rstrip(';')

        gyro_data['timestamp'] = gyro_data['timestamp'].astype(int)
        gyro_data['user_id'] = gyro_data['user_id']
        user_id = gyro_data['user_id'][0]

        merged_file_path = os.path.join(gyro_folder, f'merged_user_{user_id}.csv')
        gyro_data.to_csv(merged_file_path, index=False)

        merged_files.append(merged_file_path)

all_data = []
for file in merged_files:
    df = pd.read_csv(file)
    all_data.append(df)

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv('wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/gyro_data.csv', index=False)

print('Gyroscope data done')

# Final Merge
smartwatch_acc = pd.read_csv('wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/accel_data.csv')
smartwatch_gyro = pd.read_csv('wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/gyro_data.csv')
merged_data = pd.merge(smartwatch_acc, smartwatch_gyro, on=['user_id', 'activity', 'timestamp'])
merged_data = merged_data[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]

merged_data.to_csv('data_wisdm.csv', index=False)
print("Saved merged data")

