import pandas as pd
import matplotlib.pyplot as plt
import os

path = "combined_dreamt_25Hz.csv"
if not os.path.exists(f'plots_dreamt'):
    os.makedirs(f'plots_dreamt')
print('hereee')

data = pd.read_csv(path)
data = data[['activity', 'accel_x', 'accel_y', 'accel_z']]
# data[['accel_x', 'accel_y', 'accel_z']] = data[['accel_x', 'accel_y', 'accel_z']] / 1000
data = data.dropna()
unique_activities = data['activity'].unique()

data['accel_x_diff'] = data['accel_x'].diff()
print("---- accel_x_diff ----")
print(data['accel_x_diff'].describe())
data = data.drop(['accel_x_diff'], axis=1)

data['accel_y_diff'] = data['accel_y'].diff()
print("---- accel_y_diff ----")
print(data['accel_y_diff'].describe())
data = data.drop(['accel_y_diff'], axis=1)

data['accel_z_diff'] = data['accel_z'].diff()
print("---- accel_z_diff ----")
print(data['accel_z_diff'].describe())
data = data.drop(['accel_z_diff'], axis=1)

for activity in unique_activities:
    subset = data[data['activity'] == activity].iloc[200:800]
    subset = subset.drop(['activity'], axis=1)

    subset.plot(subplots=True, figsize=(10, 10))
    plt.xlabel('Time')
    plt.savefig(f'plots_dreamt/unscaled_{activity}.png')