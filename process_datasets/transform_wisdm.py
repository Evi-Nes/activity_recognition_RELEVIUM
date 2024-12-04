import numpy as np
import pandas as pd

# A Walking
# B Jogging
# C Stairs
# D Sitting
# E Standing
# F Typing
# G Brushing Teeth
# H Eating Soup
# I Eating Chips
# J Eating Pasta
# K Drinking from Cup
# L Eating Sandwich
# M Kicking (Soccer Ball)
# O Playing Catch w/Tennis Ball
# P Dribblinlg (Basketball)
# Q Writing
# R Clapping
# S Folding Clothes


df = pd.read_csv('data_wisdm.csv')
df['user_id'] = df['user_id'].astype(str) + 'W'
# print(df.head())

# remove activities
desired_activities = ['A', 'B', 'C', 'D', 'E']
df = df[df['activity'].isin(desired_activities)]

# for activity in desired_activities:
#     print(len(df[df['activity'] == activity]))

# rename activities
activity_mapping = {
    'A': 'walking', 'B': 'running', 'C': 'stairs', 'D': 'sitting', 'E': 'standing'}
df['activity'] = df['activity'].replace(activity_mapping)

df = df[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]
unique_activities = df['activity'].unique()
print(unique_activities)
# print(df.head())
# print(len(df))

# Increase frequency 20Hz -> 25Hz
print(df.index.duplicated().sum())
print('\n')
# Set timestamp as index
df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
print(f"Timestamp Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

df.set_index('datetime', inplace=True)
print(df.head())
# print(df['timestamp'].diff().describe())

print(df.index.duplicated().sum())
print('\n')

# df.asfreq(freq='40ms', method='ffill')
# df = df.reset_index()
df = df.resample('40ms').ffill()  # Forward-fill missing values
df = df.reset_index()
# df = df.reset_index().rename(columns={'index': 'timestamp'})


