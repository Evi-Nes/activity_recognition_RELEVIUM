import numpy as np
import pandas as pd
from numpy.ma.core import count

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
df['user_id'] = df['user_id'].astype(int)
df['user_id'] = df['user_id'].astype(str) + 'W'
# print(df.head())

# remove activities
desired_activities = ['A', 'B', 'C', 'D', 'E']
df = df[df['activity'].isin(desired_activities)]

# rename activities
activity_mapping = {
    'A': 'walking', 'B': 'running', 'C': 'stairs', 'D': 'sitting', 'E': 'standing'}
df['activity'] = df['activity'].replace(activity_mapping)

df = df[['timestamp', 'activity', 'user_id', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]
unique_activities = df['activity'].unique()
print(df.head)

for activity in unique_activities:
    print(activity)
    print(count(df[df['activity'] == activity]))

df.to_csv('final_wisdm.csv', index=False)




