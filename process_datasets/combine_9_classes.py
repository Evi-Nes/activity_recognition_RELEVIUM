import pandas as pd
import os
import contextlib

# Redirect stderr to /dev/null to silence warnings
devnull = open(os.devnull, 'w')
contextlib.redirect_stderr(devnull)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


if __name__ == '__main__':
    frequency = 25
    time_required_ms = 10000
    samples_required = int(time_required_ms * frequency / 1000)
    class_labels = ['cycling', 'dynamic_exercising', 'lying', 'running', 'sitting', 'standing', 'static_exercising', 'walking', 'sleeping']

    train_path = "train_data.csv"
    test_path = "test_data.csv"
    dreamt_path = "combined_dreamt_25Hz.csv"

    print('---- Initial Dreamt Data ----')
    dreamt_data = pd.read_csv(dreamt_path)
    dreamt_data = dreamt_data[['activity', 'accel_x', 'accel_y', 'accel_z', 'user_id']]
    dreamt_data = dreamt_data.dropna()
    mapping = {'W': 'lying', 'N1': 'sleeping', 'N2': 'sleeping', 'N3': 'sleeping', 'R': 'sleeping'}
    dreamt_data['activity'] = dreamt_data['activity'].replace(mapping)
    dreamt_data = dreamt_data[:int((len(dreamt_data)*0.3))]
    
    unique_activities = dreamt_data['activity'].unique()
    for activity in unique_activities:
        print(f"Activity {activity}: {len(dreamt_data[dreamt_data['activity'] == activity])}")

    # Activity Train Data
    print('\n---- Train Data ----')
    train_data = pd.read_csv(train_path)
    train_data = train_data[['activity', 'accel_x', 'accel_y', 'accel_z']]
    train_data = train_data.dropna()
    train_data = train_data[train_data['activity'] != 'lying']
    
    unique_activities = train_data['activity'].unique()
    for activity in unique_activities:
        print(f"Activity {activity}: {len(train_data[train_data['activity'] == activity])}")
    
    # Activity Test Data
    print('\n---- Test Data----')
    test_data = pd.read_csv(test_path)
    test_data = test_data[['activity', 'accel_x', 'accel_y', 'accel_z']]
    test_data = test_data.dropna()
    test_data = test_data[test_data['activity'] != 'lying']

    unique_activities = test_data['activity'].unique()
    for activity in unique_activities:
        print(f"Activity {activity}: {len(test_data[test_data['activity'] == activity])}")

    print('\n---- Dreamt Train Test Split ----')
    train_dreamt_data = dreamt_data[:int(0.8*len(dreamt_data))]
    lying_data = train_dreamt_data[train_dreamt_data['activity'] == 'lying']
    sleeping_data = train_dreamt_data[train_dreamt_data['activity'] == 'sleeping']
    sleeping_data = sleeping_data[:int(len(sleeping_data) * 0.5)]
    train_dreamt_data = pd.concat([lying_data, sleeping_data])

    unique_activities = train_dreamt_data['activity'].unique()
    for activity in unique_activities:
        print(f"Activity {activity}: {len(train_dreamt_data[train_dreamt_data['activity'] == activity])}")
    print('\nUnique Users: ', train_dreamt_data['user_id'].unique())

    test_dreamt_data = dreamt_data[int(0.8*len(dreamt_data)):]
    lying_data = test_dreamt_data[test_dreamt_data['activity'] == 'lying']
    lying_data = lying_data[int(len(lying_data) * 0.4):]
    sleeping_data = test_dreamt_data[test_dreamt_data['activity'] == 'sleeping']
    sleeping_data = sleeping_data[:int(len(sleeping_data) * 0.5)]
    test_dreamt_data = pd.concat([lying_data, sleeping_data])

    unique_activities = test_dreamt_data['activity'].unique()
    for activity in unique_activities:
        print(f"Activity {activity}: {len(test_dreamt_data[test_dreamt_data['activity'] == activity])}")
    print('\nUnique Users: ', test_dreamt_data['user_id'].unique())

    # Combine data
    print('\n---- Combined Train Data ----')
    combined_train_data = pd.concat([train_data, train_dreamt_data])
    combined_train_data.to_csv('train_data_9.csv', index=False)
    unique_activities = combined_train_data['activity'].unique()
    for activity in unique_activities:
        print(f"Activity {activity}: {len(combined_train_data[combined_train_data['activity'] == activity])}")

    print('\n---- Combined Test Data ----')
    combined_test_data = pd.concat([test_data, test_dreamt_data])
    combined_test_data.to_csv('test_data_9.csv', index=False)
    unique_activities = combined_test_data['activity'].unique()
    for activity in unique_activities:
        print(f"Activity {activity}: {len(combined_test_data[combined_test_data['activity'] == activity])}")




