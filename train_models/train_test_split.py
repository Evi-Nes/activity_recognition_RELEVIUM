import pandas as pd

df = pd.read_csv('../process_datasets/combined_dataset.csv')
print(df['activity'].value_counts())
print(df.value_counts())

# print(df['user_id'].unique())
unique_users = df['user_id'].unique()
domino_users = []
wisdm_users = []
pamap_users = []
dreamt_users = []
exercising_user = '3DE'
walking_running_user = '1E'

for user in unique_users:
    if user.endswith('D'):
        domino_users.append(user)
    elif user.endswith('W'):
        wisdm_users.append(user)
    elif user.endswith('P'):
        pamap_users.append(user)
    elif user.endswith('DR'):
        dreamt_users.append(user)

train_users = []
test_users = []
train_users.append(domino_users[:int(0.8*len(domino_users))]) 
test_users.append(domino_users[int(0.8*len(domino_users)):])

train_users.append(wisdm_users[:int(0.8*len(domino_users))]) 
test_users.append(wisdm_users[int(0.8*len(domino_users)):])

train_users.append(dreamt_users[:int(0.8*len(domino_users))]) 
test_users.append(dreamt_users[int(0.8*len(domino_users)):])

train_users.append(pamap_users[:int(0.8*len(domino_users))]) 
test_users.append(pamap_users[int(0.8*len(domino_users)):])

train_users_flat = [element for sublist in train_users for element in sublist]
test_users_flat = [element for sublist in test_users for element in sublist]

walking_data =  df[df['user_id'] == walking_running_user]
train_walking = walking_data[:int(0.8 * len(walking_data))]
test_walking = walking_data[int(0.8 * len(walking_data)):]

static_exercising_data =  df[df['activity'] == 'static_exercising']
train_static = static_exercising_data[:int(0.8 * len(static_exercising_data))]
test_static = static_exercising_data[int(0.8 * len(static_exercising_data)):]

dynamic_exercising_data =  df[df['activity'] == 'dynamic_exercising']
train_dynamic = dynamic_exercising_data[:int(0.8 * len(dynamic_exercising_data))]
test_dynamic = dynamic_exercising_data[int(0.8 * len(dynamic_exercising_data)):]

train_users_data = df[df['user_id'].isin(train_users_flat)]
test_users_data = df[df['user_id'].isin(test_users_flat)]

train_data = pd.concat([train_users_data, train_walking, train_static, train_dynamic], axis=0, ignore_index=True)
test_data = pd.concat([test_users_data, test_walking, test_static, test_dynamic], axis=0, ignore_index=True)

print(train_data['activity'].value_counts())
print(test_data['activity'].value_counts())
print(len(train_data))
print(len(test_data))

train_data.to_csv('../process_datasets/train_data.csv', index=False)
test_data.to_csv('../process_datasets/test_data.csv', index=False)