import pandas as pd
from sklearn.model_selection import train_test_split
path = "../process_datasets/combined_dataset.csv"
df = pd.read_csv(path)

print(df['activity'].value_counts())
print(df.value_counts())

# Separate activities with only one user
special_activities = ['static_exercising', 'dynamic_exercising']
special_data = df[df['activity'].isin(special_activities)]

# Perform temporal split for these activities
special_train = special_data.groupby('activity').apply(
    lambda group: group.iloc[:int(0.8 * len(group))]
).reset_index(drop=True)

special_test = special_data.groupby('activity').apply(
    lambda group: group.iloc[int(0.8 * len(group)):]
).reset_index(drop=True)

# Remove these activities from the main dataset
df_remaining = df[~df['activity'].isin(special_activities)]


user_activity_groups = df_remaining.groupby(['user_id', 'activity']).size().reset_index(name='counts')
user_activity_map = user_activity_groups[['user_id', 'activity']].drop_duplicates()

# Stratified splitting of users
train_users, test_users = train_test_split(
    user_activity_map['user_id'].unique(),
    test_size=0.2,
    random_state=42)

# Filter the data based on the split
train_data = df_remaining[df_remaining['user_id'].isin(train_users)]
test_data = df_remaining[df_remaining['user_id'].isin(test_users)]

final_train = pd.concat([train_data, special_train], ignore_index=True)
final_test = pd.concat([test_data, special_test], ignore_index=True)

# Ensure all activities are present in both sets
print("Training data")
print(final_train['activity'].value_counts())
print("Testing data")
print(final_test['activity'].value_counts())

assert len(set(train_data['user_id']) & set(test_data['user_id'])) == 0, "Train and test sets have overlapping users!"
