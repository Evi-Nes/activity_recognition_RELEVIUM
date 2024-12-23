import pandas as pd

df = pd.read_csv('combined_dataset.csv')
print(df['activity'].value_counts())
print(df.value_counts())

print(df['user_id'].unique())