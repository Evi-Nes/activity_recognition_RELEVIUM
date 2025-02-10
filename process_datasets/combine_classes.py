import pandas as pd

data = pd.read_csv('combined_dreamt_25Hz.csv')
mapping = {'W': 'lying', 'N1': 'sleeping', 'N2': 'sleeping', 'N3': 'sleeping', 'R': 'sleeping'}
data['activity'] = data['activity'].replace(mapping)
unique_activities = data['activity'].unique()
for activity in unique_activities:
    data = data.loc[data['activity'] == activity, :int(len(data)*0.7)]

print(data['activity'].value_counts())
data.to_csv('final_dreamt_25Hz.csv', index=False)

