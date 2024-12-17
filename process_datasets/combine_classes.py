import pandas as pd

data = pd.read_csv('combined_dataset.csv')
mapping = {'sleeping': 'lying'}
data['activity'] = data['activity'].replace(mapping)

print(data['activity'].value_counts())
data.to_csv('combined_dataset_8_classes.csv', index=False)

