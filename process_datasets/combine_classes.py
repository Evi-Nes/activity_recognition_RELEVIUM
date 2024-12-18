import pandas as pd

data = pd.read_csv('combined_dreamt_25Hz.csv')
mapping = {'W': 'lying', 'N1': 'lying', 'N2': 'lying', 'N3': 'lying', 'R': 'lying'}
data['activity'] = data['activity'].replace(mapping)

print(data['activity'].value_counts())
data.to_csv('final_dreamt_25Hz.csv', index=False)

