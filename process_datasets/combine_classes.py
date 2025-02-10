import pandas as pd

data = pd.read_csv('combined_dreamt_25Hz.csv')

data = data[:int(len(data)*0.6)]
print(data['activity'].value_counts())

mapping = {'W': 'lying', 'N1': 'sleeping', 'N2': 'sleeping', 'N3': 'sleeping', 'R': 'sleeping'}
data['activity'] = data['activity'].replace(mapping)

data.to_csv('final_dreamt_25Hz.csv', index=False)

