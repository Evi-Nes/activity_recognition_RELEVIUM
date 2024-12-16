import pandas as pd

df = pd.read_csv('final_domino.csv')
print(df['activity'].value_counts())
print(df.value_counts())
