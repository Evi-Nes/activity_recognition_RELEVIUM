import pandas as pd
from numpy.ma.core import count

df_domino = pd.read_csv('final_domino.csv')
df_pamap2 = pd.read_csv('final_pamap2.csv')
df_exercise = pd.read_csv('final_exercise.csv')

combined_df = pd.concat([df_domino, df_pamap2, df_exercise], axis=0, ignore_index=True)
print(combined_df.head())

unique_activities = combined_df['activity'].unique()

for activity in unique_activities:
    print(activity)
    print(count(combined_df[combined_df['activity'] == activity]))

combined_df.to_csv('combined_dataset2.csv', index=False)