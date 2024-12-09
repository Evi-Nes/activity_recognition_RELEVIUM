import pandas as pd
from numpy.ma.core import count

df_domino = pd.read_csv('final_domino.csv')
df_pamap2 = pd.read_csv('final_pamap2.csv')
df_exercise1 = pd.read_csv('final_walk_running.csv')
df_wisdm = pd.read_csv('final_wisdm.csv')

combined_df = pd.concat([df_domino, df_pamap2, df_wisdm, df_exercise1], axis=0, ignore_index=True)
print(combined_df.head())

unique_activities = combined_df['activity'].unique()

for activity in unique_activities:
    print(activity)
    print(count(combined_df[combined_df['activity'] == activity]))

combined_df.to_csv('combined_dataset4.csv', index=False)