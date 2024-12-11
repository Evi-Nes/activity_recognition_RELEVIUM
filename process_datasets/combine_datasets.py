import pandas as pd
from numpy.ma.core import count

df_domino = pd.read_csv('final_domino.csv')
df_pamap2 = pd.read_csv('final_pamap2.csv')
df_wisdm = pd.read_csv('final_wisdm.csv')
df_walk_running = pd.read_csv('final_walk_running.csv')
df_static_exercising = pd.read_csv('final_static_exercise.csv')
df_dynamic_exercising = pd.read_csv('final_dynamic_exercise.csv')
# df_multilevel_1_2 = pd.read_csv('final_multilevel_1-2.csv')
# df_multilevel_sleep = pd.read_csv('final_multilevel_sleep.csv')

combined_df = pd.concat([df_domino, df_pamap2, df_wisdm, df_walk_running, df_static_exercising, df_dynamic_exercising], axis=0, ignore_index=True)
print(combined_df.head())

unique_activities = combined_df['activity'].unique()

for activity in unique_activities:
    print(activity)
    print(count(combined_df[combined_df['activity'] == activity]))

combined_df.to_csv('combined_dataset8.csv', index=False)