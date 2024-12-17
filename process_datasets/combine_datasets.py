import pandas as pd

df_domino = pd.read_csv('final_domino.csv')
df_pamap2 = pd.read_csv('final_pamap2.csv')
df_wisdm = pd.read_csv('final_wisdm.csv')
df_walk_running = pd.read_csv('final_walking_running.csv')
df_static_exercising = pd.read_csv('final_static_exercising.csv')
df_dynamic_exercising = pd.read_csv('final_dynamic_exercising.csv')
df_dreamt = pd.read_csv('final_dreamt.csv')

combined_df = pd.concat([df_domino, df_pamap2, df_wisdm, df_walk_running, df_static_exercising, df_dynamic_exercising, df_dreamt], axis=0, ignore_index=True)
print(combined_df.head())

print(combined_df['activity'].value_counts())

combined_df.to_csv('combined_dataset.csv', index=False)