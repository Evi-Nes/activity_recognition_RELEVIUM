import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv("data_domino.csv")
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
size = len(df)
df =  df.iloc[int(size*0.6):]

# Calculate yesterday's date
yesterday_date = datetime.now().date() - timedelta(days=1)

# Update the date in each timestamp while preserving the original time
df['new_datetime'] = df['datetime'].apply(lambda dt: datetime.combine(yesterday_date, dt.time()))

# Convert the updated datetime back to milliseconds
df['new_timestamp'] = df['new_datetime'].astype(int) // 10**6
df = df.drop(columns=['datetime', 'new_datetime', 'timestamp'])


column_to_move = 'new_timestamp'
cols = [column_to_move] + [col for col in df.columns if col != column_to_move]
df = df[cols]
df.rename(columns={"new_timestamp": "timestamp"}, inplace=True)

df.to_csv("new_domino.csv", index=False)

print("Timestamps updated to yesterday's date and saved to 'new_domino.csv'")