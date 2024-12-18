import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../process_datasets/final_dreamt_1Hz.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
print(df)
window_size = '30s'

def calculate_features(window):
    features = {}

    # Accelerometer Magnitude (vector magnitude of accelerometer data)
    window['accel_magnitude'] = np.sqrt(window['accel_x'] ** 2 + window['accel_y'] ** 2 + window['accel_z'] ** 2)

    features['accel_mean'] = window['accel_magnitude'].mean()
    features['accel_std'] = window['accel_magnitude'].std()
    features['accel_min'] = window['accel_magnitude'].min()
    features['accel_max'] = window['accel_magnitude'].max()
    features['accel_range'] = features['accel_max'] - features['accel_min']

    # Heart Rate Features
    features['hr_mean'] = window['hr'].mean()
    features['hr_std'] = window['hr'].std()


    # Sleep Stage Mode (N1, N2, N3, R, W)
    features['activity'] = window['activity'].mode().iloc[0]  # Most common activity in the window

    return pd.Series(features)


# Apply feature extraction over rolling windows
# Process each user_id group
processed_dfs = []
for user_id, group in df.groupby('user_id'):
    print(f"Processing user_id: {user_id}")
    user_features = group.resample(window_size).apply(calculate_features)
    # user_features['user_id'] = user_id
    processed_dfs.append(user_features)

# Combine all user data into a single DataFrame
actigraphy_features = pd.concat(processed_dfs).reset_index()

# actigraphy_features = df.resample(window_size).apply(calculate_features).reset_index()
actigraphy_features.dropna(inplace=True)

print(actigraphy_features)
actigraphy_features.to_csv('actigraphy_features.csv')


actigraphy_features['activity'] = actigraphy_features['activity'].replace({
    'W': 0, 'N1': 1, 'N2': 1, 'N3': 1, 'R': 1})

X = actigraphy_features.drop(['activity', 'timestamp'], axis=1)  # Drop non-feature columns
y = actigraphy_features['activity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf))

conf_matrix = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Lying', 'Sleeping'], yticklabels=['Lying', 'Sleeping'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()