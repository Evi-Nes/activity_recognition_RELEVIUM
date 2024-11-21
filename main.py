from influxdb_client import InfluxDBClient, WritePrecision
import pandas as pd
import time
import numpy as np
import keras
import os
import contextlib
import pymongo

from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Redirect stderr to /dev/null to silence warnings
devnull = open(os.devnull, 'w')
contextlib.redirect_stderr(devnull)


def preprocess_data(data, timesteps):
    # data = data.drop(['timestamp'], axis=1)
    data = data.drop(['user_id'], axis=1)
    data = data.dropna()

    columns_to_scale = ['accel_x', 'accel_y', 'accel_z']
    scaler = RobustScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    X_seq = []
    y_seq = []
    time_seq = []
    for i in range(0, len(data) - timesteps, timesteps//2):
        X_seq.append(data.loc[i:(i+timesteps), ['accel_x', 'accel_y', 'accel_z']])
        y_seq.append(data.loc[i+timesteps, 'activity'])
        time_seq.append(data.loc[i+timesteps, 'timestamp'])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    time_seq = np.array(time_seq)

    return X_seq, y_seq, time_seq


def predict_activity(X_data, y_data, chosen_model, class_labels):

    file_name = f'saved_models/acc_{chosen_model}_model.h5'
    model = keras.models.load_model(file_name)

    # print(model.summary())

    y_pred = model.predict(X_data)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_pred_classes = np.empty(len(y_pred_labels), dtype=object)

    for i in range(0, len(y_pred_labels)-1):
        y_pred_classes[i] = class_labels[y_pred_labels[i]]

    # for i in range(len(y_pred_classes)):
    #     print(y_pred_classes[i])
    #     print(y_data[i])

    return y_pred_labels, y_pred_classes



if __name__ == '__main__':
    print('starting')

    bucket = 'my_bucket'
    org = 'my_org'
    client = InfluxDBClient(url="http://localhost:8086", token="2o0W-fSz6Cv2EH1rKkAOotZ8oaBAr_r2V09-kTA6zBdhMGC_9fx6yl49HF7T1ndRGjFhRs__YhkL0Dy8uNm2WQ==", org=org)
    query_api = client.query_api()

    frequency = 25
    time_required_ms = 3500
    samples_required = int(time_required_ms * frequency / 1000)

    class_labels = ['Brushing teeth', 'Cycling', 'Lying', 'Moving by car', 'Running', 'Sitting', 'Stairs', 'Standing', 'Walking']
    chosen_model = 'gru_2'

    users_range = [16, 19]
    for user in users_range:
        try:
            query = f'''
                from(bucket: "{bucket}")
                  |> range(start: 0, stop: -3d)\
                  |> filter(fn: (r) => r["_measurement"] == "accelerometer")\
                  |> filter(fn: (r) => r["_field"] == "x" or r["_field"] == "y" or r["_field"] == "z")\
                  |> filter(fn: (r) => r["patient_id"] == "{user}")\
                  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn: "_value")
                  |> keep(columns: ["_time", "x", "y", "z", "activity", "patient_id"])  
                '''

            result = query_api.query(org=org, query=query)
            data = []

            for table in result:
                for record in table.records:

                    data.append({
                        'timestamp': record.get_time(),
                        'accel_x': record["x"],
                        'accel_y': record["y"],
                        'accel_z': record["z"],
                        'activity': record["activity"],
                        'user_id': record["patient_id"]
                    })
                    patient_id = record["patient_id"]

            loaded_df = pd.DataFrame(data)
            print(f"User {user}: data have been downloaded")

            # loaded_df.to_csv(f'user_{patient_id}_downloaded_data.csv', index=False)

            X_seq_data, y_seq_data, time_seq_data = preprocess_data(loaded_df, samples_required)
            y_labels, y_classes = predict_activity(X_seq_data, y_seq_data, chosen_model, class_labels)

            # print("The predicted activities are:", y_classes)
            print(f"User {user}: data have been processed")

            time.sleep(2)

            # Connect to MongoDB
            mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")

            db = mongo_client["ReleviumData"]
            collection = db[f"accelerometer_for_user_{user}"]

            documents = [{"recorder activity": item, "recorded time": time_seq_data[i], "patient id": user, "sensor": 'accelerometer'} for i, item in enumerate(y_classes)]
            collection.insert_many(documents)

            print(f"User {user}: data have been uploaded")

            time.sleep(2)

        except Exception as e:
            print(e)

