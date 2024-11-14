import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import os
import contextlib
import pickle
import pymongo

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from keras.src.layers import MaxPooling1D, Conv1D

from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing

# Redirect stderr to /dev/null to silence warnings
devnull = open(os.devnull, 'w')
contextlib.redirect_stderr(devnull)


def preprocess_data(data, timesteps):
    data = data.drop(['timestamp'], axis=1)
    data = data.dropna()

    columns_to_scale = ['accel_x', 'accel_y', 'accel_z']
    scaler = RobustScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    X_seq = []
    for i in range(0, len(data) - timesteps, timesteps//2):
        X_seq.append(data.iloc[i:(i+timesteps)].values)

    X_seq = np.array(X_seq)

    return X_seq


def train_sequential_model(X_data, chosen_model, class_labels):
    if not os.path.exists('../saved_models'):
        os.makedirs('../saved_models')

    file_name = f'saved_models/acc_domino_{chosen_model}_model.h5'

    model = keras.models.load_model(file_name)

    # print(model.summary())

    y_pred = model.predict(X_data)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_pred_classes = np.empty(len(y_pred_labels), dtype=object)

    for i in range(0, len(y_pred_labels)):
        y_pred_classes[i] = class_labels[y_pred_labels[i]]

    return y_pred_labels, y_pred_classes


if __name__ == '__main__':
    from influxdb_client import InfluxDBClient

    bucket = 'actual_data'
    org = 'local_test'
    client = InfluxDBClient(url="http://localhost:8085", token="ax1hjMD3MVseMkM4Zg1t12sPvakLlyj_bLmHjEMDshXCPEjfN1fIW_owMNQs4VSk-JDiDswUD7HSF2jUIAcEGw==", org=org)
    query_api = client.query_api()

    def fetch_data():
        while True:
            query = f'''
            from(bucket: "{bucket}")
              |> range(start: -5m)\
              |> filter(fn: (r) => r["_field"] == "x" or r["_field"] == "y" or r["_field"] == "z")\
              |> filter(fn: (r) => r["patient_id"] == "RELEVIUM-MAINZ-01-01")\
              |> filter(fn: (r) => r["device"] == "Smartwatch")\
                |> filter(fn: (r) => r["_measurement"] == "accelerometer")\
              |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn: "_value")
            '''

            result = query_api.query(org=org, query=query)
            data = []

            for table in result:
                for record in table.records:
                    data.append({
                        'timestamp': record.get_time(),
                        'accel_x': record["x"],
                        'accel_y': record["y"],
                        'accel_z': record["z"]
                    })
                    patient_id = record["patient_id"]
                    device = record["device"]
                    sensor = record["_measurement"]

            loaded_df = pd.DataFrame(data)
            if len(loaded_df) < samples_required:
                print('Not enough samples yet')
                continue

            chosen_model = 'gru_2'
            X_seq_data = preprocess_data(loaded_df, samples_required)
            y_labels, y_classes = train_sequential_model(X_seq_data, chosen_model, class_labels)

            print("The predicted activities are:", y_classes)

            time.sleep(2)

            # Connect to MongoDB
            client = pymongo.MongoClient("mongodb://localhost:27017/")

            db = client["ReleviumData"]
            collection = db["accelerometer"]

            documents = [{"recorder activity": item, "patient id": patient_id, "device": device, "sensor": sensor} for item in y_classes]
            collection.insert_many(documents)

            print("Data uploaded successfully!")

            time.sleep(10)


    frequency = 25
    time_required_ms = 3500
    samples_required = int(time_required_ms * frequency / 1000)

    class_labels = ['Cycling', 'Lying', 'Running', 'Sitting', 'Standing', 'Walking']

    try:
        fetch_data()
    except KeyboardInterrupt:
        print("Live data fetching stopped.")
    finally:
        client.close()


