import logging
from datetime import timedelta
import mongoengine as mongo
from influxdb_client import InfluxDBClient, QueryApi
import json
import os
import pandas as pd
import numpy as np
import keras
from mongoengine import Document, StringField, DateTimeField, IntField, FloatField
from sklearn.preprocessing import RobustScaler


def connect_to_mongodb():
    username = 'root'
    password = 'rootpassword'
    host = '172.17.0.1'
    port = '27017'

    mongo.connect(
        host=f"mongodb://{username}:{password}@{host}:{port}/",
        authSource='relevium',
        authMechanism='SCRAM-SHA-256'
    )
    print(f"Connecting to MongoDB at {os.getenv('MONGODB_PORT')}")


def connect_to_influxdb():
    token = '2o0W-fSz6Cv2EH1rKkAOotZ8oaBAr_r2V09-kTA6zBdhMGC_9fx6yl49HF7T1ndRGjFhRs__YhkL0Dy8uNm2WQ=='
    org_name = 'my_org'
    host = 'http://172.17.0.1'
    port = '8086'

    influx_client = InfluxDBClient(
        url=f"{host}:{port}",
        token=token,
        org=org_name,
    )

    print(f"Connected to InfluxDB in Org: {org_name}")
    query_api = influx_client.query_api()
    return influx_client, query_api


def fetch_influxdb_data(query_api, bucket_name, org_name):

    ##### Retrieve active patients list #####
    # patient_ids_query = f'''
    #     from(bucket: "{bucket_name}")
    #       |> range(start: 0)
    #       |> filter(fn: (r) => r["_measurement"] == "accelerometer")
    #       |> keep(columns: ["patient_id"])
    #       |> group(columns: ["patient_id"])
    #       |> distinct(column: "patient_id")
    #     '''
    # result = query_api.query(org=org_name, query=patient_ids_query)
    # patient_ids = []
    # for table in result:
    #     for record in table.records:
    #         patient_ids.append(record["patient_id"])

    query = f"""
            from(bucket: "{bucket_name}")
              |> range(start: 0, stop: -3d)
              |> filter(fn: (r) => r._measurement == "accelerometer")
              |> filter(fn: (r) => r.patient_id == "4")
              |> filter(fn: (r) => r["_field"] == "x" or r["_field"] == "y" or r["_field"] == "z")
              |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn: "_value")
              |> keep(columns: ["patient_id", "device", "date", "timewindow", "tz", "x", "y", "z", "_time"])
            """

    tables = query_api.query(query=query, org=org_name)
    print(f"Data querying successful")

    data = []
    for table in tables:
        for record in table.records:
            data.append({
                'patient_id': record["patient_id"],
                'device': record["device"],
                'date': record["date"],
                'timewindow': record["timewindow"],
                'tz': record["tz"],
                'accel_x': record["x"],
                'accel_y': record["y"],
                'accel_z': record["z"],
                'datetime': record["_time"]
            })

    print(f"Fetched {len(data)} size of data from InfluxDB")

    loaded_df = pd.DataFrame(data)
    print(loaded_df[["patient_id", "accel_x", "datetime"]].head)
    return data, loaded_df


def preprocess_data(data_df, frequency=25, time_required_ms=3500):
    timesteps = int(time_required_ms * frequency / 1000)
    data_df = data_df.dropna()
    columns_to_scale = ['accel_x', 'accel_y', 'accel_z']
    scaler = RobustScaler()
    data_df[columns_to_scale] = scaler.fit_transform(data_df[columns_to_scale])

    X_seq = []
    time_seq = []
    for i in range(0, len(data_df) - timesteps, timesteps):
        X_seq.append(data_df.loc[i:(i+timesteps), ['accel_x', 'accel_y', 'accel_z']])
        time_seq.append(data_df.loc[i, 'datetime'])

    X_seq = np.array(X_seq)
    time_seq = np.array(time_seq)

    return X_seq, time_seq


def predict_activity(X_data, chosen_model='gru_2'):
    class_labels = ['Brushing teeth', 'Cycling', 'Lying', 'Moving by car', 'Running', 'Sitting', 'Stairs', 'Standing', 'Walking']
    file_name = f'/code/consumer/api/acc_{chosen_model}_model.h5'
    model = keras.models.load_model(file_name)

    y_pred = model.predict(X_data)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_pred_classes = np.empty(len(y_pred_labels), dtype=object)

    for i in range(0, len(y_pred_labels)):
        y_pred_classes[i] = class_labels[y_pred_labels[i]]

    print(f"Model prediction successful")
    return y_pred_labels, y_pred_classes


def transform_data_for_mongo(time_seq_data, predicted_classes, patient_id):
    documents_data = []
    for i in range(0, len(predicted_classes)):
        documents_data.append({
            "patient_id": patient_id,
            "start_date": time_seq_data[i],
            "end_date": time_seq_data[i],
            "activity_type": predicted_classes[i],
            "duration": 3500,
            "distance": 0,
            "rpe": 0,
            "rpp": 0,
            "patient_msg": 'n/a',
            "source": 'ai'
        })

    print(f"Transformed {len(documents_data)} data")
    return documents_data


def insert_into_mongodb(data):
    class Activity(Document):
        patient_id = StringField(required=True)
        start_date = DateTimeField(required=True)
        end_date = DateTimeField(required=True)
        activity_type = StringField(default='none')
        duration = IntField(default=1200)
        distance = FloatField(default=0)
        rpe = IntField(default=0)
        rpp = IntField(default=0)
        patient_msg = StringField(default='n/a')
        source = StringField(default='ai')

        meta = {'collection': 'TestPatientActivities'}

    for record in data:
        activity = Activity(**record)
        activity.save()

    print(f"Inserted {len(data)} records into MongoDB")



def main():
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    # Connect to databases
    connect_to_mongodb()
    influx_client, query_api = connect_to_influxdb()

    bucket_name = 'my_bucket'
    org_name = 'my_org'

    # Fetch data from InfluxDB
    influx_data, influx_df = fetch_influxdb_data(query_api, bucket_name=bucket_name, org_name=org_name)

    # Process data from InfluxDB and make predictions
    X_seq_data, time_seq_data = preprocess_data(influx_df)
    y_labels, y_classes = predict_activity(X_seq_data)

    # Transform data to MongoDB format
    mongo_data = transform_data_for_mongo(time_seq_data, y_classes, influx_df.loc[0, "patient_id"])

    # Insert data into MongoDB
    insert_into_mongodb(mongo_data)

