import pandas as pd
from datetime import datetime
from influxdb_client import InfluxDBClient, WriteOptions


def timestamp_to_date_and_timewindow(millis, window_size=10):
    dt = datetime.utcfromtimestamp(millis / 1000)
    date = dt.strftime("%Y%m%d")
    minutes_since_midnight = dt.hour * 60 + dt.minute
    timewindow = (minutes_since_midnight // window_size) + 1

    return date, timewindow

if __name__ == '__main__':
    create_data = True

    INFLUX_HOST = 'http://localhost'
    INFLUX_PORT = '8086'
    INFLUX_TOKEN = '2o0W-fSz6Cv2EH1rKkAOotZ8oaBAr_r2V09-kTA6zBdhMGC_9fx6yl49HF7T1ndRGjFhRs__YhkL0Dy8uNm2WQ=='
    INFLUX_ORG = 'my_org'
    INFLUX_BUCKET = 'my_bucket'

    if create_data == True:
        date = 20180105
        timewindow = 135

        df = pd.read_csv("data_domino.csv")
        size = len(df)
        df =  df.iloc[int(size*0.5):]

        df.drop('gyro_x', axis=1, inplace=True)
        df.drop('gyro_y', axis=1, inplace=True)
        df.drop('gyro_z', axis=1, inplace=True)
        df.drop('activity', axis=1, inplace=True)
        df.rename(columns={"user_id": "patient_id", "accel_x": "x", "accel_y": "y", "accel_z": "z"}, inplace=True)
        df = df[['patient_id', 'x', 'y', 'z', 'timestamp']]
        df.insert(1, "device", 'Smartwatch', allow_duplicates=True)
        df.insert(2, "date", date)
        df.insert(3, "timewindow", timewindow)
        df.insert(4, "tz", '02:00', allow_duplicates=True)

        print(df.head())

        df = df.reset_index()
        for index, row in df.iterrows():
            date, timewindow = timestamp_to_date_and_timewindow(row['timestamp'], window_size=10)
            df.at[index, "date"] = date
            df.at[index, "timewindow"] = timewindow

        print(df.head())

        # df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        tag_columns = ["patient_id", "device", "date", "timewindow", "tz"]
        for col in tag_columns:
            df[col] = df[col].astype(str)

        df.to_csv("formated_domino.csv", index=False)
        print("File updated and saved to 'formated_domino.csv'")


    with InfluxDBClient(url=f"{INFLUX_HOST}:{INFLUX_PORT}", token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        for df in pd.read_csv("formated_domino.csv", chunksize=10_000):
            with client.write_api() as write_api:
                try:
                    write_api.write(
                        record=df,
                        bucket=INFLUX_BUCKET,
                        data_frame_measurement_name="accelerometer",
                        data_frame_tag_columns=["patient_id", "device", "date", "timewindow", "tz"],
                        data_frame_timestamp_column="timestamp",
                        write_precision='ms'
                    )
                    print("Uploaded data to InfluxDB")
                except InfluxDBError as e:
                    print(e)
