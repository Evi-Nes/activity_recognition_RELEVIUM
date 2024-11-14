from influxdb_client import InfluxDBClient
import pandas as pd

bucket = 'my_bucket'
org = 'my_org'
client = InfluxDBClient(url="http://localhost:8086", token="2o0W-fSz6Cv2EH1rKkAOotZ8oaBAr_r2V09-kTA6zBdhMGC_9fx6yl49HF7T1ndRGjFhRs__YhkL0Dy8uNm2WQ==", org=org)
query_api = client.query_api()

users_range = [4, 16, 19]
for user in users_range:
    try:

        query = f'''
            from(bucket: "{bucket}")
              |> range(start: 0)\
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
        loaded_df.to_csv(f'user_{patient_id}_downloaded_data.csv', index=False)

        print(len(loaded_df))

    except Exception as e:
        print(e)

