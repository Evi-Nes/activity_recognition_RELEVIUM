import time
from influxdb_client import InfluxDBClient, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

bucket = "my_bucket"
org = "my_org"
client = InfluxDBClient(url="http://localhost:8086", token="2o0W-fSz6Cv2EH1rKkAOotZ8oaBAr_r2V09-kTA6zBdhMGC_9fx6yl49HF7T1ndRGjFhRs__YhkL0Dy8uNm2WQ==", org=org)

write_api = client.write_api(write_options=SYNCHRONOUS)
is_header = 1

try:
    with open('data_new_domino.csv', 'r') as file:

        for line in file:
            if is_header == 1:
                is_header = 0
                continue

            if line:
                timestamp_part, measurement_part = line.split(',', 1)
                accel_x, accel_y, accel_z, activity, gyro_x, gyro_y, gyro_z, user_id = measurement_part.split(',')
                user_id = str(user_id).strip()

                # Create a data point in the required line protocol format
                line_protocol = (
                    f"accelerometer,patient_id={user_id},activity={activity} "
                    f"x={accel_x},y={accel_y},z={accel_z} "
                    f"{timestamp_part}"
                )

                write_api.write(bucket=bucket, org=org, record=line_protocol, write_precision=WritePrecision.MS)
                print(f"Uploaded data: {line_protocol}")

except KeyboardInterrupt:
    print("Live data upload stopped.")

finally:
    client.close()




# df = pd.read_csv("data_domino.csv")
# size = len(df)
# new_df =  df.iloc[int(size*0.6):]
#
# new_df.to_csv("data_new_domino.csv", index=False)