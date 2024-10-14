# with open('a_s_RELEVIUM-MAINZ-01-01_20200202_001', 'r') as file:
#     content = file.read()
# print(content)

# with open('a_s_RELEVIUM-MAINZ-01-01_20200202_001', 'r') as file:
#     # Loop through each line in the file
#     for line in file:
#
#         print(line.strip())


import time
from datetime import datetime
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

bucket = "static_data"
org = "local_test"
client = InfluxDBClient(url="http://localhost:8085", token="ax1hjMD3MVseMkM4Zg1t12sPvakLlyj_bLmHjEMDshXCPEjfN1fIW_owMNQs4VSk-JDiDswUD7HSF2jUIAcEGw==", org=org)

write_api = client.write_api(write_options=SYNCHRONOUS)

try:
    while True:
        patient_id = "RELEVIUM-MAINZ-01-01"
        device = "Smartwatch"
        date = "20241014"
        # timewindow = 1
        tz = "03:00"
        sensor_value_x = 1.6855
        sensor_value_y = 4.6731
        sensor_value_z = 5.0779
        dt = datetime.strptime("2024-10-14 17:30:00", "%Y-%m-%d %H:%M:%S")
        timestamp = int(dt.timestamp() * 1000)   # to milliseconds

        # Create a data point in the required line protocol format
        line_protocol = (
            f"accelerometer,patient_id={patient_id},device={device},"
            f"date={date},tz={tz} "
            f"x={sensor_value_x},y={sensor_value_y},z={sensor_value_z} "
            f"{timestamp}"
        )
        # timewindow={timewindow},
        # Write the data point to InfluxDB
        write_api.write(bucket=bucket, org=org, record=line_protocol, precision='ms')

        print(f"Uploaded data: {line_protocol}")

        time.sleep(1)

except KeyboardInterrupt:
    print("Live data upload stopped.")

finally:
    # Close the InfluxDB client connection
    client.close()





