import time
from datetime import datetime
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

bucket = "actual_data"
org = "local_test"
client = InfluxDBClient(url="http://localhost:8085", token="ax1hjMD3MVseMkM4Zg1t12sPvakLlyj_bLmHjEMDshXCPEjfN1fIW_owMNQs4VSk-JDiDswUD7HSF2jUIAcEGw==", org=org)

write_api = client.write_api(write_options=SYNCHRONOUS)

try:
    with open('a_s_RELEVIUM-MAINZ-01-01_20200202_001', 'r') as file:

        for line in file:
            if line:
                # Split the line into the measurement, the fields and the timestamp
                measurement_part, fields_part, timestamp_part = line.split(' ', 2)

                accelerometer, tags_part = measurement_part.split(',', 1)
                tags = tags_part.split(',')
                tags_dict = {tag.split('=')[0]: tag.split('=')[1] for tag in tags}

                # Split fields part by commas to get field values
                fields = fields_part.split(',')
                fields_dict = {field.split('=')[0]: float(field.split('=')[1]) for field in fields}

            patient_id = tags_dict['patient_id']
            device = tags_dict['device']
            date = "20241014"
            tz = tags_dict['tz']
            sensor_value_x = fields_dict['x']
            sensor_value_y = fields_dict['y']
            sensor_value_z = fields_dict['z']
            timestamp = int(time.time() * 1_000_000_000)

            # Create a data point in the required line protocol format
            line_protocol = (
                f"accelerometer,patient_id={patient_id},device={device},"
                f"date={date},tz={tz} "
                f"x={sensor_value_x},y={sensor_value_y},z={sensor_value_z} "
                f"{timestamp}"
            )

            write_api.write(bucket=bucket, org=org, record=line_protocol, precision='ms')

            print(f"Uploaded data: {line_protocol}")

            time.sleep(0.5)

except KeyboardInterrupt:
    print("Live data upload stopped.")

finally:
    client.close()





