This folder contains the code files that create, train and test the models. Each code file process a specific type of data.

Before running these code files, it's necessary to run the files in the **process_datasets folder**.
## accelerometer_models_8_classes.py
Processes the data using 8 classes and their corresponding categories, from files train_data_9.csv, test_data_9.csv

class_labels = ['cycling', 'exercising', 'lying', 'running', 'sitting', 'sleeping', 'standing', 'walking']
category_labels = ['exercising', 'idle', 'lying', 'running', 'walking']

This is the final code that creates the desired models used in physical_activity_recognition repo.

## accelerometer_hr_models.py
Processes data only from DREAMT dataset, the only dataset with heart rate data. Only 2 classes:

sleeping, lying

Unfortunately, there were no more datasets with heart rate data so we can't use them for now. Needs to be seen later on.


## fine_tune_models.py
This file will be used when we acquire the real accelerometer data from the smartwatches in order to 
fine-tune the existing models for the new data. This is an important step that must be implemented in order to achieve maximum
accuracy on the new data.

Before running this file, it's necessary to save the acquired data to a csv file. Then, update the test path and run the code.
