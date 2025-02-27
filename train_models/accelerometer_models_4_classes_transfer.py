import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import os
import contextlib
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from keras.src.layers import MaxPooling1D, Conv1D, Layer, Dense
from pickle import dump, load
from scipy.signal import butter, filtfilt
from keras import Model

# Redirect stderr to /dev/null to silence warnings
devnull = open(os.devnull, 'w')
contextlib.redirect_stderr(devnull)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def apply_lowpass_filter(data, cutoff_freq=11, fs=25, order=4):
    """
    Apply a low-pass Butterworth filter to the signal.

    Args:
        data: Input signal array
        cutoff_freq: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
    """
    nyquist = fs / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype='low')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def display_data(path, filename, testing):
    """
    This function plots subsets of the data as timeseries, to visualize the form of the data.
    """
    if not os.path.exists(f'files_{filename}/plots_{filename}'):
        os.makedirs(f'files_{filename}/plots_{filename}')

    data = pd.read_csv(path)
    data = data[['activity', 'accel_x', 'accel_y', 'accel_z']]
    data = data.dropna()
    unique_activities = data['activity'].unique()

    for activity in unique_activities:
        subset = data[data['activity'] == activity].iloc[200:600]
        subset = subset.drop(['activity'], axis=1)

        subset.plot(subplots=True, figsize=(10, 10))
        plt.xlabel('Time')
        plt.savefig(f'files_{filename}/plots_{filename}/unscaled_{activity}.png')

    # scaler = load(open(f'robust_scaler.pkl', 'rb'))
    # data[['accel_x', 'accel_y', 'accel_z']] = scaler.transform(data[['accel_x', 'accel_y', 'accel_z']])
    #
    # for activity in unique_activities:
    #     subset = data[data['activity'] == activity].iloc[200:600]
    #     subset = subset.drop(['activity'], axis=1)
    #
    #     subset.plot(subplots=True, figsize=(10, 10))
    #     plt.xlabel('Time')
    #     plt.savefig(f'files_{filename}/plots_{filename}/scaled_{activity}2.png')


def add_noise_and_scale(data, noise_level=0.02, scale_range=(0.9, 1.1)):
    """
    Adds random noise and scale to accelerometer data.
    Returns:
    - Jittered data
    """
    # std_dev = np.std(data, axis=(1, 2), keepdims=True)  # Compute per-sample std deviation
    noise = np.random.normal(loc=0.0, scale=noise_level, size=data.shape)
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    jittered_data = (data + noise) * scale_factor
    return jittered_data


def create_sequences(X_data, Y_data, timesteps, unique_activities):
    """
    This function takes the X, Y data as time instances and transforms them to small timeseries.
    For each activity, creates sequences using sliding windows with 50% overlap.
    :returns: data as timeseries
    """
    X_seq, Y_seq = [], []
    for activity in unique_activities:
        for i in range(0, len(X_data) - timesteps, timesteps // 2):
            if Y_data.iloc[i] != activity or Y_data.iloc[i + timesteps] != activity:
                continue

            window_data = X_data.iloc[i:(i + timesteps)].values
            X_seq.append(window_data)
            Y_seq.append(activity)

    X_seq, Y_seq = np.array(X_seq), np.array(Y_seq)
    return X_seq, Y_seq.reshape(-1, 1)


def process_data(path, timesteps, testing):
    """
    Process time series data in windows
    """
    data = pd.read_csv(path)
    data = data[['activity', 'accel_x', 'accel_y', 'accel_z']]
    data = data.dropna()
    data['activity'] = data['activity'].replace('static_exercising', 'dynamic_exercising')
    if not testing:
        data = data[data['activity'] != 'dynamic_exercising']
        data = data[data['activity'] != 'sleeping']
        data = data[data['activity'] != 'lying']
        data = data[data['activity'] != 'cycling']
        data = data[data['activity'] != 'running']

    print(data['activity'].value_counts())
    unique_activities = data['activity'].unique()
    data_train = []
    data_train = pd.DataFrame(data_train)
    data_train_sample = []
    data_test = []
    data_test = pd.DataFrame(data_test)
    data_test_sample = []
    for activity in unique_activities:
        sample = data[data['activity'] == activity]
        data_train_sample = sample[:int(len(sample) * 0.7)]
        data_train_sample = pd.DataFrame(data_train_sample)
        data_train = pd.concat([data_train, data_train_sample])

        data_test_sample = sample[int(len(sample) * 0.7):]
        data_test_sample = pd.DataFrame(data_test_sample)
        data_test = pd.concat([data_test, data_test_sample])


    # data_train = [item for row in data_train for item in row]
    print(data_train.shape)
    data_train = pd.DataFrame(data_train, columns=['activity', 'accel_x', 'accel_y', 'accel_z'])
    data_test = pd.DataFrame(data_test, columns=['activity', 'accel_x', 'accel_y', 'accel_z'])

    x_data_train, y_data_train = create_sequences(data_train[['accel_x', 'accel_y', 'accel_z']], data_train['activity'], timesteps, unique_activities)
    x_data_test, y_data_test = create_sequences(data_test[['accel_x', 'accel_y', 'accel_z']], data_test['activity'], timesteps, unique_activities)
    print(x_data_test.shape)

    # Apply low-pass filter to each window
    filtered_windows = np.zeros_like(x_data_train)
    for i in range(len(x_data_train)):
        for j in range(3):  # For each axis
            filtered_windows[i, :, j] = apply_lowpass_filter(x_data_train[i, :, j])

    # Scale the entire dataset
    reshaped_data = filtered_windows.reshape(-1, 3)
    scaler = load(open(f'robust_scaler.pkl', 'rb'))
    scaled_data = scaler.transform(reshaped_data)
    final_x_data_train = scaled_data.reshape(len(x_data_train), timesteps, 3)

    # Testing data
    # Apply low-pass filter to each window
    filtered_windows_test = np.zeros_like(x_data_test)
    for i in range(len(x_data_test)):
        for j in range(3):  # For each axis
            filtered_windows_test[i, :, j] = apply_lowpass_filter(x_data_test[i, :, j])

    # Scale the entire dataset
    reshaped_data = filtered_windows_test.reshape(-1, 3)
    scaler = load(open(f'robust_scaler.pkl', 'rb'))
    scaled_data = scaler.transform(reshaped_data)
    final_x_data_test = scaled_data.reshape(len(x_data_test), timesteps, 3)

    return final_x_data_train, y_data_train, final_x_data_test, y_data_test, unique_activities


def onehot_encode_data(y_train, y_test):
    hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    y_train = hot_encoder.fit_transform(y_train)
    y_test = hot_encoder.transform(y_test)

    return y_train, y_test


def create_sequential_model(X_train, y_train, chosen_model, input_shape, file_name):
    """
    This function is used to create the sequential models. Given the chosen_model param, it chooses the appropriate
    structure and then compiles the model.
    :return: the chosen sequential model
    """
    model = keras.Sequential()
    if chosen_model == 'cnn_lstm':
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=4))
        model.add(keras.layers.LSTM(units=32, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.4))
    elif chosen_model == 'cnn_gru':
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=4))
        model.add(keras.layers.GRU(units=32, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.4))
    elif chosen_model == 'cnn_cnn_lstm':
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu', input_shape=input_shape))
        model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(keras.layers.LSTM(units=64, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.4))
    elif chosen_model == 'cnn_cnn_gru':
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu', input_shape=input_shape))
        model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(keras.layers.GRU(units=64, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.4))
    elif chosen_model == 'cnn_cnn':
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu', input_shape=input_shape))
        model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(keras.layers.Dropout(rate=0.4))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(rate=0.4))

    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=[keras.metrics.CategoricalAccuracy()])

    model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2, verbose=2)
    model.save(f'{file_name}')

    return model


def train_sequential_model(X_test, y_test, chosen_model, class_labels, filename, train_model):
    """
    This function is used to train the sequential models. If train_model == True, then it trains the model using
    X-train, y_train, else it loads the model from the existing file. Then, it evaluates the model and prints the
    classification report.
    :return: y_test_labels, y_pred_labels containing the actual y_labels of test set and the predicted ones.
    """
    if not os.path.exists(f'files_{filename}/saved_models_{filename}'):
        os.makedirs(f'files_{filename}/saved_models_{filename}')

    file_name = f'files_{filename}/saved_models_{filename}/acc_{chosen_model}_model.keras'
    model = keras.models.load_model(file_name)

    probabilities = model.predict(X_test)
    print(probabilities)
    print(y_test)
    window_size = 3
    threshold = 0.7
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(probabilities, axis=1)
    print(y_test_labels)
    print(y_pred_labels)

    smoothed_predictions = []


    # Calculate accuracy and other metrics
    print("Accuracy with initial predictions: ", round(100 * accuracy_score(y_test_labels, y_pred_labels), 2))
    print("F1 score with initial predictions :",
          round(100 * f1_score(y_test_labels, y_pred_labels, average='weighted'), 2))
    print("\nClassification Report for initial predictions: :")
    print(classification_report(y_test_labels, y_pred_labels, target_names=class_labels))

    activity_predictions_true = np.empty(len(y_test_labels), dtype=object)
    for i in range(0, len(y_test_labels)):
        activity_predictions_true[i] = class_labels[y_test_labels[i]]

    activity_predictions = np.empty(len(y_pred_labels), dtype=object)
    for i in range(0, len(y_pred_labels)):
        activity_predictions[i] = class_labels[y_pred_labels[i]]

    smoothed_predictions = np.array(smoothed_predictions)
    return y_test_labels, y_pred_labels, smoothed_predictions


def plot_confusion_matrix(y_test_labels, y_pred_labels, smoothed_predictions, class_labels, chosen_model, filename):
    """
    This function plots the confusion matrices, visualising the results of the sequential models. Using the y_test_labels
    and y_pred_labels parameters, it creates and saves the confusion matrix.
    """
    path = f'files_{filename}/plots_{filename}'
    if not os.path.exists(path):
        os.makedirs(path)

    # Initial Values
    normalize_cm = [None]
    for norm_value in normalize_cm:
        if norm_value == 'true':
            format = '.2f'
            plot_name = f'acc_{chosen_model}_cm_norm_inital_transfer.png'
        else:
            format = 'd'
            plot_name = f'acc_{chosen_model}_cm_initial_transfer.png'

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test_labels, y_pred_labels,
            display_labels=class_labels,
            normalize=norm_value,
            xticks_rotation=70,
            values_format=format,
            cmap=plt.cm.Blues
        )

        plt.figure(figsize=(8, 10))
        plt.title(f'Confusion Matrix for {chosen_model}')
        disp.plot(cmap=plt.cm.Blues, values_format=format)
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.savefig(f'{path}/{plot_name}', bbox_inches='tight', pad_inches=0.1)


def plot_confusion_matrix_grouped(y_test_labels, y_pred_labels, smoothed_predictions, class_labels, chosen_model, filename):
    """
    This function plots the confusion matrices, visualising the results of the sequential models. Using the y_test_labels
    and y_pred_labels parameters, it creates and saves the confusion matrix.
    """
    path = f'files_{filename}/plots_{filename}'
    if not os.path.exists(path):
        os.makedirs(path)

    # Initial Values
    normalize_cm = [None]
    for norm_value in normalize_cm:
        if norm_value == 'true':
            format = '.2f'
            plot_name = f'acc_{chosen_model}_cm_norm_inital_grouped.png'
        else:
            format = 'd'
            plot_name = f'acc_{chosen_model}_cm_initial_grouped.png'

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test_labels, y_pred_labels,
            display_labels=class_labels,
            normalize=norm_value,
            xticks_rotation=70,
            values_format=format,
            cmap=plt.cm.Blues
        )

        plt.figure(figsize=(8, 10))
        plt.title(f'Confusion Matrix for {chosen_model}')
        disp.plot(cmap=plt.cm.Blues, values_format=format)
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.savefig(f'{path}/{plot_name}', bbox_inches='tight', pad_inches=0.1)

    # Smoothed Values
    for norm_value in normalize_cm:
        if norm_value == 'true':
            format = '.2f'
            plot_name = f'acc_{chosen_model}_cm_norm_smooth_grouped.png'
        else:
            format = 'd'
            plot_name = f'acc_{chosen_model}_cm_smooth_grouped.png'

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test_labels, smoothed_predictions,
            display_labels=class_labels,
            normalize=norm_value,
            xticks_rotation=70,
            values_format=format,
            cmap=plt.cm.Blues
        )

        plt.figure(figsize=(8, 10))
        plt.title(f'Confusion Matrix for {chosen_model}')
        disp.plot(cmap=plt.cm.Blues, values_format=format)
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.savefig(f'{path}/{plot_name}', bbox_inches='tight', pad_inches=0.1)


def group_categories(y_labels, class_labels):
    """
    This function takes the activity labels and groups them to more generic categories, exercising, idle, walking, sleeping
    """
    predicted_categories = []
    exercising_activities = ['running', 'cycling', 'static_exercising', 'dynamic_exercising']
    idle_activities = ['sitting', 'lying']
    standing_activities = ['walking', 'standing']
    y_labels = [class_labels[label] for label in y_labels]

    for activity in y_labels:
        if activity in exercising_activities:
            predicted_categories.append('exercising')
        elif activity in idle_activities:
            predicted_categories.append('idle')
        elif activity in standing_activities:
            predicted_categories.append('walking')
        else:
            predicted_categories.append(activity)

    predicted_categories = np.array(predicted_categories)
    # print(predicted_categories)

    return predicted_categories


def retrain_model(X_train, y_train, X_test, y_test, chosen_model):

    pretrained_model = keras.models.load_model(f"files_{filename}/saved_models_{filename}/acc_{chosen_model}_model.keras")

    model_without_last_layer = Model(inputs=pretrained_model.inputs, outputs=pretrained_model.layers[-2].output)

    # Fine-tune pretrained model
    new_layer = Dense(y_train.shape[1], activation='softmax')(model_without_last_layer.output)
    new_model = Model(inputs=model_without_last_layer.input, outputs=new_layer)

    # Compile the new model
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])
    new_model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=64, verbose=2)
    new_model.save(f'files_{filename}/saved_models_{filename}/acc_retrained_{chosen_model}_model.keras')
    # new_model.summary()

    loss, accuracy = new_model.evaluate(X_train, y_train, verbose=0)
    print("Train Accuracy: %d%%, Train Loss: %d%%" % (100*accuracy, 100*loss))

    y_pred = new_model.predict(X_test, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
    print("Test Accuracy: %d%%" % (100*accuracy))
    print("Test F1 Score: %d%%" % (100*f1))

    report = classification_report(y_test_labels, y_pred_labels, target_names=class_labels)
    print(report)

    return y_test_labels, y_pred_labels


if __name__ == '__main__':
    frequency = 25
    time_required_ms = 10000
    samples_required = int(time_required_ms * frequency / 1000)
    # class_labels = ['cycling', 'exercising', 'lying', 'running', 'sitting', 'sleeping', 'standing', 'walking']
    class_labels = ['lying', 'running', 'sitting', 'standing', 'walking']
    category_labels = ['exercising', 'idle', 'walking']
    my_test_path = "../process_datasets/final_my_data_collector.csv"
    filename = f"{time_required_ms}ms_5_classes_transfer"

    # print(f'\nTraining 8 classes from file: {train_path}')
    print('Timesteps per timeseries: ', time_required_ms)
    print(f"folder path: files_{filename}")

    # Implemented models
    models = ['cnn_cnn_lstm']
    # models = ['cnn_lstm','cnn_gru', 'cnn_cnn_lstm', 'cnn_cnn_gru']
    # X_train, y_train, unique_activities = process_data(train_path, samples_required, False)
    # X_test, y_test, _ = process_data(test_path, samples_required, True)
    X_train, y_train, X_test, y_test, _ = process_data(my_test_path, samples_required, True)

    # display_data(train_path, filename, False)
    # display_data(my_test_path, filename, True)

    y_train, y_test = onehot_encode_data(y_train, y_test)

    for chosen_model in models:
        print(f'\n{chosen_model=}')
        y_test_labels, y_pred_labels = retrain_model(X_train, y_train, X_test, y_test, chosen_model)
        smoothed_predictions = []
        plot_confusion_matrix(y_test_labels, y_pred_labels, smoothed_predictions, class_labels, chosen_model, filename)

        # Make predictions with generic categories
        predicted_y_categories = group_categories(y_test_labels, class_labels)
        predicted_categories = group_categories(y_pred_labels, class_labels)
        predicted_categories_smooth = group_categories(smoothed_predictions, class_labels)

        print("\nAccuracy with initial predictions: ",
              round(100 * accuracy_score(predicted_y_categories, predicted_categories), 2))
        print("F1 score with initial predictions :",
              round(100 * f1_score(predicted_y_categories, predicted_categories, average='weighted'), 2))
        print("\nClassification Report for initial predictions: :")
        print(classification_report(predicted_y_categories, predicted_categories, target_names=category_labels))

        plot_confusion_matrix_grouped(predicted_y_categories, predicted_categories, predicted_categories_smooth, category_labels, chosen_model, filename)
