import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import os
import contextlib
import tensorflow as tf
import sys

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
np.set_printoptions(threshold=sys.maxsize)


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
    data['activity'] = data['activity'].replace('static_exercising', 'dynamic_exercising')
    data['activity'] = data['activity'].replace('dynamic_exercising', 'exercising')
    unique_activities = data['activity'].unique()

    for activity in unique_activities:
        subset = data[data['activity'] == activity].iloc[200:600]
        subset = subset.drop(['activity'], axis=1)

        subset.plot(subplots=True, figsize=(10, 10))
        plt.xlabel('Time')
        plt.savefig(f'files_{filename}/plots_{filename}/unscaled_{activity}2.png')

    scaler = load(open(f'robust_scaler.pkl', 'rb'))
    data[['accel_x', 'accel_y', 'accel_z']] = scaler.transform(data[['accel_x', 'accel_y', 'accel_z']])

    for activity in unique_activities:
        subset = data[data['activity'] == activity].iloc[200:600]
        subset = subset.drop(['activity'], axis=1)

        subset.plot(subplots=True, figsize=(10, 10))
        plt.xlabel('Time')
        plt.savefig(f'files_{filename}/plots_{filename}/scaled_{activity}2.png')


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


def process_data(path, timesteps):
    """
    Process time series data in windows
    """
    data = pd.read_csv(path)
    data = data[['activity', 'accel_x', 'accel_y', 'accel_z']]
    data = data.dropna()
    data['activity'] = data['activity'].replace('static_exercising', 'dynamic_exercising')
    unique_activities = data['activity'].unique()
    train_data = []
    test_data = []

    for activity in unique_activities:
        activity_data = data[data['activity'] == activity]
        train_activity = activity_data[:int(0.8 * len(activity_data))]
        test_activity = activity_data[int(0.8 * len(activity_data)):]
        train_data.append(train_activity)
        test_data.append(test_activity)

    train_data_df = pd.concat(train_data, ignore_index=True, sort=False)
    test_data_df = pd.concat(test_data, ignore_index=True, sort=False)

    x_data_train, y_data_train = create_sequences(train_data_df[['accel_x', 'accel_y', 'accel_z']], train_data_df['activity'], timesteps, unique_activities)
    x_data_test, y_data_test = create_sequences(test_data_df[['accel_x', 'accel_y', 'accel_z']], test_data_df['activity'], timesteps, unique_activities)

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

    np.random.seed(42)
    random = np.arange(0, len(y_data_train))
    np.random.shuffle(random)
    final_x_data_train = final_x_data_train[random]
    y_data_train = y_data_train[random]

    # For test data
    filtered_windows = np.zeros_like(x_data_test)
    for i in range(len(x_data_test)):
        for j in range(3):  # For each axis
            filtered_windows[i, :, j] = apply_lowpass_filter(x_data_test[i, :, j])

    # Scale the entire dataset
    reshaped_data = filtered_windows.reshape(-1, 3)
    scaler = load(open(f'robust_scaler.pkl', 'rb'))
    scaled_data = scaler.transform(reshaped_data)
    final_x_data_test = scaled_data.reshape(len(x_data_test), timesteps, 3)

    return final_x_data_train, y_data_train, final_x_data_test, y_data_test, unique_activities


def onehot_encode_data(y_train, y_test):
    hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    hot_encoder = hot_encoder.fit(y_train)
    y_train = hot_encoder.transform(y_train)
    y_test = hot_encoder.transform(y_test)

    return y_train, y_test


def retrain_model(X_train, y_train, X_test, y_test, chosen_model, class_labels):

    pretrained_model = keras.models.load_model(f"files_{filename}/saved_models_{filename}/acc_{chosen_model}_model.keras")

    model_without_last_layer = Model(inputs=pretrained_model.inputs, outputs=pretrained_model.layers[-2].output)

    # Fine-tune pretrained model
    new_layer = Dense(y_train.shape[1], activation='softmax')(model_without_last_layer.output)
    new_model = Model(inputs=model_without_last_layer.input, outputs=new_layer)

    # Compile the new model
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])
    new_model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=64, verbose=2)
    new_model.save(f'files_{filename}/saved_models_{filename}/acc_retrained_{chosen_model}_model.keras')

    loss, accuracy = new_model.evaluate(X_train, y_train, verbose=0)
    print("Train Accuracy: %d%%, Train Loss: %d%%" % (100*accuracy, 100*loss))

    probabilities = new_model.predict(X_test, verbose=0)
    window_size = 3
    threshold = 0.7
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(probabilities, axis=1)
    smoothed_probs = np.zeros_like(probabilities)

    for i in range(len(probabilities)):
        start = max(0, i - window_size + 1)
        end = i + 1
        smoothed_probs[i] = np.mean(probabilities[start:end], axis=0)

    smoothed_predictions = []
    for i, probs in enumerate(smoothed_probs):
        max_prob = np.max(probs)
        if max_prob >= threshold:
            smoothed_predictions.append(np.argmax(probs))
        else:
            # If below threshold, retain previous prediction or mark as uncertain (-1)
            smoothed_predictions.append(smoothed_predictions[-1] if smoothed_predictions else 1)

    # Calculate accuracy and other metrics
    print("Accuracy with initial predictions: ", round(100 * accuracy_score(y_test_labels, y_pred_labels), 2))
    print("F1 score with initial predictions :",
          round(100 * f1_score(y_test_labels, y_pred_labels, average='weighted'), 2))
    print("Accuracy with smoothed predictions: ", round(100 * accuracy_score(y_test_labels, smoothed_predictions), 2))
    print("F1 score with smoothed predictions: ",
          round(100 * f1_score(y_test_labels, smoothed_predictions, average='weighted'), 2))

    print("\nClassification Report for initial predictions: :")
    print(classification_report(y_test_labels, y_pred_labels, target_names=class_labels))
    print("\nClassification Report for smoothed predictions: :")
    print(classification_report(y_test_labels, smoothed_predictions, target_names=class_labels))

    activity_predictions_true = np.empty(len(y_test_labels), dtype=object)
    for i in range(0, len(y_test_labels)):
        activity_predictions_true[i] = class_labels[y_test_labels[i]]

    activity_predictions = np.empty(len(y_pred_labels), dtype=object)
    for i in range(0, len(y_pred_labels)):
        activity_predictions[i] = class_labels[y_pred_labels[i]]

    activity_predictions_smoothed = np.empty(len(smoothed_predictions), dtype=object)
    for i in range(0, len(smoothed_predictions)):
        activity_predictions_smoothed[i] = class_labels[smoothed_predictions[i]]

    smoothed_predictions = np.array(smoothed_predictions)
    return y_test_labels, y_pred_labels, smoothed_predictions



def plot_confusion_matrix(y_test_labels, y_pred_labels, smoothed_predictions, class_labels, chosen_model, filename):
    """
    This function plots the confusion matrices, visualising the results of the sequential models. Using the y_test_labels
    and y_pred_labels parameters, it creates and saves the confusion matrix.
    """
    path = f'files_{filename}/plots_{filename}_retrained'
    if not os.path.exists(path):
        os.makedirs(path)

    # Initial Values
    normalize_cm = [None]
    for norm_value in normalize_cm:
        if norm_value == 'true':
            format = '.2f'
            plot_name = f'acc_{chosen_model}_cm_norm_inital.png'
        else:
            format = 'd'
            plot_name = f'acc_{chosen_model}_cm_initial.png'

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
            plot_name = f'acc_{chosen_model}_cm_norm_smooth.png'
        else:
            format = 'd'
            plot_name = f'acc_{chosen_model}_cm_smooth.png'

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


def merge_activity_periods(y_labels, class_labels):
    """
    This function takes the activity labels and combine activities with the same label to create larger periods of a certain activity.
    """
    grouped_labels = [y_labels[0]]
    for label in y_labels[1:]:
        if label != grouped_labels[-1]:
            grouped_labels.append(label)

    grouped_labels = np.array(grouped_labels)
    grouped_class_labels = [class_labels[label] for label in grouped_labels]
    # print(len(grouped_class_labels)
    return grouped_class_labels


def group_categories(y_labels, class_labels):
    """
    This function takes the activity labels and groups them to more generic categories, exercising, idle, walking, sleeping
    """
    predicted_categories = []
    # 'running',
    exercising_activities = ['cycling', 'static_exercising', 'dynamic_exercising']
    idle_activities = ['sitting', 'standing']
    lying_activities = ['sleeping', 'lying']
    y_labels = [class_labels[label] for label in y_labels]

    for activity in y_labels:
        if activity in exercising_activities:
            predicted_categories.append('exercising')
        elif activity in idle_activities:
            predicted_categories.append('idle')
        elif activity in lying_activities:
            predicted_categories.append('lying')
        else:
            predicted_categories.append(activity)

    predicted_categories = np.array(predicted_categories)

    # unique, counts = np.unique(predicted_categories, return_counts=True)
    # print(np.asarray((unique, counts)).T)

    return predicted_categories


if __name__ == '__main__':
    frequency = 25 # Hz
    time_window = 10000 # in ms
    samples_per_window = int(time_window * frequency / 1000)
    class_labels = ['cycling', 'exercising', 'lying', 'running', 'sitting', 'sleeping', 'standing', 'walking']
    category_labels = ['exercising', 'idle', 'lying', 'running', 'walking']
    test_path = "../process_datasets/test_data_9.csv"
    filename = f"{time_window}ms_8_classes_final"

    print('Timesteps per timeseries: ', time_window)
    print(f"folder path: files_{filename}")

    models = ['cnn_lstm','cnn_gru', 'cnn_cnn_lstm', 'cnn_cnn_gru']
    X_train, y_train, X_test, y_test, _ = process_data(test_path, samples_per_window)

    # display_data(train_path, filename, False)
    # display_data(test_path, filename, True)

    y_train, y_test = onehot_encode_data(y_train, y_test)

    for chosen_model in models:
        print(f'\n{chosen_model=}')
        y_test_labels, y_pred_labels, smoothed_predictions = retrain_model(X_train, y_train, X_test, y_test, chosen_model, class_labels)

        plot_confusion_matrix(y_test_labels, y_pred_labels, smoothed_predictions, class_labels, chosen_model, filename)

        # Merge activity periods
        # grouped_class_y_labels = merge_activity_periods(y_test_labels, class_labels)
        # grouped_class_labels = merge_activity_periods(y_pred_labels, class_labels)
        # grouped_class_labels_smooth = merge_activity_periods(smoothed_predictions, class_labels)

        # Make predictions with generic categories
        y_labels_categories = group_categories(y_test_labels, class_labels)
        predicted_categories = group_categories(y_pred_labels, class_labels)
        predicted_categories_smooth = group_categories(smoothed_predictions, class_labels)

        print("\nAccuracy with categories predictions: ",
              round(100 * accuracy_score(y_labels_categories, predicted_categories), 2))
        print("F1 score with categories predictions:",
              round(100 * f1_score(y_labels_categories, predicted_categories, average='weighted'), 2))
        print("\nClassification Report for categories predictions:")
        print(classification_report(y_labels_categories, predicted_categories, target_names=category_labels))

        plot_confusion_matrix_grouped(y_labels_categories, predicted_categories, predicted_categories_smooth, category_labels, chosen_model, filename)
