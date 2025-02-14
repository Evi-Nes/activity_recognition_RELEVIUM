import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import os
import contextlib
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from keras.src.layers import MaxPooling1D, Conv1D

from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing

# Redirect stderr to /dev/null to silence warnings
devnull = open(os.devnull, 'w')
contextlib.redirect_stderr(devnull)

np.set_printoptions(threshold=np.inf)

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

            X_seq.append(X_data.iloc[i:(i + timesteps)].values)
            Y_seq.append(activity)
    X_seq, Y_seq = np.array(X_seq), np.array(Y_seq)
    return X_seq, Y_seq.reshape(-1, 1)


def add_noise_and_scale(data, noise_level=0.02, scale_range=(0.9, 1.1)):
    """
    Adds random noise and scale to accelerometer data.
    Returns:
    - Jittered data
    """
    noise = np.random.normal(loc=0.0, scale=noise_level, size=data.shape)
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    jittered_data = (data + noise) * scale_factor
    return jittered_data


def jittering_data(X_train, y_train):
    # Add noise and scale original data
    X_train_jittered = add_noise_and_scale(X_train, noise_level=0.02)
    y_train_jittered = np.copy(y_train)

    # Concatenate original and augmented data
    X_train_augmented = np.concatenate((X_train, X_train_jittered), axis=0)
    y_train_augmented = np.concatenate((y_train, y_train_jittered), axis=0)

    return X_train_augmented, y_train_augmented


def preprocessing_data(X_train_augmented, y_train_augmented, X_test, y_test):
    scaler = RobustScaler()
    X_train_flat = X_train_augmented.reshape(-1, X_train_augmented.shape[-1])
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_train_augmented = X_train_flat.reshape(X_train_augmented.shape)

    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_flat = scaler.transform(X_test_flat)
    X_test = X_test_flat.reshape(X_test.shape)

    hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    hot_encoder = hot_encoder.fit(y_train_augmented)
    y_train_augmented = hot_encoder.transform(y_train_augmented)
    y_test = hot_encoder.transform(y_test)

    return X_train_augmented, y_train_augmented, X_test, y_test


def train_test_split(path, timesteps, to_balance):
    """
    This function splits the data to train-test sets. After reading the csv file, it creates the train and test sets.
    :return: train_data, test_data, unique_activities
    """
    data = pd.read_csv(path)

    # columns_to_scale = ['accel_x', 'accel_y', 'accel_z', 'hr']
    # scaler = RobustScaler()
    # data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    data = data[['activity', 'accel_x', 'accel_y', 'accel_z', 'hr']]
    data = data.dropna()

    # remove after new DREAMT data
    if to_balance:
        lying_data = data[data['activity'] == 'lying']
        sleeping_data = data[data['activity'] == 'sleeping']
        sleeping_data = sleeping_data[:int(len(data) * 0.6)]
        data = pd.concat([lying_data, sleeping_data])

    # data = data.replace({'W': 0, 'N1': 1, 'N2': 1, 'N3': 1, 'R': 1})
    unique_activities = data['activity'].unique()

    data_seq, activities_seq = create_sequences(data[['accel_x', 'accel_y', 'accel_z', 'hr']], data['activity'], timesteps, unique_activities)

    np.random.seed(42)
    original_indices = np.arange(len(activities_seq))
    shuffled_indices = original_indices.copy()
    np.random.shuffle(shuffled_indices)
    
    data_seq = data_seq[shuffled_indices]
    activities_seq = activities_seq[shuffled_indices]

    size = len(activities_seq)
    test_indices = shuffled_indices[int(size * 0.8):]

    X_train = data_seq[:int(size * 0.8)]
    y_train = activities_seq[:int(size * 0.8)]
    X_test = data_seq[int(size * 0.8):]
    y_test = activities_seq[int(size * 0.8):]

    restored_test_order = np.argsort(test_indices)
    X_test_restored = X_test[restored_test_order]
    y_test_restored = y_test[restored_test_order]
    
    print(X_train.shape, X_test_restored.shape)
    print(y_train.shape, y_test_restored.shape)

    # Add noise to original data
    X_train_augmented, y_train_augmented = jittering_data(X_train, y_train)
    X_train_augmented, y_train_augmented, X_test_restored, y_test_restored = preprocessing_data(X_train_augmented, y_train_augmented, X_test_restored, y_test_restored)
    print(X_train.shape, X_test_restored.shape)
    print(y_train.shape, y_test_restored.shape)
    
    # for activity in unique_activities:
    #     print(f'Train Activity {activity}: {len(y_train[y_train == activity])}')
    #     print(f'Test Activity {activity}: {len(y_test[y_test == activity])}')

    # hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # hot_encoder = hot_encoder.fit(y_train)
    # y_train = hot_encoder.transform(y_train)
    # y_test_restored = hot_encoder.transform(y_test_restored)

    return X_train_augmented, y_train_augmented, X_test_restored, y_test_restored, unique_activities


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

    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=2)
    model.save(f'{file_name}.keras')

    return model


def train_sequential_model(X_train, y_train, X_test, y_test, chosen_model, class_labels, filename, train_model):
    """
    This function is used to train the sequential models. If train_model == True, then it trains the model using
    X-train, y_train, else it loads the model from the existing file. Then, it evaluates the model and prints the
    classification report.
    :return: y_test_labels, y_pred_labels containing the actual y_labels of test set and the predicted ones.
    """
    if not os.path.exists(f'files_{filename}/saved_models_{filename}'):
        os.makedirs(f'files_{filename}/saved_models_{filename}')

    file_name = f'files_{filename}/saved_models_{filename}/acc_{chosen_model}_sleeping_model.h5'

    if train_model:
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_sequential_model(X_train, y_train, chosen_model, input_shape, file_name)
    else:
        model = keras.models.load_model(f'{file_name}.keras')

    loss, accuracy = model.evaluate(X_train, y_train)
    print("Train Accuracy: %d%%, Train Loss: %d%%" % (100 * accuracy, 100 * loss))

    probabilities = model.predict(X_test)

    window_size = 6
    threshold = 0.8
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
            smoothed_predictions.append(smoothed_predictions[-1] if smoothed_predictions else 0)

    # Calculate accuracy and other metrics
    print("#### Sleeping-Lying ####")
    print("Accuracy with initial predictions: ", round(100 * accuracy_score(y_test_labels, y_pred_labels), 2))
    print("F1 score with initial predictions :", round(100 * f1_score(y_test_labels, y_pred_labels, average='weighted'), 2))
    print("Accuracy with smoothed predictions: ", round(100 * accuracy_score(y_test_labels, smoothed_predictions), 2))
    print("F1 score with smoothed predictions: ", round(100 * f1_score(y_test_labels, smoothed_predictions, average='weighted'), 2))
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

    # print(activity_predictions_true)
    # print(activity_predictions_smoothed)

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
            plot_name = f'acc_{chosen_model}_cm_norm_sleeping_inital.png'
        else:
            format = 'd'
            plot_name = f'acc_{chosen_model}_cm_sleeping_initial.png'

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
            plot_name = f'acc_{chosen_model}_cm_norm_sleeping_smooth.png'
        else:
            format = 'd'
            plot_name = f'acc_{chosen_model}_cm_sleeping_smooth.png'

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


if __name__ == '__main__':
    frequency = 25
    time_required_ms = 10000
    samples_required = int(time_required_ms * frequency / 1000)
    path = "../process_datasets/final_dreamt_25Hz.csv"
    to_balance = True
    class_labels = ['lying', 'sleeping']

    print(f'\nTraining with 2 classes with balanced data: {to_balance} from file: {path}')
    print('Timesteps per timeseries: ', time_required_ms)
    print(f'Frequency: {frequency} Hz \n')

    if to_balance:
        filename = f"sleeping_{frequency}Hz_balanced_with_hr"
        print(filename)
    else:
        filename = f"sleeping_{frequency}Hz_unbalanced"
        print(filename)

    # Implemented models
    # models = ['cnn_cnn_lstm']
    models = ['cnn_lstm', 'cnn_gru', 'cnn_cnn_lstm', 'cnn_cnn_gru']
    X_train, y_train, X_test, y_test, unique_activities = train_test_split(path, samples_required, to_balance)

    for chosen_model in models:
        print(f'\n{chosen_model=}')
        y_test_labels, y_pred_labels, smoothed_predictions = train_sequential_model(X_train, y_train, X_test, y_test, chosen_model,
                                                                class_labels, filename, train_model=True)

        plot_confusion_matrix(y_test_labels, y_pred_labels, smoothed_predictions, class_labels, chosen_model, filename)
