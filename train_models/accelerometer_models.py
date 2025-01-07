import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import os
import contextlib
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from keras.src.layers import MaxPooling1D, Conv1D, Layer, Dense

from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from scipy.fft import fft
from scipy.stats import skew, kurtosis

# Redirect stderr to /dev/null to silence warnings
devnull = open(os.devnull, 'w')
contextlib.redirect_stderr(devnull)


def compute_frequency_features(data, sampling_rate=25):
    """
    Compute frequency domain features from the data.
    """
    # Apply FFT
    fft_vals = np.abs(fft(data))
    freqs = np.fft.fftfreq(len(data), d=1/sampling_rate)
    
    # Keep only positive frequencies
    positive_indices = np.where(freqs > 0)[0]  # Find indices of positive frequencies
    positive_freqs = freqs[positive_indices]
    positive_fft_vals = fft_vals[positive_indices]
    
    # Ensure positive_freqs is not empty
    if len(positive_freqs) == 0 or len(positive_fft_vals) == 0:
        dominant_freq = 0
        spectral_energy = 0
        spectral_entropy = 0
    else:
        # Compute frequency domain features
        dominant_freq = positive_freqs[np.argmax(positive_fft_vals)]
        spectral_energy = np.sum(positive_fft_vals**2)
        spectral_entropy = -np.sum((positive_fft_vals / np.sum(positive_fft_vals)) * 
                                   np.log(positive_fft_vals / np.sum(positive_fft_vals) + 1e-12))
    
    return np.array([dominant_freq, spectral_energy, spectral_entropy])


def compute_statistical_features(data):
    """
    Compute statistical features from the data.
    """
    mean_val = np.mean(data)
    variance_val = np.var(data)
    skewness_val = skew(data)
    kurtosis_val = kurtosis(data)
    min_val = np.min(data)
    max_val = np.max(data)
    rms_val = np.sqrt(np.mean(data**2))
    
    return np.hstack((
        np.array([mean_val, variance_val]),
        np.array([skewness_val]).flatten(),  # Ensure skewness is flattened
        np.array([kurtosis_val]).flatten(),  # Ensure kurtosis is flattened
        np.array([min_val, max_val, rms_val])
    ))


def plot_data_distribution(y_train, y_test, unique_activities, filename):
    """
    This function plots the number of instances per activity (the distribution of the data).
    """
    if not os.path.exists(f'plots_{filename}'):
        os.makedirs(f'plots_{filename}')
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    data = pd.concat([y_train, y_test], ignore_index=True)
    data = data.replace({'0': 'cycling', '1': 'dynamic_exercising', '2': 'lying', '3': 'running', '4': 'sitting', '5': 'standing', '6': 'static_exercising', '7': 'walking'})
    class_counts = data.value_counts()

    plt.figure(figsize=(10, 10))
    class_counts.plot(kind='bar')
    plt.xlabel('Activity')
    plt.ylabel('Number of Instances')
    plt.xticks(rotation=45)
    plt.savefig(f'files_{filename}/plots_{filename}/data_distribution.png')
    # plt.show()


def create_sequences(X_data, Y_data, timesteps, unique_activities):
    """
    This function takes the X, Y data as time instances and transforms them to small timeseries.
    For each activity, creates sequences using sliding windows with 50% overlap.
    :returns: data as timeseries
    """
    X_seq, Y_seq = [], []
    features = []
    for activity in unique_activities:
        for i in range(0, len(X_data) - timesteps, timesteps // 2):
            if Y_data.iloc[i] != activity or Y_data.iloc[i + timesteps] != activity:
                continue

            window_data = X_data.iloc[i:(i + timesteps)].values
            X_seq.append(window_data)            
            Y_seq.append(activity)

             # Compute features
            freq_features = compute_frequency_features(window_data.flatten(), 25)
            stat_features = compute_statistical_features(window_data.flatten())

            # Combine features
            combined_features = np.hstack((freq_features, stat_features))
            features.append(combined_features)

    X_seq, Y_seq = np.array(X_seq), np.array(Y_seq)
    features = np.array(features)
    # print(X_seq.shape)
    # print(features.shape)
    return X_seq, Y_seq.reshape(-1, 1), features


def train_test_split(path, timesteps, testing, scaler):
    """
    This function splits the data to train-test sets. After reading the csv file, it creates the train and test sets.
    :return: train_data, test_data, unique_activities
    """
    data = pd.read_csv(path)
    columns_to_scale = ['accel_x', 'accel_y', 'accel_z']
    if not testing:
        scaler = RobustScaler()
        data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    else:
        data[columns_to_scale] = scaler.transform(data[columns_to_scale])

    data = data[['timestamp', 'activity', 'accel_x', 'accel_y', 'accel_z']]
    data = data.dropna()
    unique_activities = data['activity'].unique()

    # uncomment this if you want to plot the data as timeseries
    # display_data(data, unique_activities)

    x_data, y_data, features = create_sequences(data[['accel_x', 'accel_y', 'accel_z']], data['activity'], timesteps, unique_activities)
    # fscaler = StandardScaler()
    # features = fscaler.fit_transform(features)
    
    # # Expand features to match the time steps
    # features_expanded = np.repeat(features[:, np.newaxis, :], x_data.shape[1], axis=1)  # Shape: (9067, 250, 10)
    # x_data = np.concatenate((x_data, features_expanded), axis=-1)  # Shape: (9067, 250, 13)

    if not testing:
        np.random.seed(42)
        random = np.arange(0, len(y_data))
        np.random.shuffle(random)
        x_data = x_data[random]
        y_data = y_data[random]

    # for activity in unique_activities:
    #     print(f'Activity {activity}: {len(y_data[y_data == activity])}')

    return x_data, y_data, unique_activities, scaler


def display_data(data, unique_activities):
    """
    This function plots subsets of the data as timeseries, to visualize the form of the data.
    """
    if not os.path.exists('plots'):
        os.makedirs('plots')

    for activity in unique_activities:
        subset = data[data['activity'] == activity].iloc[400:600]
        subset = subset.drop(['activity'], axis=1)

        subset.plot(subplots=True, figsize=(10, 10))
        plt.xlabel('Time')
        plt.savefig(f'files_{filename}/plots_{filename}/scaled_{activity}_data.png')
        # plt.show()


def create_sequential_model(X_train, y_train, chosen_model, input_shape, file_name):
    """
    This function is used to create the sequential models. Given the chosen_model param, it chooses the appropriate
    structure and then compiles the model.
    :return: the chosen sequential model
    """
    model = keras.Sequential()
    if chosen_model == 'lstm_1':
        model.add(keras.layers.LSTM(units=64, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.3))
    elif chosen_model == 'gru_1':
        model.add(keras.layers.GRU(units=64, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.3))
    elif chosen_model == 'lstm_2':
        model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.4))
        model.add(keras.layers.LSTM(units=32, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.3))
    elif chosen_model == 'gru_2':
        model.add(keras.layers.GRU(units=64, return_sequences=True, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.4))
        model.add(keras.layers.GRU(units=32, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.3))
    elif chosen_model == 'cnn_lstm':
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
    elif chosen_model == '2cnn_2cnn':
        model.add(Conv1D(filters=32, kernel_size=11, activation='relu', input_shape=input_shape))
        model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(keras.layers.Dropout(rate=0.4))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(rate=0.4))

    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.3, verbose=2)
    model.save(file_name)

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

    file_name = f'files_{filename}/saved_models_{filename}/acc_{chosen_model}_model.h5'

    if train_model:
        input_shape = (X_train.shape[1], X_train.shape[2])
        # input_shape = (timesteps, features)
        model = create_sequential_model(X_train, y_train, chosen_model, input_shape, file_name)
    else:
        model = keras.models.load_model(file_name)

    loss, accuracy = model.evaluate(X_train, y_train)
    print("Train Accuracy: %d%%, Train Loss: %d%%" % (100 * accuracy, 100 * loss))

    probabilities = model.predict(X_test)

    window_size = 3
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
            smoothed_predictions.append(smoothed_predictions[-1] if smoothed_predictions else 5)

    # Calculate accuracy and other metrics
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


if __name__ == '__main__':
    frequency = 25
    time_required_ms = 10000
    samples_required = int(time_required_ms * frequency / 1000)

    train_path = "../process_datasets/train_data.csv"
    test_path = "../process_datasets/test_data.csv"
    filename = f"{time_required_ms}ms"
    print(f'\nTraining 8 classes from file: {train_path}')
    print('Timesteps per timeseries: ', time_required_ms)
    print(f"folder path: files_{filename}")
    print('\n')
    
    class_labels = ['cycling', 'dynamic_exercising', 'lying', 'running', 'sitting', 'standing', 'static_exercising', 'walking']

    # Implemented models
    models = ['gru_2', 'cnn_gru']
    # models = ['gru_2', 'cnn_lstm','cnn_gru', 'cnn_cnn_lstm', 'cnn_cnn_gru', 'cnn_cnn', '2cnn_2cnn']
    scaler = RobustScaler()
    X_train, y_train, unique_activities, scaler = train_test_split(train_path, samples_required, False, scaler)
    X_test, y_test, _, _ = train_test_split(test_path, samples_required, True, scaler)

    unique, counts = np.unique(y_train, return_counts=True)
    print(np.asarray((unique, counts)).T)
    unique, counts = np.unique(y_test, return_counts=True)
    print(np.asarray((unique, counts)).T)

    hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    hot_encoder = hot_encoder.fit(y_train)
    y_train = hot_encoder.transform(y_train)
    y_test = hot_encoder.transform(y_test)

    # Uncomment if you want to plot the distribution of the data
    # plot_data_distribution(y_train, y_test, unique_activities, filename)

    for chosen_model in models:
        print(f'\n{chosen_model=}') 
        y_test_labels, y_pred_labels, smoothed_predictions = train_sequential_model(X_train, y_train, X_test, y_test, chosen_model,
                                                                class_labels, filename, train_model=True)

        plot_confusion_matrix(y_test_labels, y_pred_labels, smoothed_predictions, class_labels, chosen_model, filename)
