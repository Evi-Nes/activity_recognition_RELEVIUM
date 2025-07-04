import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import os
import sys
import io
import contextlib
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from keras.src.layers import MaxPooling1D, Conv1D, Layer, Dense

from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from scipy.fft import fft
from scipy.stats import skew, kurtosis

# Redirect stderr to /dev/null to silence warnings
devnull = open(os.devnull, 'w')
contextlib.redirect_stderr(devnull)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def plot_data_distribution(y_train, y_test, unique_activities, filename):
    """
    This function plots the number of instances per activity (the distribution of the data).
    """
    if not os.path.exists(f'plots_{filename}'):
        os.makedirs(f'plots_{filename}')
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    data = pd.concat([y_train, y_test], ignore_index=True)
    data = data.replace(
        {'0': 'cycling', '1': 'dynamic_exercising', '2': 'lying', '3': 'running', '4': 'sitting', '5': 'standing',
         '6': 'static_exercising', '7': 'walking'})
    class_counts = data.value_counts()

    plt.figure(figsize=(10, 10))
    class_counts.plot(kind='bar')
    plt.xlabel('Activity')
    plt.ylabel('Number of Instances')
    plt.xticks(rotation=45)
    plt.savefig(f'files_{filename}/plots_{filename}/data_distribution.png')
    # plt.show()


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


def jitter_data(data, noise_level=0.01):
    """
    Adds random noise to accelerometer data.

    Parameters:
    - data: np.array of shape (num_samples, window_size, num_axes)
    - noise_level: float, standard deviation of Gaussian noise

    Returns:
    - Jittered data
    """
    # noise = np.random.normal(0, noise_level, size=data.shape)
    std_dev = np.std(data, axis=(1, 2), keepdims=True)  # Compute per-sample std deviation
    noise = np.random.normal(loc=0.0, scale=noise_level * std_dev, size=data.shape)
    jittered_data = data + noise
    return jittered_data


def scale_data(data, scale_range=(0.9, 1.1)):
    scaling_factors = np.random.uniform(scale_range[0], scale_range[1], size=(data.shape[0], 1, 1))
    return data * scaling_factors


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

    X_seq, Y_seq = np.array(X_seq), np.array(Y_seq)
    return X_seq, Y_seq.reshape(-1, 1)


def train_test_split(path, timesteps, testing):
    """
    This function splits the data to train-test sets. After reading the csv file, it creates the train and test sets.
    :return: train_data, test_data, unique_activities
    """
    data = pd.read_csv(path)
    data = data.drop(columns=['timestamp', 'hr', 'Unnamed: 0'])
    data = data[['activity', 'gyro_x', 'gyro_y', 'gyro_z']]
    data = data.dropna()
    unique_activities = data['activity'].unique()

    # uncomment this if you want to plot the data as timeseries
    # display_data(data, unique_activities)

    x_data, y_data = create_sequences(data[['gyro_x', 'gyro_y', 'gyro_z']], data['activity'], timesteps,
                                      unique_activities)

    if not testing:
        np.random.seed(42)
        random = np.arange(0, len(y_data))
        np.random.shuffle(random)
        x_data = x_data[random]
        y_data = y_data[random]

    # for activity in unique_activities:
    #     print(f'Activity {activity}: {len(y_data[y_data == activity])}')

    return x_data, y_data, unique_activities


def jittering_data(X_train, y_train):
    # Add noise to original data
    X_train_jittered = jitter_data(X_train, noise_level=0.02)
    y_train_jittered = np.copy(y_train)

    # Scale original data
    X_train_scaled = scale_data(X_train)
    y_train_scaled = np.copy(y_train)

    # Concatenate original and augmented data
    X_train_augmented = np.concatenate((X_train, X_train_scaled, X_train_jittered), axis=0)
    y_train_augmented = np.concatenate((y_train, y_train_scaled, y_train_jittered), axis=0)

    return X_train_augmented, y_train_augmented


def preprocessing_data(X_train_augmented, y_train_augmented, X_test, y_test):
    scaler = RobustScaler()
    X_train_flat = X_train_augmented.reshape(-1, X_train_augmented.shape[-1])
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_train_augmented = X_train_flat.reshape(X_train_augmented.shape)

    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_flat = scaler.transform(X_test_flat)
    X_test = X_test_flat.reshape(X_test.shape)
    
    # unique, counts = np.unique(y_train_augmented, return_counts=True)
    # print(np.asarray((unique, counts)).T)
    # unique, counts = np.unique(y_test_augmented, return_counts=True)
    # print(np.asarray((unique, counts)).T)

    hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    hot_encoder = hot_encoder.fit(y_train_augmented)
    y_train_augmented = hot_encoder.transform(y_train_augmented)
    y_test = hot_encoder.transform(y_test)

    return X_train_augmented, y_train_augmented, X_test, y_test


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
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=[keras.metrics.CategoricalAccuracy()])  # ['acc']

    model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2, verbose=2)
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

    file_name = f'files_{filename}/saved_models_{filename}/gyro_{chosen_model}_model.h5'
    # file_name = f'files_{filename}/gyro_{chosen_model}_model_1.h5'

    if train_model:
        input_shape = (X_train.shape[1], X_train.shape[2])
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

    return y_test_labels, y_pred_labels, smoothed_predictions


def cross_validation_models(X_train_init, y_train_init, X_test_init, y_test_init, chosen_model, class_labels, filename):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    fold_no = 1
    X = np.concatenate((X_train_init, X_test_init), axis=0)
    y = np.concatenate((y_train_init, y_test_init), axis=0)
    acc_per_fold = []
    loss_per_fold = []

    for train_index, test_index in kf.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        unique, counts = np.unique(y_train, return_counts=True)
        print(np.asarray((unique, counts)).T)
        unique, counts = np.unique(y_test, return_counts=True)
        print(np.asarray((unique, counts)).T)

        hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        hot_encoder = hot_encoder.fit(y_train)
        y_train = hot_encoder.transform(y_train)
        y_test = hot_encoder.transform(y_test)

        input_shape = (X_train.shape[1], X_train.shape[2])
        file_name = ''

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        model = create_sequential_model(X_train, y_train, chosen_model, input_shape, file_name)
        history = model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=40,
            validation_split=0.2, 
            verbose=2,
        )

        model.save(f"files_{filename}/gyro_{chosen_model}_model_{fold_no}.h5")
        scores = model.evaluate(X_test, y_test)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # extra metrics
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
        print("F1 score with initial predictions :",
              round(100 * f1_score(y_test_labels, y_pred_labels, average='weighted'), 2))
        print("Accuracy with smoothed predictions: ",
              round(100 * accuracy_score(y_test_labels, smoothed_predictions), 2))
        print("F1 score with smoothed predictions: ",
              round(100 * f1_score(y_test_labels, smoothed_predictions, average='weighted'), 2))
        print("\nClassification Report for initial predictions: :")
        print(classification_report(y_test_labels, y_pred_labels, target_names=class_labels))
        print("\nClassification Report for smoothed predictions: :")
        print(classification_report(y_test_labels, smoothed_predictions, target_names=class_labels))

        # Increase fold number
        fold_no = fold_no + 1

    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('\n')


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
            plot_name = f'gyro_{chosen_model}_cm_norm_inital.png'
        else:
            format = 'd'
            plot_name = f'gyro_{chosen_model}_cm_initial.png'

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
            plot_name = f'gyro_{chosen_model}_cm_norm_smooth.png'
        else:
            format = 'd'
            plot_name = f'gyro_{chosen_model}_cm_smooth.png'

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
    time_window = 10000
    samples_per_window = int(time_window * frequency / 1000)
    class_labels = ['cycling', 'dynamic_exercising', 'lying', 'running', 'sitting', 'standing', 'static_exercising', 'walking']

    train_path = "../process_datasets/train_data.csv"
    test_path = "../process_datasets/test_data.csv"
    filename = f"{time_window}ms_final_gyro"

    print(f'\nTraining 8 classes from file: {train_path}')
    print('Timesteps per timeseries: ', time_window)
    print(f"folder path: files_{filename}")

    # Implemented models
    models = ['cnn_lstm','cnn_gru', 'cnn_cnn_lstm', 'cnn_cnn_gru']
    X_train, y_train, unique_activities = train_test_split(train_path, samples_per_window, False)
    X_test, y_test, _ = train_test_split(test_path, samples_per_window, True)

    # Preprocess original and augmented data
    X_train_augmented, y_train_augmented = jittering_data(X_train, y_train)
    X_train_augmented, y_train_augmented, X_test, y_test = preprocessing_data(X_train_augmented, y_train_augmented, X_test, y_test)

    # Uncomment if you want to plot the distribution of the data
    # plot_data_distribution(y_train, y_test, unique_activities, filename)

    for chosen_model in models:
        print(f'\n{chosen_model=}')
        y_test_labels, y_pred_labels, smoothed_predictions = train_sequential_model(X_train_augmented, y_train_augmented, X_test, y_test, chosen_model,
                                                                class_labels, filename, train_model=True)
        # cross_validation_models(X_train_augmented, y_train_augmented, X_test, y_test, chosen_model, class_labels, filename)

        plot_confusion_matrix(y_test_labels, y_pred_labels, smoothed_predictions, class_labels, chosen_model, filename)
