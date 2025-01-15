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
from sklearn.model_selection import StratifiedKFold
from scipy.fft import fft
from scipy.stats import skew, kurtosis
import sys

# Redirect stderr to /dev/null to silence warnings
devnull = open(os.devnull, 'w')
contextlib.redirect_stderr(devnull)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
np.set_printoptions(threshold=sys.maxsize)

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


def train_test_split_acc(path, timesteps):
    """
    This function splits the data to train-test sets. After reading the csv file, it creates the train and test sets.
    :return: train_data, test_data, unique_activities
    """
    data = pd.read_csv(path)
    data = data.drop(columns=['timestamp', 'hr', 'Unnamed: 0'])
    # data = data.dropna(subset=['gyro_x', 'gyro_y', 'gyro_z'])
    data = data.dropna()
    data = data[['activity', 'accel_x', 'accel_y', 'accel_z']]
    unique_activities = data['activity'].unique()

    x_data, y_data = create_sequences(data[['accel_x', 'accel_y', 'accel_z']], data['activity'], timesteps,
                                      unique_activities)

    return x_data, y_data, unique_activities


def train_test_split_gyro(path, timesteps):
    """
    This function splits the data to train-test sets. After reading the csv file, it creates the train and test sets.
    :return: train_data, test_data, unique_activities
    """
    data = pd.read_csv(path)
    data = data.drop(columns=['timestamp', 'hr', 'Unnamed: 0'])
    data = data[['activity', 'gyro_x', 'gyro_y', 'gyro_z']]
    data = data.dropna()
    unique_activities = data['activity'].unique()

    x_data, y_data = create_sequences(data[['gyro_x', 'gyro_y', 'gyro_z']], data['activity'], timesteps,
                                      unique_activities)

    return x_data, y_data, unique_activities


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


def train_sequential_model(X_train_augmented, y_train_augmented, X_test_acc, y_test_acc, X_train_augmented_gyro, y_train_augmented_gyro, X_test_gyro, y_test_gyro, class_labels):
    """
    This function is used to train the sequential models. If train_model == True, then it trains the model using
    X-train, y_train, else it loads the model from the existing file. Then, it evaluates the model and prints the
    classification report.
    :return: y_test_labels, y_pred_labels containing the actual y_labels of test set and the predicted ones.
    """

    # file_name_acc = 'files_10000ms_final/saved_models_10000ms_final/acc_cnn_cnn_lstm_model.h5'
    file_name_acc = 'files_10000ms_clean/saved_models_10000ms_clean/acc_cnn_cnn_lstm_model.h5'
    model_acc = keras.models.load_model(file_name_acc)
    loss, accuracy = model_acc.evaluate(X_train_augmented, y_train_augmented)
    print("Train Accuracy: %d%%, Train Loss: %d%%" % (100 * accuracy, 100 * loss))

    file_name_gyro = 'files_10000ms_final_gyro/saved_models_10000ms_final_gyro/gyro_cnn_cnn_lstm_model.h5'
    model_gyro = keras.models.load_model(file_name_gyro)
    loss, accuracy = model_gyro.evaluate(X_train_augmented_gyro, y_train_augmented_gyro)
    print("Train Accuracy: %d%%, Train Loss: %d%%" % (100 * accuracy, 100 * loss))

    probabilities_acc = model_acc.predict(X_test_acc)
    probabilities_gyro = model_gyro.predict(X_test_gyro)
    # probabilities_comb = (probabilities_acc + probabilities_gyro) / 2

    avg_prob_acc = np.mean(probabilities_acc, axis=0)  
    avg_prob_gyro = np.mean(probabilities_gyro, axis=0) 

    # Calculate class-specific weights (based on the average probabilities)

    weights_acc_class = avg_prob_acc / (avg_prob_acc + avg_prob_gyro)
    weights_gyro_class = avg_prob_gyro / (avg_prob_acc + avg_prob_gyro)
    print(weights_acc_class, weights_gyro_class)

    probabilities_comb = (weights_acc_class * probabilities_acc +
                            weights_gyro_class * probabilities_gyro)

    window_size = 3
    threshold = 0.8
    y_test_labels = np.argmax(y_test_acc, axis=1)
    y_pred_labels = np.argmax(probabilities_comb, axis=1)
    smoothed_probs = np.zeros_like(probabilities_comb)

    for i in range(len(probabilities_comb)):
        start = max(0, i - window_size + 1)
        end = i + 1
        smoothed_probs[i] = np.mean(probabilities_comb[start:end], axis=0)

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
    class_labels = ['cycling', 'dynamic_exercising', 'lying', 'running', 'sitting', 'standing', 'static_exercising', 'walking']

    train_path = "../process_datasets/train_data.csv"
    test_path = "../process_datasets/test_data.csv"
    filename = f"{time_required_ms}ms_final_comb"

    print('Timesteps per timeseries: ', time_required_ms)
    print(f"folder path: files_{filename}")

    models = ['ccn_cnn_lstm']
    # models = ['cnn_lstm','cnn_gru', 'cnn_cnn_lstm', 'cnn_cnn_gru']
    X_train_acc, y_train_acc, unique_activities = train_test_split_acc(train_path, samples_required)
    X_test_acc, y_test_acc, unique_activities = train_test_split_acc(test_path, samples_required)
    X_train_gyro, y_train_gyro, unique_activities = train_test_split_gyro(train_path, samples_required)
    X_test_gyro, y_test_gyro, unique_activities = train_test_split_gyro(test_path, samples_required)
    print(X_test_acc.shape, X_test_gyro.shape)

    # Preprocess original and augmented data
    X_train_augmented, y_train_augmented, X_test_acc, y_test_acc = preprocessing_data(X_train_acc, y_train_acc, X_test_acc, y_test_acc)
    X_train_augmented_gyro, y_train_augmented_gyro, X_test_gyro, y_test_gyro = preprocessing_data(X_train_gyro, y_train_gyro, X_test_gyro, y_test_gyro)
  
    for chosen_model in models:
        print(f'\n{chosen_model=}')
        y_test_labels, y_pred_labels, smoothed_predictions = train_sequential_model(X_train_augmented, y_train_augmented, X_test_acc, y_test_acc, X_train_augmented_gyro, y_train_augmented_gyro, X_test_gyro, y_test_gyro,
                                                                class_labels)

        plot_confusion_matrix(y_test_labels, y_pred_labels, smoothed_predictions, class_labels, chosen_model, filename)
