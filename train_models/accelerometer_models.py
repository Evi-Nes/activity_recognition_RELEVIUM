import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import os
import contextlib
# import pickle
# import tsfel
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from keras.src.layers import MaxPooling1D, Conv1D

from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing

# Redirect stderr to /dev/null to silence warnings
devnull = open(os.devnull, 'w')
contextlib.redirect_stderr(devnull)


def plot_data_distribution(path):
    """
    This function plots the number of instances per activity (the distribution of the data).
    """
    if not os.path.exists('plots'):
        os.makedirs('plots')

    data = pd.read_csv(path)
    data = data.drop(['timestamp'], axis=1)
    data = data.drop(['user_id'], axis=1)

    undesired_activities = ['ELEVATOR_DOWN', 'ELEVATOR_UP', 'SITTING_ON_TRANSPORT', 'STAIRS_UP',
                            'STANDING_ON_TRANSPORT', 'TRANSITION']
    data = data[~data['activity'].isin(undesired_activities)]
    data = data.iloc[::4, :]

    class_counts = data['activity'].value_counts()

    plt.figure(figsize=(10, 10))
    class_counts.plot(kind='bar')
    plt.xlabel('Activity')
    plt.ylabel('Number of Instances')
    plt.xticks(rotation=45)
    plt.savefig(f'plots/data_distribution.png')
    plt.show()


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


def train_test_split(path, timesteps):
    """
    This function splits the data to train-test sets. After reading the csv file, it maps the activities to numbers,
    removes some undesired activities, sets the frequency of the data to 25 Hz and creates the train and test sets.
    :return: train_data, test_data, unique_activities
    """
    data = pd.read_csv(path)
    # data = data.dropna()

    columns_to_scale = ['accel_x', 'accel_y', 'accel_z']
    scaler = RobustScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    data = data[['timestamp', 'activity', 'accel_x', 'accel_y', 'accel_z']]
    data = data.dropna()

    unique_activities = data['activity'].unique()
    # unique_activities = unique_activities[:8]
    # unique_activities = ['cycling', 'dynamic_exercising', 'lying', 'running', 'sitting', 'standing', 'static_exercising', 'walking']
    # uncomment this if you want to plot the data as timeseries
    # display_data(data, unique_activities)
    data_seq, activities_seq = create_sequences(data[['accel_x', 'accel_y', 'accel_z']], data['activity'], timesteps, unique_activities)

    np.random.seed(42)
    original_indices = np.arange(len(activities_seq))
    shuffled_indices = original_indices.copy()
    np.random.shuffle(shuffled_indices)

    # random = np.arange(0, len(activities_seq))
    # np.random.shuffle(random)
    data_seq = data_seq[shuffled_indices]
    activities_seq = activities_seq[shuffled_indices]

    size = len(activities_seq)
    train_indices = shuffled_indices[:int(size * 0.8)]
    test_indices = shuffled_indices[int(size * 0.8):]

    X_train = data_seq[:int(size * 0.8)]
    y_train = activities_seq[:int(size * 0.8)]
    X_test = data_seq[int(size * 0.8):]
    y_test = activities_seq[int(size * 0.8):]

    restored_test_order = np.argsort(test_indices)
    X_test_restored = X_test[restored_test_order]
    y_test_restored = y_test[restored_test_order]

    for activity in unique_activities:
        print(f'Train Activity {activity}: {len(y_train[y_train == activity])}')
        print(f'Test Activity {activity}: {len(y_test[y_test == activity])}')

    # unique, counts = np.unique(y_test, return_counts=True)
    # label_distribution = pd.DataFrame({'Label': unique, 'Support': counts})
    # print(label_distribution)

    hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    hot_encoder = hot_encoder.fit(y_train)
    y_train = hot_encoder.transform(y_train)
    y_test_restored = hot_encoder.transform(y_test_restored)


    return X_train, y_train, X_test_restored, y_test_restored, unique_activities



def preprocess_data(train_data, test_data, timesteps, unique_activities):
    """
    This function pre-processes the data. It uses the create_sequences function to create small timeseries and encodes
    the data using OneHotEncoder.
    :returns: the preprocessed data that can be used by the models (X_train, y_train, X_test, y_test)
    """
    X_train, y_train = create_sequences(train_data[['accel_x', 'accel_y', 'accel_z']], train_data['activity'],
                                        timesteps, unique_activities)
    X_test, y_test = create_sequences(test_data[['accel_x', 'accel_y', 'accel_z']], test_data['activity'],
                                      timesteps, unique_activities)

    np.random.seed(42)
    random = np.arange(0, len(y_train))
    np.random.shuffle(random)
    X_train = X_train[random]
    y_train = y_train[random]

    # random = np.arange(0, len(y_test))
    # np.random.shuffle(random)
    # X_test = X_test[random]
    # y_test = y_test[random]

    for activity in unique_activities:
        print(f'Train Activity {activity}: {len(y_train[y_train == activity])}')
        print(f'Test Activity {activity}: {len(y_test[y_test == activity])}')

    hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    hot_encoder = hot_encoder.fit(y_train)
    y_train = hot_encoder.transform(y_train)
    y_test = hot_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test


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
        plt.savefig(f'plots/scaled_{activity}_data.png')
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


def train_sequential_model(X_train, y_train, X_test, y_test, chosen_model, class_labels, time_required_ms, train_model):
    """
    This function is used to train the sequential models. If train_model == True, then it trains the model using
    X-train, y_train, else it loads the model from the existing file. Then, it evaluates the model and prints the
    classification report.
    :return: y_test_labels, y_pred_labels containing the actual y_labels of test set and the predicted ones.
    """
    if not os.path.exists(f'saved_models8_{time_required_ms}'):
        os.makedirs(f'saved_models8_{time_required_ms}')

    file_name = f'saved_models8_{time_required_ms}/acc_{chosen_model}_model.h5'

    if train_model:
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_sequential_model(X_train, y_train, chosen_model, input_shape, file_name)
    else:
        model = keras.models.load_model(file_name)

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
    print("Accuracy with initial predictions: ", accuracy_score(y_test_labels, y_pred_labels))
    print("Accuracy with smoothed predictions: ", accuracy_score(y_test_labels, smoothed_predictions))
    print("\nClassification Report for initial predictions: :")
    print(classification_report(y_test_labels, y_pred_labels, target_names=class_labels))
    print("\nClassification Report for smoothed predictions: :")
    print(classification_report(y_test_labels, smoothed_predictions, target_names=class_labels))

    activity_predictions_true = np.empty(len(y_test_labels), dtype=object)
    for i in range(0, len(y_test_labels)):
        activity_predictions_true[i] = class_labels[y_test_labels[i]]
    # print(activity_predictions_true)

    activity_predictions = np.empty(len(y_pred_labels), dtype=object)
    for i in range(0, len(y_pred_labels)):
        activity_predictions[i] = class_labels[y_pred_labels[i]]
    # print(activity_predictions)

    # smoothed_predictions = np.argmax(smoothed_probs, axis=1)
    activity_predictions_smoothed = np.empty(len(smoothed_predictions), dtype=object)
    for i in range(0, len(smoothed_predictions)):
        activity_predictions_smoothed[i] = class_labels[smoothed_predictions[i]]
    # print(activity_predictions_smoothed)

    # format = 'd'
    # plot_name = f'acc_{chosen_model}_smooth_cm.png'
    #
    # disp = ConfusionMatrixDisplay.from_predictions(
    #     y_test_labels, smoothed_predictions,
    #     display_labels=class_labels,
    #     normalize=None,
    #     xticks_rotation=70,
    #     values_format=format,
    #     cmap=plt.cm.Blues
    # )
    #
    # plt.figure(figsize=(8, 10))
    # plt.title(f'Confusion Matrix for {chosen_model}')
    # disp.plot(cmap=plt.cm.Blues, values_format=format)
    # plt.xticks(rotation=70)
    # plt.tight_layout()
    # plt.savefig(f'plots4_8000/{plot_name}', bbox_inches='tight', pad_inches=0.1)

    # loss, accuracy = model.evaluate(X_train, y_train)
    # print("Train Accuracy: %d%%, Train Loss: %d%%" % (100 * accuracy, 100 * loss))
    #
    # y_pred = model.predict(X_test)
    # y_pred_labels = np.argmax(y_pred, axis=1)
    # y_test_labels = np.argmax(y_test, axis=1)
    # accuracy = accuracy_score(y_test_labels, y_pred_labels)
    # f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
    # print("Test Accuracy: %d%%" % (100 * accuracy))
    # print("Test F1 Score: %d%%" % (100 * f1))
    #
    # report = classification_report(y_test_labels, y_pred_labels, target_names=class_labels)
    # print(report)

    return y_test_labels, y_pred_labels


def plot_confusion_matrix(y_test_labels, y_pred_labels, class_labels, chosen_model, time_required_ms):
    """
    This function plots the confusion matrices, visualising the results of the sequential models. Using the y_test_labels
    and y_pred_labels parameters, it creates and saves the confusion matrix.
    """
    path = f'plots8_{time_required_ms}'
    if not os.path.exists(path):
        os.makedirs(path)

    normalize_cm = [None, 'true']
    for norm_value in normalize_cm:
        if norm_value == 'true':
            format = '.2f'
            plot_name = f'acc_{chosen_model}_cm_norm.png'
        else:
            format = 'd'
            plot_name = f'acc_{chosen_model}_cm.png'

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
        # plt.show()


if __name__ == '__main__':
    frequency = 25
    time_required_ms = 8000
    samples_required = int(time_required_ms * frequency / 1000)

    path = "combined_dataset7.csv"
    # class_labels = ['walking', 'running', 'cycling', 'standing', 'sitting', 'lying']
    #class_labels = ['cycling', 'dynamic_exercising', 'lying', 'running', 'sitting', 'standing', 'static_exercising', 'walking', 'sleeping']
    class_labels = ['cycling', 'dynamic_exercising', 'lying', 'running', 'sitting', 'sleeping', 'standing', 'static_exercising', 'walking']
    # Uncomment if you want to plot the distribution of the data
    # plot_data_distribution(path)

    # Implemented models
    models = ['gru_2']
#    models = ['gru_2', 'cnn_lstm', 'cnn_gru', 'cnn_cnn_lstm', 'cnn_cnn_gru', 'cnn_cnn', '2cnn_2cnn']
    X_train, y_train, X_test, y_test, unique_activities = train_test_split(path, samples_required)
    # X_train, y_train, X_test, y_test = preprocess_data(train_set, test_set, samples_required, unique_activities)

    for chosen_model in models:
        print(f'{chosen_model=}')
        y_test_labels, y_pred_labels = train_sequential_model(X_train, y_train, X_test, y_test, chosen_model,
                                                              class_labels, time_required_ms, train_model=False)

        plot_confusion_matrix(y_test_labels, y_pred_labels, class_labels, chosen_model, time_required_ms)
