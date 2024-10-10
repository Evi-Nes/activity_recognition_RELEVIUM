import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import os
import contextlib
import pickle

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from keras.src.layers import MaxPooling1D, Conv1D

from sklearn import preprocessing

# Redirect stderr to /dev/null to silence warnings
devnull = open(os.devnull, 'w')
contextlib.redirect_stderr(devnull)


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


def train_test_split(path):
    """
    This function splits the data to train-test sets. After reading the csv file, it maps the activities to numbers,
    removes some undesired activities, sets the frequency of the data to 25 Hz and creates the train and test sets.
    :return: train_data, test_data, unique_activities
    """
    data = pd.read_csv(path)
    data = data.drop(['timestamp'], axis=1)
    data = data.drop(['user_id'], axis=1)

    letter_to_number = {'BRUSHING_TEETH': 1, 'CYCLING': 2, 'ELEVATOR_DOWN': 3, 'ELEVATOR_UP': 4, 'LYING': 5,
                        'MOVING_BY_CAR': 6, 'RUNNING': 7, 'SITTING': 8, 'SITTING_ON_TRANSPORT': 9, 'STAIRS_DOWN': 10,
                        'STAIRS_UP': 11, 'STANDING': 12, 'STANDING_ON_TRANSPORT': 13, 'WALKING': 14, 'TRANSITION': 15}

    data['activityId'] = data['activity'].map(letter_to_number)

    undesired_activities = [1, 3, 4, 6, 9, 10, 11, 13, 15]
    data = data[~data['activityId'].isin(undesired_activities)]
    unique_activities = data['activityId'].unique()
    data = data.iloc[::4, :]

    columns_to_scale = ['accel_x', 'accel_y', 'accel_z']
    scaler = RobustScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    size = len(data)
    train_data = data.iloc[0:int(size*0.8)]
    test_data = data.iloc[int(size*0.8):]

    return train_data, test_data, unique_activities


def preprocess_data(train_data, test_data, timesteps, unique_activities):
    """
    This function pre-processes the data. It uses the create_sequences function to create small timeseries and encodes
    the data using OneHotEncoder.
    :returns: the preprocessed data that can be used by the models (X_train, y_train, X_test, y_test)
    """
    X_train, y_train = create_sequences(train_data[['accel_x', 'accel_y', 'accel_z']], train_data['activityId'],
                                        timesteps, unique_activities)
    X_test, y_test = create_sequences(test_data[['accel_x', 'accel_y', 'accel_z']], test_data['activityId'],
                                      timesteps, unique_activities)

    np.random.seed(42)
    random = np.arange(0, len(y_train))
    np.random.shuffle(random)
    X_train = X_train[random]
    y_train = y_train[random]

    random = np.arange(0, len(y_test))
    np.random.shuffle(random)
    X_test = X_test[random]
    y_test = y_test[random]

    # for activity in unique_activities:
    #     print(f'Train Activity {activity}: {len(y_train[y_train == activity])}')
    #     print(f'Test Activity {activity}: {len(y_test[y_test == activity])}')

    hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    hot_encoder = hot_encoder.fit(y_train)
    y_train = hot_encoder.transform(y_train)
    y_test = hot_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test


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

    model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.3, verbose=2)
    model.save(file_name)

    return model


def train_sequential_model(X_train, y_train, X_test, y_test, chosen_model, class_labels, train_model):
    """
    This function is used to train the sequential models. If train_model == True, then it trains the model using
    X-train, y_train, else it loads the model from the existing file. Then, it evaluates the model and prints the
    classification report.
    :return: y_test_labels, y_pred_labels containing the actual y_labels of test set and the predicted ones.
    """
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    file_name = f'saved_models/acc_domino_{chosen_model}_model.h5'

    if train_model:
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_sequential_model(X_train, y_train, chosen_model, input_shape, file_name)
    else:
        model = keras.models.load_model(file_name)

    # print(model.summary())

    loss, accuracy = model.evaluate(X_train, y_train)
    print("Train Accuracy: %d%%, Train Loss: %d%%" % (100*accuracy, 100*loss))

    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
    print("Test Accuracy: %d%%" % (100*accuracy))
    print("Test F1 Score: %d%%" % (100*f1))

    report = classification_report(y_test_labels, y_pred_labels, target_names=class_labels)
    print(report)

    return y_test_labels, y_pred_labels


def plot_confusion_matrix(y_test_labels, y_pred_labels, class_labels, chosen_model):
    """
    This function plots the confusion matrices, visualising the results of the sequential models. Using the y_test_labels
    and y_pred_labels parameters, it creates and saves the confusion matrix.
    """
    if not os.path.exists('plots'):
        os.makedirs('plots')

    normalize_cm = [None, 'true']
    for norm_value in normalize_cm:
        if norm_value == 'true':
            format = '.2f'
            plot_name = f'acc_domino_{chosen_model}_cm_norm.png'
        else:
            format = 'd'
            plot_name = f'acc_domino_{chosen_model}_cm.png'

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
        plt.savefig(f'plots/{plot_name}', bbox_inches='tight', pad_inches=0.1)
        # plt.show()


if __name__ == '__main__':
    frequency = 25
    time_required_ms = 3500
    samples_required = int(time_required_ms * frequency / 1000)

    path = "../domino_dataset/data_domino.csv"
    class_labels = ['Cycling', 'Lying', 'Running', 'Sitting', 'Standing', 'Walking']

    # Choose the model
    models = ['lstm_1', 'gru_1', 'lstm_2', 'gru_2', 'cnn_lstm', 'cnn_gru', 'cnn_cnn_lstm', 'cnn_cnn_gru', 'cnn_cnn', '2cnn_2cnn']
    models = models[0:2]

    for chosen_model in models:
        print(f'{chosen_model=}')

        train_set, test_set, unique_activities = train_test_split(path)
        X_train, y_train, X_test, y_test = preprocess_data(train_set, test_set, samples_required, unique_activities)
        y_test_labels, y_pred_labels = train_sequential_model(X_train, y_train, X_test, y_test, chosen_model,
                                                              class_labels, train_model=True)

        # Uncomment if you want to create the confusion matrices for the results
        plot_confusion_matrix(y_test_labels, y_pred_labels, class_labels, chosen_model)

