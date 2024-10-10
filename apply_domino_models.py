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

from sklearn.feature_selection import VarianceThreshold
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

    undesired_activities = [0, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 24]
    data = data[~data['activity'].isin(undesired_activities)]
    data = data.dropna()

    data['activity'] = pd.Categorical(data['activity'], categories=[6, 1, 5, 2, 3, 4], ordered=True)
    data = data.sort_values(by='activity')

    number_to_number = {6: 1, 1: 2, 5: 3, 2: 4, 3: 5, 4: 6}
    data['activity'] = data['activity'].map(number_to_number)

    unique_activities = data['activity'].unique()
    data = data.iloc[::4, :]

    columns_to_scale = ['accel_x', 'accel_y', 'accel_z']
    scaler = RobustScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    test_data = data

    return test_data, unique_activities


def preprocess_data(test_data, timesteps, unique_activities):
    """
    This function pre-processes the data. It uses the create_sequences function to create small timeseries and encodes
    the data using OneHotEncoder.
    :returns: the preprocessed data that can be used by the models (X_train, y_train, X_test, y_test)
    """
    X_test, y_test = create_sequences(test_data[['accel_x', 'accel_y', 'accel_z']], test_data['activity'],
                                      timesteps, unique_activities)

    np.random.seed(42)
    random = np.arange(0, len(y_test))
    np.random.shuffle(random)
    X_test = X_test[random]
    y_test = y_test[random]

    # for activity in unique_activities:
    #     print(f'Train Activity {activity}: {len(y_train[y_train == activity])}')
    #     print(f'Test Activity {activity}: {len(y_test[y_test == activity])}')

    hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    hot_encoder = hot_encoder.fit(y_test)
    y_test = hot_encoder.transform(y_test)

    return X_test, y_test


def train_sequential_model(X_test, y_test, chosen_model, class_labels):
    """
    This function is used to train the sequential models. If train_model == True, then it trains the model using
    X-train, y_train, else it loads the model from the existing file. Then, it evaluates the model and prints the
    classification report.
    :return: y_test_labels, y_pred_labels containing the actual y_labels of test set and the predicted ones.
    """
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    file_name = f'saved_models/acc_domino_{chosen_model}_model.h5'

    model = keras.models.load_model(file_name)

    # print(model.summary())

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
            plot_name = f'acc_applied_{chosen_model}_cm_norm.png'
        else:
            format = 'd'
            plot_name = f'acc_applied_{chosen_model}_cm.png'

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

    path = "../pamap2_dataset/data_pamap2.csv"
    class_labels = ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling']

    # Choose the model
    models = ['lstm_1', 'gru_1', 'lstm_2', 'gru_2', 'cnn_lstm', 'cnn_gru', 'cnn_cnn_lstm', 'cnn_cnn_gru', 'cnn_cnn', '2cnn_2cnn', 'rf', 'knn']
    models = models[0:2]

    for chosen_model in models:
        print(f'{chosen_model=}')

        test_set, unique_activities = train_test_split(path)
        X_test, y_test = preprocess_data(test_set, samples_required, unique_activities)
        y_test_labels, y_pred_labels = train_sequential_model(X_test, y_test, chosen_model, class_labels)

        # Uncomment if you want to create the confusion matrices for the results
        # plot_confusion_matrix(y_test_labels, y_pred_labels, class_labels, chosen_model)
