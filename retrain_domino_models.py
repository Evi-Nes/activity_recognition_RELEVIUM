import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import os
import contextlib

from keras import Model
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from keras.src.layers import MaxPooling1D, Conv1D, Dense

from tensorflow.keras.models import load_model


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
    data = data.iloc[::4, :]

    columns_to_scale = ['accel_x', 'accel_y', 'accel_z']
    scaler = RobustScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    size = len(data)
    train_data = data.iloc[0:int(size*0.8)]
    test_data = data.iloc[int(size*0.8):]

    train_data['activity'] = pd.Categorical(train_data['activity'], categories=[6, 1, 5, 2, 3, 4], ordered=True)
    train_data = train_data.sort_values(by='activity')

    number_to_number = {6: 1, 1: 2, 5: 3, 2: 4, 3: 5, 4: 6}
    train_data['activity'] = train_data['activity'].map(number_to_number)

    test_data['activity'] = pd.Categorical(test_data['activity'], categories=[6, 1, 5, 2, 3, 4], ordered=True)
    test_data = test_data.sort_values(by='activity')

    number_to_number = {6: 1, 1: 2, 5: 3, 2: 4, 3: 5, 4: 6}
    test_data['activity'] = test_data['activity'].map(number_to_number)

    unique_activities = test_data['activity'].unique()

    return train_data, test_data, unique_activities


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


def retrain_model(chosen_model):

    pretrained_model = load_model(f"saved_models/acc_domino_{chosen_model}_model.h5")

    model_without_last_layer = Model(inputs=pretrained_model.input, outputs=pretrained_model.layers[-2].output)

    # Fine-tune pretrained model
    new_layer = Dense(y_train.shape[1], activation='softmax')(model_without_last_layer.output)
    new_model = Model(inputs=model_without_last_layer.input, outputs=new_layer)

    # Compile the new model
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    new_model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=2)
    new_model.save(f'models/acc_retrained_{chosen_model}_model.h5')
    # new_model.summary()

    loss, accuracy = new_model.evaluate(X_train, y_train)
    print("Train Accuracy: %d%%, Train Loss: %d%%" % (100*accuracy, 100*loss))

    y_pred = new_model.predict(X_test)
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
            plot_name = f'acc_retrained_{chosen_model}_cm_norm.png'
        else:
            format = 'd'
            plot_name = f'acc_retrained_{chosen_model}_cm.png'

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
    class_labels = ['Cycling', 'Lying', 'Running', 'Sitting', 'Standing', 'Walking']

    models = ['lstm_1', 'gru_1', 'lstm_2', 'gru_2', 'cnn_lstm', 'cnn_gru', 'cnn_cnn_lstm', 'cnn_cnn_gru']
    models = models[0:2]

    train_set, test_set, unique_activities = train_test_split(path)
    X_train, y_train, X_test, y_test = preprocess_data(train_set, test_set, samples_required, unique_activities)

    for chosen_model in models:
        print(f'{chosen_model=}')

        y_test_labels, y_pred_labels = retrain_model(chosen_model)

        plot_confusion_matrix(y_test_labels, y_pred_labels, class_labels, chosen_model)

