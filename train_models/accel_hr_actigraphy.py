import numpy as np
import pandas as pd
from pywt import threshold
from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tsfel
import pickle
from sklearn.feature_selection import VarianceThreshold


def train_test_split(path):
    data = pd.read_csv(path)
    data = data.drop(['timestamp'], axis=1)
    data = data.drop(['user_id'], axis=1)

    # data = data.replace({'W': 0, 'N1': 1, 'N2': 1, 'N3': 1, 'R': 1})

    columns_to_scale = ['accel_x', 'accel_y', 'accel_z']
    scaler = RobustScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    size = len(data)
    train_data = data.iloc[0:int(size*0.7)]
    test_data = data.iloc[int(size*0.7):]

    return train_data, test_data


def extract_features(train_data, test_data, frequency, samples_required, train_features):
    """
    This function uses the tsfel package to extract statistical features from the data and preprocessed the data.
    If train_features == True, then it extracts statistical features from the data, else it loads the features from the
    existing file.
    :returns: the extracted features (X_train_features, y_train_features, X_test_features, y_test_features)
    """
    X_train_sig, y_train_sig = train_data[['accel_x', 'accel_y', 'accel_z', 'hr']], train_data['activity']
    X_test_sig, y_test_sig = test_data[['accel_x', 'accel_y', 'accel_z', 'hr']], test_data['activity']

    if train_features:
        if not os.path.exists('saved_features_1Hz'):
            os.mkdir('saved_features_1Hz')

        cfg_file = tsfel.get_features_by_domain('statistical')
        X_train_features = tsfel.time_series_features_extractor(cfg_file, X_train_sig, fs=frequency, window_size=samples_required)
        X_test_features = tsfel.time_series_features_extractor(cfg_file, X_test_sig, fs=frequency, window_size=samples_required)
        X_train_features.to_csv(f'saved_features_1Hz/X_train_acc.csv', index=False)
        X_test_features.to_csv(f'saved_features_1Hz/X_test_acc.csv', index=False)
    else:
        X_train_features = pd.read_csv(f'saved_features_1Hz/X_train_acc.csv')
        X_test_features = pd.read_csv(f'saved_features_1Hz/X_test_acc.csv')

    X_train_columns = X_train_features.copy(deep=True)
    y_train_features = y_train_sig[::samples_required]
    if len(y_train_features) > len(X_train_features):
        y_train_features = y_train_features.drop(y_train_features.tail(1).index)

    y_test_features = y_test_sig[::samples_required]
    if len(y_test_features) > len(X_test_features):
        y_test_features = y_test_features.drop(y_test_features.tail(1).index)

    # Highly correlated features are removed
    corr_features = tsfel.correlated_features(X_train_features)
    X_train_features.drop(corr_features, axis=1, inplace=True)
    X_test_features.drop(corr_features, axis=1, inplace=True)

    # Remove low variance features
    selector = VarianceThreshold(threshold=0.2)
    X_train_features = selector.fit_transform(X_train_features)
    X_test_features = selector.transform(X_test_features)

    cols_idxs = selector.get_support(indices=True)
    X_train_columns = X_train_columns.iloc[:, cols_idxs]

    scaler = preprocessing.StandardScaler()
    X_train_features = scaler.fit_transform(X_train_features)
    X_test_features = scaler.transform(X_test_features)

    return X_train_features, y_train_features, X_test_features, y_test_features


def train_feature_model(X_train, y_train, X_test, y_test, chosen_model, class_labels, train_model):
    """
    This function is used to train the models based on the extracted features. If train_model == True, then it trains
    the model using the features from the extract_features function, else it loads the model from the existing file.
    Then, it evaluates the model and prints the classification report.
    :return: y_test, y_test_predict containing the actual y_labels of test set and the predicted ones.
    """
    if train_model:
        classifier = RandomForestClassifier(n_estimators=40, min_samples_split=15, min_samples_leaf=4, max_depth=None, bootstrap=True, n_jobs=-1, random_state=42)
        classifier.fit(X_train, y_train.ravel())

        file = open(f'saved_features_1Hz/acc_{chosen_model}_model.pkl', 'wb')
        pickle.dump(classifier, file)
    else:
        file = open(f'saved_features_1Hz/acc_{chosen_model}_model.pkl', 'rb')
        classifier = pickle.load(file)

    classifier.fit(X_train, y_train.ravel())
    y_pred_train = classifier.predict(X_train)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    print("Training Accuracy: %.2f%%" % (round(train_accuracy*100, 2)))

    y_test_predict = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_predict)
    # f1 = f1_score(y_test, y_test_predict, average='weighted')
    print("Test Accuracy: %.2f%% " % (100 * round(accuracy, 2)))

    report = classification_report(y_test, y_test_predict, target_names=class_labels)
    print(report)

    return y_test, y_test_predict



class_labels = ['Lying', 'Sleeping']
# class_labels = ['W', 'N1', 'N2', 'N3', 'R']
path = '../process_datasets/final_dreamt_1Hz.csv'

train_set, test_set = train_test_split(path)
X_train, y_train, X_test, y_test = extract_features(train_set, test_set, 1, 60, train_features=False)
y_train = y_train.replace({'W': 0, 'N1': 1, 'N2': 1, 'N3': 1, 'R': 1})
y_test = y_test.replace({'W': 0, 'N1': 1, 'N2': 1, 'N3': 1, 'R': 1})

y_test_labels, y_pred_rf = train_feature_model(X_train, y_train, X_test, y_test, 'rf', class_labels, train_model=False)

conf_matrix = confusion_matrix(y_test_labels, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Lying', 'Sleeping'], yticklabels=['Lying', 'Sleeping'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(f'saved_features_1Hz/acc_hr_rf_cm.png', bbox_inches='tight', pad_inches=0.1)
# plt.show()