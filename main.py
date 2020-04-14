import numpy as np
from sklearn import svm
from sklearn import preprocessing
from scipy.sparse import dok_matrix
from scipy import sparse
from sklearn.metrics import f1_score, classification_report
from scipy.sparse import dok_matrix
from scipy.sparse import hstack
import pandas as pd
import scipy.sparse as sp
import csv
from data import *
from bag_of_words import *

def compute_accuracy(gt_labels, predicted_labels):
    accuracy = np.sum(predicted_labels == gt_labels) / len(predicted_labels)
    return accuracy

#######  normalization function #######
def normalize_data(train_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')

    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')

    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        return (scaled_train_data, scaled_test_data)
    else:
        print("No scaling was performed. Raw data is returned.")
        return (train_data, test_data)

#######  Load data #######
train_samples = data()
train_labels = pd.read_csv('data/train_labels.txt', sep='\t',   header=None)
train_labels.columns = ['id', 'nationalitate']
validation_s_samples = data()
validation_s_labels = pd.read_csv('data/validation_source_labels.txt', sep='\t',   header=None)
validation_s_labels.columns = ['id', 'nationalitate']
validation_t_samples = data()
validation_t_labels = pd.read_csv('data/validation_target_labels.txt', sep='\t',   header=None)
validation_t_labels.columns = ['id', 'nationalitate']
test_samples = data()
train_samples.config_data("data/train_samples.txt")
validation_s_samples.config_data("data/validation_source_samples.txt")
validation_t_samples.config_data("data/validation_target_samples.txt")
test_samples.config_data("data/test_samples.txt")

#######  Create Bag of Words Model #######
bow_model = BagOfWords()
bow_model.build_vocabulary(train_samples)
validation_s_samples_features = bow_model.get_features(validation_s_samples).tocsr()
validation_t_samples_features = bow_model.get_features(validation_t_samples).tocsr()
train_samples_features = bow_model.get_features(train_samples).tocsr()
test_samples_features = bow_model.get_features(test_samples).tocsr()
scaled_train_data, scaled_test_data = normalize_data(train_samples_features, test_samples_features, type='l2')
scaled_validation_s_samples, scaled_validation_t_samples = normalize_data(validation_s_samples_features, validation_t_samples_features, type='l2')

svm_model = svm.SVC(C=1, kernel='linear')
svm_model.fit(scaled_train_data, train_labels['nationalitate'])

predicted_labels_svm = svm_model.predict(scaled_validation_s_samples)
print(predicted_labels_svm)
model_accuracy_svm = compute_accuracy(np.asarray(validation_s_labels['nationalitate']), predicted_labels_svm)
print("SVM model accuracy: ", model_accuracy_svm * 100)
predicted_labels_svm = svm_model.predict(scaled_validation_t_samples)
print(predicted_labels_svm)
model_accuracy_svm = compute_accuracy(np.asarray(validation_t_labels['nationalitate']), predicted_labels_svm)
print("SVM model accuracy: ", model_accuracy_svm * 100)

Concat_Train_Labels = pd.concat([train_labels,validation_s_labels,validation_t_labels], ignore_index=True)
Concat_Train_Samples = sp.vstack((scaled_train_data, scaled_validation_s_samples), format='csr')
Concat_Train_Samples = sp.vstack((Concat_Train_Samples, scaled_validation_t_samples), format='csr')
print(Concat_Train_Samples.shape[0])
print(len(Concat_Train_Labels))




svm_model.fit(Concat_Train_Samples, Concat_Train_Labels['nationalitate'])
predicted_labels_svm = svm_model.predict(scaled_test_data)
print(predicted_labels_svm)

with open("data/sample_submission.csv", 'w', newline='') as subm:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(subm, fieldnames=fieldnames)
    writer.writeheader()
    for i, j in zip(test_samples.id,predicted_labels_svm):
        writer.writerow({'id': i, 'labels': j})




