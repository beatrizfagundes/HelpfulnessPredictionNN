import os
import sys
import math
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM, GRU, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.text import one_hot

"""
Parse arguments sent in the command line
"""
parser = argparse.ArgumentParser(description='Single task network')

parser.add_argument("--data", type=str, default='', help="Full path of the dataset that will be splitted intro train, validation and test sets")
#parser.add_argument("--train_data", type=str, default='', help="Full path of the train dataset")
#parser.add_argument("--test_data", type=str, default='', help="Full path of the test dataset")
#parser.add_argument("--val_data", type=str, default='', help="Full path of the validation dataset")
parser.add_argument("--outputdir", type=str, default='results/single_task/', help="Output directory to store results")

# classification or regression
parser.add_argument("--learning_task", type=str, default='classification', help="Classification or Regression")
# balanced (1) or imbalanced (0)
parser.add_argument("--balanced", type=int, default='0', help="If the data should be balanced")
# value between 0 and 1
parser.add_argument("--vocab_percent", type=float, default='0.3', help="Percentage of the vocabulary size")
parser.add_argument("--max_sent_len", type=float, default='1000', help="Max sentence length")
# 16, 32, 64, 128
parser.add_argument("--batch_size", type=int, default='32', help="Size of the batches")
# 10, 50, 128, 300
parser.add_argument("--embedding_dim", type=int, default='50', help="Embeddings dimension")
parser.add_argument("--dropout", type=float, default='0.5', help="Dropout rate")
# MLP, LSTM, BILSTM, CONV, LSTM_CONV, BILSTM_CONV, GRU, GRU_CONV
parser.add_argument("--nn_type", type=str, default='MLP', help="Network architecture")
parser.add_argument("--patience", type=int, default='3', help="Patience to early stop")
parser.add_argument("--num_epochs", type=int, default='100', help="Number of epochs")
# adam, rmsprop
parser.add_argument("--optimizer", type=str, default='adam', help="Number of epochs")
# LSTM arguments
parser.add_argument("--lstm_output", type=int, default='50', help="Number of dimensions from the LSTM layer")
# Convolutional arguments
parser.add_argument("--kernel_size", type=int, default='5', help="Kernel size")
parser.add_argument("--filters", type=int, default='64', help="Number of filters")
parser.add_argument("--pool_size", type=int, default='4', help="Pool size")

params, _ = parser.parse_known_args()

print(params)

np.random.seed(42)

"""
Convert the votes to helpfulness ratio if the task is regression
Sampling a balanced dataset if the task is classification and if requested in the args
"""
if params.data == '':
    print('Missing --data argument')
    sys.exit(0)

df = pd.read_csv(params.data)
if params.learning_task == 'regression':
    # Remove reviews without votes
    df = df[df['totalVotes'] > 0]
    df['reviewClass'] = float(df['helpfulVotes']) / float(df['totalVotes'])
elif params.learning_task == 'classification':
    reviewClass = df['reviewClass']
    df = df.drop('reviewClass', axis=1)
    df['reviewClass'] = reviewClass
    # Change class -1 to 0
    df.loc[df['reviewClass'] == -1, 'reviewClass'] = 0
    if params.balanced:
        print('Sampling a balanced data...')
        # Force dataset to be balanced by the less frequent class
        neg_class_count = df[df['reviewClass'] == 0].shape[0]
        pos_class = df[df['reviewClass'] == 1]
        pos_class_count = pos_class.shape[0]
        pos_indices = pos_class.index.values
        drop_indices = np.random.choice(pos_indices, pos_class_count-neg_class_count, replace=False)
        df = df.drop(drop_indices)
else:
    sys.exit(0)

"""
Build train and test sets
"""
X = df['reviewText'].values
y = df['reviewClass'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

print('%i samples for training' % (X_train.shape[0]))
print('%i samples for validation' % (X_val.shape[0]))
print('%i samples for testing' % (X_test.shape[0]))

if params.vocab_percent > 1 || vocab_percent < 0:
    sys.exit(0)

vectorizer = CountVectorizer()
vectorizer.fit(X_train)
vocab_size = int(len(vectorizer.vocabulary_) * params.vocab_percent)

print('%i tokens in the vocabulary from training + validation data' % (vocab_size))

"""
One-hot encoding
Map every token in the vocabulary to an integer value
"""
X_train = [one_hot(d, vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',lower=True, split=' ') for d in X_train]
X_val = [one_hot(d, vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',lower=True, split=' ') for d in X_val]
X_test = [one_hot(d, vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',lower=True, split=' ') for d in X_test]

"""
Use padding to make all instances (reviews) to be represented by a vector of same size
"""
maxlen = params.max_sent_len
X_train = pad_sequences(X_train, maxlen=maxlen)
X_val = pad_sequences(X_val, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

"""
Structure data with batches
"""
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
batch = params.batch_size
padded_shapes = ([None],())
train_batches = train_data.shuffle(1000).padded_batch(batch, padded_shapes = padded_shapes)
val_batches = val_data.shuffle(1000).padded_batch(batch, padded_shapes = padded_shapes)

"""
Build sequential model, train and test
"""
k = 5
if params.learning_task == 'classification':
    acc = np.zeros(k, dtype=np.float32)
    auc = np.zeros(k, dtype=np.float32)
    prec_pos = np.zeros(k, dtype=np.float32)
    recall_pos = np.zeros(k, dtype=np.float32)
    prec_neg = np.zeros(k, dtype=np.float32)
    recall_neg = np.zeros(k, dtype=np.float32)
elif params.task_learning == 'regression':
    mse = np.zeros(k, dtype=np.float32)

for i in range(k):
    print('Run %i' % (i))
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=params.embedding_dim, input_length=maxlen))

    nn_type = params.nn_type
    if nn_type == 'MLP':
        model.add(Dense(units=10, activation='relu'))
        model.add(Dropout(params.dropout))
    elif nn_type == 'LSTM':
        model.add(LSTM(units=params.lstm_output, dropout=params.dropout))
    elif nn_type == 'GRU':
        model.add(GRU(units=params.lstm_output, dropout=params.dropout))
    elif nn_type == 'LSTM_CONV':
        model.add(Dropout(0.5))
        model.add(Conv1D(params.filters, params.kernel_size, padding='valid', activation='relu', strides=1))
        model.add(MaxPooling1D(pool_size=params.pool_size))
        model.add(LSTM(units=params.lstm_output))
    else:
        print('Invalid neural network layer')
        sys.exit(0)

    if params.learning_task == 'classification':
        model.add(Dense(1, activation='sigmoid'))
    elif params.learning_task == 'regression':
        model.add(Dense(1))

    print(model.summary())

    if params.learning_task == 'classification':
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    elif params.learning_task == 'regression':
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # Train model
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=params.patience)
    model.fit(train_batches, epochs=params.num_epochs, validation_data=val_batches, validation_steps=None, callbacks=[early_stop])

    # Test model
    if params.learning_task == 'classification':
        predictions = model.predict_classes(X_test)
        acc[i] = accuracy_score(y_test, predictions)
        auc[i] = roc_auc_score(y_test, predictions)
        prec_pos[i] = precision_score(y_test, predictions)
        recall_pos[i] = recall_score(y_test, predictions)
        prec_neg[i] = precision_score(y_test, predictions, pos_label=0)
        recall_neg[i] = recall_score(y_test, predictions, pos_label=0)
    elif params.learning_task == 'regression':
        predictions = model.predict(X_test)
        mse[i] = mean_squared_error(y_test, predictions)

if params.learning_task == 'classification':
    results = [
        {'Accuracy': acc},
        {'AUC': auc},
        {'Precision Positive Class': prec_pos},
        {'Recall Positive Class': recall_pos},
        {'Precision Negative Class': prec_neg},
        {'Recall Negative Class': recall_neg}
    ]
elif params.learning_task == 'regression':
    results = [
        {'MSE': mse}
    ]

print(results)
for metric_results in results:
    for metric in metric_results:
        print('Mean %s: %f' % (metric, np.mean(metric_results[metric], dtype=np.float64)))

