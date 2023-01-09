import tensorflow as tf
import os
from tensorflow import keras
from os import listdir, walk
from os.path import isfile, join, exists
import glob
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from keras import layers, optimizers
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.optimizers import Adam, Adamax
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, GRU, RNN, Dropout, Flatten, Input, Conv1D, MaxPooling1D
from keras.losses import SparseCategoricalCrossentropy
from keras.activations import relu, elu, softmax, selu
from keras.utils import to_categorical
from matplotlib import pyplot as plt

def extract_audio_data(folder, file_pattern):
    files = glob.glob(join(folder, file_pattern)) # extract the audio files in each file

    data = np.load(files[0])
    x = data["x"];
    y = data["y"];
    return x, y


def preprocess_data(x, y):
    # Normalize the features
    #scaler = StandardScaler()
    #x = scaler.fit_transform(x)
    # Save the scaler to a file
    #with open('scaler.pkl', 'wb') as f:
    #  pickle.dump(scaler, f)
    #print("X Normalize ready")

    # Use StratifiedShuffleSplit to split the data into train, validation, and test sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=121)

    for train_index, test_index in sss.split(x, y):
        x_train1, x_test = x[train_index], x[test_index]
        y_train1, y_test = y[train_index], y[test_index]

    print("Test is ready")
    # Use StratifiedShuffleSplit again to split the training data into train and validation sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=121)
    for train_index, val_index in sss.split(x_train1, y_train1):
        x_train, x_val = x_train1[train_index], x_train1[val_index]
        y_train, y_val = y_train1[train_index], y_train1[val_index]

    print("Train and Val are ready")
    # Reshape the features to have a third dimension (required by the Conv1D layer)
    x_train = np.reshape(x_train, newshape=(len(x_train), 1, 193))
    x_val = np.reshape(x_val, newshape=(len(x_val), 1, 193))
    x_test = np.reshape(x_test, newshape=(len(x_test), 1, 193))

    # Label encode Y because some values of Y are not in [0, num_features]
    encoder = LabelEncoder()

    y_train = y_train.ravel()
    y_train = encoder.fit_transform(y_train)

    y_val = y_val.ravel()
    y_val = encoder.transform(y_val)

    y_test = y_test.ravel()
    y_test = encoder.transform(y_test)
    print("Y encoded")

    # Save the encoder instance to a file
    with open('label_encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)

    return x_train, y_train, x_val, y_val, x_test, y_test, encoder

#CNN w/ SeLU
def build_model(categories_length):
    with tf.device("/cpu:0"):
        keras.backend.clear_session()
        keras.layers.BatchNormalization._USE_V2_BEHAVIOR = False

        # Define model as Sequential class
        model = Sequential()

        # Block 1

        # first layer: convolutional layer with filter size of 2 and strid of 2, due to
        # the features having shape divisable by 2. selu shown to work better than relu
        # and leaky relu without CNN.
        # Max Pool to get only the important features
        # add dropout to combat overfitting. rate = 10% of neurons to drop
        model.add(Conv1D(128, (2), strides=(2), padding='same',
                             input_shape = (1,193), activation= 'selu'))
        model.add(MaxPooling1D((2), strides=(2), padding='same'))
        model.add(Dropout(rate = 0.1))

        # Block 2
        model.add(Conv1D(256, (2), strides=(2), padding='same', activation= 'selu'))
        model.add(MaxPooling1D((2), strides=(2), padding='same'))
        model.add(Dropout(rate = 0.1))

        # Block 3
        model.add(Conv1D(256, (2), strides=(2), padding='same', activation= 'selu'))
        model.add(Conv1D(512, (2), strides=(2), padding='same',activation= 'selu'))
        model.add(MaxPooling1D((2), strides=(2), padding='same'))

        # Block 4
        model.add(Conv1D(512, (2), strides=(2), padding='same', activation= 'selu'))
        model.add(MaxPooling1D((2), strides=(2), padding='same'))
        model.add(Dropout(rate = 0.1))

        # Block 5
        model.add(Conv1D(512, (2), strides=(2), padding='same',activation= 'selu'))
        model.add(MaxPooling1D((2), strides=(2), padding='same'))

        # Block 6
        model.add(Conv1D(256, (2), strides=(2), padding='same', activation= 'selu'))
        model.add(MaxPooling1D((2), strides=(2), padding='same'))

        # Block 7
        model.add(Conv1D(256, (2), strides=(2), padding='same', activation= 'selu'))
        model.add(MaxPooling1D((2), strides=(2), padding='same'))
        model.add(Dropout(rate = 0.1))

        # Block 8
        model.add(Conv1D(256, (2), strides=(2), padding='same', activation= 'selu'))
        model.add(MaxPooling1D((2), strides=(2), padding='same'))

        # Output Block
        model.add(layers.Flatten())
        model.add(Dense(categories_length, activation = 'softmax'))

        # Compile the model with a specified Adamax optimizer
        opt = Adamax(learning_rate = 1e-3, decay = 1e-5) # Adamax has shown to yield faster learning than Adam and SGD
        model.compile(optimizer=opt, loss=SparseCategoricalCrossentropy(), metrics = ['accuracy'])

        return model

def train_model(model, x_train, y_train, x_test, y_test, x_val, y_val):
    with tf.device("/cpu:0"):
        # Add automated stopping after val_loss difference from epoch t and t-1 is
        # more than 0.001; give it three more epochs to try and get back on track (patience)
        earlystop = EarlyStopping(monitor='val_loss',
                                  min_delta=0.001,
                                  patience=3,
                                  verbose=0, mode='auto')

        # fit the data and save it with history variable.
        history = model.fit(x_train,
                        y_train,
                        epochs = 100,
                        batch_size = 25,
                        validation_data= (x_val,y_val), callbacks = [earlystop])

        save_model(model)


        l, a = model.evaluate(x_test, y_test, verbose = 0)
        print("Loss: {0} | Accuracy: {1}".format(l, a))

        hist_dense = history.history # contains information from fitting model

        return hist_dense


def save_model(model):
    # Save best model so don't have to rerun in scenario of crash
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model_weights.h5")
    print("Saved model to drive")


folder = './test_data'
file_pattern = 'data.npz'
x, y = extract_audio_data(folder, file_pattern);


x_train, y_train, x_val, y_val, x_test, y_test, encoder = preprocess_data(x, y)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)
dimensions = len(encoder.classes_)
print(dimensions)

cnn_model = build_model(dimensions);

hist_dense = train_model(cnn_model, x_train, y_train, x_val, y_val, x_test, y_test);

def plot(hist_dense):
    loss_prelu = hist_dense['loss']
    val_loss_prelu = hist_dense['val_loss']
    acc_prelu = hist_dense['accuracy']
    val_acc_prelu = hist_dense['val_accuracy']

    epochs = range(1, len(loss_prelu) + 1)

    # Plot Losses of training and validation sets v. epochs
    plt.figure(figsize=(25,6))
    plt.subplot(131)
    plt.plot(epochs, loss_prelu, 'k', label = 'Training loss')
    plt.plot(epochs, val_loss_prelu, 'r', label = 'Validation loss')

    # Add the good stuff
    plt.title('Training and validation loss over epochs')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy scores over epochs for training and validation sets
    plt.subplot(132)
    plt.plot(epochs, acc_prelu, 'k', label='Training acc')
    plt.plot(epochs, val_acc_prelu, 'r', label= 'Validation acc')

    # Add the good stuff
    plt.title('Training and validation accuracy over epochs [loss_prelu]')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()

    plt.show()

plot(hist_dense)

