from keras.models import model_from_json
from keras.optimizers import Adamax
from keras.losses import SparseCategoricalCrossentropy
import pandas as pd
from pydub import AudioSegment
from pydub.playback import play
import librosa
import librosa.feature
import random
from os import listdir
from IPython.display import Audio
import numpy as np
import csv
import pickle
from argparse import ArgumentParser

def read_model(filename):
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    return loaded_model

def load_model_weights(model, filename):
    model.load_weights(filename)

def compile_model(model):
    opt = Adamax(learning_rate = 1e-3, decay = 1e-5) # Adamax has shown to yield faster learning than Adam and SGD
    model.compile(optimizer=opt, loss=SparseCategoricalCrossentropy(), metrics = ['accuracy'])

def build_model(model_filename, weights_filename):
    # Read in model
    model = read_model(model_filename)

    # Load weights into new model
    load_model_weights(model, weights_filename)

    # Compile model
    compile_model(model)

    if (debug):
        print(model.summary())

    return model

def load_label_encoder(filename):
    # Load the encoder instance from a file
    with open(filename, 'rb') as file:
        encoder = pickle.load(file)
        return encoder


def load_names(filename):
    # Load data
    names = pd.read_csv(filename, sep=',', header=(0))

    return names


def extract_feat(filename):
    # load in audio file
    y, sr = librosa.load(filename) # y = audio file, sr = sample rate

    # extract the various features of the audio
    mfcc = np.mean(librosa.feature.mfcc(y = y, sr = sr, n_mfcc=40).T, axis = 0)
    mel = np.mean(librosa.feature.melspectrogram(y = y, sr = sr).T, axis = 0)
    stft = np.abs(librosa.stft(y))
    chroma = np.mean(librosa.feature.chroma_stft(S = stft, y = y, sr = sr).T, axis = 0)
    contrast = np.mean(librosa.feature.spectral_contrast(S = stft, y = y, sr = sr).T, axis = 0)
    tonnetz =  np.mean(librosa.feature.tonnetz(y = librosa.effects.harmonic(y), sr = sr).T, axis = 0)

    return mfcc, chroma, mel, contrast, tonnetz # shape: (40,), (12,), (128,), (7,), (6,)

def get_features_from_audio(audio_filename, scaler_filename):
    # get features for the audio
    mfcc,chroma,mel,contrast,tonnetz = extract_feat(audio_filename)

    features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
    fill = np.empty((0,193))
    row = np.vstack([fill,features]) # shape (1,193)
    if (debug):
        print(row.shape)

    # Load the scaler from the file
    #with open('scaler.pkl', 'rb') as f:
    #    scaler = pickle.load(f)
    #    row = scaler.fit_transform(row)

    #reshape row to fit into model
    features = np.expand_dims(row, axis=0) # shape (1,1,193)
    #features = np.reshape(row, newshape=(1, 1. 193)

    if (debug):
        print(features.shape)

    return features

def get_names(names, name_indices):
    result = []
    # Use index to find person in a dummified dataframe of the names
    id_column = names.id

    if (debug):
        print(id_column)
    for index in name_indices:
        person_id = f"id{index}"
        if (debug):
            print(person_id)

        result.append(names[names.id == person_id].name.unique()[0])

    return result

def get_best_guesses(y_pred, encoder_filename):
    # Get the indices of the top 5 largest elements in y_pred
    best_guesses = np.argpartition(-y_pred, 5)[:, :5].ravel()

    # Get the original labels
    encoder = load_label_encoder(encoder_filename)
    original_labels = encoder.inverse_transform(best_guesses)

    # Get the confidence values of the top 5 guesses
    confidence_values = y_pred[np.arange(len(y_pred)), best_guesses]
    confidence = [prob * 100 for prob in confidence_values]

    if (debug):
        print(best_guesses, original_labels, confidence)
    return original_labels, confidence

"""
def get_predicted_label(y_pred, encoder_filename):
    best_guesses = np.argmax(y_pred, axis=1)

    index = best_guesses[0]
    confidence = y_pred[0][index]

    # Get the original labels
    encoder = load_label_encoder(encoder_filename)
    original_labels = encoder.inverse_transform(best_guesses)
    if (debug):
        print(best_guesses, original_labels[0], confidence)
    return original_labels[0], confidence
"""

def predict_name(audio_filename, model_filename, model_weights, encoder_filename, scaler_filename, names_file):
    # Get features from input file
    features = get_features_from_audio(audio_filename, scaler_filename)

    # Build the model
    model = build_model(model_filename, model_weights)

    # Load all the names
    names = load_names(names_file);

    # Get predictions for the audio file
    predictions = model.predict(features)

    # Get predicted label
    name_indices, confidence = get_best_guesses(predictions, encoder_filename)

    result = get_names(names, name_indices)

    return result, confidence

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample", default='./sample.wav', help='Path to audio wav sample')
    parser.add_argument("--model", default='./models/colab/model.adam-0.5416649580001831.json', help='Path to .on file with serialized model')
    parser.add_argument("--weights", default='./models/colab/model_weights.adam-0.5416649580001831.json', help='Path to .h5 file with serialized model weights')
    parser.add_argument("--label_encoder", default='./models/colab/label_encoder.pkl', help='Path to .pkl file with serialized encoder for label')
    parser.add_argument("--scaler", default='./models/colab/scaler.pkl', help='Path to .pkl file with serialized scaler for input datat')
    parser.add_argument("--names", default='./test_data/voxall_meta.csv', help='Path to csv file with ids and names')
    parser.add_argument("--debug", default=False, help='Print debug info')

    args = parser.parse_args()

    global debug
    debug = args.debug

    name, confidence = predict_name(args.sample, args.model, args.weights, args.label_encoder, args.scaler, args.names)

    print(name, confidence)
