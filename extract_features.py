import glob
import re
import wave
import itertools
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join, exists
import librosa
import librosa.feature
from tqdm import tqdm
import joblib

"""
    Extract IDs corresponding to the .wav files
"""
def compute(inpath, celeb, outpath):
    mfcc,chroma,mel,contrast,tonnetz = extract_feat(inpath)

    # get the id in integer form
    celeb_int = int(celeb[2:])

    labelledFeatures = np.hstack([celeb_int, mfcc,chroma,mel,contrast,tonnetz])

    np.save(outpath, labelledFeatures)
    return outpath

"""
#TODO: Consider using this to extract features as these might be more speech=specific

def extract_features(audio_path):
    y, sr = librosa.load(audio_path) # y = audio file, sr = sample rate

    # Extract various features of the audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=stft, y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    # Extract pitch and formants
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches, axis=1)
    formants = librosa.formant_track(y=y, sr=sr)

    #""
    #    Additional Features
    #""
    # Compute the zero crossing rate of the audio signal
    #zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

    #""
    #    The rhythm of a person's speaking voice can be an important factor in speaker identification,
    #    and it is possible that the beat structure of an audio signal
    #    could be used as a secondary or supplementary feature in a speaker identification system.
    #    However, it would not be the primary basis for identifying the speaker.
    #""
    # Extract other prosodic features such as tempo
    # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)


    # Concatenate all features
    features = np.concatenate([mfcc, mel, chroma, contrast, tonnetz, pitch, formants, zero_crossing_rate])

    return features
"""

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

    return mfcc,chroma,mel,contrast,tonnetz # shape: (40,), (12,), (128,), (7,), (6,)

def parse_files(folder, audio_df):
    out_dir = 'processed/voxceleb2'
    n_jobs = -1
    verbose=1

    celebs = [f for f in listdir(folder)] # get the filenames (ids) in the train_audio/ directory
    #for celeb in tqdm(celebs[344:]):
    for celeb in tqdm(celebs):
        records = [f for f in listdir('{}/{}'.format(folder, celeb))] # go into each celeb file

        files = glob.glob(join(folder, celeb, '*', '*.wav')) # extract the audio files in each file

        inputs = files
        outputs = [ join(out_dir, '{}-{}.npy'.format(celeb,n)) for n in range(len(inputs)) ]

        if (exists(outputs[0])):
            continue

        jobs = [ joblib.delayed(compute)(i, celeb, o) for i,o in zip(inputs, outputs) ]
        out = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)


# read the meta csv file
meta_df = pd.read_csv('../voxceleb/data/voxall_meta.csv', delimiter = '\t' )
print(meta_df.head())

# Instatiate a dataframe the train audio features will be in
columns = ['id'] + ['mfcc']*40 + ['chroma']*12 + ['mel']*128 + ['contrast']*7 + ['tonnetz']*6
audio_df = pd.DataFrame(columns = columns)

parse_files('../voxceleb/data/voxceleb2', audio_df)
