# Voice-Detect

Playing around with tensorflow to extract voice features and
train a model to recognize celebrity voices from wav file sample

I'm using miniconda to get tensorflow working on my M1

### Dependencies

```
pip install -r requirements.txt
```

In addition to the Python dependencies, `ffmpeg` must be installed on the system.

### Data

The [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) datasets are used for these experiments.

### How it works

After getting the raw datasets and preparing them, the scripts here will do the following:

#### Extract relevant audio features from the .wav files

As someone with no knowledge about the physics of sound, I found this guy's youtube channel to be pretty awesome! https://www.youtube.com/@mikexcohen1

```
python3 ./extract_features.py ...
```

This ended up taking almost 2 days on my M1

#### Take the extracted features, and feed them to a Convolutional Model

```
python3 ./train_model.py ...
```

On a subset of data (VoxCeleb1) it takes about an hour. The model and weights then get persisted to file

#### Load the trained model and test it on sound files to see if it recognizes the speaker

```
python3 ./predict.py ...
```

### Results

This model only has ~70% accuracy. There are many improvements to be made

#### Prepare the data better (better standardization)

#### Improve features extracted

#### Feed them as a proper 2D matrix to the model

#### Change the model architecture as per this: https://arxiv.org/abs/2005.07143
