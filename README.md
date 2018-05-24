# Predict customer satisfaction!

Data Challenge at Air Liquide

## Data description
The 'train' set is made of 12,144 verbatims of customers with the associated Net Promoter Scores expressed in two classes: 1 = detractor, 0 = otherwise (see this link to learn more on Net Promoter Score).  The dataset comes mostly from the Program Voice of Customer.  The verbatims have been cleared of any personal data or sensitive information which have been replaced with hashes (e.g. 87sd5f64fs, a random list of letters and integers, different for each hash). If you want more information on the process, you can have a look at this link.

Only the anonymisation of the data has been performed, the remaining data cleaning and data pre-processing (e.g. feature engineering) is part of the Data Challenge.

You are to use this data to predict the probability of a verbatim within the 'test' dataset being in the class 'Detractor'. The score is the Area Under the ROC Curve (AUC, see this excellent article if you want to learn more on this metrics), since the predicted class (Detractor) is less present in the dataset.

```
verbatim - the original verbatim
DETRACTOR - the detractor class (1: detractor, 0: otherwise): the target to predict
```

You can take a look at the inital data used ('train_with_initial_NPS'), with the initial Net Promoter Scores (NPS) assigned by the customers. The initial NPS vary between 0 (very dissatisfied) to 10 (very satisfied) and are aggregated in 3 categories:

- PROMOTER: NPS greater or equal to 9
- DETRACTOR: NPS less or equal to 6  --> this challenge focus only on this class
- NEUTRAL: remaining NPS (7 and 8)

## Usage

### Install Theano Keras

Full anaconda2 package Theano Keras (install version 1.2.2 for reproduce acc > 89 %)

```
conda install theano pygpu
pip install keras=1.2.2
```

Switch Keras backend to Theano (How-to: https://keras.io/backend/)

### Download Google pretrained word2vec file

Download from google drive (1.5 GB)

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

Unzip the file

```
gunzip GoogleNews-vectors-negative300.bin.gz
```

### Preprocess data

Clean up strings, create vocabulary with word counts, generate word matrix with word2vec embeddings using Google's pretrained word2vec model.

```
python preprocessing.py GoogleNews-vectors-negative300.bin
```


### Run RCNN Model

```
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32 python sst2_cnn_rnn.py

THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32 python sst2_cnn_rnn_keras1.py
```
