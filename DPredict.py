from __future__ import print_function
import os

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
#import pandas as pd ### For future manipulations
#import scipy as sp ### For future manipulations
#import matplotlib.pyplot as plt  #### Uncomment and use if you would like to see the traiing dataset length frequency plots
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, cross_validation, neighbors
#from sklearn.decomposition import PCA ### Uncomment if planning to do dimensionality reduction
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Input
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten, Dense, UpSampling1D
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import concatenate
from numpy import zeros, newaxis
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K

dir = os.path.dirname(os.path.realpath(__file__))

def DP(X_t,B_t):
    input_img=Input(shape=(1000,1))
    # print(X_t.shape)
    # print(Y_t)

    # input_img=Input(shape=(1000,1))
    # print(X_t.shape)

    regularization_rate=0.001
    weightinit = 'lecun_uniform'
    fc_hidden_nodes = 256
    outputdim = 2
    metrics=['accuracy']
    learning_rate=0.001
    callbacks = []


    model = load_model('0.6_prob_keras_m1A_v2.h5')

    model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=learning_rate),
                    metrics=metrics)

    modP=0
    unmodP=0
    classes = model.predict_classes(X_t, batch_size=1000)
    proba = model.predict(X_t, batch_size=300)
    del model   
    K.clear_session()   

    # print(classes)
    # print(proba)
    # for i, k in zip(proba,classes):
    #     if k == 1:
    #        print(i, k)
        # if i == 1:
        #     modP = modP + 1
        # elif i == 0:
        #     unmodP = unmodP + 1

    # print("Modified :", modP)
    # print("Unmodified :", unmodP)
    return classes, proba.tolist()
