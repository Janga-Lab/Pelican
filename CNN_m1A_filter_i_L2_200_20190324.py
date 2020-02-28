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

dir = os.path.dirname(os.path.realpath(__file__))

file=open('post_epi_hela_m1A_signals_1mer_20190324.txt','r') 

u_labels=[]
### lists of train m5c labels and signals
m_labels=[]
m_signals=[]

um_labels=[]
um_signals=[]

### lists of test m5c labels and signals-will be helpful to do a test hold
mt_labels=[]
mt_signals=[]
### control input data length and signal length by changing len(m_labels) and len(s_len) value below 
m_median = []
c_median = []
em_median = []
mm_median = []
cm_median = []

for i in file:
    i1=i.split( )
    s_len=i1[3].split('_')
    if i1[1][2:][:1] == 'A' and len(m_labels) < 12001 and len(s_len) <= 1000:
        #print(i1[1][2:][:1])
        #print(i1[3])
        ms=i1[3].split('_')

        msm=list(map(int, ms))
        #print(msm)
        #print(np.median(msm))
       # print(i1[1][:5])
        # if np.median(msm) >= 543:
        m_labels.append('m')
        u_labels.append(i1[1][1:][:1]+'m'+i1[1][3:][:1])
        mm_median.append(np.median(msm))
        m_signals.append(i1[3].split('_'))
    elif i1[1][2:][:1] != 'A'  and len(m_labels) < 4001 and len(s_len) <= 1000:
            #print(i1[1][2:][:1])
            #print(i1[3])
        um_labels.append('e')
        um_signals.append(i1[3].split('_'))
    elif i1[1][2:][:1] == 'A' and len(m_labels) >= 43001 and len(mt_labels) <= 150 and len(s_len) <= 1000:
        #print(i1[1][2:][:1])
        #print(i1[3])

        mt_labels.append('m')
        mt_signals.append(i1[3].split('_'))

uni=len(set(u_labels))
#print(len(u_labels))
#print(uni)
#print('length of m5c training data : '+str(len(m_labels)))  #### uncomment to print length of m5c training data

################################## Following section loads Unmodified A,T,G,C from input file ##########################################

file1=open('control_hela_m1A_m6A_signals_201903_24.txt','r')
#loss_out=open('loss_out.txt','a')
### lists of A,T,G,C labels and signals

A_labels=[]
A_signals=[]
uA_labels=[]
### lists of C labels and signals for test dataset hold 
At_labels=[]
At_signals=[]


### control input data length and signal length by changing len(X_labels) and len(X_len) value below 

for i in file1:
    i1=i.split( )
    s_len=i1[3].split('_')
    
    if i1[1][2:][:1] == 'A' and len(A_labels) < len(m_labels) and len(s_len) <= 1000:
        #print(i1[1][2:][:1])
        #print(i1[3])
        cs=i1[3].split('_')

        csm=list(map(int, cs))
        #print(np.median(csm))
        # if np.median(csm) <= 540:
        cm_median.append(np.median(csm))
        A_labels.append('A')
        A_signals.append(i1[3].split('_'))
        uA_labels.append(i1[1][1:][:1]+'A'+i1[1][3:][:1])
            #print(i1[1][1:][:1]+'C'+i1[1][3:][:1])
            #print(i1[1][:5])
    elif i1[1][2:][:1] == 'A' and len(A_labels) >= 43001 and len(At_labels) <= 301 and len(s_len) <= 1000:
        #print(i1[1][2:][:1])
        #print(i1[3])
        At_labels.append('C')
        At_signals.append(i1[3].split('_'))

signals=m_signals+A_signals
labels=m_labels+A_labels

print(len(m_signals))
print(len(A_signals))

### uncomment if planning to use hold data for testing
# t_signals=mt_signals+Ct_signals
# t_labels=mt_labels+Ct_labels


X = [list(map(int, i)) for i in signals]
#test_X=[list(map(int, i)) for i in t_signals]

####### Normaliazation ########
#XN=[]
#for i in X:
#    ads=[]
#    for n in i:
        
#        norm=(n-np.mean(i))/np.std(i)
#        ads.append(norm)
#    XN.append(ads)
#    #print(len(ads))
#X=XN
### Transform labels data from text to oneHot encoder
le=preprocessing.LabelEncoder()
le.fit(labels)
Y=le.transform(labels)

###Uncomment to print label classes
#print('\n'+"Print Classes of train data set: "+'\n')
#print(le.classes_)

# integer encode
label_encoder = preprocessing.LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
#print(integer_encoded)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y = onehot_encoder.fit_transform(integer_encoded)

### Post Pad sequences to a length of 3000
X=pad_sequences(X, padding="post", maxlen=1000)

### crosschecking code for signal length #########
# asd=[]
# for i in X:
#     asd.append(len(i))
# print('Max_length of signal: '+str(max(asd)))
# test_X=pad_sequences(test_X, padding="post",maxlen=max(asd))
# asd1=[]
# for i in X:
#     asd1.append(len(i))
# print('Max_length of mt_signal: '+str(max(asd1)))

#print(signals)
#print(labels)

print("Length of dataset labels",len(labels))
print("Length of dataset signals",len(signals))
X = X[:, :, newaxis]

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y,test_size=0.2)


input_img=Input(shape=(1000,1))
print(X_train.shape)

regularization_rate=0.0001
weightinit = 'lecun_uniform'
fc_hidden_nodes = 256
outputdim = 2
metrics=['accuracy']
learning_rate=0.001
callbacks = []

model = Sequential()
model.add(
    BatchNormalization(
        input_shape=(1000,1)))


# for filter_number, f_s1, in zip(filters, f_s):
#     model.add(Convolution1D(filter_number, f_s1, kernel_size=3, padding='same',
#                             kernel_regularizer=l2(regularization_rate),
#                             kernel_initializer=weightinit))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))

###1
model.add(Convolution1D(32, kernel_size=5, padding='same',
                        kernel_regularizer=l2(regularization_rate),
                        kernel_initializer=weightinit))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=1, padding='valid'))
###2
model.add(Dropout(0.5))
model.add(Convolution1D(64, kernel_size=5, padding='same',
                        kernel_regularizer=l2(regularization_rate),
                        kernel_initializer=weightinit))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=1, padding='valid'))
###3
#model.add(Dropout(0.5))
#model.add(Convolution1D(128, kernel_size=5, padding='same',
#                        kernel_regularizer=l2(regularization_rate),
#                        kernel_initializer=weightinit))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling1D(pool_size=2, strides=1, padding='valid'))
###4
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(units=fc_hidden_nodes,
                kernel_regularizer=l2(regularization_rate),
                kernel_initializer=weightinit))  # Fully connected layer
model.add(Activation('relu'))  # Relu activation

model.add(Dropout(0.5))
model.add(Dense(units=outputdim, kernel_initializer=weightinit))
model.add(BatchNormalization())
model.add(Activation("softmax"))  # Final classification layer

model.compile(loss='categorical_crossentropy',
                optimizer=Adam(lr=learning_rate),
                metrics=metrics)

model.summary()

#Save Model=ON
history = model.fit(X_train, Y_train,
          batch_size=50,
          epochs=200,
          verbose=1,
          validation_data=(X_test, Y_test),shuffle=True,callbacks=[TensorBoard(log_dir=dir+'/autoencoder200_20190324')])

score = model.evaluate(X_test, Y_test, verbose=0)
model.save(dir+'/0.6_prob_keras_m1A_v2_200_20190324.h5')  # creates a HDF5 file 'my_model.h5'

#print loss and accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])
