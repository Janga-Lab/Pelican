#!/usr/bin/python 

import re
import sys
import getopt


import sys, getopt

import os

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.preprocessing.sequence import pad_sequences
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
#from mlxtend.plotting import plot_decision_regions
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#################################### Modificaion data ########################################
dir = os.path.dirname(os.path.realpath(__file__))


def MLpred(x_sigs):
    mod_seq=''
    mod_prob=''
    x_sig = [list(map(int, x_sigs))] 
    X_pred = pad_sequences(x_sig, padding="post", maxlen=3000)
    #print(X_pred)
    R_model = joblib.load(dir+'/models/'+'m5c_flter_Random Forest.sav')
    N_model = joblib.load(dir+'/models/'+'m5c_flter_Neural Net.sav')
    NN_model = joblib.load(dir+'/models/'+'m5c_flter_Nearest Neighbors.sav')
    
    
    y_label_R = R_model.predict(X_pred)
    y_prob_R = R_model.predict_proba(X_pred)
    y_label_N = N_model.predict(X_pred)
    y_prob_N = N_model.predict_proba(X_pred)
    y_label_NN = NN_model.predict(X_pred)
    y_prob_NN = NN_model.predict_proba(X_pred)
    
    
    if y_label_NN[0] == 1 and y_prob_NN[0][0] >= 0.3:
        
        #print(y_label)
        mod_seq='F'
        # print("X=%s, Predicted=%s" % (X_pred[0], y_label_NN[0]), y_prob_NN[0][0])
        mod_prob=str(y_prob_NN[0][0])
    else:
        mod_seq='C'
    mod_prob=''.join(mod_prob)
    return mod_seq, mod_prob