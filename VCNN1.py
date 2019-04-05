from __future__ import print_function
import os

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
#import pandas as pd ### For future manipulations
#import scipy as sp ### For future manipulations
#import matplotlib.pyplot as plt  #### Uncomment and use if you would like to see the traiing dataset length frequency plots
from sklearn.preprocessing import StandardScaler
# from sklearn import preprocessing, cross_validation, neighbors
#from sklearn.decomposition import PCA ### Uncomment if planning to do dimensionality reduction
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import OneHotEncoder

dir = os.path.dirname(os.path.realpath(__file__))

def predictLabel(X,bt):

    testX=X
    
    #print(trainX)
    #### Batch generator fucntion for dataset ########
    def btch(x, batch_size):
        i=0
        while i<len(x):
            start = i
            end = i+batch_size
            batch_x = np.array(x[start:end])
    #        batch_y = np.array(y[start:end])
            i += batch_size
            #print(len(batch_x))
            return batch_x
    
    #### Random Batch generator fucntion for dataset ########
    
    def next_batch(num, data, labels):
        '''
        Return a total of `num` random samples and labels. 
        '''
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[ i] for i in idx]
        labels_shuffle = [labels[ i] for i in idx]
    
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    
    
    ######################### Parameters for Neural nets##########################
    tf.reset_default_graph()
    batch_size = bt# Batch size of dataset
    v_batch_size = bt# Batch size of dataset
    
    n_classes = 2 # Number of label classes
    n_steps=1 # Chunk size (1 dimension) 
    features=600 # Feature size (length of signal-2 dimension)
    epsilon = 1e-3
     #input('\n'+'Save new model or Load from Previous checkpoint (type True for new & False for previous model): ')
    ######################### CNN Code ##########################
    x = tf.placeholder('float', shape=[batch_size,features])
    y = tf.placeholder('float')
    
    keep_rate = 0.5
    keep_prob = tf.placeholder(tf.float32)
    
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
    def maxpool2d(x):
        #                        size of window         movement of window
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    
    ##### CNN function with 3 5*5 layers #####
    def convolutional_neural_network(x):
        weights = {'W_conv1':tf.Variable(tf.random_normal([1,5,1,32])),
                   'W_conv2':tf.Variable(tf.random_normal([1,5,32,64])),
                   'W_conv3':tf.Variable(tf.random_normal([1,5,64,128])),
                  
                   'W_fc':tf.Variable(tf.random_normal([9600,1024])),
                   'out':tf.Variable(tf.random_normal([1024, n_classes]))}
    
        biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
                   'b_conv2':tf.Variable(tf.random_normal([64])),
                   'b_conv3':tf.Variable(tf.random_normal([128])),
    
                   
                   'b_fc':tf.Variable(tf.random_normal([1024])),
                   'out':tf.Variable(tf.random_normal([n_classes]))}
    
        x = tf.reshape(x, shape=[-1, n_steps, features, 1])
    
        conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = maxpool2d(conv1)
    #    print(conv1.shape) ## uncomment to check the shape of 1st CNN layer
        
        conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = maxpool2d(conv2)
    #    print(conv2.shape) ## uncomment to check the shape of 1st CNN layer
    
        conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
        conv3 = maxpool2d(conv3)
    #    print(conv3.shape) ## uncomment to check the shape of 1st CNN layer
    
        #conv2 == tf.contrib.layers.flatten(conv2)
        #print(conv2.shape)
        conv3s = conv3.get_shape().as_list()
        fc = tf.reshape(conv3,[-1, conv3s[1]*conv3s[2]*conv3s[3]])
        print(fc.shape)
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
        fc = tf.nn.dropout(fc, keep_rate)
    #    print(fc.shape)
        output = tf.matmul(fc, weights['out'])+biases['out']
    #    print(output.shape) ## uncomment to print shape of output
        return output
        
    train_sess=True # change to False if you wanna load from previous checkpoint
    
    loss_v=[]
    accu=[]
    label=[]
    def train_neural_network(x):
        with tf.name_scope('Model1'):
            prediction = convolutional_neural_network(x)
        #print(prediction)
        with tf.name_scope('Loss'):
            cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
        
        with tf.name_scope('OPtimizer'):
            optimizer = tf.train.AdamOptimizer().minimize(cost)
    
        with tf.name_scope('Accuracy'):
        # # Accuracy
            acc = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))
            # predict = tf.argmax(prediction, 1)
    
        hm_epochs = 150 ### Epoch (time steps)
    
        saver = tf.train.Saver()
    
        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", cost)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar("accuracy", acc)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
    
        
    
        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess,dir +'/')
                print('Loaded latest check point....')
    #            for _ in range(int(len(testX)/v_batch_size)):
    #                test_X = btch(testX,v_batch_size)
                test_X=testX.reshape(v_batch_size,features)
    #                print(prediction)
                preds=tf.nn.softmax(prediction)
#                print("Label:" + str(preds.eval(feed_dict={x: test_X})))
                label.append(preds.eval(feed_dict={x: test_X}))
                labs=str(sess.run(tf.argmax(prediction, 1), feed_dict={x: test_X}))
#                file5.write(labs+'\n')
                accur=tf.argmax(prediction, 1)
                
    
    
    train_neural_network(x)
#    print(label)
    return label
