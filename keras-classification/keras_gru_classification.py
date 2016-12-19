# -*- coding: utf-8 -*-
'''
Created on 2016Äê11ÔÂ23ÈÕ
ref http://www.cnblogs.com/doublemystery/p/5092014.html
@author: RenaiC
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
import pickle
import os

np.random.seed(1337)  # for reproducibility
batch_size = 32
def gru():
    weights_file = '20lstm_weights.h5'
    print("Loading data...")
    f=open('train.pkl', 'r')
    X_train, Y_train = pickle.load(f)
    f.close()
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('Build model...')
    model = Sequential()
    model.add(GRU(output_dim=128,input_dim = 48, activation='tanh', inner_activation='hard_sigmoid', input_length=100))  # try using a GRU instead, for fun
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode="categorical")
    json_string = model.to_json()
    print(json_string)
    print("Train...")
    if os.path.exists(weights_file):
        model.load_weights(weights_file)
    from keras.utils.visualize_util import plot
    plot(model, to_file=r'.\data\gru-model.png')
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=4, show_accuracy=True)
    model.save_weights(weights_file, overwrite=True)
    f=open('test.pkl', 'r')
    X_test, Y_test = pickle.load(f)
    f.close()
    score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
    print('Test score:', score)
    print('Test accuracy:', acc)
if __name__ == '__main__':
    lstm_raw()
#     lstm_w2v()