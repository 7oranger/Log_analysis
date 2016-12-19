# -*- coding: utf-8 -*-
#src http://www.cnblogs.com/doublemystery/p/5092014.html 
# 多层感知机
# 2016年11月21日 
from __future__ import absolute_import,division,print_function
import numpy as np

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
import pickle
import ReadData
np.random.seed(1337)  # for reproducibility
batch_size = 32
maxlen = 100
max_features = 1000
print("Loading data...")
X_train, Y_train = ReadData.LoadOriData(r"H:\corpus_trained_model\ldagibbs-master\dataset\newsgroup20\20ng-train-stemmed.txt")
X_test, Y_test = ReadData.LoadOriData(r"H:\corpus_trained_model\ldagibbs-master\dataset\newsgroup20\20ng-test-stemmed.txt")
print(len(X_train), 'train sequences')
tokenizer = Tokenizer(nb_words=max_features)
X_train = tokenizer.sequences_to_matrix(X_train, mode="binary")
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
Y_test = np_utils.to_categorical(Y_test)
Y_train = np_utils.to_categorical(Y_train)
print('X_train shape:', X_train.shape)  #X_train shape: (11293L, 1000L)
print('Y_train shape:', Y_train.shape)
print('Build model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_features,), activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))
# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])#class_mode="categorical"
json_string = model.to_json()
print(json_string)
f = open('20mlp_model.txt', 'w')
f.write(json_string)
f.close()
print("Train...")
from keras.utils.visualize_util import plot
plot(model, to_file=r'.\data\dnn-model.png')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=5, shuffle=True)
model.save_weights('20mlp_weights.h5', overwrite=True)

print(X_test)
print(Y_test)
score, acc = model.evaluate(X_test, Y_test)
#score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
print('\n')
print('Test score:', score)
print('Test accuracy:', acc)