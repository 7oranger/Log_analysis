#-*-coding:utf-8-*-
'''
Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
2016年11月26日
从 keras例程里修改
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from sklearn.cross_validation import train_test_split  
import ReadData,pickle,os
max_features = 1000
maxlen = 1000  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
num_epoch = 10

print('Loading data...')
folder_path = r'H:\corpus_trained_model\nltk_data\corpora\movie_reviews'
'''

共读 400 个文件
max_features = 1000
maxlen = 1000  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
num_epoch = 10
data set movie review
Test score: 0.753259503841
Test accuracy: 0.5375
'''

#folder_path = r'H:\network_diagnosis_data\new_cut'
X_t, y_t,dicc=ReadData.ReadRaw2HierData(folder_path,200)
X_t = ReadData.to_num(X_t,max_features)
X_train, X_test, y_train, y_test = train_test_split(X_t,y_t, test_size=0.2, random_state= 42 )

print('Pading sequences ')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)#padding = 'post'
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)#truncating = 'post'
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Building model...')
Embedding_Dim = 50
model = Sequential()
model.add(Embedding(max_features, Embedding_Dim, dropout=0.2))
model.add(LSTM(Embedding_Dim, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
from keras.utils.visualize_util import plot
plot(model, to_file=r'.\data\lstm-model.png')
print('Training...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=num_epoch,
          validation_data=(X_test, y_test))

json_string = model.to_json()  #等价于 json_string = model.get_config()  
open('.\data\lstm-model.json','w').write(json_string)    
model.save_weights('.\data\keras-movie-lstm.h5', overwrite=True)
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
