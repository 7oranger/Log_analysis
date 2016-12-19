#-*-coding:utf-8-*-
'''
ref http://www.volodenkov.com/post/keras-lstm-sentiment-p2/
2016-11-30
'''
from __future__ import print_function,division
import os, sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten,LSTM, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from gensim.models import Doc2Vec
from gensim.models import word2vec
from keras import callbacks
from keras.models import Sequential
np.random.seed(1337)

def load_w2v():
    #_fname = "/home/walter/Dev/Data/GoogleNews-vectors-negative300.bin"
    file_path = r"H:\corpus_trained_model\GoogleNews-vectors-negative300.bin"
    #w2vModel = Doc2Vec.load_word2vec_format(_fname, binary=True)
    model = word2vec.Word2Vec.load_word2vec_format(file_path, binary=True)
    model.save_word2vec_format(r'.\data\GoogleNews-vectors-negative300.txt', binary=False)
    return w2vModel 

#tokenize with your prefered tokenizer
tokenize = Tokenizer()
tokens = tokenize(text)
filteredTokens = filter(lambda x: x in w2vModel.vocab, tokens)

class DataIterator:
    def __init__(self, data_path, batch_size = 1000):
        pos_files, neg_files = get_data_file_list(data_path)
        self.pos_iter = iter(pos_files)
        self.neg_iter = iter(neg_files)
        self.batchSize = batch_size

    def get_next(self):
        vectors = []
        values = []
        while (len(vectors) < self.batchSize):

            file = next (self.pos_iter, None)
            if file == None:
                break
            vec = np.load(self.data_path + file)
            vectors.append(vec)
            values.append([1,0])

            file = next(self.neg_iter, None)
            if file == None:
                break
            vec = np.load(self.data_path + file)
            vectors.append(vec)
            values.append([0,1])
        return np.array(vectors), np.array(values)

# creating and training models


def train():
    timesteps = 100
    dimensions = 300
    batch_size = 64
    epochs_number = 40
    model = Sequential()
    model.add(LSTM(200,  input_shape=(timesteps, dimensions),  return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(2, input_dim=200, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='rmsprop')
    fname = 'weights/keras-lstm.h5'
    model.load_weights(fname)
    cbks = [callbacks.ModelCheckpoint(filepath=fname, monitor='val_loss', save_best_only=True),
            callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    train_data_path=r'H:\corpus_trained_model\aclImdb\train'
    test_data_path=r'H:\corpus_trained_model\aclImdb\test'
    train_iterator = DataIterator(train_data_path, sys.maxint)
    test_iterator = DataIterator(test_data_path, sys.maxint)
    train_X, train_Y = train_iterator.get_next()
    test_X, test_Y = test_iterator.get_next()
    from keras.utils.visualize_util import plot
    plot(model, to_file=r'\data\pretrained-embedding-cnn-model.png')
    model.fit(train_X, train_Y, batch_size=batch_size, callbacks=cbks, nb_epoch=epochs_number,
              show_accuracy=True, validation_split=0.25, shuffle=True)
    loss, acc = model.evaluate(test_X, test_Y, batch_size, show_accuracy=True)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
    # example output Test loss / test accuracy = 0.1978 / 0.8195
if __name__ == "__main__":
    train()
