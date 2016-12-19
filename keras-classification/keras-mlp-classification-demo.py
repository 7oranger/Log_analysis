#-*-coding:utf-8-*-
'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''

from __future__ import print_function
import numpy as np
import pprint,json,pickle,os
import ReadData
from sklearn.cross_validation import train_test_split  
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

max_words = 1000#1000
batch_size = 32
nb_epoch = 5
min_count = 5
max_featues = 250
np.random.seed(1337)  # for reproducibility
print('Loading data...')
#X_train, y_train = ReadData.LoadOriData(r"H:\corpus_trained_model\ldagibbs-master\dataset\newsgroup20\20ng-train-stemmed.txt")
#X_test, y_test = ReadData.LoadOriData(r"H:\corpus_trained_model\ldagibbs-master\dataset\newsgroup20\20ng-test-stemmed.txt")
#(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2) #type list[[]]
#pickle.dump()
#X_train, y_train,dicc=ReadData.ReadRaw2HierData( r"H:\network_diagnosis_data\cut")
#X_test, y_test,dicc=ReadData.ReadRaw2HierData(r'H:\network_diagnosis_data\cut_extra')
folder_path = r'/media/workserv/498ee660-1fc8-40e8-bb02-f0a626cbfe93/renaic/cut'
folder_path = r"H:\network_diagnosis_data\cut"
folder_path = r'H:\network_diagnosis_data\new_cut'
folder_path = r'H:\corpus_trained_model\20_newsgroup'
folder_path = r'H:\network_diagnosis_data\new-cut-1000'

X_t, y_t,dicc=ReadData.ReadRaw2HierData(folder_path,500)
X_t,Y_t=ReadData.shuffle_X_Y(X_t,y_t)
X_t= ReadData.to_num(X_t,max_featues)
data_set=(X_t, y_t)
with open("new-cut-1000lines-500-random-samples", 'wb') as fx:
    pickle.dump(data_set,fx)
# with open("y_20newx-train.pkl", 'wb') as fy:
#     pickle.dump(y_t,fy)  
os.system('PAUSE')  
X_train, X_test, y_train, y_test = train_test_split(X_t,y_t, test_size=0.2, random_state= 42 )
#word_index = reuters.get_word_index(path="reuters_word_index.pkl")  
#print(word_index["million"])
# dataset can be found here C:\Users\RenaiC\.keras\datasets
print(type( X_train)) 
#print(type(word_index))
# with open(r'word_index_data-json.txt','w+') as f1:
#     json.dump(word_index,f1, indent=4, sort_keys=False, separators=(',', ':'))
# os.system('pause')
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

nb_classes = np.max(y_train)+1
print(nb_classes, 'classes')

print('Vectorizing sequence data...')
# 参考 http://keras-cn.readthedocs.io/en/latest/preprocessing/text/
tokenizer = Tokenizer(nb_words=max_words) #Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类。
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,))) # 全连接层 ，输入（，max_words）；输出（，512）
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))# 只需要定义输出层个数，分类类别个数
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=1, validation_split=0.1)
score = model.evaluate(X_test, Y_test,
                       batch_size=batch_size, verbose=1)

from keras.utils.visualize_util import plot
plot(model, to_file='mlp-model.png')
print('Test score:', score[0])
print('Test accuracy:', score[1])
print (model.predict_classes(X_test,batch_size=batch_size))
print('---------------------------------')
print(Y_test)