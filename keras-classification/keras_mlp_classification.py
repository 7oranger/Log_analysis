#-*-coding:utf-8-*-
'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''

from __future__ import print_function
import numpy as np
import pprint,json,pickle,os,time
import ReadData
from sklearn.cross_validation import train_test_split  
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

max_words = 10000#1000
max_feature= 300
batch_size = 32
nb_epoch = 5
vec_dim = 50
# min_count = 5
np.random.seed(1337)  # for reproducibility
def mlp():
    max_words = 10000#1000
    max_feature= 300
    batch_size = 32
    nb_epoch = 5
    vec_dim = 50
    print('Loading data...')
    folder_path = r'H:\network_diagnosis_data\cut-500'
    X_t, y_t,dicc=ReadData.ReadRaw2HierData(folder_path,3000)
    X_t,Y_t=ReadData.shuffle_X_Y(X_t,y_t)
    X_t= ReadData.to_num(X_t,max_feature)
    X_train, X_test, y_train, y_test = train_test_split(X_t,y_t, test_size=0.2, random_state= 42 )
    
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
    model.add(Dropout(0.3))
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
    
    print('Saving model...')
    from keras.utils.visualize_util import plot
    data_today=time.strftime('%Y-%m-%d',time.localtime(time.time()))
    plot(model, to_file=r'.\data\mlp-model'+data_today+'.png')
    json_string = model.to_json()  #等价于 json_string = model.get_config()  
    open('.\data\mlp-model'+data_today+'.json','w+').write(json_string)    
    model.save_weights('.\data\keras-mlp'+data_today+'.h5', overwrite=True)
    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
#     print (model.predict_classes(X_test,batch_size=batch_size))
    print('---------------------------------')
#     print(Y_test)
def mlp_w2v():
    file_path = r'.\data\w2v_replaced-500samples.pkl'
    t = pickle.load(open(file_path,'rb'))
    
    X_t=t[0]
    y_t=t[1]
    X_t = np.array(X_t)
    X_train, X_test, y_train, y_test = train_test_split(X_t,y_t, test_size=0.2, random_state= 42 )
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    
    nb_classes = np.max(y_train)+1
    print(nb_classes, 'classes')
    print(len(X_train[1]))
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    
    print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)
    
    print('Building model...')
    model = Sequential()
    model.add(Dense(512, input_dim=vec_dim)) # 全连接层 ，输入（，max_words）；输出（，512）
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
    
    print('Saving model...')
    from keras.utils.visualize_util import plot
    data_today=time.strftime('%Y-%m-%d',time.localtime(time.time()))
    plot(model, to_file=r'.\data\mlp-w2v-model'+data_today+'.png')
    json_string = model.to_json()  #等价于 json_string = model.get_config()  
    open('.\data\mlp-model'+data_today+'.json','w+').write(json_string)    
    model.save_weights('.\data\keras-mlp-w2v'+data_today+'.h5', overwrite=True)
    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
#     print (model.predict_classes(X_test,batch_size=batch_size))


if __name__ == '__main__':
#     mlp()
    mlp_w2v()