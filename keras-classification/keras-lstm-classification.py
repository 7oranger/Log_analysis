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
已跑通
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
import ReadData,pickle,os,time
from keras.utils.np_utils import to_categorical
max_features = 250 
maxlen = 5000  # cut texts after this number of words (among top max_features most common words)
batch_size = 10
num_epoch = 8
N = 600 # num of samples(for one class)
Embedding_Dim = 50
def lstm_raw():
    print('Loading data...')
    folder_path = r'H:\network_diagnosis_data\cut-1000'
    
    X_t, y_t,dicc=ReadData.ReadRaw2HierData(folder_path,N)
    nb_classes = np.max(y_t)+1
    X_t = ReadData.to_num(X_t,max_features)
    X_train, X_test, y_train, y_test = train_test_split(X_t,y_t, test_size=0.2, random_state= 42 )
    
    print('Pading sequences ')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)#padding = 'post'
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)#truncating = 'post'
    y_train = to_categorical (y_train,nb_classes)
    y_test = to_categorical (y_test,nb_classes)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    
    print('Building model...')
    
    model = Sequential()
    model.add(Embedding(max_features, Embedding_Dim, dropout=0.2))
    model.add(LSTM(Embedding_Dim, dropout_W=0.2, dropout_U=0.2))  # 
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy', #binary_crossentropy
                  optimizer='adam',
                  metrics=['accuracy'])
    
    from keras.utils.visualize_util import plot
    plot(model, to_file=r'.\data\lstm-model.png')
    print('Training...')
#     model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=num_epoch,
#               validation_data=(X_test, y_test))
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=num_epoch,
              validation_split=0.1,verbose=1)
    
    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    from keras.utils.visualize_util import plot
    data_today=time.strftime('%Y-%m-%d',time.localtime(time.time()))
    plot(model, to_file=r'.\data\lstm-model'+data_today+'.png')
    json_string = model.to_json()  #等价于 json_string = model.get_config()  
    open('.\data\lstm-model'+data_today+'.json','w+').write(json_string)    
    model.save_weights('.\data\keras-lstm'+data_today+'.h5', overwrite=True)
    print('model saved')
def lstm_w2v():
    print('Loading data...')
    file_path = r'.\data\w2v_replaced-1000samples.pkl'
    t = pickle.load(open(file_path,'rb'))
    X_t=t[0]
    y_t=t[1]
    X_t = np.array(X_t)
    nb_classes = np.max(y_t)+1
    X_train, X_test, y_train, y_test = train_test_split(X_t,y_t, test_size=0.2, random_state= 42 )
    y_train = to_categorical (y_train,nb_classes)
    y_test = to_categorical (y_test,nb_classes)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    
    print('Building model...')
    
    model = Sequential()
    model.add(Embedding(max_features, Embedding_Dim, dropout=0.2))
    model.add(LSTM(Embedding_Dim, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',#,categorical_crossentropy
                  optimizer='adam',
                  metrics=['accuracy'])
    
    from keras.utils.visualize_util import plot
    plot(model, to_file=r'.\data\lstm-model.png')
    print('Training...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=num_epoch,
              validation_data=(X_test, y_test))
    
    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    from keras.utils.visualize_util import plot
    data_today=time.strftime('%Y-%m-%d',time.localtime(time.time()))
    plot(model, to_file=r'.\data\lstm-model'+data_today+'.png')
    json_string = model.to_json()  #等价于 json_string = model.get_config()  
    open('.\data\lstm-model'+data_today+'.json','w+').write(json_string)    
    model.save_weights('.\data\keras-lstm'+data_today+'.h5', overwrite=True)
    print('model saved')

if __name__ == '__main__':
    lstm_raw()
#     lstm_w2v()
