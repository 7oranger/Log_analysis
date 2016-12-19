#-*-coding:utf-8-*-
from __future__ import print_function,division
'''
Created on 2016��11��20��
https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py
notes can be found here http://keras-cn.readthedocs.io/en/latest/blog/word_embedding/
https://kiseliu.github.io/2016/08/03/using-pre-trained-word-embeddings-in-a-keras-model/
https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
chines document http://keras-cn.readthedocs.io/en/latest/blog/word_embedding/
all of these can be found via Youdao noets
@author: RenaiC
'''
'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''


import os,sys,json,pickle
import numpy as np
import ReadData
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model


# GLOVE_DIR = BASE_DIR + '/glove.6B/'
GLOVE_DIR = r"H:\corpus_trained_model\glove.6B\\"

# TEXT_DATA_DIR = r"H:\corpus_trained_model\20_newsgroup\\"
# TEXT_DATA_DIR = r'H:\network_diagnosis_data\cut-1000'
TEXT_DATA_DIR = r'H:\corpus_trained_model\movie_reviews'

MAX_SEQUENCE_LENGTH = 500
MAX_NB_WORDS = 1000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {} # embedding��Ĵʵ�
f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
# f= open H:\corpus_trained_model\glove.6B
# construction dictionary
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')
# texts,labels,word_dict = ReadData.ReadRaw2HierData(TEXT_DATA_DIR,50)
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
N = 100
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path): # �Ƿ��� Ŀ¼
        label_id = len(labels_index)
        labels_index[name] = label_id
        j = 0
        for fname in sorted(os.listdir(path)):
            if j < N: 
                fpath = os.path.join(path, fname)
                f = open(fpath, 'r')
                texts.append(f.read())
                f.close()
                labels.append(label_id)
                j = j + 1

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
# ref http://keras-cn.readthedocs.io/en/latest/preprocessing/text/
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS) # ѡ��ǰ MAX_NB_WORDS����Ƶ��
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index # dict  like hello:23
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)#������ÿ���ĵ�����󵥴����������� nparray

labels = to_categorical(np.asarray(labels))# ת��� 0 1 ����
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set �ȴ��� ���֣�����Ҫ��ô�鷳�����������ط�ʵ�֣�
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index)) # ÿ�ζ��ٴʣ��ܴ�unique�����е�С��
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))# embedding ���� ����ʼ�� Ϊ0����û�еĵ��ʴ�������0
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        '��Ƶ��Ĳ�����'
        continue
    embedding_vector = embeddings_index.get(word) # �ôʶ�Ӧ�Ĵ�����
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector # ���б�ʾ�ôʵĴ�������ע������i��Ӧ�ض��Ĵ�

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
# ref http://keras-cn.readthedocs.io/en/latest/layers/embedding_layer/
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)# ע��weight ��������ʽ

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
class_num = len(np.unique(labels))
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
from keras.utils.visualize_util import plot
plot(model, to_file=r'.\data\pretrained-embedding-cnn-model.png')
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=1, batch_size=BATCH_SIZE)
print('Saving model')
from keras.utils.visualize_util import plot
data_today=time.strftime('%Y-%m-%d',time.localtime(time.time()))
plot(model, to_file=r'.\data\cnn-embedding-model'+data_today+'.png')
json_string = model.to_json()  #�ȼ��� json_string = model.get_config()  
open('.\data\cnn-embedding-model'+data_today+'.json','w+').write(json_string)    
model.save_weights('.\data\keras-cnn-embedding'+data_today+'.h5', overwrite=True)
# ������ͨ�����غ�ʱ̫��
#����Ҳ���Բ����������ʹ��Ԥ��ѵ���õĴ����������Ǵ�ͷ��ʼ��ʼ��Embedding�㣬
#��ѵ���Ĺ�����ѧϰ����ֵ��׼ȷ�ʻ���Σ�����ֻ��Ҫ�����еĴ����滻Embedding�㣺
# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             input_length=MAX_SEQUENCE_LENGTH)
