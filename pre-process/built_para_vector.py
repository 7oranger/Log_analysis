#-*-coding:utf-8-*-
'''
Created on 2016年11月21日
tutorial http://radimrehurek.com/gensim/models/word2vec.html
'使用word2vec来构建文本向量'
@author: RenaiC
'''
# import modules & set up logging
from numpy import average
import gensim, logging, os, json, pickle,ReadData
from multiprocessing import cpu_count
from sklearn.linear_model import LogisticRegression
import logging,sys,numpy
import pprint,json,pickle,os
import ReadData
from sklearn.cross_validation import train_test_split  
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def average_word_vec(model_path,x):
    'x:each line is a sample,has been tokenized x should be a lilst or np array'
    model = gensim.models.Word2Vec.load(model_path)
    vocab = list(model.vocab.keys()) # model.vocab 是一个dict了 存储例子： raining： Vocab(count:7, index:11550, sample_int:4294967296L)
    vd= model.vocab
    # for key in vd.keys():
    #     print key,vd[key]
    #     os.system('pause')
    all_words =model.index2word  
    # prepare a dictionary for these vector:
    word_vec = {}
    for item in all_words:
        #if word in all_words.keys(): # takes too long, use the latter expression
        if not word_vec.has_key(item):
            word_vec[item] = model[item]  
    dim = len(model[vocab[1]])
    # 将每行内容平均，得到每行的代表
    new_x = []
    for i in xrange(len(x)):
        len_sample = len(x[i])
        weight = 1./len_sample
        xx = [0]*dim
        for item in x[i]:
            if word_vec.has_key(item):
                '否则 不操作，类似于加0'
                xx = xx + word_vec[item]
        xx = [tt*weight for tt in xx]
        new_x.append(xx)
    
    return new_x
def test_log(x,y):
    y_t= numpy.array(y)
    X_t= numpy.array(x)
    print X_t[0],'\ty:', y_t[0]
    print type(X_t),type(y_t)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_t,y_t, test_size=0.2, random_state= 42 )
    print(len(X_train), 'train samples')
    print(len(X_test), 'test samples')
    print 'Fitting'
    classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, penalty='l2', random_state=None, tol=0.001)
    classifier.fit(X_train,y_train)
    print classifier.score( X_test, y_test)
        
if __name__ == '__main__':
    model_path = r'.\data\movie_review-50D-word-vector'
    model_path = r'.\data\cut1000-all-50D-w2v'
    file_path =r'H:\network_diagnosis_data\cut-500'
    x,y,d=ReadData.ReadRaw2HierData(file_path,5000)
    new_x = average_word_vec(model_path,x)
    test_log(new_x, y)
    new_data = [new_x,y]
    print new_x[0]
    pickle.dump( new_data, open(r'.\data\w2v_replaced-500samples.pkl','wb') )