#-*-coding:utf-8-*-
'''
Created on 2016��10��18��
to test the created vord2vec model
@author: RENAIC225
'''
from gensim.models import word2vec
import logging

w2v_model = word2vec.Word2Vec.load("processed_data_v5.model")
y2 = w2v_model.most_similar("one", topn=20)  # 20������ص�
for word in y2:
    print word[0], word[1]
print "--------\n"