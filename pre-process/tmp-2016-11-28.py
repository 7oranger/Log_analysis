#-*-coding:utf-8-*-
'''
Created on 2016��11��28��

@author: RenaiC
'''
import gensim, logging, os
model_path = r'all-default-w2v'
model = gensim.models.Word2Vec.load(model_path)
ap=model['ap']
print ap,len(ap)