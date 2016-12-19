#-*-coding:utf-8-*-
'''
Created on 2016��11��22��
basically these are pre-processing operations 
 ReadRaw2HierData(����Ŀ¼��
 word 2 num ת��Ϊ��Ƶ����Ƶ��ߵ�Ϊ 1��������Ϊ 2��...�������������С��Ƶ��
 �򽫴�Ƶ̫�ٵĴ� תΪΪ 0

@author: RenaiC
'''
from __future__ import division 
import os
import nltk
import re
import time 
import pprint
import json,numpy
import pickle,cPickle
import random  #random.shuffle (list or tuple )
from random import shuffle

def formatting(src, dst):
    fdst = open(dst, "w+")
    files = os.listdir(src)
    for file in files:  # different sample 
        fobj = open(src +os.sep + file, 'r')
        raw = fobj.read()  # discard the beginning and the ending 
        fdst.write(raw)
        fdst.write('\n')
        fobj.close()
    
    fdst.close()

if __name__ == "__main__": 
    src =r'H:\corpus_trained_model\aclImdb\train\neg'
    dst = r'H:\corpus_trained_model\aclImdb\train-neg.txt'
    formatting(src,dst)
    src = r'H:\corpus_trained_model\aclImdb\train\pos'
    dst = r'H:\corpus_trained_model\aclImdb\train-pos.txt'
    formatting(src,dst)
    src = r'H:\corpus_trained_model\aclImdb\test\neg'
    dst = r'H:\corpus_trained_model\aclImdb\test-neg.txt'
    formatting(src,dst)
    src = r'H:\corpus_trained_model\aclImdb\test\pos'
    dst = r'H:\corpus_trained_model\aclImdb\test-pos.txt'
    formatting(src,dst)
    src = r'H:\corpus_trained_model\aclImdb\train\unsup'
    dst = r'H:\corpus_trained_model\aclImdb\train-unsup.txt'
    formatting(src,dst)
   
