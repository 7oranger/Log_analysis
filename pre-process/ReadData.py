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
import random  #random.shuffle (lst or tuple )
from random import shuffle


def LoadOriData(src):
    '��ȡ���ݸ�ʽ��20ng-train-stemmed.txt��20ng-test-stemmed.txt'
    x_set = []
    y_set = []
    with open(src,'r') as f:
        for x in f:
            #os.system('pause')
            index = x.find('\t')
            y_set.append(x[0:index])
            word_list = re.split(r'\W+',x[index+1:])
            #y_set.append(word_list[0]+word_list[1])
            #word_list = re.split(r'\W+',x)
            x_set.append(word_list)
            #y_set.append(word_list[0]+word_list[1])
    yy = set(y_set)
    n=[]
    for i in yy:
        n.append(i)
    for i in xrange(len(y_set)):
        indexx = n.index(y_set[i])
        y_set[i] = indexx
        
    return x_set, y_set
    
def ReadRaw2HierData(src,n):
    'src������Ŀ¼src-kindA-textA������ X,Y(list),dict'
    'ע������ļ�̫���ڴ�ᱬ������'
    folder_path = src
    folder_list = os.listdir(folder_path)
#     random.shuffle(folder_list)
#     print type (folder_list)
#     os.system('pause')
    class_list = [] ##������[0,1,...]�������ļ�����
    nClass = 0 # classtype: 0 1 2 3 etc. 
    x_set = []
    y_set = []
    all_words = {}
    class_num = len(folder_list) # �м��࣬�м���Ŀ¼
    for i in range(class_num): # different category
        new_folder_path = folder_path + '\\' + folder_list[i]
        if os.path.isdir(new_folder_path): # �Ƿ��� Ŀ¼
            files = os.listdir(new_folder_path)
            random.shuffle(files)
            class_label = str(folder_list[i]) #  or use nClass
            
            j = 0 
            for file in files: # different sample 
                if j < n: 
                    fobj = open(new_folder_path+'\\'+file, 'r')
                    #raw = fobj.readlines()[0:-20]
                    #raw = fobj.read()[100:-100] # discard the beginning and the ending 
                    raw = fobj.read()# discard the beginning and the ending 
                    #word_list = nltk.tokenize(raw)
                    #word_list = re.split(r'\W+',raw)[0:-100] # split, remain a-z0-9 plu _��waive the last 100 items   
                    word_list = re.split(r'\W+',raw) # split, remain a-z0-9 plu _��waive the last 100 items   
                    for word in word_list: # made word list 
                        #if word in all_words.keys(): # takes too long, use the latter expression
                        if all_words.has_key(word):
                            all_words[word] += 1
                        else:
                            all_words[word] = 1    # ��0��Ϊ1  
                    
                    x_set.append(word_list)
                    y_set.append(nClass)   
                    j += 1
            
        nClass += 1
    #all_words_list = sorted(all_words.items(), key=lambda e:e[1], reverse=True) # to sort the list 
    return x_set, y_set, all_words  # , class_num 
def to_num(src,nb_words):
    'src= list,each item is a paragraph'
    '2016-11-26���������дʣ������޸ı����Ƶ��,���� nb_words�����ʣ���������޸�Ϊ����Ƶ��������'
    '��� 0  �������κδʣ�1 ��Ƶ����ߵĴ�'
    all_words = {}
    for line in src:
        for word in line:
            if all_words.has_key(word):
                all_words[word] += 1
            else:
                all_words[word] = 1    # new word:1��0��Ϊ1     
                
    all_words_list = sorted(all_words.items(), key=lambda e:e[1], reverse=True)
    di = []
    di.append(None)
    for x in all_words_list:
        di.append(x[0])  

    word_dic = dict(enumerate(di))
    word_dic_n={value:key for key, value in word_dic.items()}#������޸ģ�������Ƶ�� ,����ָ��nb_word֮��value �滻Ϊ0
    if isinstance(nb_words, int):
        for key in word_dic_n:
            if word_dic_n[key] >= nb_words:
                word_dic_n[key] = 0
                
    height = len(src)
    for x in xrange(height):
        width = len(src[x])
        for y in xrange(width):
            src[x][y] = word_dic_n[src[x][y]]
    return src
    #return src,word_dic_n return the dictioinary at the same time 
    #num_mat= numpy.zeros(height,nb_words)

def shuffle_X_Y(x,y):  
    'input:corresonding x and y,return the shuffled ones ' 
    data_set=[]
    for i in xrange(len(y)):
        data_set.append( (x[i],y[i]) )   
#     pprint.pprint(data_set)
#     z=[x,y]
#     with open('test-json--.txt','w+') as f1:
#         #json.dump(data_set,f1, indent=4, sort_keys=False, separators=(',', ':'))
#         json.dump(data_set,f1)
  
#     for i in xrange(len(data_set)):
#         pprint.pprint(data_set[i][1])
    shuffle(data_set)
#     with open('test-json-shuffled--.txt','w+') as f2:
#         #json.dump(data_set,f2, indent=4, sort_keys=False, separators=(',', ':'))
#         json.dump(data_set,f2)
    
#     print '--------------','\n',len(data_set)
    x_n = []
    y_n = []
    for i in xrange(len(data_set)):
        x_n.append(data_set[i][0])
        y_n.append(data_set[i][1])
    
    return x_n, y_n   

if __name__ == "__main__":
    
    src = r'H:\network_diagnosis_data\cut'
    src =r'H:\corpus_trained_model\movie_reviews'
    x,y,dic=ReadRaw2HierData(src,5)
    print y
    x_n,y_n = shuffle_X_Y(x,y)  
    print y_n
#     x_data,y_data,word_list = ReadRaw2HierData(src)
#     with open('cut-FreDict-json.txt',"w+") as f:
#         json.dump(word_list, f, indent=4, sort_keys=False, separators=('.', ':'))
#     src = r"H:\corpus_trained_model\ldagibbs-master\dataset\newsgroup20\20ng-test-stemmed.txt"
#     x , y = LoadOriData(src)
#     x,y,dic=ReadRaw2HierData(src,5)
#     pprint.pprint(dic)
#     f=open("x_movies-review.pkl",'rb')
#     src= pickle.load( f ) 
#     mat= to_num(src,2000)
#     pprint.pprint(mat)
    #print set(y)
