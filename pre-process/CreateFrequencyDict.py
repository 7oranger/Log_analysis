#-*-coding:utf-8-*-
'''
Created on 2016年11月22日

@author: RenaiC
'''
from __future__ import division 
import os
import nltk
import re
import pprint
import json
def CreateFrequencyDict(src_dir):
    'Two-level directory, returns the dict table of frequencies'
# 两级别目录下文件的所有词:注意，是两级目录
    folder_path = src_dir
    folder_list = os.listdir(folder_path)
    all_words = {}
    for i in range(len(folder_list)):
        new_folder_path = folder_path + '\\' + folder_list[i]
        files = os.listdir(new_folder_path)
        for file in files:
            fobj = open(new_folder_path+'\\'+file, 'r')
            raw = fobj.read()
            #word_list = nltk.tokenize(raw)
            #word_list = re.split(r'\W+',raw)[0:-100] # split, remain a-z0-9 plu _，waive the last 100 items 
            word_list = re.split(r'\W+',raw)
            for word in word_list: # made word list 
                #if word in all_words.keys(): # takes too long, use the latter expression
                if all_words.has_key(word):
                    all_words[word] += 1
                else:
                    all_words[word] = 0   
    ## 根据word的词频排序
    all_words_list = sorted(all_words.items(), key=lambda e:e[1], reverse=True)
    return all_words_list #返回降序词典(type list)  [['sample1',5],['sample2',11]]


# H:\corpus_trained_model\ldagibbs-master\dataset\newsgroup20
if __name__ == "__main__":
    # trans2lower()
    src = r'H:\network_diagnosis_data\cut'
    src = r'H:\network_diagnosis_data\cut1000'
    dicc = CreateFrequencyDict(src)
    print type(dicc)
    with open(os.path.join(src, 'FreDict-json.txt'),"w+") as f:
        json.dump(dicc, f, indent=4, sort_keys=False, separators=('.', ':'))
              
    
   