#-*-coding:utf-8-*-
'''
Created on 2016��11��22��

@author: RenaiC
'''
import CreateFrequencyDict
import json
src = r'H:\network_diagnosis_data\cut'
d = CreateFrequencyDict.CreateFrequencyDict(src)
with open('word-list-jason.txt','w+') as f:
    json.dump(d, f, indent=2, sort_keys=False, separators=('.', ':'))