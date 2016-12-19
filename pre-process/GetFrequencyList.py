#-*-coding:utf-8-*-
'''
Created on 2016年10月18日
包括：
tokenization http://stackoverflow.com/questions/10677020/real-word-count-in-nltk
@author: RENAIC225
'''
from collections import Counter
import json
#this section is to translate upper letters into lower one
num=[
          'zero','one','two','three',
          'four','five','six','seven',
          'eight','nine'
          ]
_FORMATLETTER = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
_LOWER_LETTER = 'abcdefghijklmnopqrstuvwxyz_'


def tokenization(txt):
    '简单分词'
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    #text = "This is my text. It icludes commas, question marks? and other stuff. Also U.S.."
    tokens = nltk.tokenizer.tokenize(txt)
    return tokens # list

def format_string(string):
    '对读入的一行string进行操作，大小写转换，去除符号，数字转换成英文单词等等'
    for c in string:
#         if c in '01213456789':
#         #replace number with its English word,add a blank
#             c_tmp = num[int(c)]+' '
#             string = string.replace(c,c_tmp)
#         if c.isupper():
#         #change upper letters into lower cases 
#             c_tmp = c.lower()
#             string = string.replace(c,c_tmp)
#         if c not in _LOWER_LETTER:
#         # remove all puctuations with 
#             string = string.replace(c, ' ')
        if c not in _FORMATLETTER:
            string = string.replace(c, ' ')
    retstring = ' '.join(string.split())  # 合并多余的空格
    return retstring

def split_txt(file_name):
    '拆分文本'
    f1 = open(file_name, "r")
    f2= open(file_name+"-01 .txt", "w+")
    count = 0
    #get number of lines in the file_name
    with open(file_name) as f:
        for x in f:
            count = count + 1
            #line = f.readline()
            f2.write(x)
            f2.write('\n')
#             if count > 20000:
#                 break
    f2.write('Num of lines:')
    f2.write(count)
    print count
    
    #get number of lines in the file_name
#     count1 = -1
#     for count1,line in enumerate(open(file_name,'rU')):  
#         pass  
#     count1 += 1  
#     print count1
    f1.close()
    f2.close()

def get_dictionary(file_path):
    f = open(file_path, "rb")
    wf = open(file_path+"-dictionary.txt", "w+")
    #context = f.readlines()
    my_counter = Counter()
    dic = {}
    for line in f:
        sentence = line.strip().split(" ")
        for word in sentence:
       
              #for word in Words:
            my_counter.update(word)
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1

    ok_dic = sorted(dic.items(), key=lambda e: e[1], reverse=True)
    print len(ok_dic)
    for item in ok_dic:
        wf.write("%s,%d\n" % (item[0], item[1]))
  
  
def get_text(path):
    #path = '..\data\BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1.log'
    f1 = open(path,'rb')
    #f2= open('..\\data\\BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1-dic.txt', "w+")
    #context = f1.readlines()#[1:100]
    my_counter = Counter()
    #with open('test.txt', 'r') as fin:
    #fdist.update(word_tokenize(fin.read().lower()))
    count = 0
    for line in f1:
        #if line == "":
        #   f2.write("\n")
        s = format_string(line)
        #if s != "":
        #   f2.write(s + "\n")
        s = s.strip().split(" ")
        my_counter.update(s)
        count = count +1
#         if count == 100:
#             break
    my_counter_order = sorted(my_counter.items(), key=lambda e: e[1], reverse=True)
    print "line num",count
    with open(path+'-dic.txt', 'w+') as handle:
         handle.writelines([
                            "%s %s\n" % item 
                            for item in my_counter.items()
                            ])
    
    #for k,v in my_counter.items():
    #    f2.write( k + str(v))
    
    
    # use json to save a dictionary 
    #with open('data.json', 'w') as outfile:
    #json.dump(data, outfile, indent=4, sort_keys=True, separators=(',', ':'))
    json.dump(my_counter, open(path+'-dic-jason.txt', 'w+'),indent=1,sort_keys = True, ) 
    #jason.dump() or jason.dumps(): the latter one creat a str,and you need to save the str again
    f1.close()
    #f2.close()
    '''
    while True:
        c=f1.read(1)
        #point=f1.tell()
        if len(c)<1:
            # end of file
            break
        if c.isupper():
            c=c.lower()
        if c in '01213456789':
            c=num[int(c)]+' '
        f2.write(c)
    f1.close()
    f2.close()
'''

'''
    for line in context:
        s = format_string(line)
        if s != "":
            wf.write(s + "\n")
def trans2lower(src_file="new_data_no_punctuation.txt",dest_file='new_data_no_punctuation_lower.txt'):
    f1 = open(src_file, 'r')
    f2 = open(dest_file, 'w+')
    while True:
        c=f1.read(1)
        #point=f1.tell()
        if len(c)<1:
            # end of file
            break
        if c.isupper():
            c=c.lower()
        f2.write(c)
    f1.close()
    f2.close()
'''  
# def num2word(src_file="new_data_no_punctuation.txt",dest_file='new_data_no_punctuation_num2word.txt'):
#     
# def remove_punctuation:(src_file="raw_data.txt",dest_file='new_data_no_punctuation.txt'):    
#  
if __name__ == "__main__":
    # trans2lower()
    src = r"H:\network_diagnosis_data\BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1_new_With_Tag.log"
    get_text(src)
    #split_txt('.\data\BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1.log')