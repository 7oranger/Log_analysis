#-*-coding:utf-8-*-
'''
Created on 2016Äê10ÔÂ18ÈÕ

@author: RENAIC225
'''

#this section is to translate upper letters into lower one
num=[
          'zero','one','two','three',
          'four','five','six','seven',
          'eight','nine'#,'ten'
          ]
FORMATLETTER = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
_LOWER_LETTER = 'abcdefghijklmnopqrstuvwxyz'

def format_string(string):
    for c in string:
        if c in '01213456789':
        #replace number with its English word,add a blank
            c_tmp = num[int(c)]+' '
            string = string.replace(c,c_tmp)
        if c.isupper():
        #change upper letters into lower cases 
            c_tmp = c.lower()
            string = string.replace(c,c_tmp)
        if c not in _LOWER_LETTER:
        # remove all puctuations with 
            string = string.replace(c, ' ')
    retstring = ' '.join(string.split())
    return retstring
def split_txt(file_name):
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
    print count
    
    #get number of lines in the file_name
#     count1 = -1
#     for count1,line in enumerate(open(file_name,'rU')):  
#         pass  
#     count1 += 1  
#     print count1
    f1.close()
    f2.close()
    
    
def get_text():
    f1 = open("raw_data.log", "r")
    f2= open("processed_data_v5.txt", "w+")
    context = f1.readlines()#[1:100]
    for line in context:
        if line == "":
            f2.write("\n")
        s = format_string(line)
        if s != "":
            f2.write(s + "\n")
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
    # [220:1725]+[1731:13425]  
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
def num2word(src_file="new_data_no_punctuation.txt",dest_file='new_data_no_punctuation_num2word.txt'):
    
def remove_punctuation:(src_file="raw_data.txt",dest_file='new_data_no_punctuation.txt'):    
'''   
if __name__ == "__main__":
    # trans2lower()
    #get_text()
    split_txt('.\data\BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1.log')