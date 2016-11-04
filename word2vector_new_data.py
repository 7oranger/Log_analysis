#-*-coding:utf-8-*-
'''
Created on 2016��10��18��

@author: RENAIC225
'''
from gensim.models import word2vec
import logging
#newfile=open("new_data_no_punctuation.txt","wb") 
# to remove all punctuation



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#sentences = word2vec.Text8Corpus("sample1.txt")  # ��������
sentences = word2vec.Text8Corpus("processed_data_v5.txt")  # ��������
model = word2vec.Word2Vec(sentences, size=50)  # ѵ��skip-gramģ��; Ĭ��window=5

# ���������ʵ����ƶ�/��س̶�
y1 = model.similarity("recv", "send")
print "siimiliraty between recv and send��", y1
print "--------\n"

# ����ĳ���ʵ���ش��б�
y2 = model.most_similar("recv", topn=2)  # 20������ص�
print "words related to recv��\n"
for item in y2:
    print item[0], item[1]
print "--------\n"
'''
# Ѱ�Ҷ�Ӧ��ϵ
print ' "boy" is to "father" as "girl" is to ...? \n'
y3 = model.most_similar(['enb ', 'ap'], ['boy'], topn=3)
for item in y3:
    print item[0], item[1]
print "--------\n"

more_examples = ["he his she", "big bigger bad", "going went being"]
for example in more_examples:
    a, b, x = example.split()
    predicted = model.most_similar([x, b], [a])[0][0]
    print "'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted)
print "--------\n"
'''
# Ѱ�Ҳ���Ⱥ�Ĵ�
y4 = model.doesnt_match("debug top mme mme nas init attach wait crt ses rsp timer stop for ue".split())
print u"outlier", y4
print "--------\n"

# ����ģ�ͣ��Ա�����
model.save("processed_data_v5.model")
# ��Ӧ�ļ��ط�ʽ
# model_2 = word2vec.Word2Vec.load("new_data_no_punctuation_lower.model")

# ��һ��C���Կ��Խ�������ʽ�洢������
model.save_word2vec_format("processed_data_v5.model.bin", binary=True)