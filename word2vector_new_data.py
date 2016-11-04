#-*-coding:utf-8-*-
'''
Created on 2016年10月18日

@author: RENAIC225
'''
from gensim.models import word2vec
import logging
#newfile=open("new_data_no_punctuation.txt","wb") 
# to remove all punctuation



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#sentences = word2vec.Text8Corpus("sample1.txt")  # 加载语料
sentences = word2vec.Text8Corpus("processed_data_v5.txt")  # 加载语料
model = word2vec.Word2Vec(sentences, size=50)  # 训练skip-gram模型; 默认window=5

# 计算两个词的相似度/相关程度
y1 = model.similarity("recv", "send")
print "siimiliraty between recv and send：", y1
print "--------\n"

# 计算某个词的相关词列表
y2 = model.most_similar("recv", topn=2)  # 20个最相关的
print "words related to recv：\n"
for item in y2:
    print item[0], item[1]
print "--------\n"
'''
# 寻找对应关系
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
# 寻找不合群的词
y4 = model.doesnt_match("debug top mme mme nas init attach wait crt ses rsp timer stop for ue".split())
print u"outlier", y4
print "--------\n"

# 保存模型，以便重用
model.save("processed_data_v5.model")
# 对应的加载方式
# model_2 = word2vec.Word2Vec.load("new_data_no_punctuation_lower.model")

# 以一种C语言可以解析的形式存储词向量
model.save_word2vec_format("processed_data_v5.model.bin", binary=True)