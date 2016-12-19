#-*-coding:utf-8-*-
'''
Created on 2016年11月21日
tutorial http://radimrehurek.com/gensim/models/word2vec.html
@author: RenaiC
'''
# import modules & set up logging
import gensim, logging, os, json, random
from multiprocessing import cpu_count
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# sentences = [['first', 'sentence'], ['second', 'sentence']]
# # train word2vec on the two sentences
# model = gensim.models.Word2Vec(sentences, min_count=1)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
class MySentences2Hier(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        first_hier = os.listdir(self.dirname)
        random.shuffle(first_hier)
        for folder_name in first_hier:
            new_folder_path = self.dirname + os.sep + folder_name
            second_hier = os.listdir(new_folder_path)
            random.shuffle(second_hier)
            for fname in second_hier:
                for line in open(os.path.join(new_folder_path, fname)):
                    yield line.split()
file_path = r'H:\corpus_trained_model\movie_reviews\all'
file_path = r'H:\network_diagnosis_data\all'

file_path = r'H:\network_diagnosis_data\cut1000\all'
file_path = r'H:\corpus_trained_model\movie_reviews_mixed\all'
file_path = r'H:\network_diagnosis_data\cut-500'
# sentences = MySentences(file_path) # a memory-friendly iterator
# sentences = gensim.models.word2vec.LineSentence(sentence)
sentences = MySentences2Hier(file_path) # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences,size= 50, iter= 10, window=5, min_count= 3, workers=cpu_count())
# model = gensim.models.Word2Vec(sentences,size= 50, iter= 10, window=5, min_count=5,sorted_vocab=1, workers=cpu_count())
model.save(r'.\data\movie_review-50D-word-vector')  
# model.sort_vocab()
# model.build_vocab(sentences)
# vocab = list(model.vocab.keys())
# with open(r'.\data\cut1000-all-wordlist-w2v.json',"w+") as f:
#     json.dump(vocab,f)
# model.init_sims(replace=True)  
# model.save(r'.\data\cut1000-all-50D-w2v')  
# a='ap'
# print model[a]
#model.build_vocab(some_sentences)  # can be a non-repeatable, 1-pass generator
#model.train(other_sentences)  # can be a non-repeatable, 1-pass generator
# model = gensim.models.Word2Vec(sentences, min_count=10)  # default value is 5
# model.save('outp1-mov')  
# model = gensim.models.Word2Vec(sentences, size=200)  # default value is 100
# model.save('outp2-mov')  
# model = gensim.models.Word2Vec(sentences, workers=4) # default = 1 worker = no parallelization
# model.save('outp3-mov')  
## Memory
#model.accuracy('/tmp/questions-words.txt')
#new_model = gensim.models.Word2Vec.load('/tmp/mymodel')
# 载入C训练的模型
#model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)
#using gzipped/bz2 input works too, no need to unzip:
#model = Word2Vec.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)
#online training,  Resuming training
#model = gensim.models.Word2Vec.load('/tmp/mymodel')
#model.train(more_sentences)
#Using the model
#########################################################
##train a wiki model

# # -*- coding: utf-8 -*-  
# import logging  
# import os.path  
# import sys  
# import multiprocessing  
#    
# from gensim.corpora import  WikiCorpus  
# from gensim.models import Word2Vec  
# from gensim.models.word2vec import LineSentence  
#    
#    
# if __name__ == '__main__':  
#     program = os.path.basename(sys.argv[0])  
#     logger = logging.getLogger(program)  
#    
#     logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')  
#     logging.root.setLevel(level=logging.INFO)  
#     logger.info("running %s" % ' '.join(sys.argv))  
#    
#     # check and process input arguments  
#    
#     if len(sys.argv) < 3:  
#         print globals()['__doc__'] % locals()  
#         sys.exit(1)  
#     inp, outp = sys.argv[1:3]  
#    
#     model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())  
#    
#     # trim unneeded model memory = use (much) less RAM  
#     model.init_sims(replace=True)  
#    
#     model.save(outp)  