#-*-coding:utf-8-*-
'''
Created on 2016年11月30日
tutorial http://radimrehurek.com/gensim/models/word2vec.html
https://rare-technologies.com/doc2vec-tutorial/
@author: RenaiC
'''
# import modules & set up logging
import gensim, logging, os,pickle
from gensim.models import doc2vec
from multiprocessing import cpu_count
from gensim.models.doc2vec import TaggedDocument,Doc2Vec,LabeledSentence
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
KK = 0
FILE_COUNT = -1
sentence_map={}
class MyDocuments(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        global KK,FILE_COUNT
        for fname in os.listdir(self.dirname):
            k = 0
            FILE_COUNT = FILE_COUNT + 1 
            if FILE_COUNT < 100:
                for line in open(os.path.join(self.dirname, fname)):
                     words = line.split()
                     file_type = fname.split('.')[0]
                     file_no = fname.split('.')[1]
                     #yy = TaggedDocument(words,['SENT_%s' % k,file_type,file_no])  # one line each, how to use the later tag?
                     #yy = TaggedDocument(words,['SENT_%s' % k+str(file_type)+str(file_no)])  # one line each,works well
                     label = 'SENT_%s' % k+str(file_type)+str(file_no)
                     sentence_map[label] = KK
                     yy = TaggedDocument(words,[KK])
    #                  print yy
    #                  os.system('pause')
                     k = k  + 1
                     KK = KK+1
                     yield yy
# class LabeledLineSentence(object): # doesn't work 
#     def __init__(self, filename):
#         self.filename = filename
#     def __iter__(self):
#         for uid, line in enumerate(open(filename)):
#             print uid
#             print line
#             words=line.split()
#             tt = TaggedDocument(words, ['SENT_%s' % uid])
#             print tt
#             os.system('pause')
#             yield t
file_path = r'H:\corpus_trained_model\movie_reviews_mixed\all'   
file_path = r'H:\network_diagnosis_data\cut-1000\gtpc'   
file_path =r'H:\network_diagnosis_data\test' #test two sample     
file_path = r'H:\network_diagnosis_data\cut-1000-all\all'
dim = 100
# sentences = LabeledLineSentence(file_path+'\cv000_29416.txt')
# sentence = LabeledSentence(words=[u'some', u'words', u'here'], labels=[u'SENT_1'])            
sentences = MyDocuments(file_path)
model = Doc2Vec(sentences,alpha=0.025,size = dim, min_alpha=0.005, workers=cpu_count())  # use fixed learning rate
# model = Doc2Vec(sentences)
# store the model to mmap-able files
model.save('./data/all-line-tagged.doc2vec')
pickle.dump(sentence_map,open('./data/all-line-tagged.doc2vec.setence_map.pkl','wb'))
# load the model back
model_loaded = Doc2Vec.load('./data/all-line-tagged.doc2vec')

print '模型和句子字典已保存'
# voc= model.vocab
# print list(voc.keys())
# ind = model.index2word
# print ind  # same as word2vec
# print model.docvecs['SENT_0']
# print  model.docvecs['SENT_1',0]
print  '总共句子向量个数:',model.docvecs.count 
# print model.most_similar('label')
# i = model.docvecs['SENT_1']
# ii = model.docvecs[1] # 1 not  same;  0 the same  
# print i-ii
# print model.docvecs['SENT_0','GTPC_TUNNEL_PATH_BROKEN',3] #一次print 3个向量出来
