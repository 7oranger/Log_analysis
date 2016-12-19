#-*-coding:utf-8-*-
'''
Created on 2016年11月27日

@author: RenaiC
'''
#-*-coding:utf-8-*-
'''
Created on 2016年11月21日
tutorial http://radimrehurek.com/gensim/models/word2vec.html
@author: RenaiC
'''
# import modules & set up logging
import gensim, logging, os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
file_path = r'H:\network_diagnosis_data\cut\GTPC'
model1 = gensim.models.Word2Vec.load('outp1')
print model1['ap']
model2 = gensim.models.Word2Vec.load('outp2')
print model2['ap']
model3 = gensim.models.Word2Vec.load('outp3')
print model3['ap']

# sentences = MySentences(file_path) # a memory-friendly iterator
# model = gensim.models.Word2Vec(sentences)
# model.save('GTPC_WORD2VEC.TXT')  
# model = gensim.models.Word2Vec(iter=1)  # an empty model, no training yet
#model.build_vocab(some_sentences)  # can be a non-repeatable, 1-pass generator
#model.train(other_sentences)  # can be a non-repeatable, 1-pass generator
# model = gensim.models.Word2Vec(sentences, min_count=10)  # default value is 5
# model.save('outp1')  
# model = gensim.models.Word2Vec(sentences, size=200)  # default value is 100
# model.save('outp2')  
# model = gensim.models.Word2Vec(sentences, workers=4) # default = 1 worker = no parallelization
# model.save('outp3')  