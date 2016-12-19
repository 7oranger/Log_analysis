#-*-coding:utf-8-*-

from gensim.models import doc2vec
from collections import namedtuple

# Load data

doc1 = ["This is a sentence", "This is another sentence"]
# Transform data (you can add more data preprocessing steps) 
docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(doc1):
    words = text.lower().split()
    print words
    tags = ['test']
#     tags = ['hello'+str(i)]
    print tags
    docs.append(analyzedDocument(words, tags))
# Train model (set min_count = 1, if you want the model to work with the provided example data set)
model = doc2vec.Doc2Vec(docs, size = 100, window = 300, min_count = 1, workers = 4)

# Get the vectors
print model.docvecs['test']
# print model.docvecs['hello0']
print model.docvecs[0]
print model.docvecs[1]  # 这两种方法的结果是一样 的
# 标签必须 unique？
'''
sentences=doc2vec.TaggedLineDocument(file_path)
model = doc2vec.Doc2Vec(sentences,size = 100, window = 300, min_count = 10, workers=4)
docvec = model.docvecs[99] 
'''