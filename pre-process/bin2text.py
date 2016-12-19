#-*-coding:utf-8-*-
'''
http://stackoverflow.com/questions/27324292/convert-word2vec-bin-file-to-text
convert a word2vec binary file to a readable text 
'''
import codecs
from gensim.models import Word2Vec
import json

import gensim, logging, os, json, pickle,ReadData
from multiprocessing import cpu_count
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def main():
    path_to_model = r'.\data\cut1000-all-50D-w2v'
    output = r'.\data\vector_list_cut1000-all-50D-w2v.txt'
    w2v_to_txt(path_to_model,output)
#     export_to_file(path_to_model, output)

def w2v_to_txt(file_path,output):
    '将 word2vec模型存成文本'
    '格式为 hello 0 2 3.1 '
    model = gensim.models.Word2Vec.load(file_path)
    vocab = list(model.vocab.keys()) # model.vocab 是一个dict了 存储例子： raining： Vocab(count:7, index:11550, sample_int:4294967296L)
    vd= model.vocab
    # for key in vd.keys():
    #     print key,vd[key]
    #     os.system('pause')
    all_words =model.index2word 
    # prepare a dictionary for these vector:
    word_vec = {}
    for item in all_words:
        #if word in all_words.keys(): # takes too long, use the latter expression
        if not word_vec.has_key(item):
            word_vec[item] = model[item]  
#     with open(output+'.pkl',"w+") as ff:
#         pickle.dump(word_vec,ff)
    with open(output,"w+") as f:
        for d,x in word_vec.items():
            f.write(d+' ')
            tt=''
            xx =  [ str(i) for i in x ]
            tt= ' '.join(xx)
            f.write(tt)
            f.write('\n')
            
def export_to_file(path_to_model, output_file):
    output = codecs.open(output_file, 'w' , 'utf-8')
    model = Word2Vec.load_word2vec_format(path_to_model, binary=True)
    print('done loading Word2Vec')
    vocab = model.vocab
    for mid in vocab:
        #print(model[mid])
        #print(mid)
        vector = list()
        for dimension in model[mid]:
            vector.append(str(dimension))
        #line = { "mid": mid, "vector": vector  }
        vector_str = ",".join(vector)
        line = mid + "\t"  + vector_str
        #line = json.dumps(line)
        output.write(line + "\n")
    output.close()

if __name__ == "__main__":
    main()
    #cProfile.run('main()') # if you want to do some profiling