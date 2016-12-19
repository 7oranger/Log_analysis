#-*-coding:utf-8-*-
from __future__ import division
from gensim.models.doc2vec import Doc2Vec,TaggedLineDocument
# import numpy.linalg
import numpy as np
import math
import scipy
from PIL import Image,ImageDraw
input_file = r"H:\network_diagnosis_data\test\GTPC_TUNNEL_PATH_BROKEN.3054.txt"
sentences = TaggedLineDocument(input_file)
dim = 1000
model = Doc2Vec(alpha=0.025, min_alpha=0.025,size = dim) # default 300 ά
model.build_vocab(sentences)

for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
model.save(r'.\data\test_d2v')
# print model.infer_vector([u'people', u'like', u'words'])

total_num = model.docvecs.count
# print total_num
# print len( model.docvecs[0] )
para_vec = []
for i in xrange(total_num):
    if i == 0:
        para_vec = model.docvecs[i];
        continue
    para_vec = np.vstack((para_vec, model.docvecs[i]))
print  para_vec
print para_vec.shape
print np.linalg.norm(para_vec,1)
print np.max(para_vec),np.min(para_vec) # 0.114164 -0.110262
'''
Generate a image and save it , and then do FFT, save the fft iamge 
def data_shift(src):
    data_in = np.matrix(src)
    min_val = np.min(data_in)
    data_in = data_in - np.abs(min_val)*np.sign(min_val) 
    max_val = np.max(data_in)
    data_in = data_in * 255 /max_val
    return np.ceil(data_in)


data_fft = np.fft.fft2(para_vec) 
im_data = data_shift( np.absolute(data_fft) )
im_fft = Image.fromarray((im_data))
# im_fft.show()
# im_data = np.matrix(para_vec)
# data = np.reshape(data,(512,512))
new_im = Image.fromarray(data_shift(para_vec))
# new_im.show()
if new_im.mode != 'RGB':
    new_im = new_im.convert('RGB')
if im_fft.mode != 'RGB':
    im_fft = im_fft.convert('RGB')
# with open('GTPC_TUNNEL_PATH_BROKEN-3054.bmp','w') as f:
#     new_im.save(f,format='bmp')
# 
# new_im.save('.\data\GTPC_TUNNEL_PATH_BROKEN.3054.png',format='png')
# im_fft.save('.\data\GTPC_TUNNEL_PATH_BROKEN.3054-fft.png',format='png')
'''
