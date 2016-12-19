#!/usr/bin/python2.7

"""
To get started, save this file to a directory using the link above, then run:

chmod +x embeddings.py
mkdir data
./embeddings.py -d data --demo
"""

from __future__ import print_function

import json
import os
import sys
import argparse
import numpy as np

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from keras.engine import Input
from keras.layers import Embedding, merge
from keras.models import Model

# tokenizer: can change this as needed
tokenize = lambda x: simple_preprocess(x)


def create_embeddings(data_dir, embeddings_path, vocab_path, **params):
    """
    Generate embeddings from a batch of text
    :param embeddings_path: where to save the embeddings
    :param vocab_path: where to save the word-index map
    """

    assert os.path.exists(data_dir), 'Directory not found: {}'.format(data_dir)
    assert os.listdir(data_dir), 'No files found in "{}"'.format(data_dir)

    class SentenceGenerator(object):
        def __init__(self, dirname):
            self.dirname = dirname

        def __iter__(self):
            for fname in os.listdir(self.dirname):
                for line in open(os.path.join(self.dirname, fname)):
                    yield tokenize(line)

    sentences = SentenceGenerator(data_dir)

    model = Word2Vec(sentences, **params)
    weights = model.syn0
    np.save(open(embeddings_path, 'wb'), weights)

    vocab = dict([(k, v.index) for k, v in model.vocab.items()])
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))


def load_vocab(vocab_path):
    """
    Load word -> index and index -> word mappings
    :param vocab_path: where the word-index map is saved
    :return: word2idx, idx2word
    """

    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word


def word2vec_embedding_layer(embeddings_path):
    """
    Generate an embedding layer word2vec embeddings
    :param embeddings_path: where the embeddings are saved (as a numpy file)
    :return: the generated embedding layer
    """

    weights = np.load(open(embeddings_path, 'rb'))
    layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights])
    return layer

def main():
    parser = argparse.ArgumentParser(description='Train and test Keras embeddings')
    parser.add_argument('-d', '--data', help='Path to text files to train on', required=True)
    parser.add_argument('-e', '--embeddings', help='Where to save embeddings (default: embeddings.npz)', default='embeddings.npz')
    parser.add_argument('-v', '--vocab', help='Where to save vocabulary map (default: map.json)', default='map.json')
    parser.add_argument('-m', '--demo', help='Download some demo data', action='store_true', default=False)
    parser.add_argument('-n', '--no-train', dest='train', help='Don\'t train embeddings', action='store_false', default=True)
    parser.add_argument('-t', '--tokens', dest='print_tokens', help='Print tokens', action='store_true', default=False)

    options = parser.parse_args()

    if options.demo:
        import urllib2
        mach = 'http://www.gutenberg.org/cache/epub/1232/pg1232.txt'
        save_path = os.path.join(options.data, 'machiavelli.txt')
        if not os.path.exists(save_path):
            print('Downloading demo data...')
            with open(save_path, 'w') as f:
                response = urllib2.urlopen(mach)
                f.write(response.read())

    # variable arguments are passed to gensim's word2vec model
    if options.train:
        print('Training Word2Vec...')
        create_embeddings(options.data, options.embeddings, options.vocab, size=100, min_count=5, window=5, sg=1, iter=25)

    word2idx, idx2word = load_vocab(options.vocab)
    if options.print_tokens:
        print('Tokens:', ', '.join(word2idx.keys()))

    # cosine similarity model
    print('Building model...')
    input_a = Input(shape=(1,), dtype='int32', name='input_a')
    input_b = Input(shape=(1,), dtype='int32', name='input_b')
    embeddings = word2vec_embedding_layer(options.embeddings)
    embedding_a = embeddings(input_a)
    embedding_b = embeddings(input_b)
    similarity = merge([embedding_a, embedding_b], mode='cos', dot_axes=2)
    model = Model(input=[input_a, input_b], output=similarity)
    model.compile(optimizer='sgd', loss='mse') # optimizer and loss don't matter

    while True:
        word_a = raw_input('First word: ')
        if word_a not in word2idx:
            print('"%s" is not in the index' % word_a)
            continue
        word_b = raw_input('Second word: ')
        if word_b not in word2idx:
            print('"%s" is not in the index' % word_b)
            continue
        output = model.predict([np.asarray([word2idx[word_a]]), np.asarray([word2idx[word_b]])])
        print('%f' % output)

if __name__ == '__main__':
    main()