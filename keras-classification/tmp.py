# -*- coding: utf-8 -*-
#2016年11月25日
# bayes classification

from __future__ import division 
import os
import nltk
import re
import time 
import pprint
import json
import pickle,cPickle
import pygraphviz as pgv
G=pgv.AGraph(strict=False,directed=True)
G.add_node('a')
G.add_node('b')
G.add_edge('b','c')
nodelist=['f','g','h']
G.add_nodes_from(nodelist)
#attributes
G.graph_attr['label']="simple nodes and edge"
G.node_attr['shape']='circle'
G.edge_attr['color']='red'
s=G.to_string()
G.write("first.dot")
G.layout(prog='dot')
G.draw('first.png')