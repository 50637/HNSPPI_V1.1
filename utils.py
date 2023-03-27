# -*- coding: utf-8 -*-
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import copy
import itertools
import random

import networkx as nx
import numpy as np

import OpenNE.graph as og
import csv
from gensim.models import KeyedVectors
from OpenNE import  node2vec

def embedding_training(args, train_graph_filename):
    '''
    load features from network.
    '''
    G = og.Graph()
    G.read_edgelist(train_graph_filename, weighted=False)
    _embedding_training(args, G_=G)
    return

def _embedding_training(args, G_=None):
    '''
    extract features from network.
    '''
    seed = args.seed
    model = node2vec.Node2vec(graph=G_, path_length=args.walk_length,
                              num_paths=args.number_walks, dim=args.dimensions,
                              workers=args.workers, p=args.p, q=args.q, window=args.window_size)
    model.save_embeddings(args.output)
    return

def split_train_test_graph(input_edgelist,fu_edgelist, seed, testing_ratio=0.2):
    '''
    split train and test data.
    '''
    G1 = nx.read_edgelist(input_edgelist)
    G0 = nx.read_weighted_edgelist(fu_edgelist)
    node_num1, edge_num1 = len(G1.nodes), len(G1.edges)
    G_train = copy.deepcopy(G1)
    # remove isolate nodes
    G_train.remove_nodes_from(list(nx.isolates(G_train)))
    # #assert node_num1 == node_num2
    train_graph_filename = 'graph_train.edgelist'
    nx.write_edgelist(G_train, train_graph_filename, data=False)
    node_num1, edge_num1 = len(G_train.nodes), len(G_train.edges)
    L1 = list(G_train.nodes())
    L0 = list(G0.nodes())
    for i in range(len(L0)):
        if L0[i] in L1:
            continue
        else:
            G0.remove_node(L0[i])
    G0.remove_nodes_from(list(nx.isolates(G0)))
    node_num0, edge_num0 = len(G0.nodes), len(G0.edges)
    # The number of negative examples to build
    num = edge_num1-edge_num0
    G = nx.Graph()
    threshold = 10
    #construct fully connected graph
    G.add_edges_from(itertools.combinations(L1, 2))
    #remove positive cases
    G.remove_edges_from(G_train.edges())
    G.remove_edges_from(G0.edges())
    random.seed(seed)
    for edge in G.edges:
        node_u, node_v = edge
        if (G.degree(node_u) > threshold and G.degree(node_v) > threshold):
            G.remove_edge(node_u, node_v)
    print(len(G.edges))
    neg_edges = random.sample(G.edges, num)
    G0.add_edges_from(neg_edges)
    node_num0, edge_num0 = len(G0.nodes), len(G0.edges)
    print('dealt neg Graph: nodes:', node_num0, 'edges:', edge_num0)
    testing_edges_num = int(len(G_train.edges) * testing_ratio)
    random.seed(seed)
    testing_pos_edges = random.sample(G_train.edges, testing_edges_num)
    G_aux = copy.deepcopy(G_train)
    G_aux.remove_edges_from(testing_pos_edges)
    train_pos_edges=G_aux.edges()
    return G1, G_train, testing_pos_edges, train_graph_filename,G0,train_pos_edges

def generate_neg_edges(G0, edges_num, seed):
    '''
    generate negative edges
    '''
    random.seed(seed)
    neg_edges = random.sample(G0.edges, edges_num)
    return neg_edges

def getKmers(sequence, size):
    '''
    seperate protein sequence
    '''
    return [sequence[x:x + size] for x in range(len(sequence) - size + 1)]

def load_embedding(embedding_file_name,species):
    '''
    Generate feature vectors for each protein
    '''
    dict_club = {}
    f = open('seperate.txt', 'w')
    with open('data/'+species+'/'+species+'.csv', 'r') as csvfile:  #
        reader = csv.reader(csvfile)
        for row in reader:
            s = getKmers(str(row), 3)
            try:
                del (s[0])
                del (s[0])
                del (s[-1])
                del (s[-1])
                f.write(str(s))
                f.write('\r\n')
            except:
                continue
    f.close()
    sentences = LineSentence(f)
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1 , sg=1 , workers=multiprocessing.cpu_count()-1 )
    with open('data/'+species+'/'+species+'.csv','r',encoding='utf-8')as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            dict_club[row[0]] = row[1:]
    seqfeature = {}
    with open('data/'+species+'/'+species+'.csv', 'r',encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row = ','.join(str(i) for i in row)
            arr1 = np.zeros(100)
            for x in range(int(len(row) / 3)):
                list1 = model[str(row[x * 3:x * 3 + 3])]
                arr1 += list1
                seqfeature[row] = arr1 / (int(len(row) / 3))
    with open(embedding_file_name) as f:
        node_num, emb_size = f.readline().split()
        print('Nodes with embedding: %s'%node_num)
        embedding_look_up = {}
        for line in f:
            vec = line.strip().split()
            node_id = vec[0]
            embeddings = vec[1:]
            seq1 = dict_club[node_id]
            seq1=' '.join(seq1)
            emb = [float(x) for x in embeddings]
            emb=np.append(emb,seqfeature[seq1])
            emb = emb / np.linalg.norm(emb)
            emb[np.isnan(emb)] = 0
            embedding_look_up[node_id] = list(emb)
        assert int(node_num) == len(embedding_look_up)
        f.close()
        return embedding_look_up





