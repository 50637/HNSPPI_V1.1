# -*- coding: utf-8 -*-
import multiprocessing
import copy
import itertools
import random
from OpenNE import  node2vec
import networkx as nx
import numpy as np

import OpenNE.graph as og
import csv
from gensim.models import KeyedVectors

def read_for_OpenNE(filename):
    G = og.Graph()
    G.read_edgelist(filename=filename)
    return G

def split_train_test_graph(input_edgelist,fu_edgelist, seed, testing_ratio=0.2):
    G1 = nx.read_edgelist(input_edgelist)
    G0 = nx.read_weighted_edgelist(fu_edgelist)
    node_num1, edge_num1 = len(G1.nodes), len(G1.edges)
    print('Original Graph: nodes:', node_num1, 'edges:', edge_num1)
    G_train = copy.deepcopy(G1)
    G_train.remove_nodes_from(list(nx.isolates(G_train)))
    train_graph_filename = 'graph_train.edgelist'
    nx.write_edgelist(G_train, train_graph_filename, data=False)
    node_num1, edge_num1 = len(G_train.nodes), len(G_train.edges)
    print('delt Graph: nodes:', node_num1, 'edges:', edge_num1)
    L1 = list(G_train.nodes())
    L0 = list(G0.nodes())
    for i in range(len(L0)):
        if L0[i] in L1:
            continue
        else:
            G0.remove_node(L0[i])
    G0.remove_nodes_from(list(nx.isolates(G0)))
    node_num0, edge_num0 = len(G0.nodes), len(G0.edges)
    num = edge_num1-edge_num0
    G = nx.Graph()
    G.add_edges_from(itertools.combinations(L1, 2))
    G.remove_edges_from(G_train.edges())
    G.remove_edges_from(G0.edges())
    random.seed(seed)
    for edge in G.edges:
        node_u, node_v = edge
        if (G.degree(node_u) > 10 and G.degree(node_v) > 10):
            G.remove_edge(node_u, node_v)
    neg_edges = random.sample(G.edges, num)
    G0.add_edges_from(neg_edges)
    node_num0, edge_num0 = len(G0.nodes), len(G0.edges)
    testing_edges_num = int(len(G_train.edges) * testing_ratio)
    random.seed(seed)
    testing_pos_edges = random.sample(G_train.edges, testing_edges_num)
    G_aux = copy.deepcopy(G_train)
    G_aux.remove_edges_from(testing_pos_edges)
    train_pos_edges=G_aux.edges()
    return G1, G_train, testing_pos_edges, train_graph_filename,G0,train_pos_edges


def generate_neg_edges(G0, edges_num, seed):
    random.seed(seed)
    neg_edges = random.sample(G0.edges, edges_num)
    return neg_edges

def load_embedding(embedding_file_name, node_list=None):
    dict_club = {}
    with open('data/mouse/mouse.csv','r',encoding='utf-8')as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            dict_club[row[0]] = row[1:]
    model = KeyedVectors.load_word2vec_format('data/mouse/mouse.txt', binary=False)
    seqfeature = {}
    with open('data/mouse/mouse.csv', 'r',encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row=row[1]
            arr1 = np.zeros(model.vector_size)
            for x in range(int(len(row) / 3)):
                try:
                    list1 = model[str(row[x * 3:x * 3 + 3])]
                    arr1 += list1
                except:
                    continue
            seqfeature[row] = arr1 / (int(len(row) / 3))

    with open(embedding_file_name) as f:
        node_num, emb_size = f.readline().split()
        embedding_look_up = {}
        for line in f:
            vec = line.strip().split()
            node_id = vec[0]
            embeddings = vec[1:]
            seq1 = dict_club[node_id]
            seq1 = ' '.join(seq1)
            emb = [float(x) for x in embeddings]
            emb=np.append(emb,seqfeature[seq1])
            emb = emb / np.linalg.norm(emb)
            emb[np.isnan(emb)] = 0
            embedding_look_up[node_id] = list(emb)
        assert int(node_num) == len(embedding_look_up)
        f.close()
        return embedding_look_up

def embedding_training(args, train_graph_filename):
    g = read_for_OpenNE(train_graph_filename)
    _embedding_training(args, G_=g)
    return

def _embedding_training(args, G_=None):
    model = node2vec.Node2vec(graph=G_, path_length=64,
                              num_paths=32, dim=100,
                              workers=8, p=1, q=1, window=10)
    model.save_embeddings(args.output)
    return
