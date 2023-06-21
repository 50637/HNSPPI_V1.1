# -*- coding: utf-8 -*-

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from utils import *
from evaluation import PPIPrediction

def main(args):
    G1, G_train, testing_pos_edges, train_graph_filename,G0,training_pos_edges = split_train_test_graph(args.input1,args.input2, args.seed)
    embedding_training(args, train_graph_filename)
    embedding_look_up = load_embedding(args.output,args.species)
    #start predicting...
    PPIPrediction(embedding_look_up, G1, G_train, G0 ,testing_pos_edges, args.seed,training_pos_edges)
    os.remove(train_graph_filename)
    
if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input1', default="data/mouse/mouse-pos.edgelist",
                        help='Input graph file（positive）. Only accepted edgelist format.')
    parser.add_argument('--input2', default="data/mouse/mouse-neg.edgelist",
                        help='Input graph file（negative）. Only accepted edgelist format.')
    parser.add_argument('--output',default="mouse",
                        help='Output graph embedding file')
    parser.add_argument('--species',default="mouse",
                        help='species name')
    parser.add_argument('--seed',default=0, type=int,  help='seed value')
    args = parser.parse_known_args()[0]
    main(args)
