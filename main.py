# -*- coding: utf-8 -*-

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from utils import *
from evaluation import PPIPrediction

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input1', required=True,
                        help='read positive edges')
    parser.add_argument('--input2', required=True,
                        help='read negative edges')
    parser.add_argument('--output',
                        help='embedding file', required=True)
    parser.add_argument('--species',
                        help='species name', required=True)
    parser.add_argument('--number-walks', default=32, type=int,
                        help='Number of random walks to start at each node. ')
    parser.add_argument('--walk-length', default=64, type=int,
                        help='Length of the random walk started at each node. ')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes. ')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of word2vec model. ')
    parser.add_argument('--seed',default=0, type=int,  help='seed value')
    args = parser.parse_args()

    return args

def main(args):
        G1, G_train, testing_pos_edges, train_graph_filename,G0,training_pos_edges = split_train_test_graph(args.input1,args.input2, args.seed)
        embedding_training(args, train_graph_filename)
        embedding_look_up = load_embedding(args.output,args.species)
        PPIPrediction(embedding_look_up, G1, G_train, G0 ,testing_pos_edges, args.seed,training_pos_edges)
        os.remove(train_graph_filename)

def more_main():
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    main(parse_args())


if __name__ == "__main__":
    more_main()


