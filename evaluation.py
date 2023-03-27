
# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Sequential
import time
from sklearn.model_selection import KFold,train_test_split,cross_val_score


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn import datasets
from sklearn import svm
from utils import *
import keras
from keras_applications.densenet import layers
from utils import *
import numpy as np


def calculate_performace(test_num, pred_y, labels):
    '''
    Evaluate algorithm performance.
    '''
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    accuracy = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp + 1e-06)
    sensitivity = float(tp) / (tp + fn + 1e-06)
    recall = float(tp) / (tp + fn + 1e-06)
    specificity = float(tn) / (tn + fp + 1e-06)
    f1_score = float(2 * tp) / (2 * tp + fp + fn + 1e-06)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return  accuracy, precision, sensitivity, recall, specificity, MCC, f1_score

def PPIPrediction(embedding_look_up, original_graph, train_graph,G0, test_pos_edges, seed,training_pos_edges):
    '''
    Predict new PPIs.
    '''
    random.seed(seed)
    train_neg_edges = generate_neg_edges(G0, len(training_pos_edges), seed)
    G_aux = copy.deepcopy(G0)
    for edge in train_neg_edges:
        node1 = edge[0]
        node2 = edge[1]
        G_aux.remove_edge(node1,node2)

    test_neg_edges = G_aux.edges() #generate_neg_edges(G_aux, len(test_pos_edges), seed)
    print(len(test_pos_edges))
    print(len(training_pos_edges))

    # construct X_train, y_train, X_test, y_test
    X_train = []
    y_train = []
    for edge in training_pos_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_train.append(feature_vector)
        y_train.append(1)
    for edge in train_neg_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_train.append(feature_vector)
        y_train.append(0)


    X_test = []
    y_test = []
    for edge in test_pos_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_test.append(feature_vector)
        y_test.append(1)
    for edge in test_neg_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_test.append(feature_vector)
        y_test.append(0)


    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    c = list(zip(X, y))
    random.shuffle(c)
    X, y = zip(*c)
    folds = 10
    X_folds = np.array_split(X, folds)
    y_folds = np.array_split(y, folds)
    print(X_folds)
    accsum=[]
    presum=[]
    specisum=[]
    recallsum=[]
    f1sum=[]
    aucsum=[]
    prcsum=[]
    for i in range(folds):
        X_train = np.vstack(X_folds[:i] + X_folds[i + 1:])
        y_train = np.hstack(y_folds[:i] + y_folds[i + 1:])
        y_val = y_folds[i]
        X_val = X_folds[i]
        clf = svm.SVC()  # svm class
        clf.fit(X_train, y_train)  # training the svc model

        print('Start predicting...')
        pred_y= clf.predict(X_val)
        auc_test = roc_auc_score(y_val, pred_y)
        pr_test = average_precision_score(y_val, pred_y)
        test_num=len(y_val)
        accuracy, precision, sensitivity, recall, specificity, MCC, f1_score = calculate_performace(test_num, pred_y, y_val)
        accsum.append(accuracy)
        presum.append(precision)
        specisum.append(sensitivity)
        recallsum.append(recall)
        f1sum.append(f1_score)
        aucsum.append(auc_test)
        prcsum.append(pr_test)
    with open('result'+str(seed)+'.csv','w') as f:
        f.write('acc')
        f.write('\n')
        f.write(str(np.mean(accsum)))
        f.write('\n')
        f.write(str(np.std(accsum)))
        f.write('\n')
        f.write('pre')
        f.write('\n')
        f.write(str(np.mean(presum)))
        f.write('\n')
        f.write(str(np.std(presum)))
        f.write('\n')
        f.write('recall')
        f.write('\n')
        f.write(str(np.mean(recallsum)))
        f.write('\n')
        f.write(str(np.std(recallsum)))
        f.write('\n')
        f.write('speci')
        f.write('\n')
        f.write(str(np.mean(specisum)))
        f.write('\n')
        f.write(str(np.std(specisum)))
        f.write('\n')
        f.write('f1')
        f.write('\n')
        f.write(str(np.mean(f1sum)))
        f.write('\n')
        f.write(str(np.std(f1sum)))
        f.write('\n')
        f.write('auc')
        f.write('\n')
        f.write(str(np.mean(accsum)))
        f.write('\n')
        f.write(str(np.std(accsum)))
        f.write('\n')
        f.write('prc')
        f.write('\n')
        f.write(str(np.mean(prcsum)))
        f.write('\n')
        f.write(str(np.std(prcsum)))


       







