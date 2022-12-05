# direct import  hadamrd matrix from scipy
from scipy.linalg import hadamard  
import torch
import statistics
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
import time
import random

# top-level interface for metric calculation
def compute_metrics(query_dataloader, net):
    ''' Labeling Strategy:
    Closest Cell Anchor:
    Label the query using the label associated to the nearest cell anchor
    - Computation Complexity:
    O(m) << O(n) per puery
    m = number of classes in database
    - Less Accurate
    '''
    # print("Compute result using gpu")
    binaries_query, labels_query = compute_result(query_dataloader, net)
    
    labels_pred_CHC = get_labels_pred_closest_cell_anchor(binaries_query.cpu().numpy(), labels_query.numpy(),
                                                        net.cell_anchors.numpy())
 
    # (1) labeling accuracy
    labeling_accuracy_CHC = compute_labeling_strategy_accuracy(labels_pred_CHC, labels_query.numpy())
    
    # (2) F1_score, average = (per class, weighted)
    F1_score_weighted_average_CHC = f1_score(labels_query, labels_pred_CHC, average='weighted')
    F1_score_per_class_CHC = f1_score(labels_query, labels_pred_CHC, average=None)
    target_names = [i for i in net.trainer.datamodule.label_mapping.keys()]
    class_report = classification_report(labels_query, labels_pred_CHC, labels=[i for i in range(len(target_names))], target_names=target_names)

    # (3) F1_score median
    F1_score_median_CHC = statistics.median(F1_score_per_class_CHC)

    # (4) precision, recall
    precision = precision_score(labels_query, labels_pred_CHC, average="weighted")
    recall = recall_score(labels_query, labels_pred_CHC, average="weighted")

    CHC_metrics = (labeling_accuracy_CHC, 
                F1_score_weighted_average_CHC, F1_score_median_CHC, F1_score_per_class_CHC,
                precision, recall, class_report)

    return CHC_metrics

        
def compute_result(dataloader, net):
    binariy_codes, labels = [],[]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    for data in dataloader:
        img = data[0].to(device)
        binariy_codes.append(net(img))
        labels.append(data[1])
    return torch.vstack(binariy_codes),torch.cat(labels)

def test_compute_metrics(query_dataloader, net):
    ''' Labeling Strategy:
    Closest Cell Anchor:
    Label the query using the label associated to the nearest cell anchor
    - Computation Complexity:
    O(m) << O(n) per puery
    m = number of classes in database
    - Less Accurate
    '''
    # print("Compute result using gpu")
    ######################## modified here
    start = time.time()
    # original function
    # binaries_query, labels_query = compute_result(query_dataloader, net)
    binaries_query,labels_query = compute_result(query_dataloader, net)
    hashing_time = time.time() - start
    print("################### compute_result time: ", hashing_time)
    ######################## modified here    
    
    start = time.time() 
    labels_pred_CHC = get_labels_pred_closest_cell_anchor(binaries_query.cpu().numpy(), labels_query.numpy(),
                                                        net.cell_anchors.numpy())
    cell_assign_time = time.time() - start
    
    query_time = cell_assign_time+hashing_time
    print("################### cell_assign time: ",cell_assign_time)
    print("################### query time: ",query_time)
       
    # (1) labeling accuracy
    labeling_accuracy = compute_labeling_strategy_accuracy(labels_pred_CHC, labels_query.numpy())
    
    # (2) F1_score, average = (micro, macro, weighted)
    f1 = f1_score(labels_query, labels_pred_CHC, average='weighted')
    F1_score_per_class_CHC = f1_score(labels_query, labels_pred_CHC, average=None)
    f1_median = statistics.median(F1_score_per_class_CHC)

    # (4) precision, recall
    precision = precision_score(labels_query, labels_pred_CHC, average="weighted")
    recall = recall_score(labels_query, labels_pred_CHC, average="weighted")

    return labeling_accuracy,precision,recall,f1, hashing_time, cell_assign_time, query_time, f1_median

# generate cell anchors
def get_cell_anchors(n_class, bit):
    H_K = hadamard(bit)
    H_2K = np.concatenate((H_K, -H_K), 0)
    hash_targets = torch.from_numpy(H_2K[:n_class]).float()

    if H_2K.shape[0] < n_class:
        hash_targets.resize_(n_class, bit)
        for k in range(20):
            for index in range(H_2K.shape[0], n_class):
                ones = torch.ones(bit)
                # Bernouli distribution
                sa = random.sample(list(range(bit)), bit // 2)
                ones[sa] = -1
                hash_targets[index] = ones
            # to find average/min pairwise distance
            c = []
            for i in range(n_class):
                for j in range(n_class):
                    if i < j:
                        TF = sum(hash_targets[i] != hash_targets[j])
                        c.append(TF)
            c = np.array(c)

            # choose min(c) in the range of K/4 to K/3
            # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
            # but it is hard when bit is  small
            if c.min() > bit / 4 and c.mean() >= bit / 2:
                print(c.min(), c.mean())
                break
    return hash_targets


# Predict label using Closest Cell Anchor strategy (b)
def get_labels_pred_closest_cell_anchor(query_binaries, query_labels, cell_anchors):
    labels_pred = []
    for binary_query, label_query in zip(query_binaries, query_labels):
          dists = CalcHammingDist(binary_query, cell_anchors)
          closest_class = np.argmin(dists)
          labels_pred.append(closest_class)
    return labels_pred

# calcuate hamming distance，B1 is a vector，B2 is a matrix
def CalcHammingDist(B1, B2):
    B1 = np.tile(B1, (B2.shape[0], 1))
    distH = np.abs(B1 - B2)
    distH = distH.sum(axis=1)
    return distH

# simply get the accuracy
def compute_labeling_strategy_accuracy(labels_pred, labels_query):
    same = 0

    for i in range(len(labels_pred)):
      if (labels_pred[i] == labels_query[i]).all():
        same += 1

    return same / labels_query.shape[0]