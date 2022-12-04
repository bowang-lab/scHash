# direct import  hadamrd matrix from scipy
from scipy.linalg import hadamard  
import torch
from collections import Counter
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
    start_time_CHC = time.time()

    # print("Compute result using gpu")
    binaries_query, labels_query = compute_result(query_dataloader, net)
    
    labels_pred_CHC = get_labels_pred_closest_cell_anchor(binaries_query.cpu().numpy(), labels_query.numpy(),
                                                        net.cell_anchors.numpy())
    CHC_duration = time.time() - start_time_CHC
    query_num = binaries_query.shape[0]
    # if show_time:
    #     print("\n")
    #     print("  - Time spent on annotating {} test data: {:.2f}s".format(query_num, CHC_duration))
    #     print("  - CHC query speed: {:.2f} queries/s".format(query_num/CHC_duration))
    
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
    start_time_CHC = time.time()

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
    
    CHC_duration = time.time() - start_time_CHC
    query_num = binaries_query.shape[0]
#     if show_time:
#         print("\n")
#         print("  - Time spent on annotating {} test data: {:.2f}s".format(query_num, CHC_duration))
#         print("  - CHC query speed: {:.2f} queries/s".format(query_num/CHC_duration))
    
    
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
    num_query = query_labels.shape[0]
    labels_pred = []
    for binary_query, label_query in zip(query_binaries, query_labels):
          dists = CalcHammingDist(binary_query, cell_anchors)
          closest_class = np.argmin(dists)
        #   m = np.min(dists)
        #   count = np.count_nonzero(dists==m)
        #   print("Min dist =", m, "Count =", count)
          labels_pred.append(closest_class)
    return labels_pred

# calcuate hamming distance，B1 is a vector，B2 is a matrix
def CalcHammingDist(B1, B2):
    # q = B2.shape[1]
    # distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    # return distH
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

# ################## -------------------------------------------------##############
# ################## 从这里往下都是没用上的function ####################################

# def test_speed(dataloaders, net, size=280):
#     # get data samples and evaluate them
#     # Concatenate all data smaples from dataloader list
#     data, labels = None, None
#     # net.cpu()
#     net.eval()
#     for dataloader in dataloaders:
#         data_batch, labels_batch = [], []
#         if dataloader == None: 
#             break
#         for img, label in dataloader:
#             data_batch.append(img)
#             labels_batch.append(label)
#         dataloader_data = torch.cat(data_batch)[:size]
#         dataloader_labels = torch.cat(labels_batch)[:size]
#         if data == None:
#             data = dataloader_data
#             labels = dataloader_labels
#         else:
#             data = torch.cat((data, dataloader_data))[:size]
#             labels = torch.cat((labels, dataloader_labels))[:size]

#     # Take the average of 6 runs as to calculate query speed
#     times = []
#     rep_num = 6
#     for i in range(rep_num):
#         start_time_CHC = time.time()
#         binary_codes = (net(data.cuda())).data
#         labels_pred_CHC = get_labels_pred_closest_cell_anchor(binary_codes.cpu().numpy(), labels.numpy(), net.cell_anchors.numpy())
#         CHC_duration = time.time() - start_time_CHC
#         times.append(CHC_duration)
#     times = np.array(times)
#     query_num = binary_codes.shape[0]
#     print("  - Average CHC Query Speed with {} test data: {:.2f} queries/s, time = {:.2f}".format(query_num, query_num/times.mean(), times.mean()))
#     print("  - Each time =", times)

# def compute_retrieval_speed(_binaries_database, _binaries_query, size):
#     print("-------Start computing retrieval speed---------")
#     binaries_database, binaries_query = _binaries_database.cpu(), _binaries_query.cpu()
#     binaries_database_oversample = binaries_database
#     actual_size = binaries_database.shape[0]
#     print("actual database size =", actual_size)
#     binaries_database_oversample = torch.cat((binaries_database_oversample, binaries_database))[:size]
#     while binaries_database_oversample.shape[0] < size:
#         binaries_database_oversample = torch.cat((binaries_database_oversample, binaries_database))[:size]
#     print("oversampled database size =", binaries_database_oversample.shape[0])

#     binaries_query_np = binaries_query.numpy()
#     binaries_database_oversample_np = binaries_database_oversample.numpy()

#     start_time = time.time()
#     for iter in range(binaries_query.shape[0]):
#         hamm_dists = CalcHammingDist(binaries_query_np[iter, :], binaries_database_oversample_np)
#         hamm_indexes = np.argsort(hamm_dists)
#     duration = time.time() - start_time
#     print("Duration =", duration, "s")
#     print("Time per query cell =", duration/binaries_query.shape[0] * 1000, "ms")

# def save_retreival_result(retrieval_binaries, query_binaries, labels_database, labels_query, topk):
#     num_query = labels_query.shape[0]
#     labels_database_ranked_all = []
#     for iter in range(num_query):
#         hamm_dists = CalcHammingDist(query_binaries[iter, :], retrieval_binaries)
#         hamm_indexes = np.argsort(hamm_dists)
#         labels_database_ranked = labels_database[hamm_indexes]
#         labels_database_ranked = labels_database_ranked[0:topk]
#         labels_database_ranked_all.append(labels_database_ranked)

#     labels_database_ranked_all = np.stack(labels_database_ranked_all)
#     result_path = 'retrival_result.csv'
#     label_path ='qurty_label.csv'
#     pd.DataFrame(labels_database_ranked_all).to_csv(result_path)
#     pd.DataFrame(labels_query).to_csv(label_path)
#     return labels_database_ranked_all, labels_query

# # understanding Top K：https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
# def compute_MAP(retrieval_binaries, query_binaries, retrieval_labels, query_labels, topk):
#     num_query = query_labels.shape[0]
#     topK_ave_precision_per_query = 0
#     for iter in range(num_query):
#         # Given a query label，find the entries with the same labels Ex: [1,0,0,0,1,1,1,1,0,0]
#         ground_truths = (np.dot(query_labels[iter,:], retrieval_labels.transpose()) > 0).astype(np.float32)

#         # Given a query binary，calculate the hamming distances to all entries in database Ex: [2,10,14,9,1,2,1,2,1,4,6]
#         hamm_dists = CalcHammingDist(query_binaries[iter, :], retrieval_binaries)
        
#         # sort hamming distance，return indexs
#         hamm_indexes = np.argsort(hamm_dists)

#         # ideal case: [1,1,1,1,1,0,0,0,0,0]
#         # hamming distance: [1,1,1,2,2,4,6,9,10,14]
#         ground_truths = ground_truths[hamm_indexes]

#         topK_ground_truths = ground_truths[0:topk]

#         # Ex: topK_ground_truths = 5
#         topK_ground_truths_sum = np.sum(topK_ground_truths).astype(int)

#         if topK_ground_truths_sum == 0:
#             continue

#         # Ex: [1,2,3,4,5]
#         matching_binaries = np.linspace(1, topK_ground_truths_sum, topK_ground_truths_sum)

#         # ground truths position in 1 ~ n
#         ground_truths_pos = np.asarray(np.where(topK_ground_truths == 1)) + 1.0

#         topK_ave_precision_per_query_ = np.mean(matching_binaries / (ground_truths_pos))

#         topK_ave_precision_per_query += topK_ave_precision_per_query_
        
#     topK_map = topK_ave_precision_per_query / num_query

#     return topK_map





# # # compute Binary and get labels
# # def compute_result(dataloader, net):
# #     binariy_codes, labels = [], []
# #     net.eval()
# #     for img, label in dataloader:
# #         labels.append(label)
# #         binariy_codes.append((net(img.cuda())).data)
# #     return torch.cat(binariy_codes).tanh(), torch.cat(labels)

# #  modified here ##############################################################################################
# # compute Binary and get labels
# def compute_result(dataloader, net):
#     binariy_codes, labels = [], []
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
#     for data in dataloader:
#         img, label = data[0].to(device), data[1]
#         labels.append(label)
#         binariy_codes.append(net(img))
#     return torch.vstack(binariy_codes).tanh(), torch.cat(labels)
# #############################################################################################################################################

# # compute Binary and get labels
# def compute_result_cpu(dataloader, net):
#     binariy_codes, labels = [], []
#     net.cpu()
#     net.eval()
#     for img, label in dataloader:
#         labels.append(label.cpu())
#         binariy_codes.append((net(img.cpu())).data)
#     return torch.cat(binariy_codes).tanh(), torch.cat(labels)





# def find_most_common_label(labels):
#   labels_tuple = [tuple(label) for label in labels]

#   most_common_label_tuple = Counter(labels_tuple).most_common(1)[0][0]
#   return np.array(most_common_label_tuple)


# # turn categorical labels to one-hot
# def categorical_to_onehot(labels, numOfClass):
#     labels = labels.reshape(labels.shape[0], 1)
#     labels = (labels == torch.arange(numOfClass).reshape(1, numOfClass)).int()
#     return labels

# # # T-sne visualizations
# # def t_sne_visualization_2d(data, labels):
# #     from sklearn.manifold import TSNE
# #     from matplotlib import pyplot as plt
# #     import matplotlib.cm as cm

# #     # Calculate TSNE embeded space
# #     embedded_data = TSNE(n_components=2,
# #                         n_iter=5000, 
# #                         learning_rate=100,
# #                         early_exaggeration=10,
# #                         perplexity=50
# #                         ).fit_transform(data)

# #     # Visualization 
# #     plt.figure(figsize=(6, 5))

# #     # unique_classes = set(labels)
# #     # colors = cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    
# #     # for label, c in zip(unique_classes, colors):
# #     #     plt.scatter(embedded_data[labels == label, 0], 
# #     #                 embedded_data[labels == label, 1], 
# #     #                 linewidths=0.03,
# #     #                 color=c, 
# #     #                 label=label)
        
# #     plt.scatter(x = embedded_data[:,0], y = embedded_data[:,1], c = labels, s = 1.5, cmap='viridis')
# #     plt.title('t-SNE visualization of hash code generation')

# #     plt.legend()
# #     plt.show()
# #     return

# def calculate_gene_grad(model):
#     print("---Get gene grad---")
#     model.cuda()
#     gradient_genes_per_cell_type = []
    
#     # Calculate gradient for each cell type
#     for query_label in range(model.trainer.datamodule.N_CLASS):
#         print("--query_label =", query_label)
#         gradients_gene = []
#         for img, label in model.trainer.datamodule.test_dataloader():
#             # Calculate binary codes
#             img.requires_grad = True
#             hash_codes = (model(img.cuda())).tanh()
#             hash_codes_clone = torch.clone(hash_codes)
#             hash_codes_clone = hash_codes_clone.cpu().detach().numpy()
#             predicted_labels = []

#             # Find hit cell labels
#             for binary_query in hash_codes_clone:
#                 dists = CalcHammingDist(binary_query, model.cell_anchors.cpu().numpy())
#                 closest_class = np.argmin(dists)
#                 predicted_labels.append(closest_class)
#             predicted_labels = torch.tensor(predicted_labels)
#             hit_index = (predicted_labels == label) * (query_label == label)
#             # print("true_positives = ", hit_index)

#             # Calculate deviation
#             type_cell_anchor = model.cell_anchors[query_label]
#             deviation = torch.sum(torch.abs((type_cell_anchor.cuda() - hash_codes[hit_index].cuda())))
#             deviation.backward()

#             # Get gradient with respect to each gene
#             g = torch.sum(img.grad, axis=0)
#             gradients_gene.append(g)
#         gradients_gene = torch.stack(gradients_gene)
#         gradients_gene_sum = torch.sum(gradients_gene, axis=0)
#         print("gradients_gene_sum =", gradients_gene_sum)
#         gradient_genes_per_cell_type.append(gradients_gene_sum)
#     gradient_genes_per_cell_type = torch.stack(gradient_genes_per_cell_type).numpy()
#     print("gradient_genes_per_cell_type =\n", gradient_genes_per_cell_type)
#     print("shape = ", gradient_genes_per_cell_type.shape)


# def output_result(model):
#     print("Out put result for TM")
#     model.cuda()
#     binaries_train, labels_train = compute_result(model.trainer.datamodule.train_dataloader(), model)
#     binaries_val, labels_val = compute_result(model.trainer.datamodule.val_dataloader(), model)
#     binaries_database, labels_database = torch.cat([binaries_train, binaries_val]), torch.cat([labels_train, labels_val])

#     binaries_test, labels_test = compute_result(model.trainer.datamodule.test_dataloader(), model)

#     binaries_database, labels_database = binaries_database.cpu(), labels_database.cpu()
#     binaries_test, labels_test = binaries_test.cpu(), labels_test.cpu()

#     print("binaries_database size =", binaries_database.shape)
#     print("binaries_test size =", binaries_test.shape)

#     binaries_database = pd.DataFrame(binaries_database.numpy())
#     labels_train = pd.DataFrame(labels_train.numpy())
#     binaries_test = pd.DataFrame(binaries_test.numpy())
#     labels_test = pd.DataFrame(labels_test.numpy())

#     binaries_database.to_csv('temp1.csv')
#     labels_train.to_csv('temp2.csv')
#     binaries_test.to_csv('temp3.csv')
#     labels_test.to_csv('temp4.csv')
