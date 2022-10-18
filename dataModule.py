from collections import Counter
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, Dataset
from torch.nn import functional as F
from collections import Counter
from scipy.sparse import csr_matrix

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import anndata as ad
import random 
import scanpy as sc
###------------------------------Utility function for DataLoader---------------------------------###
# Perform stratified split on a dataset into two sets based on indices
def stratified_split(remaining_indices, full_labels, set1_split_percentage,random_state=42):
    target_labels = [full_labels[i] for i in remaining_indices]
    set1_indices, set2_indices = train_test_split(
        remaining_indices, train_size=set1_split_percentage, stratify=target_labels,random_state=random_state)
    return set1_indices, set2_indices

# Split a full datasets in a stratified way into test, train, validation and database sets


def split_test_train_val_database_sets(full_dataset, train_percentage, val_percentage, test_percentage):
    full_labels = full_dataset.labels
    full_indices = range(len(full_labels))

    train_indices, remaining_indices = stratified_split(full_indices, full_labels, train_percentage)
    val_indices, test_indices = stratified_split(remaining_indices, full_labels, val_percentage/(test_percentage + val_percentage))

    TM_database, TM_train, TM_val, TM_test = (None,
                                              Subset(full_dataset,
                                                     train_indices),
                                              Subset(full_dataset,
                                                     val_indices),
                                              Subset(full_dataset, test_indices))
    return TM_database, TM_train, TM_val, TM_test

# Split a full datasets in a stratified way into train, validation sets
def split_train_val_database_sets(full_dataset, train_idx, train_percentage):
    full_labels = full_dataset.labels
    full_indices = range(len(full_labels))

    train_indices, val_indices = stratified_split(train_idx, full_labels, train_percentage)
    train_set, val_set  = (Subset(full_dataset, train_indices), Subset(full_dataset, val_indices))
    
    return train_set, val_set

def label_encoder(labels):
    labels = np.unique(labels).astype(str).tolist()
    if 'unknown' not in labels:
        labels.append('unknown')
        
    df = pd.DataFrame(columns=['labels','num'])
    df['labels']=labels
    df['num']=list(range(0,len(labels)))

    label_dic = {}
    for i in df.iterrows():
        label_dic[i[1][0]]=i[0]
    return label_dic

def label_transform(label_dic,test_labels):
    test_transformed_labels = []
    for i in test_labels:
        if i not in label_dic.keys():
            test_transformed_labels.append(label_dic['unknown'])
        else:
            test_transformed_labels.append(label_dic[i])
    return test_transformed_labels

class Cross_DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', cell_type_key: str = 'cell_type', batch_key:str = '',batch_size=256, num_workers=2,  query:str = 'celseq', hvg:bool = False, log_norm:bool = False, normalize:bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "Custom"
        self.label_mapping = None  
        self.query = query
        self.hvg = hvg
        self.log_norm = log_norm
        self.normalize = normalize
        self.cell_type_key = cell_type_key
        self.batch_key = batch_key

    def setup(self):
        # Step #1: Read in all labels and keep cells with count > 10
        print(self.data_dir)
        full_data = ad.read_h5ad(self.data_dir)
        unique = np.unique(full_data.obs[self.cell_type_key],return_counts=True)
        idx  = [unique[1]<10]
        classes = unique[0][idx]
        full_data = full_data[~full_data.obs[self.cell_type_key].isin(classes),]
        
        if self.log_norm:
            full_data.X.data = np.log10(full_data.X.data+1).astype(np.float32)
        
        test = full_data[full_data.obs[self.batch_key] == self.query,]  
        full_data = full_data[full_data.obs[self.batch_key]!=self.query,]
        
        # # hvg selection
        if self.hvg:
            if self.batch_key:
                sc.pp.highly_variable_genes(full_data,n_top_genes=1000,batch_key=self.batch_key,flavor='seurat_v3')
            else:
                sc.pp.highly_variable_genes(full_data,n_top_genes=1000, flavor='seurat_v3')
                
            test = test[:,full_data.var['highly_variable'].values].copy()
            full_data = full_data[:,full_data.var['highly_variable'].values].copy()
            
        if self.normalize:
            scaler_train = StandardScaler(with_mean=False).fit(full_data.X)

            full_data.var['means'] = scaler_train.mean_
            full_data.var['std'] = scaler_train.scale_
            # full_data.X = csr_matrix(((full_data.X - full_data.var.means.values)/full_data.var['std'].values).astype(np.float32))
            full_data.X = scaler_train.transform(full_data.X)

            scaler_test = StandardScaler(with_mean=False).fit(test.X)
            # test.X = scaler.transform(test.X)
            test.var['means'] = scaler_test.mean_
            test.var['std'] = scaler_test.scale_

            b = (full_data.var['std']/test.var['std'])
            a = (full_data.var['means'] - b*test.var['means'])
            A = np.tile(a.to_numpy().reshape(test.X.shape[1],1),test.X.shape[0])
            test.X = csr_matrix((A + np.diag(b) * test.X.T).astype(np.float32).T)
            # test.X = csr_matrix(((test.X - full_data.var.means.values)/full_data.var['std'].values).astype(np.float32))
            test.X = scaler_train.transform(test.X) 
        
        
        self.N_FEATURES = full_data.X.shape[1]
        full_labels = full_data.obs[self.cell_type_key]
        remaining_labels = full_data.obs[self.cell_type_key ]

        # Step #2: Turning string class labels into int labels and store the mapping
        self.label_mapping = label_encoder(remaining_labels)
        
        int_labels =  label_transform(self.label_mapping,remaining_labels)
        print(np.unique(int_labels,return_counts=True))
        self.N_CLASS = int(np.unique(int_labels)[0])+1
        full_labels = np.asarray(int_labels)
        remaining_labels = None
        int_labels = None

        # Step #3: Read in data based on selected label indices
        full_dataset = SparseCustomDataset(data=full_data.X, labels=full_labels)
        
        test_labels = label_transform(self.label_mapping,test.obs[self.cell_type_key ])
        self.data_test = SparseCustomDataset(data=test.X, labels=np.asarray(test_labels))
          
        full_indices = range(len(full_labels))
        train_indices, val_indices = stratified_split(full_indices, full_labels, 0.8)
        self.data_train, self.data_val = (Subset(full_dataset,
                                                     train_indices),
                                              Subset(full_dataset,
                                                     val_indices))
        print("train size =", len(self.data_train))
        print("val size =", len(self.data_val))
        # print("test size =", len(self.data_test))

        # Calculate sample count in each class for training dataset
        samples_in_each_class_dict = Counter([data[1] for data in self.data_train])
        print("training samples in each class =", samples_in_each_class_dict)
        samples_in_each_class_dict_val = Counter([data[1] for data in self.data_val])
        print("val samples in each class =", samples_in_each_class_dict_val)
        samples_in_each_class_dict_test = Counter([data[1] for data in self.data_test])
        print("test samples in each class =", samples_in_each_class_dict_test)
        self.N_CLASS = len(samples_in_each_class_dict)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
           self.samples_in_each_class[index] = count


    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    
    
class Intra_DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', cell_type_key:str = 'cell_type', batch_key:str = '', batch_size=64, num_workers=2, fold_num=0, hvg:bool = False, log_norm:bool = False, normalize = True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "Custom"
        self.label_mapping = None
        self.fold_num = fold_num # fold_num 可以是 0-4
        self.hvg = hvg
        self.log_norm = log_norm
        self.cell_type_key = cell_type_key
        self.batch_key = batch_key
        self.normalize = normalize

    def setup(self):
        # Step #1: Read in all labels and keep cells with count > 10
        print(self.data_dir)
        full_data = ad.read_h5ad(self.data_dir)
        unique = np.unique(full_data.obs[self.cell_type_key],return_counts=True)
        idx  = [unique[1]<10]
        classes = unique[0][idx]
        full_data = full_data[~full_data.obs[self.cell_type_key].isin(classes),]
        
        if self.log_norm:
            full_data.X.data = np.log10(full_data.X.data+1).astype(np.float32)
        
        
        # # hvg selection
        if self.hvg:
            if self.batch_key:
                sc.pp.highly_variable_genes(full_data,n_top_genes=1000,batch_key=self.batch_key,flavor='seurat_v3')
            else:
                sc.pp.highly_variable_genes(full_data,n_top_genes=1000,flavor='seurat_v3')
            full_data = full_data[:,full_data.var['highly_variable'].values].copy()
            
        if self.normalize:
            scaler_train = StandardScaler(with_mean=False).fit(full_data.X)
            full_data.var['means'] = scaler_train.mean_
            full_data.var['std'] = scaler_train.scale_
            full_data.X = csr_matrix(((full_data.X - full_data.var.means.values)/full_data.var['std'].values).astype(np.float32))
            
        
        self.N_FEATURES = full_data.X.shape[1]
        full_labels = full_data.obs[self.cell_type_key]
        remaining_labels = full_data.obs[self.cell_type_key]

        # Step #2: Turning string class labels into int labels and store the mapping
        self.label_mapping = label_encoder(remaining_labels)
        
        int_labels =  label_transform(self.label_mapping,remaining_labels)
        print(np.unique(int_labels,return_counts=True))
        self.N_CLASS = int(np.unique(int_labels)[0])+1
        full_labels = np.asarray(int_labels)
        remaining_labels = None
        int_labels = None
        
        full_dataset = SparseCustomDataset(data=full_data.X, labels=full_labels)
        random.seed(42)
        _, self.data_train, self.data_val, self.data_test = split_test_train_val_database_sets(full_dataset, train_percentage=0.64, val_percentage=0.16, test_percentage=0.2)


        # Calculate sample count in each class for training dataset
        samples_in_each_class_dict = Counter([data[1] for data in self.data_train])
        print("training samples in each class =", samples_in_each_class_dict)
        samples_in_each_class_dict_val = Counter([data[1] for data in self.data_val])
        print("val samples in each class =", samples_in_each_class_dict_val)
        samples_in_each_class_dict_test = Counter([data[1] for data in self.data_test])
        print("test samples in each class =", samples_in_each_class_dict_test)
        self.N_CLASS = len(samples_in_each_class_dict)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
           self.samples_in_each_class[index] = count


    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)
    
class CustomDataset(Dataset):
    'A dataset base class for PyTorch Lightening'

    def __init__(self, data, labels):
        'Dataset Class Initialization'
        # Number of data and labels should match
        assert len(data) == len(labels)
        self.labels = labels
        self.data = data

    def __len__(self):
        'Returns the total number of samples'
        return len(self.data)

    def __getitem__(self, index: int):
        # Load data and get label
        return self.data[index], self.labels[index]

class SparseCustomDataset(Dataset):
    "A dataset base class for PyTorch Lightening"

    def __init__(self, data, labels):
        "Dataset Class Initialization"
        # Number of data and labels should match
        assert data.shape[0] == labels.shape[0]
        self.labels = labels
        self.data = data

    def __len__(self):
        "Returns the total number of samples"
        return self.data.shape[0]

    def __getitem__(self, index: int):
        # Load data and get label
        return self.data[index].toarray()[0], self.labels[index]