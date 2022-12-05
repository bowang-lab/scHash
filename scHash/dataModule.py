from collections import Counter
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,  Subset, Dataset
from torch.nn import functional as F
from collections import Counter
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import anndata as ad
import random 
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

###------------------------------Utility function for DataLoader---------------------------------###
# helper function
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

def label_encoder(labels):
    labels = np.unique(labels).astype(str).tolist()
    # allow unknown cases to happen
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


###------------------------------ DataLoader ---------------------------------###

class Cross_DataModule(pl.LightningDataModule):

    def __init__(self, train_data, cell_type_key: str = 'cell_type', batch_key:str = '',batch_size=128, num_workers=2, hvg:bool = True, log_norm:bool = True, normalize:bool = True):
        super().__init__()
        self.train_data = train_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "Custom"
        self.label_mapping = None  
        self.hvg = hvg
        self.log_norm = log_norm
        self.normalize = normalize
        self.cell_type_key = cell_type_key
        self.batch_key = batch_key
        self.data_val = None
        self.high_variable_genes = None
        self.test_data = None # the raw AnnData Object for query
        self.data_test = None # a transformed version of the test data that could fit into the model

    def setup(self, stage = None):
        # Step #1: Read in all labels and keep cells with count > 10
        full_data = self.train_data
        unique = np.unique(full_data.obs[self.cell_type_key],return_counts=True)
        idx  = [unique[1]<10]
        classes = unique[0][idx]
        full_data = full_data[~full_data.obs[self.cell_type_key].isin(classes),]
        
        if  self.hvg:
            self.N_FEATURES = 1000
        else:
            self.N_FEATURES = full_data.X.shape[1]
            
        remaining_labels = full_data.obs[self.cell_type_key]

        # Step #2: Turning string class labels into int labels and store the mapping
        self.label_mapping = label_encoder(remaining_labels)
        
        if self.log_norm:
            full_data.X.data = np.log10(full_data.X.data+1).astype(np.float32)

        # # hvg selection
        if self.hvg:
            if self.batch_key:
                sc.pp.highly_variable_genes(full_data,n_top_genes=1000,batch_key=self.batch_key,flavor='seurat_v3')
            else:
                sc.pp.highly_variable_genes(full_data,n_top_genes=1000, flavor='seurat_v3')

            self.high_variable_genes = full_data.var['highly_variable'].values
            full_data = full_data[:, self.high_variable_genes].copy()

        if self.normalize:
            self.scaler_train = StandardScaler(with_mean=False).fit(full_data.X)
            full_data.var['means'] = self.scaler_train.mean_
            full_data.var['std'] = self.scaler_train.scale_
            # full_data.X = csr_matrix(((full_data.X - full_data.var.means.values)/full_data.var['std'].values).astype(np.float32))
            full_data.X = self.scaler_train.transform(full_data.X)

        remaining_labels = full_data.obs[self.cell_type_key]
        int_labels =  label_transform(self.label_mapping, remaining_labels)
        full_labels = np.asarray(int_labels)

        # Step #3: Read in data based on selected label indices
        full_dataset = SparseCustomDataset(data=full_data.X, labels=full_labels)

        full_indices = range(len(full_labels))
        train_indices, val_indices = stratified_split(full_indices, full_labels, 0.8)
        self.data_train, self.data_val = (Subset(full_dataset,
                                                     train_indices),
                                              Subset(full_dataset,
                                                     val_indices))
        
        
        samples_in_each_class_dict = Counter([data[1] for data in self.data_train])
        self.N_CLASS = len(self.label_mapping)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
           self.samples_in_each_class[index] = count
        

    def train_dataloader(self):
        
        return DataLoader(self.data_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):

        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    
    def test_dataloader(self):
        if self.data_test == None:
            if self.log_norm:
                self.test_data.X.data = np.log10(self.test_data.X.data+1).astype(np.float32)

            # # hvg selection
            if self.hvg:    
                self.test_data = self.test_data[:,self.high_variable_genes].copy()

            # # perform normalization
            if self.normalize:
                scaler_test = StandardScaler(with_mean=False).fit(self.test_data.X)
                self.test_data.var['means'] = scaler_test.mean_
                self.test_data.var['std'] = scaler_test.scale_

                b = (self.scaler_train.scale_/self.test_data.var['std'])
                a = (self.scaler_train.mean_ - b*self.test_data.var['means'])
                A = np.tile(a.to_numpy().reshape(self.test_data.X.shape[1],1),self.test_data.X.shape[0])
                self.test_data.X = csr_matrix((A + np.diag(b) * self.test_data.X.T).astype(np.float32).T)
                self.test_data.X = self.scaler_train.transform(self.test_data.X) 

            test_labels = label_transform(self.label_mapping, self.test_data.obs[self.cell_type_key])
            self.data_test = SparseCustomDataset(data=self.test_data.X, labels=np.asarray(test_labels))

    
        return DataLoader(self.data_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)
    
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