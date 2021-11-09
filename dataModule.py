from collections import Counter
import torch
from torchvision import models
from torch import nn
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torch.nn import functional as F
from torchvision import datasets, transforms
import os
from collections import Counter
import statistics
import scipy.sparse

from torchvision.datasets.utils import download_and_extract_archive
import pandas as pd
import numpy as np
from sklearn import preprocessing

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

###------------------------------Utility function for DataLoader---------------------------------###
# selecting genes, input is the pandas dataframe
def gene_selection(df, num_of_gene=10000):
  print("Before feature selection:")
  print(df)
  topMeanGene = df.mean().sort_values(ascending=False)
  topVarGene = df.var().sort_values(ascending=False)
  selectedgene = list(set(topMeanGene.index[0:num_of_gene]).union(set(topVarGene.index[0:num_of_gene])))
  print("After feature selection:")
  print(df[selectedgene])
  return df[selectedgene]


# Used to discard labels with occurence smaller than 10
def preprocess_data(full_labels, import_size=None):

    # Step #1: Convert from string into integer labels
    le = preprocessing.LabelEncoder()
    le.fit(full_labels.ravel())
    int_labels = le.transform(full_labels.ravel())

    # Step #2: Prepare indices for the proportion of data that we are going to read in
    full_indices = range(len(full_labels))
    if import_size == 1 or import_size is None:
        import_indices = full_indices
        discarded_indices = []
    else:
        import_indices, discarded_indices = train_test_split(
            full_indices, train_size=import_size, stratify=int_labels, random_state=21)

    print("Number of data to import:", len(import_indices))
    print("Number of total data:", len(full_labels))

    # Step #3: Preprocess data and only keep cells with population larger than 10
    imported_labels = [int_labels[i] for i in import_indices]
    occurence_dict = Counter(imported_labels)
    remove_labels = []
    for label in occurence_dict:
        num_of_occurence = occurence_dict[label]
        if num_of_occurence < 10:
            remove_labels.append(label)
    imported_labels = None
    import_indices = None

    for label in remove_labels:
        remove_temp = [i for i, x in enumerate(int_labels) if x == label]
        discarded_indices.extend(remove_temp)

    remaining_indices = [e for e in full_indices if e not in discarded_indices]
    remaining_labels = [int_labels[i] for i in remaining_indices]
    print("Number of data after filtering:", len(remaining_indices))
    print("Number of classes after filtering:",
          len(np.unique(remaining_labels)))

    remaining_labels = list(le.inverse_transform(remaining_labels))

    return remaining_labels, discarded_indices

# Perform stratified split on a dataset into two sets based on indices


def stratified_split(remaining_indices, full_labels, set1_split_percentage):
    target_labels = [full_labels[i] for i in remaining_indices]
    set1_indices, set2_indices = train_test_split(
        remaining_indices, train_size=set1_split_percentage, stratify=target_labels)
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


###------------------------------Data Module---------------------------------###

class TMDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', batch_size=64, num_workers=2, import_size=0.4, fold_num=0, feature_selection=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "TM"
        self.import_size = import_size # Percentage of original dataset, Total dataset size = 54865. Went to 0.4 without crashing
        self.label_mapping = None
        self.N_FEATURES = 19791
        self.N_CLASS = 55
        self.fold_num = fold_num # fold_num 可以是 0-4

        self.feature_selection = feature_selection
        if feature_selection:
          DataPath = self.data_dir + "/" + self.data_name + "/Filtered_TM_data.csv"
          data = pd.read_csv(DataPath, index_col=0, sep=',')
          data = gene_selection(data)
          self.N_FEATURES = data.shape[1]

    def prepare_data(self):
        # download
        if not os.path.exists(f"{self.data_dir}/{self.data_name}"):
          print("Start downloading TM!")
          url = 'https://github.com/Aprilhuu/Deep-Learning-in-Single-Cell-Analysis/raw/main/TM.zip'
          download_and_extract_archive(url, f"{self.data_dir}", f"{self.data_dir}", filename=self.data_name + '.zip')

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders

        print("Using fold_num {}".format(self.fold_num))

        DataPath = self.data_dir + "/" + self.data_name + "/Filtered_TM_data.csv"
        LabelsPath = self.data_dir + "/" + self.data_name + "/Labels.csv"

        rf = robjects.r['load'](self.data_dir + "/" + self.data_name + "/CV_folds.RData")

        with localconverter(robjects.default_converter + pandas2ri.converter):
          test_idx = robjects.conversion.rpy2py(robjects.r['Test_Idx'])
          train_idx = robjects.conversion.rpy2py(robjects.r['Train_Idx'])
          cells_to_keep = robjects.conversion.rpy2py(robjects.r['Cells_to_Keep'])

        # Step #1: Read in all labels and keep cells with count > 10
        cells_to_keep = np.array(cells_to_keep, dtype=bool)
        full_labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',')
        full_labels = np.asarray(full_labels)

        remaining_labels = full_labels[cells_to_keep].ravel()

        # Step #2: Turning string class labels into int labels and store the mapping
        self.label_mapping = preprocessing.LabelEncoder()
        self.label_mapping.fit(remaining_labels)

        label_mapping_dict = dict()
        for i in range(len(self.label_mapping.classes_)):
          label_mapping_dict[i] = self.label_mapping.classes_[i]
        print(label_mapping_dict)

        int_labels = self.label_mapping.transform(remaining_labels)
        full_labels = np.asarray(int_labels)

        remaining_labels = None
        int_labels = None

        # Step #3: Read in data based on selected label indices
        data = pd.read_csv(DataPath, index_col=0, sep=',')
        if self.feature_selection:
          data = gene_selection(data)

        full_data = np.asarray(data, dtype=np.float32)
        full_data = full_data[cells_to_keep]

        full_dataset = CustomDataset(data=full_data, labels=full_labels)
        test_idx = np.array(test_idx[self.fold_num]) - 1
        train_idx = np.array(train_idx[self.fold_num]) - 1

        self.TM_test = Subset(full_dataset, test_idx)
        self.TM_train, self.TM_val = split_train_val_database_sets(full_dataset, train_idx, train_percentage=0.75)
        print("train size =", len(self.TM_train))
        print("val size =", len(self.TM_val))
        print("test size =", len(self.TM_test))

        # Calculate sample count in each class for training dataset
        samples_in_each_class_dict = Counter([data[1] for data in self.TM_train])
        print("training samples in each class =", samples_in_each_class_dict)
        samples_in_each_class_dict_val = Counter([data[1] for data in self.TM_val])
        print("val samples in each class =", samples_in_each_class_dict_val)
        samples_in_each_class_dict_test = Counter([data[1] for data in self.TM_test])
        print("test samples in each class =", samples_in_each_class_dict_test)
        self.N_CLASS = len(samples_in_each_class_dict)
        print("Changing N_CLASS =", self.N_CLASS)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
           self.samples_in_each_class[index] = count

    def train_dataloader(self):
        return DataLoader(self.TM_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.TM_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.TM_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)


class BaronHumanDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', batch_size=64, num_workers=2, fold_num=0, feature_selection=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "BaronHuman"
        self.label_mapping = None
        self.fold_num = fold_num # fold_num 可以是 0-4
        self.feature_selection = feature_selection
        self.N_FEATURES = 17499
        if feature_selection:
          DataPath = self.data_dir + "/" + self.data_name + "/Filtered_Baron_HumanPancreas_data.csv"
          data = pd.read_csv(DataPath, index_col=0, sep=',')
          data = gene_selection(data)
          self.N_FEATURES = data.shape[1]

    def prepare_data(self):
        # download
        if not os.path.exists(f"{self.data_dir}/{self.data_name}"):
          print("Start downloading Baron Human!")
          url = 'https://github.com/Aprilhuu/Deep-Learning-in-Single-Cell-Analysis/raw/main/BaronHuman.zip'
          download_and_extract_archive(url, f"{self.data_dir}", f"{self.data_dir}", filename=self.data_name + '.zip')

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders

        print("Using fold_num {}".format(self.fold_num))

        DataPath = self.data_dir + "/" + self.data_name + "/Filtered_Baron_HumanPancreas_data.csv"
        LabelsPath = self.data_dir + "/" + self.data_name + "/Labels.csv"
        print("loading..." + self.data_dir + "/" + self.data_name + "/CV_folds.RData")
        rf = robjects.r['load'](self.data_dir + "/" + self.data_name + "/CV_folds.RData")

        with localconverter(robjects.default_converter + pandas2ri.converter):
          test_idx = robjects.conversion.rpy2py(robjects.r['Test_Idx'])
          train_idx = robjects.conversion.rpy2py(robjects.r['Train_Idx'])
          cells_to_keep = robjects.conversion.rpy2py(robjects.r['Cells_to_Keep'])

        # Step #1: Read in all labels and keep cells with count > 10
        cells_to_keep = np.array(cells_to_keep, dtype=bool)
        full_labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',')
        full_labels = np.asarray(full_labels)

        remaining_labels = full_labels[cells_to_keep].ravel()

        # Step #2: Turning string class labels into int labels and store the mapping
        self.label_mapping = preprocessing.LabelEncoder()
        self.label_mapping.fit(remaining_labels)

        int_labels = self.label_mapping.transform(remaining_labels)
        full_labels = np.asarray(int_labels)

        remaining_labels = None
        int_labels = None

        # Step #3: Read in data based on selected label indices
        data = pd.read_csv(DataPath, index_col=0, sep=',')
        if self.feature_selection:
          data = gene_selection(data)

        full_data = np.asarray(data, dtype=np.float32)
        full_data = full_data[cells_to_keep]

        full_dataset = CustomDataset(data=full_data, labels=full_labels)
        test_idx = np.array(test_idx[self.fold_num]) - 1
        train_idx = np.array(train_idx[self.fold_num]) - 1

        self.Baron_Human_test = Subset(full_dataset, test_idx)
        self.Baron_Human_train, self.Baron_Human_val = split_train_val_database_sets(full_dataset, train_idx, train_percentage=0.75)
        print("train size =", len(self.Baron_Human_train))
        print("val size =", len(self.Baron_Human_val))
        print("test size =", len(self.Baron_Human_test))

        # Calculate sample count in each class for training dataset
        samples_in_each_class_dict = Counter([data[1] for data in self.Baron_Human_train])
        print("training samples in each class =", samples_in_each_class_dict)
        samples_in_each_class_dict_val = Counter([data[1] for data in self.Baron_Human_val])
        print("val samples in each class =", samples_in_each_class_dict_val)
        samples_in_each_class_dict_test = Counter([data[1] for data in self.Baron_Human_test])
        print("test samples in each class =", samples_in_each_class_dict_test)
        self.N_CLASS = len(samples_in_each_class_dict)
        print("Changing N_CLASS =", self.N_CLASS)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
           self.samples_in_each_class[index] = count

    def train_dataloader(self):
        return DataLoader(self.Baron_Human_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.Baron_Human_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.Baron_Human_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)


class Zheng68KDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', batch_size=64, num_workers=2, import_size=0.4, fold_num=0, feature_selection=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "Zheng_68K"
        self.import_size = import_size # Percentage of original dataset, Total dataset size = 54865. Went to 0.4 without crashing
        self.label_mapping = None
        self.N_FEATURES = 20387
        self.N_CLASS = 11
        self.fold_num = fold_num # fold_num 可以是 0-4
        self.feature_selection = feature_selection
        if self.feature_selection:
          DataPath = self.data_dir + "/" + self.data_name + "/Filtered_68K_PBMC_data.csv"
          data = pd.read_csv(DataPath, index_col=0, sep=',')
          data = gene_selection(data)
          self.N_FEATURES = data.shape[1]

    def prepare_data(self):
        # download
        if not os.path.exists(f"{self.data_dir}/{self.data_name}"):
          print("Start downloading Zheng 68K!")
          url = 'https://github.com/Aprilhuu/Deep-Learning-in-Single-Cell-Analysis/raw/main/Zheng_68K.zip'
          download_and_extract_archive(url, f"{self.data_dir}", f"{self.data_dir}", filename=self.data_name + '.zip')

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders
        
        print("Using fold_num {}".format(self.fold_num))

        DataPath = self.data_dir + "/" + self.data_name + "/Filtered_68K_PBMC_data.csv"
        LabelsPath = self.data_dir + "/" + self.data_name + "/Labels.csv"

        rf = robjects.r['load'](self.data_dir + "/" + self.data_name + "/CV_folds.RData")

        with localconverter(robjects.default_converter + pandas2ri.converter):
          test_idx = robjects.conversion.rpy2py(robjects.r['Test_Idx'])
          train_idx = robjects.conversion.rpy2py(robjects.r['Train_Idx'])
          cells_to_keep = robjects.conversion.rpy2py(robjects.r['Cells_to_Keep'])

        # Step #1: Read in all labels and keep cells with count > 10
        cells_to_keep = np.array(cells_to_keep, dtype=bool)
        full_labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',')
        full_labels = np.asarray(full_labels)

        remaining_labels = full_labels[cells_to_keep].ravel()

        # Step #2: Turning string class labels into int labels and store the mapping
        self.label_mapping = preprocessing.LabelEncoder()
        self.label_mapping.fit(remaining_labels)

        int_labels = self.label_mapping.transform(remaining_labels)
        full_labels = np.asarray(int_labels)

        remaining_labels = None
        int_labels = None

        # Step #3: Read in data based on selected label indices
        data = pd.read_csv(DataPath, index_col=0, sep=',')
        if self.feature_selection:
          data = gene_selection(data)

        full_data = np.asarray(data, dtype=np.float32)
        full_data = full_data[cells_to_keep]

        full_dataset = CustomDataset(data=full_data, labels=full_labels)
        test_idx = np.array(test_idx[self.fold_num]) - 1
        train_idx = np.array(train_idx[self.fold_num]) - 1

        self.Zheng_68K_test = Subset(full_dataset, test_idx)
        self.Zheng_68K_train, self.Zheng_68K_val = split_train_val_database_sets(full_dataset, train_idx, train_percentage=0.75)
        print("train size =", len(self.Zheng_68K_train))
        print("val size =", len(self.Zheng_68K_val))
        print("test size =", len(self.Zheng_68K_test))

        # Calculate sample count in each class for training dataset
        samples_in_each_class_dict = Counter([data[1] for data in self.Zheng_68K_train])
        print("training samples in each class =", samples_in_each_class_dict)
        samples_in_each_class_dict_val = Counter([data[1] for data in self.Zheng_68K_val])
        print("val samples in each class =", samples_in_each_class_dict_val)
        samples_in_each_class_dict_test = Counter([data[1] for data in self.Zheng_68K_test])
        print("test samples in each class =", samples_in_each_class_dict_test)
        self.N_CLASS = len(samples_in_each_class_dict)
        print("Changing N_CLASS =", self.N_CLASS)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
           self.samples_in_each_class[index] = count

    def train_dataloader(self):
        return DataLoader(self.Zheng_68K_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.Zheng_68K_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.Zheng_68K_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)


class AMBDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', batch_size=64, num_workers=2, annotation_level=92, fold_num=0, feature_selection=False):
        super().__init__()
        assert annotation_level in [3, 16, 92], "Annotation level must be one of 3, 16 or 92!"
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "AMB"
        self.annotation_level = annotation_level
        self.label_mapping = None
        self.fold_num = fold_num # fold_num 可以是 0-4
        self.feature_selection=feature_selection
        self.N_FEATURES = 42625

        if self.feature_selection:
          DataPath = self.data_dir + "/" + self.data_name + "/Filtered_mouse_allen_brain_data.csv"
          data = pd.read_csv(DataPath, index_col=0, sep=',')
          data = gene_selection(data)
          self.N_FEATURES = data.shape[1]

    def prepare_data(self):
        # download
        if not os.path.exists(f"{self.data_dir}/{self.data_name}"):
          print("Start downloading AMB!")
          url = 'https://github.com/Aprilhuu/Deep-Learning-in-Single-Cell-Analysis/raw/main/AMB.zip'
          download_and_extract_archive(url, f"{self.data_dir}", f"{self.data_dir}", filename=self.data_name + '.zip')

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders

        print("Using fold_num {}".format(self.fold_num))

        DataPath = self.data_dir + "/" + self.data_name + "/Filtered_mouse_allen_brain_data.csv"
        LabelsPath = self.data_dir + "/" + self.data_name + "/Labels.csv"

        rf = robjects.r['load'](self.data_dir + "/" + self.data_name + "/CV_folds.RData")

        with localconverter(robjects.default_converter + pandas2ri.converter):
          test_idx = robjects.conversion.rpy2py(robjects.r['Test_Idx'])
          train_idx = robjects.conversion.rpy2py(robjects.r['Train_Idx'])
          cells_to_keep = robjects.conversion.rpy2py(robjects.r['Cells_to_Keep'])

        # Step #1: Read in all labels and keep cells with count > 10
        cells_to_keep = np.array(cells_to_keep, dtype=bool)

        if self.annotation_level == 92:
          full_labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',', usecols = ['cluster'])
        elif self.annotation_level == 16:
          full_labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',', usecols = ['Subclass'])
        else:
          full_labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',', usecols = ['Class'])
          
        full_labels = np.asarray(full_labels)
        remaining_labels = full_labels[cells_to_keep].ravel()

        # Step #2: Turning string class labels into int labels and store the mapping
        self.label_mapping = preprocessing.LabelEncoder()
        self.label_mapping.fit(remaining_labels)

        label_mapping_dict = dict()
        for i in range(len(self.label_mapping.classes_)):
          label_mapping_dict[i] = self.label_mapping.classes_[i]

        print("Label name to integer mappings:", label_mapping_dict)

        int_labels = self.label_mapping.transform(remaining_labels)
        full_labels = np.asarray(int_labels)

        remaining_labels = None
        int_labels = None

        # Step #3: Read in data based on selected label indices
        data = pd.read_csv(DataPath, index_col=0, sep=',')
        if self.feature_selection:
          data = gene_selection(data)
        full_data = np.asarray(data, dtype=np.float32)
        full_data = full_data[cells_to_keep]

        full_dataset = CustomDataset(data=full_data, labels=full_labels)
        test_idx = np.array(test_idx[self.fold_num]) - 1
        train_idx = np.array(train_idx[self.fold_num]) - 1

        self.AMB_test = Subset(full_dataset, test_idx)
        self.AMB_train, self.AMB_val = split_train_val_database_sets(full_dataset, train_idx, train_percentage=0.75)
        print("train size =", len(self.AMB_train))
        print("val size =", len(self.AMB_val))
        print("test size =", len(self.AMB_test))

        # Calculate sample count in each class for training dataset
        samples_in_each_class_dict = Counter([data[1] for data in self.AMB_train])
        print("training samples in each class =", samples_in_each_class_dict)
        samples_in_each_class_dict_val = Counter([data[1] for data in self.AMB_val])
        print("val samples in each class =", samples_in_each_class_dict_val)
        samples_in_each_class_dict_test = Counter([data[1] for data in self.AMB_test])
        print("test samples in each class =", samples_in_each_class_dict_test)
        self.N_CLASS = len(samples_in_each_class_dict)
        print("Changing N_CLASS =", self.N_CLASS)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
           self.samples_in_each_class[index] = count

    def train_dataloader(self):
        return DataLoader(self.AMB_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.AMB_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.AMB_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)


class XinDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', batch_size=64, num_workers=2, fold_num=0, feature_selection=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "Xin"
        self.label_mapping = None
        self.fold_num = fold_num # fold_num 可以是 0-4
        self.feature_selection = feature_selection
        self.N_FEATURES = 33889

        if feature_selection:
          DataPath = self.data_dir + "/" + self.data_name + "/Filtered_Xin_HumanPancreas_data.csv"
          data = pd.read_csv(DataPath, index_col=0, sep=',')
          data = gene_selection(data)
          self.N_FEATURES = data.shape[1]

    def prepare_data(self):
        # download
        if not os.path.exists(f"{self.data_dir}/{self.data_name}"):
          print("Start downloading Xin!")
          url = 'https://github.com/Aprilhuu/Deep-Learning-in-Single-Cell-Analysis/raw/main/Xin.zip'
          download_and_extract_archive(url, f"{self.data_dir}", f"{self.data_dir}", filename=self.data_name + '.zip')

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders

        DataPath = self.data_dir + "/" + self.data_name + "/Filtered_Xin_HumanPancreas_data.csv"
        LabelsPath = self.data_dir + "/" + self.data_name + "/Labels.csv"

        rf = robjects.r['load'](self.data_dir + "/" + self.data_name + "/CV_folds.RData")

        with localconverter(robjects.default_converter + pandas2ri.converter):
          test_idx = robjects.conversion.rpy2py(robjects.r['Test_Idx'])
          train_idx = robjects.conversion.rpy2py(robjects.r['Train_Idx'])
          cells_to_keep = robjects.conversion.rpy2py(robjects.r['Cells_to_Keep'])

        # Step #1: Read in all labels and keep cells with count > 10
        cells_to_keep = np.array(cells_to_keep, dtype=bool)
        full_labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',')
        full_labels = np.asarray(full_labels)

        remaining_labels = full_labels[cells_to_keep].ravel()

        # Step #2: Turning string class labels into int labels and store the mapping
        self.label_mapping = preprocessing.LabelEncoder()
        self.label_mapping.fit(remaining_labels)

        int_labels = self.label_mapping.transform(remaining_labels)
        full_labels = np.asarray(int_labels)

        remaining_labels = None
        int_labels = None

        # Step #3: Read in data based on selected label indices
        data = pd.read_csv(DataPath, index_col=0, sep=',')
        if self.feature_selection:
          data = gene_selection(data)
        full_data = np.asarray(data, dtype=np.float32)
        full_data = full_data[cells_to_keep]

        full_dataset = CustomDataset(data=full_data, labels=full_labels)
        test_idx = np.array(test_idx[self.fold_num]) - 1
        train_idx = np.array(train_idx[self.fold_num]) - 1

        self.Xin_test = Subset(full_dataset, test_idx)
        self.Xin_train, self.Xin_val = split_train_val_database_sets(full_dataset, train_idx, train_percentage=0.75)
        print("train size =", len(self.Xin_train))
        print("val size =", len(self.Xin_val))
        print("test size =", len(self.Xin_test))

        # Calculate sample count in each class for training dataset
        samples_in_each_class_dict = Counter([data[1] for data in self.Xin_train])
        print("training samples in each class =", samples_in_each_class_dict)
        samples_in_each_class_dict_val = Counter([data[1] for data in self.Xin_val])
        print("val samples in each class =", samples_in_each_class_dict_val)
        samples_in_each_class_dict_test = Counter([data[1] for data in self.Xin_test])
        print("test samples in each class =", samples_in_each_class_dict_test)
        self.N_CLASS = len(samples_in_each_class_dict)
        print("Changing N_CLASS =", self.N_CLASS)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
           self.samples_in_each_class[index] = count

    def train_dataloader(self):
        return DataLoader(self.Xin_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.Xin_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.Xin_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)


class FetalDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', batch_size=128, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "Fetal"
        self.label_mapping = None

    def prepare_data(self):
        # download
        if not os.path.exists(f"{self.data_dir}/{self.data_name}"):
          print("No data is available!!!")


    def setup(self, stage):
        DataPath = self.data_dir + "/" + self.data_name + "/sparse_matrix.npz"
        LabelsPath = self.data_dir + "/" + self.data_name + "/type.csv"

        # Step #1: Read in all cells and labels
        full_labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',')
        full_labels = np.asarray(full_labels)

        # Step #2: Turning string class labels into int labels and store the mapping
        self.label_mapping = preprocessing.LabelEncoder()
        self.label_mapping.fit(full_labels)

        int_labels = self.label_mapping.transform(full_labels)
        full_labels = np.asarray(int_labels)

        int_labels = None

        label_mapping_dict = dict()
        for i in range(len(self.label_mapping.classes_)):
          label_mapping_dict[i] = self.label_mapping.classes_[i]

        print("Label name to integer mappings:", label_mapping_dict)

        # Step #3: Read in all data
        data = scipy.sparse.load_npz(DataPath)
        full_data = data.T
        full_data = full_data.astype(np.float32)
        print(full_data.shape)
       
        
        # Step #4: Train test split
        x_train, x_val, y_train, y_val = train_test_split(full_data, full_labels, test_size=0.2, random_state=11, stratify=full_labels)
        
        self.Fetal_train = SparseCustomDataset(data=x_train, labels=y_train)
        self.Fetal_val = SparseCustomDataset(data=x_val, labels=y_val)

        print("train size =", len(self.Fetal_train))
        print("val size =", len(self.Fetal_val))

        # Calculate sample count in each class for training dataset
        samples_in_each_class_dict = Counter([data[1] for data in self.Fetal_train])
        print("training samples in each class =", samples_in_each_class_dict)
        samples_in_each_class_dict_val = Counter([data[1] for data in self.Fetal_val])
        print("val samples in each class =", samples_in_each_class_dict_val)

        self.N_CLASS = len(samples_in_each_class_dict)
        print("Changing N_CLASS =", self.N_CLASS)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
           self.samples_in_each_class[index] = count


    def train_dataloader(self):
        return DataLoader(self.Fetal_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.Fetal_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.Fetal_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)                         


class Pbmc68kDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', batch_size=128, num_workers=2, fold_num=0, feature_selection=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "pbmc68k"
        self.label_mapping = None
        self.fold_num = fold_num # fold_num 可以是 0-4
        self.feature_selection = feature_selection
        self.N_FEATURES = 1000

    def prepare_data(self):
        # download
        if not os.path.exists(f"{self.data_dir}/{self.data_name}"):
          print("No data is available!!!")


    def setup(self, stage):
        DataPath = self.data_dir + "/" + self.data_name + "/zheng68k.csv"
        LabelsPath = self.data_dir + "/" + self.data_name + "/Labels.csv"
        DataSplitPath = self.data_dir + "/" + self.data_name + "/pbmc68k_split.npz"

        # Step #1: Read in all cells and labels
        full_labels = pd.read_csv(LabelsPath, header=0, index_col=0, sep=',')
        full_labels = np.asarray(full_labels)

        # Step #2: Turning string class labels into int labels and store the mapping
        self.label_mapping = preprocessing.LabelEncoder()
        self.label_mapping.fit(full_labels)

        int_labels = self.label_mapping.transform(full_labels)
        full_labels = np.asarray(int_labels)

        int_labels = None

        label_mapping_dict = dict()
        for i in range(len(self.label_mapping.classes_)):
          label_mapping_dict[i] = self.label_mapping.classes_[i]

        print("Label name to integer mappings:", label_mapping_dict)

        # Step #3: Read in all data
        data = pd.read_csv(DataPath, index_col=0, sep=',')
        full_data = np.asarray(data, dtype=np.float32)
        print(full_data.shape)
       
        
        # Step #4: Train test split
        datasplit = np.load(DataSplitPath, allow_pickle=True)
        train_idx = datasplit['train_idx']
        test_idx = datasplit['test_idx']

        #x_train, x_val, y_train, y_val = train_test_split(full_data, full_labels, test_size=0.2, random_state=11, stratify=full_labels)
        x_train = full_data[train_idx[self.fold_num]]
        x_test = full_data[test_idx[self.fold_num]]
        y_train = full_labels[train_idx[self.fold_num]]
        y_test = full_labels[test_idx[self.fold_num]]

        self.pbmc_train = CustomDataset(data=x_train, labels=y_train)
        self.pbmc_test = CustomDataset(data=x_test, labels=y_test)

        print("train size =", len(self.pbmc_train))
        print("test size =", len(self.pbmc_test))

        # Calculate sample count in each class for training dataset
        samples_in_each_class_dict = Counter([data[1] for data in self.pbmc_train])
        print("training samples in each class =", samples_in_each_class_dict)
        samples_in_each_class_dict_val = Counter([data[1] for data in self.pbmc_test])
        print("test samples in each class =", samples_in_each_class_dict_val)

        self.N_CLASS = len(samples_in_each_class_dict)
        print("Changing N_CLASS =", self.N_CLASS)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
           self.samples_in_each_class[index] = count


    def train_dataloader(self):
        return DataLoader(self.pbmc_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.pbmc_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.pbmc_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)     