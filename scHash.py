import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
import numpy as np
import argparse
import anndata as ad
import resource

from util import *
from dataModule import *

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def get_class_balance_loss_weight(samples_in_each_class, n_class, beta=0.9999):
    # Class-Balanced Loss on Effective Number of Samples
    # Reference Paper https://arxiv.org/abs/1901.05555
    weight = (1 - beta)/(1 - torch.pow(beta, samples_in_each_class))
    weight = weight / weight.sum() * n_class
    return weight


def test_class_balance_loss():
    print("Testing class balance loss...")
    # Sample 1
    samples_in_each_class, n_class = torch.tensor([15, 10, 10, 10, 19]), 5
    corr1 = torch.tensor(
        [0.79511815, 1.1923454, 1.1923454, 1.1923454, 0.6278458])
    ans1 = get_class_balance_loss_weight(samples_in_each_class, n_class)
    if np.array_equal(corr1.numpy(), ans1.numpy()):
        print("Test 1 passes")
    else:
        print("Test1 failed", "ans1 =", ans1.numpy(),
              "correct1 =", corr1.numpy())

    # Sample 2
    samples_in_each_class, n_class = torch.tensor([1, 1, 1, 1, 1]), 5
    corr2 = torch.tensor([1., 1., 1., 1., 1.])
    ans2 = get_class_balance_loss_weight(samples_in_each_class, n_class)
    if np.array_equal(corr2.numpy(), ans2.numpy()):
        print("Test 2 passes")
    else:
        print("Test2 failed", "ans2 =", ans2.numpy(),
              "correct2 =", corr2.numpy())

    # Sample 3
    samples_in_each_class, n_class = torch.tensor([1, 2, 4, 8, 16]), 5
    corr3 = torch.tensor(
        [2.5801828, 1.2904761, 0.64523804, 0.32269114, 0.16141175])
    ans3 = get_class_balance_loss_weight(samples_in_each_class, n_class)
    if np.array_equal(corr3.numpy(), ans3.numpy()):
        print("Test 3 passes")
    else:
        print("Test3 failed", "ans3 =", ans3.numpy(),
              "correct3 =", corr3.numpy())

    return

###------------------------------Model---------------------------------------###

class scDeepHashModel(pl.LightningModule):
    def __init__(self, n_class, n_features, batch_size=64, l_r=1e-5, lamb_da=0.0001, beta=0.9999, bit=64, lr_decay=0.9, decay_every=20, n_layers=5, weight_decay=0.0005, measure_retrieval=False, topK=-1):
        super(scDeepHashModel, self).__init__()
        print("hparam: l_r = {}, lambda = {}, beta = {}".format(l_r, lamb_da, beta))
        self.batch_size = batch_size
        self.l_r = l_r
        self.bit = bit
        self.n_class = n_class
        self.lamb_da = lamb_da
        self.beta = beta
        self.lr_decay = lr_decay
        self.decay_every = decay_every
        self.samples_in_each_class = None  # Later initialized in training step
        self.cell_anchors = get_cell_anchors(self.n_class, self.bit)
        self.n_layers = n_layers
        self.weight_decay = weight_decay
        self.measure_retrieval = measure_retrieval
        self.topK = topK
        ##### model structure ####
        if n_layers == 5:
            self.hash_layer = nn.Sequential(
                nn.Linear(n_features, 9000),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(9000, 3150),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(3150, 900),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(900, 450),
                nn.ReLU(inplace=True),
                nn.Linear(450, 200),
                nn.ReLU(inplace=True),
                nn.Linear(200, self.bit),
            )
        elif n_layers == 4:
            self.hash_layer = nn.Sequential(
                nn.Linear(n_features, 5000),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(5000, 2000),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(2000, 800),
                nn.ReLU(inplace=True),
                nn.Linear(800, 300),
                nn.ReLU(inplace=True),
                nn.Linear(300, self.bit),
            )
        elif n_layers == 3:
            self.hash_layer = nn.Sequential(
                nn.Linear(n_features, 500),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(500, 250),
                nn.ReLU(),
                nn.Linear(250, self.bit),
                )
             
            
    def forward(self, x):
        # forward pass returns prediction
        x = self.hash_layer(x)
        return x

    def get_class_balance_loss_weight(samples_in_each_class, n_class, beta=0.9999):
        # Class-Balanced Loss on Effective Number of Samples
        # Reference Paper https://arxiv.org/abs/1901.05555
        weight = (1 - beta)/(1 - torch.pow(beta, samples_in_each_class))
        weight = weight / weight.sum() * n_class
        return weight

    def loss_functions(self, hash_codes, labels):
        hash_codes = hash_codes.tanh()
        cell_anchors = self.cell_anchors[labels]
        cell_anchors = cell_anchors.type_as(hash_codes)

        if self.samples_in_each_class == None:
            self.samples_in_each_class = self.trainer.datamodule.samples_in_each_class
            self.n_class = self.trainer.datamodule.N_CLASS

        weight = get_class_balance_loss_weight(
            self.samples_in_each_class, self.n_class, self.beta)
        weight = weight[labels]
        weight = weight.type_as(hash_codes)

        # Center Similarity Loss
        BCELoss = nn.BCELoss(weight=weight.unsqueeze(1).repeat(1, self.bit))
        cell_anchor_loss = BCELoss(0.5 * (hash_codes + 1),
                         0.5 * (cell_anchors + 1))
        # Quantization Loss
        Q_loss = (hash_codes.abs() - 1).pow(2).mean()

        loss = cell_anchor_loss + self.lamb_da * Q_loss
        return loss

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        hash_codes = self.forward(data)
        loss = self.loss_functions(hash_codes, labels)
        return loss

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch
        hash_codes = self.forward(data)
        loss = self.loss_functions(hash_codes, labels)
        return loss

    def validation_epoch_end(self, outputs):

        val_loss_epoch = torch.stack([x for x in outputs]).mean()

        val_dataloader = self.trainer.datamodule.val_dataloader()
        train_dataloader = self.trainer.datamodule.train_dataloader()

        val_matrics_CHC = compute_metrics(val_dataloader, self, self.n_class)
        (val_labeling_accuracy_CHC, 
        val_F1_score_weighted_average_CHC, val_F1_score_median_CHC, val_F1_score_per_class_CHC,
        val_precision, val_recall,
        nmi, ari, map_score, _) = val_matrics_CHC

        train_matrics_CHC = compute_metrics(train_dataloader, self, self.n_class)
        (_, 
        _, train_F1_score_median_CHC, _, _,
        _, _,
        _, _, _) = train_matrics_CHC

        if not self.trainer.sanity_checking:
            print(f"Epoch: {self.current_epoch}, Val_loss_epoch: {val_loss_epoch:.2f}")
            print(f"val_F1_score_median_CHC:{val_F1_score_median_CHC:.3f}, \
                    val_labeling_accuracy_CHC:{val_labeling_accuracy_CHC:.3f},\
                    val_F1_score_weighted_average_CHC:{val_F1_score_weighted_average_CHC:.3f},\
                    val_F1_score_per_class_CHC:{[f'{score:.3f}' for score in val_F1_score_per_class_CHC]}, \
                    val_precision:{val_precision:.3f}, \
                    val_recall:{val_recall:.3f}, \
                    val_NMI: {nmi:.3f}, \
                    val_ARI: {ari:.3f}, \
                    train_F1_score_median_CHC: {train_F1_score_median_CHC:.3f}")
            print("map score =", map_score)


        value = {"step":self.current_epoch,
                 "Val_loss_epoch": val_loss_epoch, 
                  "Val_F1_score_median_CHC_epoch": val_F1_score_median_CHC,
                  "Val_labeling_accuracy_CHC_epoch": val_labeling_accuracy_CHC, 
                  "Val_F1_score_weighted_average_CHC_epoch": val_F1_score_weighted_average_CHC,
                  "Val_precision:" : val_precision,
                  "Val_recall:" : val_recall,
                  "Val_NMI:" : nmi,
                  "Val_ARI:" : ari,
                  "Train_F1_score_median_CHC:" : train_F1_score_median_CHC}

        self.log_dict(value, prog_bar=True, logger=True,on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        data, labels = test_batch
        data, labels = test_batch
        hash_codes = self.forward(data)
        loss = self.loss_functions(hash_codes, labels)

        return loss

    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x for x in outputs]).mean()
        
        test_dataloader = self.trainer.datamodule.test_dataloader()

        test_matrics_CHC = test_compute_metrics(test_dataloader, self, self.n_class, show_time=True, use_cpu=False, measure_retrieval=self.measure_retrieval, topK=self.topK)
        
        (nmi,accuracy,precision,recall,f1,hashing_time, cell_assign_time, query_time, f1_median) = test_matrics_CHC
        

        value = {"Test_NMI": nmi,
                 "Test_F1": f1,
                 "Test_F1_Median": f1_median,
                 "Test_precision" : precision,
                 "Test_recall" : recall,
                "Test_hashing_time": hashing_time,
                "Test_cell_assign_time": cell_assign_time,
                'Test_query_time': query_time,
                'Test_accuracy':accuracy }

        self.log_dict(value, prog_bar=True, logger=True,on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.l_r, weight_decay=self.weight_decay)


        # Decay LR by a factor of gamma every step_size epochs
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer, step_size=self.decay_every, gamma=self.lr_decay)

        return [optimizer], [exp_lr_scheduler]


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    # Parse parameters
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument("--l_r", type=float, default=1.2e-5,
                        help="learning rate")
    parser.add_argument("--lamb", type=float, default=0.001,
                        help="lambda of quantization loss")
    parser.add_argument("--beta", type=float, default=0.9999,
                        help="beta of class balance loss")
    parser.add_argument("--lr_decay", type=float, default=0.5,
                        help="learning rate decay")
    parser.add_argument("--decay_every", type=int, default=100,
                        help="how many epochs a learning rate happens")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="weight decay (L2 penalty)")
    parser.add_argument("--n_layers", type=int, default=5,
                        help="number of layers")
    parser.add_argument("--fold_number", type=int, default=0,
                        help = "5-fold number")
    # Training parameters
    parser.add_argument("--epochs", type=int, default=301,
                        help="number of epochs to run")
    parser.add_argument("--dataset", default='',
                        help="dataset to train against")
    # Control parameters
    parser.add_argument("--test", type=str, default='',
                        help="To test against a specific checkpoint")
    parser.add_argument("--measure_retrieval", type=bool, default=False,
                        help="Whether to measure retrieval metrics (MAP)")
    parser.add_argument("--topK", type=int, default=-1,
                        help="topK for MAP")
    parser.add_argument("--feature_selection", type=bool, default=False,
                        help="Whether to use feature selection for input data")
    parser.add_argument("--checkpoint_path", type=str,
                        help="The path to save checkpoints")
    parser.add_argument("--query", type=str,
                        help="The query dataset")
    parser.add_argument("--method", type=str, default='scDeepHash Raw data',
                        help="The query dataset")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="The query dataset")
    parser.add_argument("--data_dir", type=str, default='../../Benchmark/fivePancreas_Mapping/scDeepHash1.csv',
                        help="The query dataset")
    parser.add_argument("--hvg", type=str, default='False',
                        help="select highly variable genes")
    parser.add_argument("--log_norm", type=str, default='False',
                        help="Log(X+1) normalize all data")
    parser.add_argument("--normalize", type=str, default='False',
                        help="normalize by x-u/sigma")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size")
    parser.add_argument("--batch_key", type=str, default='',
                        help="batch key")
    args = parser.parse_args()

    l_r = args.l_r
    lamb_da = args.lamb
    beta = args.beta
    weight_decay = args.weight_decay
    n_layers = args.n_layers
    fold_number = args.fold_number

    max_epochs = args.epochs
    dataset = args.dataset
    lr_decay = args.lr_decay
    batch_size = args.batch_size
    batch_key = args.batch_key

    decay_every = args.decay_every
    test_checkpoint = args.test
    measure_retrieval = args.measure_retrieval
    topK = args.topK
    feature_selection = args.feature_selection
    checkpoint_path = args.checkpoint_path
    query = args.query
    method = args.method
    result_dir = args.result_dir
    data_dir = args.data_dir
    hvg = True if args.hvg =='True' else False
    log_norm = True if args.log_norm =='True' else False
    normalize = True if args.normalize =='True' else False
    top_genes_count = '1000_seuratv3' if hvg else ''
    

    # set up datamodule    
    if dataset in ['symphony_tms','tms']:
        datamodule = Intra_DataModule(data_dir = data_dir, cell_type_key = 'cell_ontology_class', batch_key = batch_key, num_workers=4, batch_size=batch_size, hvg = hvg,log_norm = log_norm, normalize = normalize)
    else:
        datamodule = Cross_DataModule(data_dir = data_dir, batch_key = batch_key, num_workers=4, batch_size=batch_size, query = query, hvg = hvg, log_norm = log_norm, normalize=normalize)

    datamodule.setup(None)
    N_CLASS = datamodule.N_CLASS
    N_FEATURES = datamodule.N_FEATURES

    # Init ModelCheckpoint callback
    checkpointPath = checkpoint_path + dataset
    print("Feature size =", datamodule.N_FEATURES)
    # Train
    if test_checkpoint == '':
        checkpoint_callback = ModelCheckpoint(
                                    monitor='Val_F1_score_median_CHC_epoch',
                                    dirpath=checkpointPath,
                                    filename='scDeepHash-{epoch:02d}-{Val_F1_score_median_CHC_epoch:.3f}',
                                    verbose=True,
                                    # save_last = True,
                                    mode='max'
                                    )
        early_stopping_callback = EarlyStopping(monitor="Val_F1_score_median_CHC_epoch")
        start = time.time()
        trainer = pl.Trainer(max_epochs=max_epochs,
                            gpus=1,
                            check_val_every_n_epoch=10,
                            progress_bar_refresh_rate=50,
                            # limit_train_batches=0.2,
                            # limit_val_batches=0.2,
                            callbacks=[checkpoint_callback]
                            )
        print("Number of Feature: ", N_FEATURES)
        model = scDeepHashModel(N_CLASS, N_FEATURES, l_r=l_r, lamb_da=lamb_da,
                            beta=beta, lr_decay=lr_decay, decay_every=decay_every,
                            n_layers=n_layers, weight_decay=weight_decay,
                            measure_retrieval=measure_retrieval, topK=topK)

        trainer.fit(model = model, datamodule = datamodule)
        ref_building = time.time()-start
        
        ref_memory = torch.cuda.max_memory_allocated()/ 1024 ** 3
        cpu_mem =  resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2

        # Test the best model
        best_model_path = checkpoint_callback.best_model_path
        trainer = pl.Trainer(max_epochs=max_epochs,
                gpus=1,
                check_val_every_n_epoch=5,
                callbacks=[checkpoint_callback]
                )
        best_model = scDeepHashModel.load_from_checkpoint(
            best_model_path, n_class=N_CLASS, n_features=N_FEATURES,
            l_r=l_r, lamb_da=lamb_da,
            beta=beta, lr_decay=lr_decay, decay_every=decay_every,
            n_layers=n_layers, weight_decay=weight_decay)
            
        best_model.eval()
   
        start = time.time()
        trainer.test(model=best_model, datamodule=datamodule)
        mapping = time.time()-start
        query_mem =  resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 - cpu_mem

        result = trainer.callback_metrics
        nmi = round(result['Test_NMI'].item(),3)
        p = round(result['Test_precision'].item(),3)
        r = round(result['Test_NMI'].item(),3)
        f = round(result['Test_F1'].item(),3)
        h = round(result['Test_hashing_time'].item(),3)
        c = round(result['Test_cell_assign_time'].item(),3)
        q = round(result['Test_query_time'].item(),3)
        a = round(result['Test_accuracy'].item(),3)
        f1_median = round(result['Test_F1_Median'].item(),3)
        

        result_table = pd.read_csv(result_dir)
        
        if dataset in ['symphony_tms','tms','COVID19']:
            result_table = result_table.append({'dataset':dataset,'method':method,"layer_num":n_layers, "accuracy": a, "log_norm": log_norm, "hvg":hvg, 'ref_building_time': round(ref_building,3),'ref_building_gpu_memory(GB)': round(ref_memory,2),'ref_building_cpu_memory(GB)': round(cpu_mem,2), 'query_mapping_time':q, 'hashing_time':h, 'cell_assignment_time': c, 'NMI':nmi,'precision':p,'recall':r,'f1':f,'top_genes_count':top_genes_count, 'normalize': normalize,'epoch':max_epochs,'batch_size':int(batch_size),'query_building_cpu_memory(GB)':round(query_mem,2),'f1_median':f1_median},ignore_index=True)
        else:
            result_table = result_table.append({'query_dataset':query,'method':method,"layer_num":n_layers, "accuracy": a, "log_norm": log_norm, "hvg":hvg, 'ref_building_time': round(ref_building,3),'ref_building_gpu_memory(GB)': round(ref_memory,2),'ref_building_cpu_memory(GB)': round(cpu_mem,2), 'query_mapping_time':q, 'hashing_time':h, 'cell_assignment_time': c, 'NMI':nmi,'precision':p,'recall':r,'f1':f,'top_genes_count':top_genes_count, 'normalize': normalize,'epoch':int(max_epochs),'batch_size':int(batch_size),'query_building_cpu_memory(GB)':round(query_mem,2),'batch_key':batch_key,'f1_median':f1_median},ignore_index=True)
            
        
        result_table = result_table.round({'ref_building_time':3, 'query_mapping_time':3, 'accuracy':3, 'hashing_time':3, 'NMI':3 ,'precision':3, 'recall':3, 'f1':3, 'f1_median':3, 'cell_assignment_time':3,})
        result_table.to_csv(result_dir,index=False)