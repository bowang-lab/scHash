import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import lr_scheduler
import torch.multiprocessing
from .util import *
from pytorch_lightning.callbacks import ModelCheckpoint
import time
torch.multiprocessing.set_sharing_strategy('file_system')


def get_class_balance_loss_weight(samples_in_each_class, n_class, beta=0.9999):
    # Class-Balanced Loss on Effective Number of Samples
    # Reference Paper https://arxiv.org/abs/1901.05555
    weight = (1 - beta)/(1 - torch.pow(beta, samples_in_each_class))
    weight = weight / weight.sum() * n_class
    return weight

###------------------------------Model---------------------------------------###

class scHashModel(pl.LightningModule):
    def __init__(self, datamodule, batch_size=128, l_r=1.2e-5, lamb_da=0.001, beta=0.9999, bit=64, lr_decay=0.5, decay_every=100, weight_decay=0.0001):
        super(scHashModel, self).__init__()
        self.batch_size = batch_size
        self.l_r = l_r
        self.bit = bit
        self.n_class = datamodule.N_CLASS
        self.n_features = datamodule.N_FEATURES
        self.lamb_da = lamb_da
        self.beta = beta
        self.lr_decay = lr_decay
        self.decay_every = decay_every
        self.samples_in_each_class = None  # Later initialized in training step
        self.cell_anchors = get_cell_anchors(self.n_class, self.bit)
        self.weight_decay = weight_decay
        self.datamodule = None
        self.pred_labels = None

        ##### model structure ####
        self.hash_layer = nn.Sequential(
            nn.Linear(self.n_features, 500),
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

        weight = get_class_balance_loss_weight(self.samples_in_each_class, self.n_class, self.beta)
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
        data, labels, _ = train_batch
        hash_codes = self.forward(data)
        loss = self.loss_functions(hash_codes, labels)
        return loss

    def validation_step(self, val_batch, batch_idx):
        data, labels, _ = val_batch
        hash_codes = self.forward(data)
        loss = self.loss_functions(hash_codes, labels)
        return loss

    def validation_epoch_end(self, outputs):

        val_loss_epoch = torch.stack([x for x in outputs]).mean()

        val_dataloader = self.trainer.datamodule.val_dataloader()
        train_dataloader = self.trainer.datamodule.train_dataloader()

        val_matrics_CHC = compute_metrics(val_dataloader, self)
        (val_labeling_accuracy_CHC, 
        val_F1_score_weighted_average_CHC, val_F1_score_median_CHC, val_F1_score_per_class_CHC,
        val_precision, val_recall,  _) = val_matrics_CHC

        train_matrics_CHC = compute_metrics(train_dataloader, self)
        (_, 
        _, train_F1_score_median_CHC, _, _,
        _, _) = train_matrics_CHC

        if self.current_epoch % 49 == 0:
            if not self.trainer.sanity_checking:
                print(f"Epoch: {self.current_epoch}, Val_loss_epoch: {val_loss_epoch:.2f}")


        value = {"step":self.current_epoch,
                 "Val_loss_epoch": val_loss_epoch, 
                  "Val_F1_score_median_CHC_epoch": val_F1_score_median_CHC,
                  "Val_labeling_accuracy_CHC_epoch": val_labeling_accuracy_CHC, 
                  "Val_F1_score_weighted_average_CHC_epoch": val_F1_score_weighted_average_CHC,
                  "Val_precision:" : val_precision,
                  "Val_recall:" : val_recall,
                  "Train_F1_score_median_CHC:" : train_F1_score_median_CHC}

        self.log_dict(value, prog_bar=True, logger=True,on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        data, labels = test_batch
        data, labels = test_batch
        hash_codes = self.forward(data)
        loss = self.loss_functions(hash_codes, labels)

        return loss

    def test_epoch_end(self, outputs):
        test_dataloader = self.trainer.datamodule.test_dataloader()
        # test_matrics_CHC = test_compute_metrics(test_dataloader, self)
        labels_pred_CHC, batchs_query, binaries_query, query_time = compute_labels(test_dataloader, self)
        
        # (accuracy,precision,recall,f1,hashing_time, cell_assign_time, query_time, f1_median, labels_pred_CHC, labels_true) = test_matrics_CHC
        self.pred_labels = labels_pred_CHC

        # value = {"F1": f1,
        #          "F1_Median": f1_median,
        #          "Precision" : precision,
        #          "Recall" : recall,
        #         # "Test_hashing_time": hashing_time,
        #         # "Test_cell_assign_time": cell_assign_time,
        #         'Query_time': query_time,
        #         'Accuracy':accuracy}

        # self.log_dict(value, prog_bar=True, logger=True,on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.l_r, weight_decay=self.weight_decay)


        # Decay LR by a factor of gamma every step_size epochs
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer, step_size=self.decay_every, gamma=self.lr_decay)

        return [optimizer], [exp_lr_scheduler]

# helper func
def training(model, datamodule, checkpointPath, filename:str = 'scHash-{epoch:02d}-{Val_F1_score_median_CHC_epoch:.3f}', max_epochs = 200):
    checkpoint_callback = ModelCheckpoint(
                                monitor='Val_F1_score_median_CHC_epoch',
                                dirpath=checkpointPath,
                                filename=filename,
                                verbose=True,
                                mode='max'
                                )

    trainer = pl.Trainer(max_epochs=max_epochs,
                        gpus=1,
                        check_val_every_n_epoch=10,
                        progress_bar_refresh_rate=0,
                        callbacks=[checkpoint_callback]
                        )

    start = time.time()
    trainer.fit(model = model, datamodule = datamodule)
    training_time = time.time()-start

    return trainer, checkpoint_callback.best_model_path, training_time

def testing(trainer, model, best_model_path, datamodule, return_detail = False):
    # Test the best model
    best_model = scHashModel.load_from_checkpoint(best_model_path, datamodule=datamodule, l_r=model.l_r, lamb_da=model.lamb_da, beta=model.beta, bit=model.bit, lr_decay=model.lr_decay, decay_every=model.decay_every, weight_decay=model.weight_decay)
    best_model.eval()

    test_dataloader = trainer.datamodule.test_dataloader()
    labels_pred, batchs_query, binaries_query, query_time = compute_labels(test_dataloader, best_model)
    
    label_map = {v: k for k, v in datamodule.label_mapping.items()}
    pred_labels = [label_map[i] for i in labels_pred]

    if return_detail:
        return pred_labels, binaries_query, batchs_query, query_time
    else:
        return pred_labels, binaries_query

def setup_training_data(train_data, cell_type_key:str = 'cell_type', batch_key: str = '', batch_size=128, num_workers=2, hvg:bool = True, log_norm:bool = True, normalize:bool = True):
    datamodule = Cross_DataModule(train_data = train_data, cell_type_key = cell_type_key, batch_key = batch_key, num_workers=num_workers, batch_size=batch_size, log_norm = log_norm, hvg = hvg, normalize = normalize)
    datamodule.setup(None)
    return datamodule

def compute_cell_composition(trainer, best_model_path, model, K:int = 100):
    best_model = scHashModel.load_from_checkpoint(best_model_path, datamodule=trainer.datamodule, l_r=model.l_r, lamb_da=model.lamb_da, beta=model.beta, bit=model.bit, lr_decay=model.lr_decay, decay_every=model.decay_every, weight_decay=model.weight_decay)
    best_model.eval()
    binaries_train, labels_train, batch_train = compute_result(trainer.datamodule.train_dataloader(), best_model)
    binaries_val, labels_val, batch_val = compute_result(trainer.datamodule.val_dataloader(), best_model)
    binaries_database, labels_database, labels_batch = torch.cat([binaries_train, binaries_val]), torch.cat([labels_train, labels_val]), batch_train+batch_val

    binaries_test, labels_test, _  = compute_result(trainer.datamodule.test_dataloader(), best_model)

    binaries_database, labels_database = binaries_database.cpu().detach().numpy(), labels_database.cpu().detach().numpy()
    binaries_test, _ = binaries_test.cpu().detach().numpy(), labels_test.cpu().detach().numpy()
    labels_test = [trainer.datamodule.label_mapping[i] for i in trainer.datamodule.test_data.obs.cell_type]


    K = K # number of cell to retrivel
    batch_list = ['smartseq', 'c1', 'wang', 'indrop', 'celseq', 'celseq2']
    celltype_list = ['true label', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    df_batch = pd.DataFrame(columns=batch_list)
    df_celltype = pd.DataFrame(columns=celltype_list)

    for i in range(len(binaries_test)):

        dists = CalcHammingDist(binaries_test[i], binaries_database)

        idx = np.argpartition(dists, K)[:K]
        celltype_metadata = labels_database[idx]
        batch_metadata = np.hstack(labels_batch)[idx]

        df_temp = pd.DataFrame({'celltype':celltype_metadata,'batch':batch_metadata})

        df_temp_batch = df_temp.batch.value_counts()
        df_temp_celltype = df_temp.celltype.value_counts()

        zero_data = np.zeros(shape=(1,len(batch_list)))
        df_one_batch = pd.DataFrame(zero_data, columns=batch_list)
        zero_data = np.zeros(shape=(1,len(celltype_list)))
        df_one_celltype = pd.DataFrame(zero_data, columns=celltype_list)

        for j in range(len(df_temp_batch)):
            df_one_batch[df_temp_batch.index[j]] = df_temp_batch.iloc[j]

        for k in range(len(df_temp_celltype)):
            df_one_celltype[str(df_temp_celltype.index[k])] = df_temp_celltype.iloc[k]

        df_one_celltype['true label'] = labels_test[i]

        df_batch = pd.concat([df_batch, df_one_batch], axis=0)
        df_celltype = pd.concat([df_celltype, df_one_celltype], axis=0)
        
    return df_celltype, df_batch