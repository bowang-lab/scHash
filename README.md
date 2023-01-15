# scHash 
scHash is an accurate, efficient, and interpretable deep hashing-based method that can build multi-million reference database and annotate tens of thousands of cells within seconds.


## :heavy_plus_sign: Method
scHash consists of three sequential steps:

(1) **Cell anchor generation**. 

scHash generates $K$-bit hash code for each unique cell type in the reference database, which is referred as "cell anchors". 

(2) **Hash function training**. 

scHash trains a deep hash function that maps raw gene expression to $K$-bit binary hash code subject to weighted cell-anchor loss and quantization loss. 

(3) **Interpretable cell type annotation**. 

scHash can efficiently annotate large-scale scRNA-seq dataset and offer interpretability for its annotation through the metadata of most similar reference cells and saliency map.

<br><img src="img/overview.png"/>

## :triangular_ruler: Requirements and Installation
* Linux/Unix
* Python 3.8

## Installation. 
```bash
$ pip install scHash
```

## Example Usage

We demonstrate how scHash encodes multiple datasets into hash codes for six publicly available Pancreas datasets.

The raw data for the first five datasets can be obtained from [Harmony](https://github.com/immunogenomics/harmony2019/tree/master/data/figure5).

The sixth Pancreas dataset is from [(Wang et al., 2016)](https://diabetesjournals.org/diabetes/article/65/10/3028/34922/Single-Cell-Transcriptomics-of-the-Human-Endocrine) is publicly available at [GSE83139](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE83139).

We compiled the six datasets into one AnnData object for easy demonstration. The processed data can be downloaded [here](https://drive.google.com/file/d/1shc4OYIbq2FwbyGUaYuzizuvzW-giSTs/view?usp=share_link).



```python
# load required package
import anndata as ad
import scHash

# load data path
data_dir = '../../../share_data/Pancreas_Wang/fivepancreas_wang_raw.h5ad'

# set up datamodule
# This anndata object is packed with 6 pancreas dataset. We take one of them to be a test dataset here.  
# we use one of the six datasets to be the query dataset.
query = 'smartseq'
full = ad.read_h5ad(data_dir)
train = full[full.obs.dataset!=query]
test = full[full.obs.dataset==query]

datamodule = scHash.setup_training_data(train_data = train,cell_type_key = 'cell_type', batch_key = 'dataset')

# set the query data
# this can be also set after train
datamodule.setup_test_data(test)

########### consider write into a function again
# Specify a directory for saving the best model
checkpointPath = '../checkpoint/'

# Init the model and Train
model = scHash.scHashModel(datamodule)
trainer, best_model_path, training_time = scHash.training(model = model, datamodule = datamodule, checkpointPath = checkpointPath, max_epochs = 100)
print(f'Training Time: {training_time}s')

# Test the best model and output with the predicted labels and the hash code expression of the cell.
pred_labels, hash_codes = scHash.testing(trainer, model, best_model_path, datamodule)
```


