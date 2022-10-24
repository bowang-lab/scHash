# Single-cell Hashing using deep learning (scHash)
An automatic cell type annotation and cell retrievalmethod for large-scale scRNA-seq datasets using neuralnetwork-based hashing.

## Quick Start:
- `pip3 install -r requirements.txt` Install dependencies
- `python3 scHash.py --checkpoint_path checkpoint --data_dir data.h5ad --batch_key dataset --query query --result_dir result.csv` 

*The model successfully runs with Python 3.8.1*


### Options
  - `--data_dir` Path to data source that contains both train and query data in .h5ad Anndata form
  - `--checkpoint_path` Path to save checkpoint
  - `--query`           Query dataset name in data.obs[bacth_key]
  - `--l_r`             learning rate
  - `--batch_size`      Batch Size
  - `--log_norm`        If applying log normalization log10(X+1)
  - `--hvg`             If doing the high variable gene selection
  - `--lamb`            lambda of quantization loss
  - `--lr_decay`        learning rate decay
  - `--n_layers`        Number of layers
  - `--epochs`          Number of epochs to run
  - `--batch_key`       Batch variable key in the data.obs dataframe  
  - `--cell_type_key`   Cell type key for the true labels variable key in the data.obs dataframe  
  - `--result_dir`      Path to the result csv                   

 
## Establish a venv
- `python3 -m venv .venv`
- `pip3 install -r requirements.txt`
