from torchvision.datasets.utils import download_and_extract_archive
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import torch
import os
import numpy as np
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
#from umap import UMAP
from matplotlib import pyplot as plt
from scDeepHash import scDeepHashModel
from util import get_labels_pred_closest_hash_center

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("pastel")

filename = 'Zheng_68K.zip'
data_name = "Zheng_68K"
data_dir = './data'

#CHECKPT_PATH='/scratch/gobi1/rexma/scDeepHash/checkpoints/XIN/scDeepHash-epoch=69-Val_F1_score_median_CHC_epoch=1.000.ckpt'
#CHECKPT_PATH='/scratch/gobi1/rexma/scDeepHash/checkpoints/BaronHuman/scDeepHash-epoch=169-Val_F1_score_median_CHC_epoch=0.981.ckpt'
#CHECKPT_PATH='/scratch/gobi1/rexma/scDeepHash/checkpoints/TM/scDeepHash-epoch=309-Val_F1_score_median_CHC_epoch=0.974.ckpt'
#CHECKPT_PATH='/scratch/gobi1/rexma/scDeepHash/checkpoints/AMB/scDeepHash-epoch=189-Val_F1_score_median_CHC_epoch=0.914.ckpt'
CHECKPT_PATH='/scratch/gobi1/rexma/scDeepHash/checkpoints/Zheng68K/scDeepHash-epoch=119-Val_F1_score_median_CHC_epoch=0.728.ckpt'
N_CLASS=11
N_FEATURES=20387

with open(os.path.join("label_maps", "Zheng68K", "label_mapping.json")) as f:
            label_mapping = json.load(f)

model = scDeepHashModel.load_from_checkpoint(checkpoint_path=CHECKPT_PATH,                                                        n_class=N_CLASS,
                                           n_features=N_FEATURES)

if not os.path.exists(data_dir+ '/' + data_name):
    url = "https://github.com/Aprilhuu/Deep-Learning-in-Single-Cell-Analysis/raw/main/Zheng_68K.zip"
    download_and_extract_archive(url, data_dir, filename=filename)    

print("Download succeeded!")

#DataPath = data_dir + "/" + data_name + "/Filtered_TM_data.csv"
#DataPath = data_dir + "/" + data_name + "/Filtered_Xin_HumanPancreas_data.csv"
#DataPath = data_dir + "/" + data_name + "/Filtered_Baron_HumanPancreas_data.csv"
#DataPath = data_dir + "/" + data_name + "/Filtered_mouse_allen_brain_data.csv"
DataPath = data_dir + "/" + data_name + "/Filtered_68K_PBMC_data.csv"
LabelsPath = data_dir + "/" + data_name + "/Labels.csv"

labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',')
#data = pd.read_csv(DataPath, index_col=0, sep=',')

print("Data loaded!")

remove_labels = ["CD4+ T Helper2", "CD4+/CD25 T Reg", "CD4+/CD45RA+/CD25- Naive T", "CD4+/CD45RO+ Memory"]
discarded_indices = []

full_labels = np.asarray(labels)
full_indices = range(len(full_labels))

for label in remove_labels:
   remove_temp = [i for i, x in enumerate(full_labels) if x in remove_labels]
   discarded_indices.extend(remove_temp)

remaining_indices = [e for e in full_indices if e not in discarded_indices]
remaining_labels = [full_labels[i] for i in remaining_indices]

print("Number of data after filtering:", len(remaining_indices))
print("Number of classes after filtering:", len(np.unique(remaining_labels)))
print(np.unique(remaining_labels))

discarded_indices = [x + 1 for x in discarded_indices]
data = pd.read_csv(DataPath, index_col=0, sep=',', skiprows=discarded_indices)
discarded_indices = None

input_data = torch.from_numpy(data.values).float()
binary_predict = model.forward(input_data).sign()
labels_pred_CHC = get_labels_pred_closest_hash_center(binary_predict.detach().numpy(), labels.to_numpy(), model.hash_centers.numpy())

string_labels = [label_mapping[str(int_label)] for int_label in labels_pred_CHC]

hash_centers = model.hash_centers.numpy()
num_hash_centers = len(hash_centers)
concated_predict = np.concatenate((binary_predict.detach().numpy(), hash_centers), axis=0)

print("Prediction done!")
print(binary_predict.shape)

print(model.hash_centers.numpy())

#tsne = TSNE(n_components=2, n_iter=5000, learning_rate=100, early_exaggeration=10, perplexity=50)
tsne = TSNE()
X_embedded_full = tsne.fit_transform(concated_predict)

X_embedded = X_embedded_full[:len(X_embedded_full) - num_hash_centers, :]
hash_center_tsne = X_embedded_full[len(X_embedded_full) - num_hash_centers:, :]
print(hash_center_tsne.shape)

#tsne = TSNE()
#X_embedded = tsne.fit_transform(binary_predict.detach().numpy())

sns_plot = sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=string_labels, linewidth=0, alpha=0.85)
sns_plot = sns.scatterplot(x=hash_center_tsne[:, 0], y=hash_center_tsne[:,1], color="red",s=200, marker="*", alpha=0.8, edgecolor='r')

for i in range(hash_center_tsne.shape[0]):
    sns_plot.text(x=hash_center_tsne[i, 0]+1, y=hash_center_tsne[i, 1]+1,s=label_mapping[str(i)], fontdict=dict(color='black',size=10, weight='bold'))

lgd = sns_plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.title("Zheng_68K (scDeepHash)")
plt.xlabel("tSNE-1")
plt.ylabel("tSNE-2")

sns_plot.figure.savefig("./plots/Zheng_68K_scDeepHash2.png", bbox_extra_artists=(lgd,), bbox_inches='tight')

