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
from util import get_labels_pred_closest_cell_anchor

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("pastel")

# take BaronHuman as an example
filename = 'BaronHuman.zip'
data_name = "BaronHuman"
data_dir = './data'

CHECKPT_PATH=''
N_CLASS = 13
N_FEATURES = 17499

with open(os.path.join("label_maps", data_name, "label_mapping.json")) as f:
            label_mapping = json.load(f)

model = scDeepHashModel.load_from_checkpoint(checkpoint_path=CHECKPT_PATH,                                                        n_class=N_CLASS,
                                           n_features=N_FEATURES)

if not os.path.exists(data_dir+ '/' + data_name):
    url = "https://github.com/Aprilhuu/Deep-Learning-in-Single-Cell-Analysis/raw/main/BaronHuman.zip"
    download_and_extract_archive(url, data_dir, filename=filename)    

print("Download succeeded!")


DataPath = data_dir + "/" + data_name + "/Filtered_Baron_HumanPancreas_data.csv"
LabelsPath = data_dir + "/" + data_name + "/Labels.csv"

labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',')
data = pd.read_csv(DataPath, index_col=0, sep=',')

print("Data loaded!")


input_data = torch.from_numpy(data.values).float()
binary_predict = model.forward(input_data).sign()
labels_pred_CHC = get_labels_pred_closest_cell_anchor(binary_predict.detach().numpy(), labels.to_numpy(), model.cell_anchors.numpy())

string_labels = [label_mapping[str(int_label)] for int_label in labels_pred_CHC]

cell_anchors = model.cell_anchors.numpy()
num_cell_anchors = len(cell_anchors)
concated_predict = np.concatenate((binary_predict.detach().numpy(), cell_anchors), axis=0)

print("Prediction done!")
print(binary_predict.shape)
print(model.cell_anchors.numpy())

#tsne = TSNE(n_components=2, n_iter=5000, learning_rate=100, early_exaggeration=10, perplexity=50)
tsne = TSNE()
X_embedded_full = tsne.fit_transform(concated_predict)

X_embedded = X_embedded_full[:len(X_embedded_full) - num_cell_anchors, :]
cell_anchors_tsne = X_embedded_full[len(X_embedded_full) - num_cell_anchors:, :]
print(cell_anchors_tsne.shape)


sns_plot = sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=string_labels, linewidth=0, alpha=0.85)
sns_plot = sns.scatterplot(x=cell_anchors_tsne[:, 0], y=cell_anchors_tsne[:,1], color="red",s=200, marker="*", alpha=0.8, edgecolor='r')

for i in range(cell_anchors_tsne.shape[0]):
    sns_plot.text(x=cell_anchors_tsne[i, 0]+1, y=cell_anchors_tsne[i, 1]+1,s=label_mapping[str(i)], fontdict=dict(color='black',size=10, weight='bold'))

lgd = sns_plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.title("tSNE visualization")
plt.xlabel("tSNE-1")
plt.ylabel("tSNE-2")

sns_plot.figure.savefig("BH_vis.png", bbox_extra_artists=(lgd,), bbox_inches='tight')

