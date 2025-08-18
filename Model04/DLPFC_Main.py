import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
from skimage import io
import sys
from pathlib import Path
from sklearn.metrics.cluster import adjusted_rand_score
from Component.SRLST import run
from Component.utils_func import fix_seed

# the location of R (used for the mclust clustering)
if len(sys.argv) > 1:
    param = sys.argv[1]
    os.environ['R_HOME'] = param
    print(f"Received parameter: {param}")
else:
    print("No parameter provided.")
    exit()

# sample name
data_path = "data/DLPFC"  #### to your path
data_name = '151676'  #### project name
save_path = "Results"  #### save path
seed = 101
n_domains = 5 if data_name in ['151669', '151670', '151671', '151672'] else 7
fix_seed(seed)
train = True
if train:
    deepen = run(save_path=save_path,
                 task="Identify_Domain",
                 #### SRLST includes two tasks, one is "Identify_Domain" and the other is "Integration"
                 pre_epochs=1000,  ####  choose the number of training
                 epochs=1000,  #### choose the number of training
                 use_gpu=True)
    ###### Read in 10x Visium data, or user can read in themselves.
    adata = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name)
    df_meta = pd.read_csv(os.path.join(data_path, data_name, 'metadata.tsv'), sep='\t')
    position = pd.read_csv(os.path.join(data_path, data_name, 'spatial/tissue_positions_list.csv'), header=None,
                           index_col=0)
    position = position.loc[adata.obs.index]
    histology = io.imread(os.path.join(data_path, data_name, 'full_image.tif'))
    adata.obs['imagerow'] = np.array(position[4])
    adata.obs['imagecol'] = np.array(position[5])
    adata.obs['layer_guess'] = df_meta['layer_guess']
    ###### Segment the Morphological Image
    adata = deepen.histology_enhance(adata, histology)
    # adata = deepen._get_image_crop(adata, data_name=data_name)

    ###### Data augmentation. spatial_type includes three kinds of "KDTree", "BallTree" and "LinearRegress", among which "LinearRegress"
    ###### is only applicable to 10x visium and the remaining omics selects the other two.
    ###### "use_morphological" defines whether to use morphological images.
    # adata = deepen._get_augment(adata, spatial_type="LinearRegress", use_morphological=True)

    ###### Build graphs. "distType" includes "KDTree", "BallTree", "kneighbors_graph", "Radius", etc., see adj.py
    graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="kneighbors_graph")
    histology_dict = deepen._get_graph(adata.obsm["histology"], distType="kneighbors_graph")


    ###### Enhanced data preprocessing
    data = deepen._data_process(adata, pca_n_comps=150, seed=seed)

    ###### Training models
    deepst_embed = deepen._fit(
        data=data,
        graph_dict=graph_dict,
        histology_dict=histology_dict
    )
    ###### SRLST outputs
    adata.obsm["SRLST"] = deepst_embed

    ###### Define the number of space domains, and the model can also be customized. If it is a model custom priori = False.
    deepen._get_cluster_data(adata, data_name, n_domains)

    # sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]
    # sc.write('SRLST_{}_adata.h5ad'.format(data_name), sub_adata)
    print("true")

"""sub_adata = sc.read_h5ad('SRLST_{}_adata.h5ad'.format(data_name))
from sklearn import metrics

ARI = metrics.adjusted_rand_score(sub_adata.obs['layer_guess'], sub_adata.obs['SRLST_refine_domain'])
print('Adjusted rand index = %.2f' % ARI)
color_palette = ["#ECE644", "#EFAB6D", "#7F7DB5", "#589AC4", "#A5CE9D", "#A2C8DD", "#E6756F"]

sc.pp.neighbors(sub_adata, use_rep="SRLST_embed")
sc.tl.umap(sub_adata)

plt.rcParams["figure.figsize"] = (4, 3)
sc.tl.paga(sub_adata, groups='layer_guess')
sc.pl.paga_compare(sub_adata, legend_fontsize=10, palette=color_palette, title="name", frameon=False, size=20,
                   legend_fontoutline=2, show=False)
# plt.savefig('output/{}_{}.png'.format(name, key), dpi=600)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
sc.pl.spatial(sub_adata, color='layer_guess', ax=axes[0], show=False, palette=color_palette, size=2)
sc.pl.spatial(sub_adata, color='SRLST_refine_domain', ax=axes[1], show=False, palette=color_palette, size=2)
axes[0].set_title('Manual Annotation')
axes[1].set_title('Clustering: (ARI=%.4f)' % ARI)
plt.tight_layout()
plt.show()"""
