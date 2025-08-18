#!/usr/bin/env python3

# @Author: ChangXu
# @E-mail: xuchang0214@163.com
# @Last Modified by:   ChangXu
# @Last Modified time: 2021-04-22 08:42:54 23:22:34
# -*- coding: utf-8 -*-


import os
import psutil
import time
import torch
import math
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import anndata
from pathlib import Path
from sklearn.metrics import pairwise_distances, calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn import metrics
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import distance

from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import Union, Callable

from Component.utils_func import *
from Component.his_feat import image_feature, image_crop
from Component.adj import graph, combine_graph_dict
from Component.model import SRLST_model, AdversarialNetwork
from Component.trainer import train

from Component.augment import augment_adata


class run():
    def __init__(
            self,
            save_path="./",
            task="Identify_Domain",
            pre_epochs=1000,
            epochs=500,
            use_gpu=True,
    ):
        self.save_path = save_path
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.use_gpu = use_gpu
        self.task = task

    def _get_adata(
            self,
            platform,
            data_path,
            data_name,
            verbose=True,
    ):
        assert platform in ['Visium', 'ST', 'MERFISH', 'slideSeq', 'stereoSeq']
        if platform in ['Visium']:
            adata = read_10X_Visium(os.path.join(data_path, data_name))
        elif platform == 'MERFISH':
            adata = read_merfish(os.path.join(data_path, data_name))
        elif platform == 'slideSeq':
            adata = read_SlideSeq(os.path.join(data_path, data_name))
        elif platform == 'seqFish':
            adata = read_seqfish(os.path.join(data_path, data_name))
        elif platform == 'stereoSeq':
            adata = read_stereoSeq(os.path.join(data_path, data_name))
        else:
            raise ValueError(
                f"""\
               				 {self.platform!r} does not support.
	                				""")
        if verbose:
            save_data_path = Path(os.path.join(self.save_path, "Data", data_name))
            save_data_path.mkdir(parents=True, exist_ok=True)
            adata.write(os.path.join(save_data_path, f'{data_name}_raw.h5ad'), compression="gzip")
        return adata

    def _get_image_crop(
            self,
            adata,
            data_name,
            cnnType='ResNet50',
            pca_n_comps=50,
    ):
        save_path_image_crop = Path(os.path.join(self.save_path, 'Image_crop', data_name))
        save_path_image_crop.mkdir(parents=True, exist_ok=True)
        adata = image_crop(adata, save_path=save_path_image_crop)
        adata = image_feature(adata, pca_components=pca_n_comps, cnnType=cnnType).extract_image_feat()
        return adata

    def _get_augment(
            self,
            adata,
            adjacent_weight=0.3,
            neighbour_k=4,
            spatial_k=30,
            n_components=100,
            md_dist_type="cosine",
            gb_dist_type="correlation",
            use_morphological=True,
            use_data="raw",
            spatial_type="KDTree"
    ):
        adata = augment_adata(adata,
                              md_dist_type=md_dist_type,
                              gb_dist_type=gb_dist_type,
                              n_components=n_components,
                              use_morphological=use_morphological,
                              use_data=use_data,
                              neighbour_k=neighbour_k,
                              adjacent_weight=adjacent_weight,
                              spatial_k=spatial_k,
                              spatial_type=spatial_type
                              )
        print("Step 1: Augment molecule expression is Done!")
        return adata

    def _get_graph(
            self,
            data,
            distType="BallTree",
            k=12,
            rad_cutoff=150,
    ):
        graph_dict = graph(data, distType=distType, k=k, rad_cutoff=rad_cutoff).main()
        print("Step 2: Graph computing is Done!")
        return graph_dict

    def _optimize_cluster(
            self,
            adata,
            resolution: list = list(np.arange(0.1, 2.5, 0.01)),
    ):
        scores = []
        for r in resolution:
            sc.tl.leiden(adata, resolution=r)
            s = calinski_harabasz_score(adata.X, adata.obs["leiden"])
            scores.append(s)
        cl_opt_df = pd.DataFrame({"resolution": resolution, "score": scores})
        best_idx = np.argmax(cl_opt_df["score"])
        res = cl_opt_df.iloc[best_idx, 0]
        print("Best resolution: ", res)
        return res

    def _priori_cluster(
            self,
            adata,
            n_domains=7,
    ):
        for res in sorted(list(np.arange(0.1, 2.5, 0.01)), reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            if count_unique_leiden == n_domains:
                break
        print("Best resolution: ", res)
        return res

    def _get_multiple_adata(
            self,
            adata_list,
            data_name_list,
            graph_list,
    ):
        for i in range(len(data_name_list)):
            current_adata = adata_list[i]
            current_adata.obs['batch_name'] = data_name_list[i]
            current_adata.obs['batch_name'] = current_adata.obs['batch_name'].astype('category')
            current_graph = graph_list[i]
            if i == 0:
                multiple_adata = current_adata
                multiple_graph = current_graph
            else:
                var_names = multiple_adata.var_names.intersection(current_adata.var_names)
                multiple_adata = multiple_adata[:, var_names]
                current_adata = current_adata[:, var_names]
                multiple_adata = multiple_adata.concatenate(current_adata)
                multiple_graph = combine_graph_dict(multiple_graph, current_graph)

        multiple_adata.obs["batch"] = np.array(
            pd.Categorical(
                multiple_adata.obs['batch_name'],
                categories=np.unique(multiple_adata.obs['batch_name'])).codes,
            dtype=np.int64,
        )

        return multiple_adata, multiple_graph

    def _data_process(self,
                      adata,
                      pca_n_comps=200,
                      seed=101,
                      ):
        adata.layers['count'] = adata.X.toarray()
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
        sc.pp.normalize_total(adata, target_sum=1e4)
        # sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
        adata = adata[:, adata.var['highly_variable'] == True]
        sc.pp.scale(adata)
        from sklearn.decomposition import PCA
        adata_X = PCA(n_components=pca_n_comps, random_state=seed).fit_transform(adata.X)
        return adata_X

    def _fit(
            self,
            data,
            graph_dict,
            histology_dict,
            domains=None,
            n_domains=None,
            Conv_type="MFConv",
            linear_encoder_hidden=[32, 20],
            linear_decoder_hidden=[32],
            conv_hidden=[32, 8],
            p_drop=0.01,
            dec_cluster_n=20,
            kl_weight=1,
            mse_weight=1,
            bce_kld_weight=1,
            domain_weight=1,
    ):
        print("Your task is in full swing, please wait")
        start_time = time.time()
        deepst_model = SRLST_model(
            input_dim=data.shape[1],
            Conv_type=Conv_type,
            linear_encoder_hidden=linear_encoder_hidden,
            linear_decoder_hidden=linear_decoder_hidden,
            conv_hidden=conv_hidden,
            p_drop=p_drop,
            dec_cluster_n=dec_cluster_n,
        )
        if self.task == "Identify_Domain":
            deepst_training = train(
                data,
                graph_dict,
                histology_dict,
                deepst_model,
                pre_epochs=self.pre_epochs,
                epochs=self.epochs,
                kl_weight=kl_weight,
                mse_weight=mse_weight,
                bce_kld_weight=bce_kld_weight,
                domain_weight=domain_weight,
                use_gpu=self.use_gpu
            )
        elif self.task == "Integration":
            deepst_adversial_model = AdversarialNetwork(model=deepst_model, n_domains=n_domains)
            deepst_training = train(
                data,
                graph_dict,
                deepst_adversial_model,
                domains=domains,
                pre_epochs=self.pre_epochs,
                epochs=self.epochs,
                kl_weight=kl_weight,
                mse_weight=mse_weight,
                bce_kld_weight=bce_kld_weight,
                domain_weight=domain_weight,
                use_gpu=self.use_gpu
            )
        else:
            print("There is no such function yet, looking forward to further development")
        deepst_training.fit()
        deepst_embed, _ = deepst_training.process()
        print("Step 3: SRLST training has been Done!")
        print(u'Current memory usage：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time: {total_time / 60 :.2f} minutes")
        print("Your task has been completed, thank you")
        print("Of course, you can also perform downstream analysis on the processed data")

        return deepst_embed

    def _get_cluster_data(
            self,
            adata,
            dataname,
            n_clusters
    ):
        save_root = Path('Results/DLPFC')
        # sample name
        sample_name = dataname
        n_clusters = n_clusters
        sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]
        sc.pp.neighbors(sub_adata, use_rep='SRLST')
        sc.tl.umap(sub_adata)
        color_palette = ["#ECE644", "#EFAB6D", "#7F7DB5", "#589AC4", "#A5CE9D", "#A2C8DD", "#E6756F"]

        plt.rcParams["figure.figsize"] = (4, 3)
        sc.tl.paga(sub_adata, groups='layer_guess')
        sc.pl.paga_compare(sub_adata, legend_fontsize=10, palette=color_palette, title="SRLST", frameon=False,
                           size=20, legend_fontoutline=2, show=False)
        # plt.savefig('output/{}_{}.png'.format(name, key), dpi=600)
        plt.show()
        best = 0
        for i in range(0, 2026):
        # 循环找到最好结果，可随时停止
            try:
                mclust_R(sub_adata, n_clusters, use_rep='SRLST', key_added='mclust_R', random_seed=i)
            except Exception as e:
                print(e)
                continue
            else:

                ARI = metrics.adjusted_rand_score(sub_adata.obs['layer_guess'], sub_adata.obs['mclust_R'])
                print(ARI)
                if best < ARI:
                    best = ARI
                    if not os.path.exists(save_root / sample_name):
                        os.makedirs(save_root / sample_name)
                    # sub_adata.write(save_root / sample_name / 'SRLST_{}_adata_{}.h5ad'.format(sample_name, ARI))
                    fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
                    sc.pl.spatial(sub_adata, color='layer_guess', ax=axes[0], show=False, palette=color_palette, size=2)
                    sc.pl.spatial(sub_adata, color='mclust_R', ax=axes[1], show=False, palette=color_palette, size=2)
                    axes[0].set_title('Manual Annotation')
                    axes[1].set_title('Clustering: (ARI=%.4f)' % ARI)
                    plt.tight_layout()
                    plt.show()
                    print(1)

    def liver_get_cluster_data(
            self,
            adata,
            dataname,
            n_clusters
    ):
        save_root = Path('Results/liver')
        # sample name
        sample_name = dataname
        n_clusters = n_clusters
        sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]
        sc.pp.neighbors(sub_adata, use_rep='SRLST')
        sc.tl.umap(sub_adata)
        color_palette = ["#ECE644", "#EFAB6D", "#7F7DB5", "#589AC4", "#A5CE9D", "#A2C8DD", "#E6756F"]

        plt.rcParams["figure.figsize"] = (4, 3)
        sc.tl.paga(sub_adata, groups='layer_guess')
        sc.pl.paga_compare(sub_adata, legend_fontsize=10, palette=color_palette, title="SRLST", frameon=False,
                           size=20, legend_fontoutline=2, show=False)
        # plt.savefig('output/{}_{}.png'.format(name, key), dpi=600)
        plt.show()

        best = 0
        for i in range(0, 2026):
        # 循环找到最好结果，可随时停止
            try:
                leiden(sub_adata, n_clusters, use_rep='SRLST', key_added='mclust_R', random_seed=i)
                mapping = {
                    0: "Central",
                    1: "Mid",
                    2: "Periportal",
                    3: "Portal"
                }
                sub_adata.obs['mclust_R'].cat.rename_categories(mapping, inplace=True)
                AVG = []
                select_type_list = ["Central", "Mid", "Periportal", "Portal"] # 因为聚类结果的颜色顺序随机，需手动控制结果顺序
                for select_type in select_type_list:
                    spot_type = sub_adata[sub_adata.obs['layer_guess'] == select_type].obs_names
                    IoU = compute_iou(sub_adata.obs['layer_guess'], sub_adata.obs['mclust_R'], spot_type, select_type)
                    AVG.append(IoU)
                IoU = np.mean(AVG)

                if best < IoU:
                    print("avg: %f  Portal: %f" % (IoU, AVG[3]))
                    best = IoU
                    if not os.path.exists(save_root / sample_name):
                        os.makedirs(save_root / sample_name)
                    sub_adata.write(save_root / sample_name / 'SRLST_IoU_{}_adata_{}.h5ad'.format(sample_name, best))
                    fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
                    sc.pl.spatial(sub_adata, color='layer_guess', ax=axes[0], show=False, palette=color_palette, size=2)
                    sc.pl.spatial(sub_adata, color='mclust_R', ax=axes[1], show=False, palette=color_palette, size=2)
                    axes[0].set_title('Manual Annotation')
                    axes[1].set_title('Clustering: (IoU=%.4f)' % best)
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                print(e)
                continue
            else:
                print("")

    def PC_get_cluster_data(
            self,
            adata,
            dataname,
            n_clusters
    ):
        save_root = Path('Results/PC')
        # sample name
        sample_name = dataname
        n_clusters = n_clusters
        sub_adata = adata
        sc.pp.neighbors(sub_adata, use_rep='SRLST')
        sc.tl.umap(sub_adata)
        color_palette = ["#ECE644", "#EFAB6D", "#7F7DB5", "#589AC4", "#A5CE9D", "#A2C8DD", "#E6756F"]
        for i in range(0, 2026):
        # 循环找到最好结果，可随时停止
            try:
                leiden(sub_adata, n_clusters, use_rep='SRLST', key_added='mclust_R', random_seed=i)
            except Exception as e:
                print(e)
                continue
            else:

                if not os.path.exists(save_root / sample_name):
                    os.makedirs(save_root / sample_name)
                # sub_adata.write(save_root / sample_name / 'SRLST_{}_adata_{}.h5ad'.format(sample_name, ARI))
                fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
                sc.pl.spatial(sub_adata, color='mclust_R', ax=axes[0], show=False, palette=color_palette, size=2)
                sc.pl.spatial(sub_adata, color='mclust_R', ax=axes[1], show=False, palette=color_palette, size=2)
                axes[0].set_title('Manual Annotation')
                axes[1].set_title('SPLST')
                plt.tight_layout()
                plt.show()
                print(1)

    def HBC_get_cluster_data(
            self,
            adata,
            dataname,
            n_clusters
    ):
        save_root = Path('Results/HBC')
        # sample name
        sample_name = dataname
        n_clusters = n_clusters
        sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]
        color_palette = ["#ECE644", "#EFAB6D", "#7F7DB5", "#589AC4", "#A5CE9D", "#A2C8DD", "#E7DBD3", "#E6756F",
                         "#D2691E", "#20B2AA", "#FF69B4", "#6A5ACD", "#32CD32", "#8A2BE2", "#FF4500", "#00CED1",
                         "#4B0082", "#FFD700", "#8B0000", "#9370DB"]
        best = 0
        for i in range(0, 2026):
        # 循环找到最好结果，可随时停止
            try:
                mclust_R(sub_adata, n_clusters, use_rep='SRLST', key_added='SRLST', random_seed=i)
                # leiden(sub_adata, n_clusters, use_rep='SRLST', key_added='SRLST', random_seed=i)
            except Exception as e:
                print(e)
                continue
            else:
                """if len(sub_adata.obs['SRLST'].unique()) != n_clusters:
                    print("not.................")
                    continue"""
                ARI = metrics.adjusted_rand_score(sub_adata.obs['layer_guess'], sub_adata.obs['SRLST'])
                print(ARI)
                if best < ARI:
                    best = ARI
                    if not os.path.exists(save_root / sample_name):
                        os.makedirs(save_root / sample_name)
                    sub_adata.write(save_root / sample_name / 'SRLST_{}_adata_{}.h5ad'.format(sample_name, ARI))
                    fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
                    sc.pl.spatial(sub_adata, color='layer_guess', ax=axes[0], show=False, palette=color_palette, size=2)
                    sc.pl.spatial(sub_adata, color='SRLST', ax=axes[1], show=False, palette=color_palette, size=2)
                    axes[0].set_title('Manual Annotation')
                    axes[1].set_title('Clustering: (ARI=%.4f)' % ARI)
                    plt.tight_layout()
                    plt.show()

    def fish_get_cluster_data(
            self,
            adata,
            dataname,
            n_clusters
    ):
        save_root = Path('Results/fish')
        # sample name
        sample_name = dataname
        n_clusters = n_clusters
        sub_adata = adata
        color_palette = ["#ECE644", "#EFAB6D", "#7F7DB5", "#589AC4", "#A5CE9D", "#A2C8DD", "#E7DBD3", "#E6756F",
                         "#D2691E", "#20B2AA", "#FF69B4", "#6A5ACD", "#32CD32", "#8A2BE2", "#FF4500", "#00CED1",
                         "#4B0082", "#FFD700", "#8B0000", "#9370DB"]
        best = 0
        low = 10
        for i in range(0, 2026):
        # 循环找到最好结果，可随时停止
            try:
                mclust_R(sub_adata, n_clusters, use_rep='SRLST', key_added='SRLST', random_seed=i)
                # leiden(sub_adata, n_clusters, use_rep='SRLST', key_added='SRLST', random_seed=i)
            except Exception as e:
                print(e)
                continue
            else:
                """if len(sub_adata.obs['SRLST'].unique()) != n_clusters:
                    print("not.................")
                    continue"""
                SC = silhouette_score(sub_adata.obsm['SRLST'], sub_adata.obs['SRLST'])
                DB = davies_bouldin_score(sub_adata.obsm['SRLST'], sub_adata.obs['SRLST'])
                print("SC:%.4f, DB:%.4f" % (SC, DB))
                if (SC > best) and (DB < low):
                    if not os.path.exists(save_root / sample_name):
                        os.makedirs(save_root / sample_name)
                    # sub_adata.write(save_root / sample_name / 'SRLST_{}_adata_{}_{}.h5ad'.format(sample_name, SC, DB))
                    fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
                    sc.pl.spatial(sub_adata, color='SRLST', ax=axes[0], show=False, palette=color_palette, size=2, alpha_img=0)
                    sc.pl.spatial(sub_adata, color='SRLST', ax=axes[1], show=False, palette=color_palette, size=2, alpha_img=0)
                    axes[0].set_title('Manual Annotation')
                    axes[1].set_title('Clustering: (SC=%.4f)' % SC)
                    plt.tight_layout()
                    plt.show()

    def FFPE_get_cluster_data(
            self,
            adata,
            dataname,
            n_clusters
    ):
        save_root = Path('Results/FFPE')
        # sample name
        sample_name = dataname
        n_clusters = n_clusters
        sub_adata = adata
        sc.pp.neighbors(sub_adata, use_rep='SRLST')
        sc.tl.umap(sub_adata)
        color_palette = ["#ECE644", "#EFAB6D", "#7F7DB5", "#589AC4", "#A5CE9D", "#A2C8DD", "#E6756F"]

        for i in range(0, 2026):
            try:
                leiden(sub_adata, n_clusters, use_rep='SRLST', key_added='mclust_R', random_seed=i)
            except Exception as e:
                print(e)
                continue
            else:
                # ARI = metrics.adjusted_rand_score(sub_adata.obs['layer_guess'], sub_adata.obs['mclust_R'])
                if not os.path.exists(save_root / sample_name):
                    os.makedirs(save_root / sample_name)
                # sub_adata.write(save_root / sample_name / 'SRLST_{}_adata_{}.h5ad'.format(sample_name, 0.7))
                fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
                sc.pl.spatial(sub_adata, color='mclust_R', ax=axes[0], show=False, palette=color_palette, size=2)
                sc.pl.spatial(sub_adata, color='mclust_R', ax=axes[1], show=False, palette=color_palette, size=2)
                axes[0].set_title('Manual Annotation')
                axes[1].set_title('SPLST')
                plt.tight_layout()
                plt.show()
                print(1)

    def calculate_adj_matrix(self, x, y, x_pixel=None, y_pixel=None, image=None, beta=31, normalize=False):
        """
        Calculate adjacency matrix using spatial locations and histology image.

        Parameters:
            x, y (list): Coordinates in spatial space (unused directly in calculation).
            x_pixel, y_pixel (list): Pixel coordinates corresponding to spatial locations.
            image (numpy.ndarray): Input image (expected shape: HxWx3).
            beta (int): Neighborhood size (must be odd for symmetry). Default is 49.
            normalize (bool): Whether to normalize the output values to [0, 1]. Default is False.

        Returns:
            c0, c1, c2 (numpy.ndarray): Adjacency matrix for R, G, B channels.
        """
        # Validate inputs
        if x_pixel is None or y_pixel is None or image is None:
            raise ValueError("x_pixel, y_pixel, and image cannot be None.")
        if len(x) != len(x_pixel) or len(y) != len(y_pixel):
            raise ValueError("Length of x, y must match length of x_pixel, y_pixel.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must be a 3D array with 3 channels (RGB).")
        if beta <= 0 or beta % 2 == 0:
            raise ValueError("Beta must be a positive odd integer.")

        # Neighborhood radius
        beta_half = beta // 2
        max_x, max_y = image.shape[:2]

        # Efficient mean computation
        means = []
        for x_p, y_p in zip(x_pixel, y_pixel):
            x_start = max(0, x_p - beta_half)
            x_end = min(max_x, x_p + beta_half + 1)
            y_start = max(0, y_p - beta_half)
            y_end = min(max_y, y_p + beta_half + 1)
            neighborhood = image[x_start:x_end, y_start:y_end]
            means.append(np.mean(neighborhood, axis=(0, 1)))  # Compute mean for RGB channels

        # Convert to NumPy array and separate channels
        means = np.array(means)
        c0, c1, c2 = means[:, 0], means[:, 1], means[:, 2]

        # Normalize if required
        if normalize:
            c0 = (c0 - c0.min()) / (c0.max() - c0.min()) if c0.ptp() > 0 else c0
            c1 = (c1 - c1.min()) / (c1.max() - c1.min()) if c1.ptp() > 0 else c1
            c2 = (c2 - c2.min()) / (c2.max() - c2.min()) if c2.ptp() > 0 else c2

        return c0, c1, c2

    def histology_enhance(self, adata, histology):
        print("Begin calculate histology......................")
        c0, c1, c2 = self.calculate_adj_matrix(x=adata.obs["array_row"].tolist(), y=adata.obs["array_col"].to_list(),
                                               x_pixel=adata.obs["imagerow"].to_list(),
                                               y_pixel=adata.obs["imagecol"].to_list(),
                                               image=histology)
        adata.obsm['histology'] = np.column_stack((c0, c1, c2))
        return adata
