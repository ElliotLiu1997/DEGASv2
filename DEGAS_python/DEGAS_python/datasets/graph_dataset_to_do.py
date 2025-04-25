#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Refer to: https://github.com/deepfindr/gnn-project/blob/main/dataset.py
import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.data import Dataset as GDataset
from torch_geometric.data import Data as GData
import numpy as np 
import pandas as pd
import os
from glob import glob
from typing import Callable, List
from sklearn.metrics import pairwise_distances
from .tools import *

# data augmentations, https://github.com/rish-16/grafog/blob/main/grafog/transforms/transforms.py
class Compose(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, data):
        for aug in self.transforms:
            data = aug(data)
        return data

class NodeDrop(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x
        y = data.y
        train_mask = torch.ones(x.shape[0], dtype = torch.bool)
        test_mask = torch.ones(x.shape[0], dtype = torch.bool)
        edge_idx = data.edge_index

        idx = torch.empty(x.size(0)).uniform_(0, 1)
        train_mask[torch.where(idx < self.p)] = 0
        test_mask[torch.where(idx < self.p)] = 0
        new_data = tg.data.Data(x=x, edge_index=edge_idx, y=y, train_mask=train_mask, test_mask=test_mask)

        return new_data

class EdgeDrop(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x
        y = data.y
        train_mask = torch.ones(x.shape[0], dtype = torch.bool)
        test_mask = torch.ones(x.shape[0], dtype = torch.bool)
        edge_idx = data.edge_index

        edge_idx = edge_idx.permute(1, 0)
        idx = torch.empty(edge_idx.size(0)).uniform_(0, 1)
        edge_idx = edge_idx[torch.where(idx >= self.p)].permute(1, 0)
        new_data = tg.data.Data(x=x, y=y, edge_index=edge_idx, train_mask=train_mask, test_mask=test_mask)
        return new_data

class STSWDataset(GDataset):
    # Spatial Transcriptomics Sliding Window Dataset
    def __init__(self, root="PRAD/Processed_ST_data", filenames=[], labels = [], gene_list = [], # list of genes to use
                 kernel_size = 5, loc_filenames = None, transform=None, pre_transform=None, 
                 random_seed = 0, sample_balance = True, batch_size = 200, phase = "train"):
        """
        Slide Window dataset for Spatial Transcriptomics
        """
        if not os.path.exists(os.path.join(root, "raw")):
            raise ValueError("Please save all your : {}".format(os.path.join(root, "raw")))
        assert type(filenames) is list
        self.filenames = filenames
        assert type(labels) is list
        self.labels = labels if (len(labels) == 0) else list(range(len(filenames))) # using file name order as label

        # other parameters
        self.kernel_size = kernel_size # define the slide window size
        self.loc_filenames = loc_filenames # location of ST datasets, not required 
        # we set these two hyper parameters for training
        # during the training process, we only sample a subset for training
        self.random_seed = random_seed
        self.sample_balance = sample_balance
        self.batch_size = batch_size
        self.phase = phase
        # select genes to use
        self.gene_idx = load_features(root, gene_list, feature_dim = 200, full_feature_dim = 100000) # 100000 a very large number you will never use
        
        super(STSWDataset, self).__init__(root, transform, pre_transform)
        self.node_aug = Compose([
            NodeDrop(p = 0.45),
        ])
        self.edge_aug = Compose([
            EdgeDrop(p = 0.15)
        ])
        

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filenames

    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = self.raw_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [os.path.join(self.raw_dir, f) for f in files]
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        filepaths = glob(os.path.join(self.processed_dir, "*.csv")) # only get file names
        return [f.split("/")[-1] for f in filepaths]

    def download(self):
        pass
    
    def calc_step_size(self, loc_df):
        # calculate the distance of spots on WSI
        loc_df = np.array(loc_df)
        assert loc_df.shape[1] == 2 
        assert loc_df.shape[0] > 10 # number of dots should be large enough
        
        np.random.seed(0) # for reproducibility
        Y = loc_df[np.random.choice(range(loc_df.shape[0]), loc_df.shape[0] // 10, replace = False), :]
        distance_mat = pairwise_distances(Y, loc_df)
        distance_min_max = np.max(np.sort(distance_mat, axis = 1)[:, 1]) # find the smallest step size for each sample, take the largest one for the universal step size
        return np.ceil(distance_min_max), np.ceil(distance_min_max) * (self.kernel_size // 2)
        
    def process(self):
        index = 0
        for i, (fp, fn) in enumerate(zip(self.raw_paths, self.raw_file_names)):
            if self.loc_filenames is None:
                loc_f = os.path.join(self.raw_dir, fn.split(".csv")[0] + "_loc.csv") # file name for coordinates
                if not os.path.exists(loc_f):
                    raise Exception("location file {} is not exist!".format(loc_f))
            else:
                loc_f = os.path.join(self.raw_dir, self.loc_filenames[i])
            loc_df = pd.read_csv(loc_f, index_col = 0, header = 0)
            loc_df.index = np.arange(len(loc_df)) 
            loc_df.columns = ["x", "y"]
            sz, wd = self.calc_step_size(loc_df) # step size and width of winder
            sz_thresh = int(sz * 1.2) # expand a little bit 
            # load gene features
            feat_df = pd.read_csv(fp, index_col = 0, header = 0)
            # load label
            label = self.labels[i]
            
            for x, y in zip(np.array(loc_df.iloc[:, 0]), np.array(loc_df.iloc[:, 1])):
                l, r, u, d = x - wd, x + wd, y - wd, y + wd
                window_loc_df = loc_df[(loc_df["x"] <= r) & (loc_df["x"] >= l) & (loc_df["y"] <= d) & (loc_df["y"] >= u)]
                # Get node features
                node_feats = self._get_node_features(feat_df.iloc[window_loc_df.index, :])
                # Get edge features
                edge_feats = self._get_edge_features(window_loc_df, sz_thresh)
                # Get adjacency info
                edge_index = self._get_adjacency_info(window_loc_df, sz_thresh)
                # Get labels info
                label_ = self._get_labels(label)
                # Create data object
                data = GData(x=node_feats, 
                            edge_index=edge_index,
                            edge_attr=edge_feats,
                            y=label_
                            ) 
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 'data_x_{}_y_{}_lab_{}_ind_{}.pt'.format(x, y, label, index)))
                index += 1
                
            # if finish a file, save it into the processed folder
            loc_df.to_csv(os.path.join(self.processed_dir, fn))


    def _get_node_features(self, all_node_feats):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, locs, sz):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        locs = np.array(locs)
        edge_indices = (pairwise_distances(locs) <= sz)
        all_edge_feats = [1] * (2 * len(np.where(edge_indices == True)[0]))
        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, locs, sz):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        edge_indices = []
        locs = np.array(locs)
        edge_indices_mat = (pairwise_distances(locs) <= sz)
        for i in range(locs.shape[0]):
            for j in range(locs.shape[0]):
                if edge_indices_mat[i, j]:
                    edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        self.all_files = glob(os.path.join(self.processed_dir, "*ind_*.pt"))
        stLab = [int(f.split("/")[-1].split("_")[6]) for f in self.all_files]
        if self.phase == "train":
            if self.sample_balance:
                self.idx_st = balance_sampling(np.expand_dims(np.array(stLab), 1), self.batch_size, self.random_seed)
            else:
                np.random.seed(self.random_seed)
                self.idx_st = np.random.choice(range(len(self.all_files)), self.batch_size, replace = False)
            self.all_files = [self.all_files[idx] for idx in self.idx_st]
            return len(self.all_files)
        else:
            return len(self.all_files)
    
    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        return self.get(idx)
    
    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        filename = self.all_files[idx]
        data = torch.load(filename)  
        new_data = self.edge_aug(self.node_aug(data))
        if len(self.gene_idx) <= new_data.x.shape[1]:
            new_data.x = new_data.x[:, self.gene_idx]
        # noise should be injected into data and appears in pairs
        x, y = filename.split("/")[-1].split("_")[2], filename.split("/")[-1].split("_")[4]
        return {"data": new_data, "x": int(x), "y": int(y)}

    
    
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader as GDataLoader
    filenames = ["H1_2.csv", "H1_4.csv", "H1_5.csv", "H2_1.csv", "H2_2.csv", "H2_5.csv", "V1_2.csv"]
    # filenames = ["H2_1.csv", "H2_2.csv", "H3_1.csv", "H3_2.csv", "H3_4.csv", "H3_5.csv", "H3_6.csv", 
    #             "V1_1.csv", "V1_2.csv", "V1_3.csv", "V1_4.csv", "V1_5.csv", "V1_6.csv", "V2_1.csv", "V2_2.csv"]
    st_data = GDataLoader(STSWDataset(root="erickson/select_genes", filenames=filenames, labels = list(range(7)), gene_list = "ST_high_pat_high",
                                            phase = "train", batch_size = 200, random_seed = 0), batch_size = 200, shuffle = False)
    for i, data in enumerate(st_data):
        print(i)
        print(data)
        break
        

