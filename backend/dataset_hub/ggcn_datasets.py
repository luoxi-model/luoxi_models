# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
from backend.dataset_hub.base_datasets import *
import numpy as np
import scipy.sparse as sp
import sys
sys.setrecursionlimit(99999)

class GGCNDataset(BaseDataset):
    def __init__(self,
                 args,
                 table_name,
                 shuffle_buffer_size,
                 is_test=False,
                 max_len=-1,
                 max_neighbor=-1):
        super(GGCNDataset, self).__init__(args, table_name, shuffle_buffer_size, is_test)
        self.maxlen = max_len
        self.max_neighbor = max_neighbor

def run_dfs(adj, msk, u, ind, nb_nodes):
    """
    data preprocess, run dfs_split function for each edge in adj
    input schema:
        adj, mask, index, nodes
    output schema:
         ret: nodes for each edge in adj
    """
    if msk[u] == -1:
        msk[u] = ind
        for v in adj[u, :].nonzero()[1]:
            run_dfs(adj, msk, v, ind, nb_nodes)

def dfs_split(adj):
    """
    data preprocess, split data with dfs method
    input schema:
        adj
    output schema:
         ret: split nodes
    """
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.int32)

    graph_id = 0

    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id, nb_nodes)
            graph_id += 1

    return ret

class GGCNDatasetLocal(GGCNDataset):
    def __init__(self,
                 args,
                 table_name,
                 shuffle_buffer_size=8194,
                 is_test=False,
                 max_len=100,
                 max_neighbor=10,traintype='train'):

        super(GGCNDatasetLocal,self).__init__(args, table_name, shuffle_buffer_size, is_test, max_len, max_neighbor)
        self.table_name = table_name
        self.train_adj_list = []
        self.val_adj_list = []
        self.test_adj_list = []
        self.train_feat = 0
        self.val_feat = 0
        self.test_feat = 0
        self.train_labels = 0
        self.val_labels = 0
        self.test_labels = 0
        self.train_nodes = []
        self.val_nodes = []
        self.test_nodes = []
        self.traintype = traintype
        if self.traintype == 'train':
            [self.train_adj_list, self.val_adj_list, self.test_adj_list, self.train_feat, self.val_feat, self.test_feat,
             self.train_labels, self.val_labels, self.test_labels, self.train_nodes, self.val_nodes, self.test_nodes] = torch.load('%s/traingraphL.pth'%table_name)

            self.loadData()
        else:
            [self.train_adj_list, self.val_adj_list, self.test_adj_list, self.train_feat, self.val_feat, self.test_feat,
             self.train_labels, self.val_labels, self.test_labels, self.train_nodes, self.val_nodes,
             self.test_nodes] = torch.load('%s/traingraphL.pth' % table_name)
            l = []
            # avoid pin error in sparse cuda
            for i in self.val_adj_list:
                e = i.to_dense()
                l.append(e)
            self.val_adj_list = l
            print('load val data')

    def get_total_row_count(self):
        self.caoncat = 160
        self.reader.seek(0)
        return self.caoncat * 20

    def _new_reader(self):
        if self.reader is not None:
            self.reader.close()
        print('self.table_name', self.table_name)
        reader = open('%s/ppi-id_map.json'%self.table_name, "r")
        return reader

    def _read_record(self):
        try:
            column_l = self.reader.readline().strip().split("\t")
            # print(column_l)
            assert len(column_l) == self.args.column_len, "len(column_l) must be %d, now is %d" % (self.args.column_len, len(column_l))
        except:
            self.reader.seek(0)
            column_l = self.reader.readline().strip().split("\t")
        return column_l

    def find_split(self,adj, mapping, ds_label):
        """
        split data into train/val/test following previous works,
        where get relation between id sub-graph and tran,val or test set
        input schema:
            adj, mapping, labels
        output schema:
             dict_splits: splitting dictionary
        """
        nb_nodes = adj.shape[0]
        dict_splits={}
        for i in range(nb_nodes):
            for j in adj[i, :].nonzero()[1]:
                if mapping[i]==0 or mapping[j]==0:
                    dict_splits[0]=None
                elif mapping[i] == mapping[j]:
                    if ds_label[i]['val'] == ds_label[j]['val'] and ds_label[i]['test'] == ds_label[j]['test']:

                        if mapping[i] not in dict_splits.keys():
                            if ds_label[i]['val']:
                                dict_splits[mapping[i]] = 'val'

                            elif ds_label[i]['test']:
                                dict_splits[mapping[i]]='test'

                            else:
                                dict_splits[mapping[i]] = 'train'

                        else:
                            if ds_label[i]['test']:
                                ind_label='test'
                            elif ds_label[i]['val']:
                                ind_label='val'
                            else:
                                ind_label='train'
                            if dict_splits[mapping[i]]!= ind_label:
                                print ('inconsistent labels within a graph exiting!!!')
                                return None
                    else:
                        print ('label of both nodes different, exiting!!')
                        return None
        return dict_splits

    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        """
        Convert a scipy sparse matrix to a torch sparse tensor.
        input schema:
            sparse matrix: numpy
        output schema:
            sparse_matrix_torch: torch.sparse.FloatTensor(indices, values, shape)
        """
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def sys_normalized_adjacency(self,adj):
        """
        convert adj to normalized adj
        input schema:
            adj
        output schema:
            normalized adj
        """
        adj = sp.coo_matrix(adj)
        adj = adj + sp.eye(adj.shape[0])
        row_sum = np.array(adj.sum(1))
        row_sum = (row_sum == 0) * 1 + row_sum
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

    def loadData(self):
        """
        add data, data augment with adj, feature, labels and nodes index
        input schema:
            adj, labels, nodes, features
        output schema:
            augmented adj, labels, nodes, features
        """
        i = self.train_adj_list

        newi = []
        for t in range(self.caoncat):
            newi = newi + i

        self.train_adj_list = newi
        i = self.train_feat
        newi = i.repeat([self.caoncat, 1, 1])

        self.train_feat =newi
        i = self.train_labels
        newi = i.repeat([self.caoncat, 1, 1])

        self.train_labels=(newi)
        i = self.train_nodes
        newi = np.tile(i, (self.caoncat))
        self.train_nodes = newi

        return

    def __getitem__(self, idx):
        """
        item for different type,
        contain adj, feature, label and nodes index
        """
        if self.traintype == 'train':
            adj = self.train_adj_list[idx]
            feat = self.train_feat[idx]
            labels = self.train_labels[idx]
            nodes = self.train_nodes[[idx]]
        elif self.traintype == 'eval':
            adj = self.val_adj_list[idx]
            feat = self.val_feat[idx]
            labels = self.val_labels[idx]
            nodes = self.val_nodes[[idx]]
        else:
            adj = self.test_adj_list[idx]
            feat = self.test_feat[idx]
            labels = self.test_labels[idx]
            nodes = self.test_nodes[[idx]]

        return feat, adj, labels, nodes, self.traintype

    def __len__(self):
        """
        get len for different type
        """
        if self.traintype == 'train':
            return len(self.train_adj_list)
        elif self.traintype == 'eval':
            return 2
        else:
            return 2

    def __del__(self):
        if self.reader is not None:
            self.reader.close()
