# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.metrics import f1_score


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        """
        :param in_features: input feature dimension
        :param out_features: output feature dimension
        :return:
        """
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class GGCN(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha,variant,args,onnx_step):
        super(GGCN, self).__init__()
        """
        :param in_features: input feature dimension
        :param out_features: output feature dimension
        :return:
        """
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.onnx_step = onnx_step
        input_tables = args.tables.split(",")
        [self.train_adj_list, self.val_adj_list, self.test_adj_list, self.train_feat, self.val_feat, self.test_feat,
         self.train_labels, self.val_labels, self.test_labels, self.train_nodes, self.val_nodes,
         self.test_nodes] = torch.load('%s/traingraphL.pth' % input_tables[0])

    def forward(self, x, adj):
        if self.onnx_step:
            self.onnx_step = False
            return self.onnxModel()
        _layers = []

        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.fcs[-1](layer_inner))
        return layer_inner

    def evalModel(self):
        """
        evaluate model with F1_score
        input schema:
             args: validation nodes, labels, adj, feature
        output schema:
             loss:  BCELoss of model inference
             score: F1_score of model inference
        """
        loss = 0
        score = 0
        device = torch.cuda.current_device()
        for i in range(2):
            nodes = self.val_nodes[i]
            labels = self.val_labels[i]
            labels = labels.to(device)
            feat = self.val_feat[i]
            adj = self.val_adj_list[i]
            feat = feat.to(device)
            adj = adj.to(device)
            output = self.forward(feat,adj)
            lossfn = torch.nn.BCELoss()
            loss += lossfn(output[:nodes], labels[:nodes])
            predict = np.where(output[:nodes].data.cpu().numpy() > 0.5, 1, 0)
            score += f1_score(labels[:nodes].data.cpu().numpy(), predict, average='micro')
        return loss/2, score/2

    def onnxModel(self):
        """
        evaluate onnx model with F1_score
        input schema:
             args: onnx nodes, labels, adj, feature
        output schema:
             loss:  BCELoss of model inference
             score: F1_score of model inference
        """
        loss = 0
        score = 0
        device = torch.cuda.current_device()
        for i in range(2):
            nodes = self.val_nodes[i]
            labels = self.val_labels[i]
            labels = labels.to(device)
            feat = self.val_feat[i]
            adj = self.val_adj_list[i]
            feat = feat.to(device)
            adj = adj.to(device)
            output = self.forward(feat,adj)
            lossfn = torch.nn.BCELoss()
            loss += lossfn(output[:nodes], labels[:nodes])
            predict = np.where(output[:nodes].data.cpu().numpy() > 0.5, 1, 0)
            score += f1_score(labels[:nodes].data.cpu().numpy(), predict, average='micro')
        return loss/2, score/2

    def inferModel(self):
        """
        evaluate model with test data in F1_score
        input schema:
             args: test nodes, labels, adj, feature
        output schema:
             loss:  BCELoss of model inference
             score: F1_score of model inference
        """
        loss = 0
        score = 0
        device = torch.cuda.current_device()
        for i in range(2):
            nodes = self.test_nodes[i]
            labels = self.test_labels[i]
            labels = labels.to(device)
            feat = self.test_feat[i]
            adj = self.test_adj_list[i]
            feat = feat.to(device)
            adj = adj.to(device)
            output = self.forward(feat,adj)
            lossfn = torch.nn.BCELoss()
            loss += lossfn(output[:nodes], labels[:nodes])
            predict = np.where(output[:nodes].data.cpu().numpy() > 0.5, 1, 0)
            score += f1_score(labels[:nodes].data.cpu().numpy(), predict, average='micro')
        return loss/2, score/2