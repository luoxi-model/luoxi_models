# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.parameter import Parameter

def cosine_similarity_onnx_exportable(x1, x2, dim=-1):
    cross = (x1 * x2).sum(dim=dim)
    x1_l2 = (x1 * x1).sum(dim=dim)
    x2_l2 = (x2 * x2).sum(dim=dim)
    return torch.div(cross, (x1_l2 * x2_l2).sqrt())

class Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.onnx_step = args.onnx_step
        self.args = args
        self.embs_table_dict = {}
        # Initialize field embedding tables
        table_name = ['userid']+args.user_fea_name.split(',')+args.item_fea_name.split(',')
        table_size = [int(x) for x in args.table_size.split(',')]
        dim_config = dict(zip(table_name,table_size))
        for i in dim_config:
            if self.onnx_step:
                if i != 'userid' and i in args.user_fea_name.split(','):
                    self.embs_table_dict[i] = nn.Embedding(dim_config[i], args.kv_dimension, padding_idx=0)
                    torch.nn.init.normal_(self.embs_table_dict[i].weight.data, mean=0.0, std=float(args.kv_dimension ** -0.5))
            else:
                if i != 'userid':
                    self.embs_table_dict[i] = nn.Embedding(dim_config[i], args.kv_dimension, padding_idx=0)
                    torch.nn.init.normal_(self.embs_table_dict[i].weight.data, mean=0.0, std=float(args.kv_dimension ** -0.5))
        # Initialize GNN id embedding tables
        self.model_name = args.model
        if self.model_name == 'CIGAR_WO_PN' or 'CIGAR' and (not self.onnx_step):
            self.gnn_layers = list(map(int, args.gnn_layers.split(',')))
            self.mem_dimension = args.mem_dimension
            self.gnn_layers.append(self.mem_dimension)
            for k in range(len(self.gnn_layers) - 1):
                self.embs_table_dict['user_mem_%d' % k] = nn.Embedding(dim_config['userid'],
                                                                       self.gnn_layers[k],
                                                                       padding_idx=0)
                torch.nn.init.normal_(self.embs_table_dict['user_mem_%d' % k].weight.data, mean=0.0,
                                      std=float(self.gnn_layers[k] ** -0.5))
                self.embs_table_dict['user_mem_%d' % k].weight.requires_grad = False
        self.embs_table_dict = nn.ModuleDict(self.embs_table_dict)

    def forward(self, input):
        # User emb
        user_emb_list = []
        for i in self.args.user_fea_name.split(','):
            user_emb_list.append(self.embs_table_dict[i](input[i]))
        user_emb = torch.cat(user_emb_list, -1)

        # Item emb
        if self.onnx_step:
            item_emb = torch.squeeze(input["item_embedding"], dim=1)
            mean_emb = torch.squeeze(input["mean_embedding"], dim=1)
            gnn_output = torch.squeeze(input["gnn_output"], dim=1) if "gnn_output" in input else None
            output = {
                'user_emb': user_emb,
                'item_emb': item_emb,
                'mean_emb': mean_emb,
                'gnn_output': gnn_output
            }
        else:
            item_emb_list = []
            for i in self.args.item_fea_name.split(','):
                item_emb_list.append(self.embs_table_dict[i](input[i]))
            item_emb = torch.cat(item_emb_list, -1)

            # Seq emb
            seq_emb_list = []
            for i in self.args.item_fea_name.split(','):
                seq_emb_list.append(self.embs_table_dict[i](input[i+'_seq']))
            seq_emb = torch.cat(seq_emb_list, -1)

            output = {
                'user_emb': user_emb,
                'item_emb': item_emb,
                'seq_emb': seq_emb,
                'emb_table':self.embs_table_dict
            }

            # GNN emb
            if self.model_name == 'CIGAR_WO_PN' or 'CIGAR':
                user_id_emb_list = []
                neigh_ids_emb_list = []
                for k in range(len(self.gnn_layers) - 1):
                    user_id_emb_list.append(self.embs_table_dict['user_mem_%d' % k](input['userid']))
                    neigh_ids_emb_list.append(self.embs_table_dict['user_mem_%d' % k](input['neighbor_ids']))
                output['user_id_emb_list'] = user_id_emb_list
                output['neigh_ids_emb_list'] = neigh_ids_emb_list

        return output

class AggregationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(AggregationLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.agg_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim).to(device),
            nn.Tanh()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, neigh_vecs, query, neigh_mask):
        # Aggregation: mean aggregator
        value_embeddings_flatten = self.agg_layer(neigh_vecs)
        mean_embedding = mean_pooling(value_embeddings_flatten, neigh_mask)
        return mean_embedding

class ProtypicalNet(nn.Module):
    def __init__(self, cls_num, input_dim, device):
        super(ProtypicalNet, self).__init__()
        # Initialize prototype embedding tables
        self.prototypes_para = Parameter(torch.empty(cls_num, input_dim,device=device))
        torch.nn.init.xavier_uniform_(self.prototypes_para)
        self.input_dim = input_dim
        self.cls_num = cls_num
        self.device = device

    def forward(self, input, training=True, onnx_step=False):
        input = F.normalize(input, p=2, dim=1)
        self.prototypes = F.normalize(self.prototypes_para, p=2, dim=1)
        a = input.detach().unsqueeze(1) * torch.ones(self.prototypes.unsqueeze(0).shape).to(self.device)  # (B,K,D)
        b = self.prototypes.unsqueeze(0) * torch.ones(input.unsqueeze(1).shape).to(self.device)  # (B,K,D)
        # Compute cosine similarity between input emb and prototype emb
        sim = cosine_similarity_onnx_exportable(a,b,dim=-1) \
            if onnx_step else F.cosine_similarity(a,b,dim=-1)  # (B,K)
        sim_loss, div_loss = 0, 0
        if training:
            # output the most similar prototype using Gumbel-softmax when training
            e = F.gumbel_softmax(sim, tau=0.1)  # (B,K)
            pt = torch.einsum('bk,kv->bv', e, self.prototypes)  # (B,D)
            # update the prototypes to be similar to the input
            sim_loss = torch.mean(1.0-torch.einsum('bk,bk->b', e, sim))
            # diversity loss
            for i in range(self.cls_num):
                for j in range(self.cls_num):
                    if i != j:
                        div_loss += F.cosine_similarity(self.prototypes[i], self.prototypes[j],dim=-1) \
                            if not onnx_step else cosine_similarity_onnx_exportable(self.prototypes[i], self.prototypes[j],dim=-1)
        else:
            pt_id = torch.argmax(sim,dim=-1)  # (B)
            pt = self.prototypes[pt_id]  # (B,D)
        return pt, self.prototypes, sim_loss, div_loss

class PredictionLayer(nn.Module):
    def __init__(self,pred_layers,input_size):
        super(PredictionLayer, self).__init__()
        self.pred_layers = []
        for i in range(len(pred_layers)):
            if i == 0:
                self.pred_layers.append(nn.Linear(input_size, pred_layers[i]))
                if pred_layers[i] != 1:
                    self.pred_layers.append(nn.ReLU())
            else:
                self.pred_layers.append(nn.Linear(pred_layers[i - 1], pred_layers[i]))
                if pred_layers[i] != 1:
                    self.pred_layers.append(nn.ReLU())
        self.linears = nn.ModuleList(self.pred_layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # MLP as final prediction layer
        for layer in self.linears:
            x = layer(x)
        return x

class GroupPredictionLayer(nn.Module):
    def __init__(self,hidden_dims,input_size,pt_dim):
        super(GroupPredictionLayer, self).__init__()
        self.pred_layers = []
        self.hidden_dims = hidden_dims
        # original prediction layer
        for i in range(len(hidden_dims)):
            if i == 0:
                self.pred_layers.append(nn.Sequential(nn.Linear(input_size, hidden_dims[i])))
            else:
                self.pred_layers.append(nn.Sequential(nn.Linear(hidden_dims[i - 1], hidden_dims[i])))
        # generate parameters from given prototypes
        self.weight_transform = {}
        for i in range(len(hidden_dims)):
            self.weight_transform['b%d' % i] = nn.Sequential(
                nn.Linear(pt_dim, hidden_dims[i]),
                nn.Tanh())
            if i == 0:
                self.weight_transform['w%d' % i] = nn.Sequential(
                    nn.Linear(pt_dim, input_size*hidden_dims[i]),
                    nn.Tanh())
            else:
                self.weight_transform['w%d' % i] = nn.Sequential(
                    nn.Linear(pt_dim, hidden_dims[i - 1]*hidden_dims[i]),
                    nn.Tanh())
        self.weight_transform = nn.ModuleDict(self.weight_transform)
        self.linears = nn.ModuleList(self.pred_layers)

    def forward(self, x, pt):
        # group-level personalized prediction layer
        self.weight = {}
        bz = x.size(0)
        origin_output = x
        group_output = x
        for i in range(len(self.hidden_dims)):
            b = self.weight_transform['b%d' % i](pt) * self.linears[i][0].bias.unsqueeze(0)
            self.weight['b%d' % i] = b
            w = self.weight_transform['w%d' % i](pt)
            w = torch.reshape(w, (bz, self.hidden_dims[i], -1)) * self.linears[i][0].weight.unsqueeze(0)
            self.weight['w%d' % i] = w
            origin_output = self.linears[i](origin_output)
            group_output = torch.einsum("ba,bca->bc", group_output, w) + b
            if self.hidden_dims[i] != 1:
                group_output = torch.relu(group_output)
                origin_output = torch.relu(origin_output)
        return origin_output, group_output

def get_mask(inputs, empty_value=0):
    return torch.not_equal(inputs, empty_value).float()

def get_length(inputs, masks=None, empty_value=0):
    """
    :param inputs: N-D value matrix. If mask is provided, it is ignored
    :param masks: Binary matrix has the same shape as inputs
    :param empty_value:
    :return:
    """
    if masks is None:
        masks = get_mask(inputs, empty_value)
    return torch.sum(masks, dim=-1)

def mean_pooling(inputs, mask=None, length=None):
    """
    Perform average pooling on inputs
    :param inputs: (N)-D tensor. The second to last dimension would be pooled
    :param mask: (N-1)-D binary int tensor. If set, the mask would be multiplied to inputs
    :param length: (N-2)-D int tensor. If not set, the length would be inferred from mask
    :return:
    """
    if length is None and mask is not None:
        length = get_length(None, mask)

    if mask is not None:
        inputs = inputs * torch.unsqueeze(mask, dim=-1)

    if length is None:
        return torch.mean(inputs, dim=-2)
    else:
        inputs = torch.sum(inputs, dim=-2)
        return torch.div(inputs, torch.max(torch.unsqueeze(length, dim=-1), torch.ones_like(torch.unsqueeze(length, dim=-1))))

def guarded_roc_auc_score(y_true, y_score):
    num_true = sum(y_true)
    if (num_true == 0) or (num_true >= len(y_true)):
        return -1
    return roc_auc_score(y_true=y_true, y_score=y_score)

class WindowAUC(object):
    def __init__(self, window_size=1024):
        self.window_size = window_size
        self.y_true = []
        self.y_score = []

    def extend(self, y_true, y_score):
        self.y_true.extend(y_true)
        self.y_true = self.y_true[-self.window_size:]
        self.y_score.extend(y_score)
        self.y_score = self.y_score[-self.window_size:]

    def compute_auc(self):
        return guarded_roc_auc_score(self.y_true, self.y_score)

class CIGAR_WO_PN(nn.Module):
    def __init__(self, args, device):
        super(CIGAR_WO_PN, self).__init__()
        self.device = device
        self.onnx_step = args.onnx_step
        self.task_type = args.task_type
        self.emb = Embedding(args)
        pred_layers = list(map(int, args.dim_hidden.split(',')))
        input_size = (len(args.user_fea_name.split(','))+len(args.item_fea_name.split(','))*3)*args.kv_dimension+int(args.gnn_layers.split(',')[-1])
        ##########  GNN  ###########
        self._gnn_layers = list(map(int, args.gnn_layers.split(',')))
        self.mem_dimension = args.mem_dimension
        self._gnn_layers.append(self.mem_dimension)
        self.agg_list = []
        for k in range(len(self._gnn_layers) - 1):
            self.agg_list.append(AggregationLayer(self._gnn_layers[k], self._gnn_layers[k+1],self.device))
        self.agg_list = nn.ModuleList(self.agg_list)
        ############################
        self.pred_layers = PredictionLayer(pred_layers, input_size)
        self.pred_loss = torch.nn.BCEWithLogitsLoss()
        self.mobile_auc_metric = WindowAUC(window_size=512)

    def forward(self, input, y):
        # Get embedding
        bz = input['userid'].size(0)
        emb_dict = self.emb(input)
        user_emb = emb_dict['user_emb']
        item_emb = emb_dict['item_emb']
        if not self.onnx_step:
            seq_emb = emb_dict['seq_emb']
            emb_table = emb_dict['emb_table']
            mask = get_mask(input['adgroup_id_seq'])
            neigh_mask = get_mask(input["neighbor_ids"])
            length = get_length(input['adgroup_id_seq'])
            # Sequence Encoder
            mean_emb = mean_pooling(seq_emb,mask,length) # PNN

            # GNN
            user_id_emb_list = emb_dict['user_id_emb_list']
            neigh_ids_emb_list = emb_dict['neigh_ids_emb_list']
            gnn_output_list = []
            gnn_output = 0.0
            emb_table['user_mem_0'].weight[input['userid']] = mean_emb  # assign memory
            for i in range(len(self._gnn_layers) - 1):
                if i == 0:
                    gnn_output_tmp = self.agg_list[i](neigh_ids_emb_list[i].detach().to(self.device),
                                                  user_id_emb_list[i].detach().to(self.device),
                                                  neigh_mask)
                else :
                    gnn_output_tmp = self.agg_list[i](neigh_ids_emb_list[i].detach().to(self.device),
                                                      user_id_emb_list[i].detach().to(self.device),
                                                      neigh_mask)
                if i+1 != len(self._gnn_layers) - 1:
                    emb_table['user_mem_%d' % (i+1)].weight[input['userid']] = gnn_output_tmp
                gnn_output += gnn_output_tmp
                gnn_output_list.append(gnn_output_tmp)
            gnn_output = gnn_output / (len(self._gnn_layers) - 1)
        else:
            length = input['seq_length']
            mean_emb = emb_dict["mean_emb"]
            gnn_output = emb_dict["gnn_output"]

        # Final layer
        classifier_feature = torch.cat([
            user_emb,
            item_emb,
            mean_emb,
            item_emb * mean_emb,
            gnn_output],
            -1)

        logits = self.pred_layers(classifier_feature)
        score = torch.sigmoid(logits)
        label = torch.unsqueeze(input['label'], dim=-1)
        cls_loss = self.pred_loss(logits, label)
        if self.task_type == 'train':
            if self.training:
                self.mobile_auc_metric.extend(y_true=label.view(bz).tolist(),
                                              y_score=score.view(bz).tolist())
                mobile_cls_auc = cls_loss.new_tensor(self.mobile_auc_metric.compute_auc())
                stats = [x for x in [
                    cls_loss, mobile_cls_auc
                ] if x is not None]
                return [cls_loss] + stats
            else:
                return cls_loss, label.view(bz).tolist(), score.view(bz).tolist()
        else:
            # item_id, user_id, seq_length, item_emb, mem_emb, score
            if self.onnx_step:
                return input['adgroup_id'], input['userid'], input['group_score'], score, label
            else:
                return input['adgroup_id'], input['userid'], length, item_emb, mean_emb, gnn_output, score

    def _rand_prop(self):
        pass

class CIGAR_WO_CDGNN(nn.Module):
    def __init__(self, args, device):
        super(CIGAR_WO_CDGNN, self).__init__()
        self.device = device
        self.onnx_step = args.onnx_step
        self.task_type = args.task_type
        self.emb = Embedding(args)
        self.pt_dim = len(args.item_fea_name.split(',')) * args.kv_dimension
        dim_hidden = list(map(int, args.dim_hidden.split(',')))
        input_size = (len(args.user_fea_name.split(','))+len(args.item_fea_name.split(','))*3) * args.kv_dimension

        self.PN = ProtypicalNet(args.prototype_num, self.pt_dim, self.device)

        self.group_pred_layers = GroupPredictionLayer(dim_hidden, input_size, self.pt_dim)
        self.pred_loss = torch.nn.BCEWithLogitsLoss()
        self.origin_auc_metric = WindowAUC(window_size=512)
        self.group_auc_metric = WindowAUC(window_size=512)

    def forward(self, input, y):
        bz = input['userid'].size(0)
        emb_dict = self.emb(input)
        user_emb = emb_dict['user_emb']
        item_emb = emb_dict['item_emb']
        # Get embedding
        if not self.onnx_step:
            seq_emb = emb_dict['seq_emb']
            emb_table = emb_dict['emb_table']
            mask = get_mask(input['adgroup_id_seq'])
            neigh_mask = get_mask(input["neighbor_ids"])
            length = get_length(input['adgroup_id_seq'])
            # Sequence Encoder
            mean_emb = mean_pooling(seq_emb, mask, length)  # PNN
        else:
            length = input['seq_length']
            mean_emb = emb_dict["mean_emb"]

        # Final layer
        classifier_feature = torch.cat([
            user_emb,
            item_emb,
            mean_emb,
            item_emb * mean_emb],
            -1)
        # generate prototype given user embedding
        pt, prototypes, sim_loss, div_loss = self.PN(mean_emb, self.task_type == 'train' and not self.onnx_step,self.onnx_step)
        # get original and personalized prediction score
        origin_logits, group_logits = self.group_pred_layers(classifier_feature, pt.detach())
        origin_score = torch.sigmoid(origin_logits)
        group_score = torch.sigmoid(group_logits)

        label = torch.unsqueeze(input['label'], dim=-1)
        origin_cls_loss = self.pred_loss(origin_logits, label)
        group_cls_loss = self.pred_loss(group_logits, label)
        if self.task_type == 'train':
            if self.training:
                self.origin_auc_metric.extend(y_true=label.view(bz).tolist(),
                                              y_score=origin_score.view(bz).tolist())
                self.group_auc_metric.extend(y_true=label.view(bz).tolist(),
                                              y_score=group_score.view(bz).tolist())
                origin_cls_auc = origin_cls_loss.new_tensor(self.origin_auc_metric.compute_auc())
                group_cls_auc = origin_cls_loss.new_tensor(self.group_auc_metric.compute_auc())
                cls_loss = 2 * group_cls_loss + origin_cls_loss + 0.1*div_loss + sim_loss
                stats = [x for x in [
                    origin_cls_loss,group_cls_loss,div_loss ,sim_loss ,origin_cls_auc,  group_cls_auc
                ] if x is not None]
                return [cls_loss] + stats
            else:
                return group_cls_loss, label.view(bz).tolist(), group_score.view(bz).tolist()
        else:
            if self.onnx_step:
                return input['adgroup_id'], input['userid'], input['group_score'], group_score, label
            else:
                return input['adgroup_id'], input['userid'], length, item_emb, mean_emb, group_score

    def _rand_prop(self):
        pass

class CIGAR(nn.Module):
    def __init__(self, args, device):
        super(CIGAR, self).__init__()
        self.device = device
        self.onnx_step = args.onnx_step
        self.task_type = args.task_type
        self.emb = Embedding(args)
        self.pt_dim = len(args.item_fea_name.split(',')) * args.kv_dimension
        dim_hidden = list(map(int, args.dim_hidden.split(',')))
        input_size = (len(args.user_fea_name.split(','))+len(args.item_fea_name.split(','))*3) * args.kv_dimension + int(args.gnn_layers.split(',')[-1])

        ##########  GNN  ###########
        if not self.onnx_step:
            self._gnn_layers = list(map(int, args.gnn_layers.split(',')))
            self.mem_dimension = args.mem_dimension
            self._gnn_layers.append(self.mem_dimension)
            self.agg_list = []
            for k in range(len(self._gnn_layers) - 1):
                self.agg_list.append(AggregationLayer(self._gnn_layers[k], self._gnn_layers[k + 1], self.device))
            self.agg_list = nn.ModuleList(self.agg_list)
        ############################

        self.PN = ProtypicalNet(args.prototype_num, self.pt_dim, self.device)

        self.group_pred_layers = GroupPredictionLayer(dim_hidden, input_size, self.pt_dim)
        self.pred_loss = torch.nn.BCEWithLogitsLoss()
        self.origin_auc_metric = WindowAUC(window_size=512)
        self.group_auc_metric = WindowAUC(window_size=512)

    def forward(self, input, y):
        # Get embedding
        bz = input['userid'].size(0)
        emb_dict = self.emb(input)
        user_emb = emb_dict['user_emb']
        item_emb = emb_dict['item_emb']
        if not self.onnx_step:
            seq_emb = emb_dict['seq_emb']
            emb_table = emb_dict['emb_table']
            mask = get_mask(input['adgroup_id_seq'])
            neigh_mask = get_mask(input["neighbor_ids"])
            length = get_length(input['adgroup_id_seq'])
            # Sequence Encoder
            mean_emb = mean_pooling(seq_emb, mask, length)  # PNN
            # GNN
            user_id_emb_list = emb_dict['user_id_emb_list']
            neigh_ids_emb_list = emb_dict['neigh_ids_emb_list']
            gnn_output_list = []
            gnn_output = 0.0
            emb_table['user_mem_0'].weight[input['userid']] = mean_emb  # assign memory
            for i in range(len(self._gnn_layers) - 1):
                if i == 0:
                    gnn_output_tmp = self.agg_list[i](neigh_ids_emb_list[i].to(self.device).detach(),
                                                      user_id_emb_list[i].to(self.device).detach(),
                                                      neigh_mask)
                else:
                    gnn_output_tmp = self.agg_list[i](neigh_ids_emb_list[i].to(self.device).detach(),
                                                      user_id_emb_list[i].to(self.device).detach(),
                                                      neigh_mask)
                if i + 1 != len(self._gnn_layers) - 1:
                    emb_table['user_mem_%d' % (i + 1)].weight[input['userid']] = gnn_output_tmp
                gnn_output += gnn_output_tmp
                gnn_output_list.append(gnn_output_tmp)
            gnn_output = gnn_output / (len(self._gnn_layers) - 1)
        else:
            length = input['seq_length']
            mean_emb=emb_dict["mean_emb"]
            gnn_output=emb_dict["gnn_output"]

        # Final layer
        classifier_feature = torch.cat([
            user_emb,
            item_emb,
            mean_emb,
            item_emb * mean_emb,
            gnn_output
        ],-1)
        # generate prototype given user embedding
        pt, prototypes, sim_loss, div_loss = self.PN(mean_emb, self.task_type == "train" and not self.onnx_step,self.onnx_step)
        # get original and personalized prediction score
        origin_logits, group_logits = self.group_pred_layers(classifier_feature, pt.detach())
        origin_score = torch.sigmoid(origin_logits)
        group_score = torch.sigmoid(group_logits)
        label = torch.unsqueeze(input["label"], dim=-1)
        origin_cls_loss = self.pred_loss(origin_logits, label)
        group_cls_loss = self.pred_loss(group_logits, label)
        if self.task_type == "train":
            if self.training:
                self.origin_auc_metric.extend(y_true=label.view(bz).tolist(),
                                              y_score=origin_score.view(bz).tolist())
                self.group_auc_metric.extend(y_true=label.view(bz).tolist(),
                                              y_score=group_score.view(bz).tolist())
                origin_cls_auc = origin_cls_loss.new_tensor(self.origin_auc_metric.compute_auc())
                group_cls_auc = origin_cls_loss.new_tensor(self.group_auc_metric.compute_auc())
                cls_loss = 2 * group_cls_loss + origin_cls_loss + 0.1*div_loss + sim_loss
                stats = [x for x in [
                    origin_cls_loss,group_cls_loss ,origin_cls_auc,  group_cls_auc
                ] if x is not None] if self.onnx_step else [x for x in [
                    origin_cls_loss,group_cls_loss,div_loss ,sim_loss ,origin_cls_auc,  group_cls_auc
                ] if x is not None]
                return [cls_loss] + stats
            else:
                return group_cls_loss, label.view(bz).tolist(), group_score.view(bz).tolist()
        else :
            # item_id, user_id, seq_length, item_emb, mem_emb, gnn_output, score
            if self.onnx_step:
                return input['adgroup_id'], input['userid'], input['group_score'], group_score, label
            else:
                return input['adgroup_id'], input['userid'], length, item_emb, mean_emb, gnn_output, group_score

    def _rand_prop(self):
        pass

class PNN(nn.Module):
    def __init__(self, args, device):
        super(PNN, self).__init__()
        self.device = device
        self.onnx_step = args.onnx_step
        self.task_type = args.task_type
        self.emb = Embedding(args)
        pred_layers = list(map(int, args.dim_hidden.split(',')))
        input_size = (len(args.user_fea_name.split(','))+len(args.item_fea_name.split(','))*3)*args.kv_dimension
        self.pred_layers = PredictionLayer(pred_layers, input_size)
        self.pred_loss = torch.nn.BCEWithLogitsLoss()
        self.mobile_auc_metric = WindowAUC(window_size=512)

    def forward(self, input, y):
        # Get embedding
        bz = input['userid'].size(0)
        emb_dict = self.emb(input)
        user_emb = emb_dict['user_emb']
        item_emb = emb_dict['item_emb']
        if not self.onnx_step:
            seq_emb = emb_dict['seq_emb']
            mask = get_mask(input['adgroup_id_seq'])
            length = get_length(input['adgroup_id_seq'])

            # Sequence Encoder
            mean_emb = mean_pooling(seq_emb,mask,length) # PNN
        else:
            length = input['seq_length']
            mean_emb = emb_dict["mean_emb"]
            gnn_output = emb_dict["gnn_output"]

        # Final layer
        classifier_feature = torch.cat([
            user_emb,
            item_emb,
            mean_emb,
            item_emb * mean_emb],
            -1)

        logits = self.pred_layers(classifier_feature)
        score = torch.sigmoid(logits)
        label = torch.unsqueeze(input['label'], dim=-1)
        if self.task_type == 'train':
            if self.training:
                cls_loss = self.pred_loss(logits, label.detach())
                self.mobile_auc_metric.extend(y_true=label.view(bz).tolist(),
                                              y_score=score.view(bz).tolist())
                mobile_cls_auc = cls_loss.new_tensor(self.mobile_auc_metric.compute_auc())
                stats = [x for x in [
                    cls_loss, mobile_cls_auc
                ] if x is not None]
                return [cls_loss] + stats
            else:
                cls_loss = self.pred_loss(logits, label.detach())
                return cls_loss, label.view(bz).tolist(), score.view(bz).tolist()
        else:
            if self.onnx_step:
                return input['adgroup_id'], input['userid'], input['group_score'], score, label
            else:
                # item_id, user_id, seq_length, item_emb, mem_emb, score
                return input['adgroup_id'], input['userid'], length, item_emb, mean_emb, score


    def _rand_prop(self):
        pass