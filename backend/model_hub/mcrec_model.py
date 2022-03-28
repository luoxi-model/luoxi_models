# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
import torch.nn as nn

class Embedding(nn.Module):
	def __init__(self, num, dim):
		"""
		build embedding table
		input schema:
			num: number of the embedding vectors
			dim: dimension of the embedding vector
		output schema:
			return a embedding table for lookup
		"""
		super(Embedding, self).__init__()
		self.emb = nn.Embedding(num, dim, padding_idx=0)

	def forward(self, idx):
		output = self.emb(idx)
		return output

class Attention(nn.Module):
	def __init__(self, dim_i, dim_o):
		"""
		build the target-aware attention
		input schema:
			dim_i: the dimension of the input feature vector
			dim_o: the dimension of the output feature vector
		output schema:
			return a aggregated vector from the context k, v of q 
		"""
		super(Attention, self).__init__()
		self.Q = nn.Linear(dim_i, dim_o)
		self.K = nn.Linear(dim_i, dim_o)
		self.V = nn.Linear(dim_i, dim_o)

	def forward(self, hist_seq_emb, hist_seq_mask, cand_emb):
		q, k, v = self.Q(cand_emb), self.K(hist_seq_emb), self.V(hist_seq_emb)

		# q: B x d; k: B x L x d; v: B x L x d
		# hist_seq_mask: B x L
		logits = torch.sum(q.unsqueeze(1) * k, dim=2)
		logits = logits * hist_seq_mask + logits * (1-hist_seq_mask) * (-2**32.0)
		scores = torch.softmax(logits, dim=1)

		output = torch.sum(scores.unsqueeze(2) * v, dim=1)

		return output

class CRec(nn.Module):
	def __init__(self, args):
		"""
		build the cloud-based recommendation model
		input schema:
			args: parameters for the model initialization
			hist_seq: the user historical sequence
			cand: the target item for scoring 
			label: click or not
		output:
			return the loss in the "train" mode or prediction in the other mode 
		"""
		super(CRec, self).__init__()
		self.task_type = args.task_type
		self.dim = args.dim
		self.num = args.num
		self.emb = Embedding(self.num, self.dim)
		self.att = Attention(self.dim, self.dim)
		self.projection = nn.Linear(self.dim, self.dim)
		self.classifier = nn.Linear(self.dim, 2)
		self.loss = nn.CrossEntropyLoss()			

	def forward(self, hist_seq, cand, label):
		hist_seq_emb = self.emb(hist_seq)
		hist_seq_mask = torch.where(hist_seq == 0, torch.zeros_like(hist_seq), torch.ones_like(hist_seq))
		cand_emb = self.emb(cand)

		agg_emb = self.att(hist_seq_emb, hist_seq_mask, cand_emb)
		logits = self.classifier(self.projection(agg_emb))

		if self.task_type!='train':
			pred = torch.softmax(logits, dim=1)[:,1]
			return pred
		else:	
			loss = torch.mean(self.loss(logits, label))
			return loss

class ORec(nn.Module):
	def __init__(self, args):
		"""
		build the on-device recommendation model
		input schema:
			args: parameters for the model initialization
			hist_seq: the user historical sequence
			cand: the target item for scoring 
			prior_score: the prior prediction from the agnostic cloud-based model
			label: click or not
		output:
			return the loss in the "train" mode or prediction in the other mode 
		"""
		super(ORec, self).__init__()
		self.task_type = args.task_type
		self.dim = args.dim
		self.num = args.num
		self.emb = Embedding(self.num, self.dim)
		self.att = Attention(self.dim, self.dim)
		self.projection = nn.Linear(self.dim, self.dim)
		self.classifier = nn.Linear(self.dim, 2)
		self.loss = nn.CrossEntropyLoss()			

	def forward(self, hist_seq, cand, prior_score, label):
		hist_seq_emb = self.emb(hist_seq)
		hist_seq_mask = torch.where(hist_seq == 0, torch.zeros_like(hist_seq), torch.ones_like(hist_seq))
		cand_emb = self.emb(cand)

		agg_emb = self.att(hist_seq_emb, hist_seq_mask, cand_emb)
		logits_res = self.classifier(self.projection(agg_emb))

		# thresholding the interval so that it does not overflow
		score = prior_score.unsqueeze(1)
		score_0 = 1.0 - score
		score_1 = score
		score_0 = score_0 * (1.0 - 1e-3) + 1e-4
		score_1 = score_1 * (1.0 - 1e-3) + 1e-4	
		logits_main = torch.cat([-torch.log(1.0/score_0 - 1.0), -torch.log(1.0/score_1 - 1.0)], dim=1)

		logits = logits_res + logits_main

		if self.task_type != "train":
			pred = torch.softmax(logits, dim=1)[:,1]
			return pred
		else:	
			loss = torch.mean(self.loss(logits, label))
			return loss

class Controller(nn.Module):
	def __init__(self, args):
		"""
		build the controller model
		input schema:
			args: parameters for the model initialization
			hist_seq: the user historical sequence
			label: which mechanism (e.g., the cloud-based session recommendation, the cloud-based refresh or the on-device recommendation) to invoke for recommendation
		output:
			return the loss in the "train" mode or prediction in the other mode 
		"""
		super(Controller, self).__init__()
		self.task_type = args.task_type
		self.dim = args.dim
		self.num = args.num
		self.emb = Embedding(self.num, self.dim)
		self.att = Attention(self.dim, self.dim)
		self.projection = nn.Linear(self.dim, self.dim)
		self.classifier = nn.Linear(self.dim, 3)
		self.loss = nn.CrossEntropyLoss()			

	def forward(self, hist_seq, label):
		hist_seq_emb = self.emb(hist_seq)
		hist_seq_mask = torch.where(hist_seq == 0, torch.zeros_like(hist_seq), torch.ones_like(hist_seq))
		# we built a pseudo averaging cand embedding as the query in target-aware attention
		cand_emb = torch.sum(hist_seq_emb * hist_seq_mask.unsqueeze(2), dim=1)/torch.sum(hist_seq_mask, dim=1, keepdim=True)

		agg_emb = self.att(hist_seq_emb, hist_seq_mask, cand_emb)
		logits = self.classifier(self.projection(agg_emb))

		if self.task_type != "train":
			pred = torch.softmax(logits, dim=1)
			return pred
		else:	
			loss = torch.mean(self.loss(logits, label))
			return loss
