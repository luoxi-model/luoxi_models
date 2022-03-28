# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch

def mock_orec_data():
    num_samples = 2
    mock_data = {}

    hist_seq = [[int(e) for e in "44,172,602,602,163,258,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0".strip().split(",")]]
    mock_data['hist_seq'] = torch.LongTensor(hist_seq).repeat((num_samples, 1))

    cand = [[int("672")]]
    mock_data['cand'] = torch.LongTensor(cand).repeat((num_samples, 1)).squeeze(1)

    prior_score = [[float("0.1192")]]
    mock_data['prior_score'] = torch.Tensor(prior_score).repeat((num_samples, 1)).squeeze(1)

    label = [[int("0")]]
    mock_data['label'] = torch.LongTensor(label).repeat((num_samples, 1)).squeeze(1)

    return mock_data

