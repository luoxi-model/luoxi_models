# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
from torch import nn
from . import common

class TargetAttention(nn.Module):
    def __init__(self, key_dimension, value_dimension):
        super(TargetAttention, self).__init__()
        self._target_key_transform = common.Linear(key_dimension, key_dimension, bias=False)
        self._item_key_transform = common.Linear(key_dimension, key_dimension, bias=False)
        self._value_transform = common.Linear(value_dimension, value_dimension, bias=False)
        self._scaler = key_dimension ** 0.5

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, target_key, item_keys, item_values, mask):
        """
        :param target_key: B * D
        :param item_keys: B * L * D
        :param item_values: B * L * D
        :param mask: B * L
        :return:
        """
        assert item_keys.shape[1] == item_values.shape[1]
        assert target_key.shape[-1] == item_keys.shape[-1]
        assert target_key.shape[0] == item_keys.shape[0] == item_values.shape[0]

        target_key = self._target_key_transform(target_key)[:, None, :]
        item_keys = self._item_key_transform(item_keys)
        item_values = self._value_transform(item_values)

        atten_weights = torch.sum(target_key * item_keys, dim=-1, keepdim=True) / self._scaler
        if mask is not None:
            atten_weights += -1e8 * (1 - mask[:, :, None])
            
        atten_weights = torch.softmax(atten_weights, dim=1)
        return torch.sum(atten_weights * item_values, dim=1)

if __name__ == '__main__':
    target_embed = torch.randn(16, 8)
    item_keys = torch.randn(16, 10, 8)
    item_values = torch.randn(16, 10, 23)

    m = TargetAttention(
        key_dimension=8,
        value_dimension=23,
        value_out_dimension=32
    )
    mask = torch.cat([torch.ones([16, 4]), torch.zeros([16, 6])], dim=1)
    data = m(target_embed, item_keys, item_values, mask)
    print(data.shape)