# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch

# trunk model init
def default_weight_init(tensor):
    # torch.nn.init.xavier_uniform(tensor)
    torch.nn.init.xavier_uniform_(tensor)

def default_bias_init(tensor):
    torch.nn.init.constant_(tensor, 0)

# lite plugin model init
def default_lite_plugin_init(layer):
    # torch.nn.init.xavier_uniform(layer.weight, gain=0.001)
    torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
    # torch.nn.init.constant_(layer.weight, 0)
    torch.nn.init.constant_(layer.bias, 0)

# naive plugin model init
def default_naive_plugin_init(layer):
    torch.nn.init.constant_(layer.weight, 0)
    torch.nn.init.constant_(layer.bias, 0)

if __name__ == '__main__':
    # model.apply(weight_init_normal)
    dimension = 10
    plugin_layer = torch.nn.Linear(dimension, dimension // 2, True)
    print("-" * 50)
    print("original")
    print("plugin_layer.weight", plugin_layer.weight)
    print("plugin_layer.bias", plugin_layer.bias)
    default_weight_init(plugin_layer.weight)
    default_bias_init(plugin_layer.bias)
    print("-" * 50)
    print("trunk_init")
    print("plugin_layer.weight", plugin_layer.weight)
    print("plugin_layer.bias", plugin_layer.bias)
    default_lite_plugin_init(plugin_layer)
    print("-" * 50)
    print("lite_plugin_init")
    print("plugin_layer.weight", plugin_layer.weight)
    print("plugin_layer.bias", plugin_layer.bias)
    default_naive_plugin_init(plugin_layer)
    print("-" * 50)
    print("naive_plugin_init")
    print("plugin_layer.weight", plugin_layer.weight)
    print("plugin_layer.bias", plugin_layer.bias)