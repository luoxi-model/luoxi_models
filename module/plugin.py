# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
from . import initializer


class Plugin(torch.nn.Module):
    def __init__(self, dimension, activation_func):
        super(Plugin, self).__init__()
        self._activation_func = activation_func()
        self.downsampling_layer = torch.nn.Linear(dimension, dimension//2, True)
        self.upsampling_layer = torch.nn.Linear(dimension//2, dimension, True)

        initializer.default_lite_plugin_init(self.downsampling_layer)
        initializer.default_lite_plugin_init(self.upsampling_layer)

        module_lst = [
            self.downsampling_layer,
            activation_func(),
            self.upsampling_layer
        ]
        self.net = torch.nn.Sequential(*module_lst)

    def forward(self, x):
        residual = self.net(x)
        return self._activation_func(x + residual)


if __name__ == '__main__':
    plugin_model = Plugin(32, torch.nn.Tanh)
    print("=" * 50)
    print("plugin_model.downsampling_layer", plugin_model.downsampling_layer.weight,
          plugin_model.downsampling_layer.bias, sep="\n")
    print("-" * 50)
    print("plugin_model.upsampling_layer", plugin_model.upsampling_layer.weight,
          plugin_model.upsampling_layer.bias, sep="\n")
