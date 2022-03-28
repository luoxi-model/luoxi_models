# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from collections import defaultdict
import torch.nn
import torch.nn as nn
from module import attention, encoder, common, plugin
import logging

logger = logging.getLogger(__name__)

__all__ = ["ModelMeta", "get_model_meta", "model"]

consts = {
        "FIELD_USER_ID": "user_id",
        "FIELD_TARGET_ID": "target_id",
        "FIELD_CLK_SEQUENCE": "clk_sequence",
        "FIELD_LABEL": "label",
        "FIELD_GROUP_ID": "group_id"
    }


class ModelMeta(object):
    def __init__(self, config_parser=None, data_loader=None, model_builder=None):
        """
        Build parent class
        """
        self._config_parser = config_parser
        self._data_loader = data_loader
        self._model_builder = model_builder

    @property
    def arch_config_parser(self):
        return self._config_parser

    def set_arch_config_parser(self, parser):
        self._check(self._config_parser, "Config parser has been set")
        self._config_parser = parser

    @property
    def data_loader_builder(self):
        return self._data_loader

    def set_data_loader_builder(self, loader):
        self._check(self._data_loader, "Data loader builder has been set")
        self._data_loader = loader

    @property
    def model_builder(self):
        return self._model_builder

    def set_model_builder(self, model_builder):
        self._check(self._model_builder, "Model builder has been set")
        self._model_builder = model_builder

    def _check(self, value, message):
        if value is not None:
            raise ValueError(message)

    def __setitem__(self, k, v):
        self.k = v


# Each model consists of two parts: model and config
class MetaType(object):
    """
    Build model type
    Each model consists of two parts: ConfigParser and ModelBuilder
    """
    ConfigParser = ModelMeta.set_arch_config_parser
    ModelBuilder = ModelMeta.set_model_builder


class _ModelMetaRegister(object):
    def __init__(self):
        """
        Register different models，to facilitate further expansion in the future
        input schema:
            name: model name
            setter: which part of the model is defined
        output schema:
            return a model
        """
        self._register_map = defaultdict(ModelMeta)

    def get(self, name):
        return self._register_map.get(name)

    def __call__(self, name, setter):
        model_meta = self._register_map[name]

        def _executor(func):
            setter(model_meta, func)
            return func

        return _executor


model = _ModelMetaRegister()
get_model_meta = model.get


@model("pinrec", MetaType.ModelBuilder)
class DeepInterestNetwork(nn.Module):
    def __init__(self, model_conf, group_num):
        """
        Main algorithm，based on DIN
        input schema:
            model_conf: configuration file
            group_num: number of user groups
        output schema:
            return a score
        """
        super(DeepInterestNetwork, self).__init__()

        assert isinstance(model_conf, ModelConfig)
        self._plugin_index = 0
        self._group_num = group_num
        self._id_encoder = encoder.IDEncoder(
            model_conf.id_vocab,
            model_conf.id_dimension,
        )
        self._target_emb_plugin = nn.ModuleList(
            [plugin.Plugin(model_conf.id_dimension, torch.nn.Tanh) for i in range(self._group_num)]
        )
        self._target_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension], [torch.nn.Tanh]
        )
        self._seq_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension], [torch.nn.Tanh]
        )
        self._target_attention = attention.TargetAttention(
            key_dimension=model_conf.id_dimension,
            value_dimension=model_conf.id_dimension,
        )
        self._atten_aggregated_embed_plugin = nn.ModuleList(
            [plugin.Plugin(model_conf.id_dimension, torch.nn.Tanh) for i in range(self._group_num)]
        )
        self._classifier = common.StackedDense(
            model_conf.id_dimension * 2,
            model_conf.classifier + [1],
            ([torch.nn.Tanh] * len(model_conf.classifier)) + [None]
        )

    def __setitem__(self, k, v):
        self.k = v

    def set_plugin_index(self, plugin_index):
        self._plugin_index = plugin_index

    def forward(self, features, plugin=True):
        # Encode target item
        # B * D
        target_embed = self._id_encoder(features[consts["FIELD_TARGET_ID"]])
        if plugin:
            target_embed = self._target_emb_plugin[self._plugin_index](target_embed)
        target_embed = self._target_trans(target_embed)

        # Encode user historical behaviors
        with torch.no_grad():
            mask = torch.not_equal(features[consts["FIELD_CLK_SEQUENCE"]], 0).to(dtype=torch.float32)
        # B * L * D
        hist_embed = self._id_encoder(features[consts["FIELD_CLK_SEQUENCE"]])
        if plugin:
            hist_embed = self._target_emb_plugin[self._plugin_index](hist_embed)
        hist_embed = self._seq_trans(hist_embed)

        # Target attention
        atten_aggregated_embed = self._target_attention(
            target_key=target_embed,
            item_keys=hist_embed,
            item_values=hist_embed,
            mask=mask
        )
        if plugin:
            atten_aggregated_embed = self._atten_aggregated_embed_plugin[self._plugin_index](atten_aggregated_embed)
        classifier_input = torch.cat([target_embed, atten_aggregated_embed], dim=1)
        return self._classifier(classifier_input)


class ModelConfig(object):
    def __init__(self):
        """
        Main algorithm，based on DIN
        input schema:
            json_obj: json object read from configuration file
        output schema:
            return an instance of the ModelConfig class
        """
        self.id_dimension = 8
        self.id_vocab = 500
        self.classifier = [64, 32]
        self.add_plugin = False

    @staticmethod
    @model("pinrec", MetaType.ConfigParser)
    def parse(json_obj):
        conf = ModelConfig()
        conf.id_dimension = json_obj.get("id_dimension")
        conf.id_vocab = json_obj.get("id_vocab")
        conf.classifier = json_obj.get("classifier")
        conf.add_plugin = json_obj.get("add_plugin")

        return conf

