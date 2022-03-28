# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import numpy as np
import torch
import os
import onnxruntime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.utils import to_numpy
from backend.model_hub import pinrec_model

consts = pinrec_model.consts


def mock_data():
    batch_size = 2
    seq_lens = 50
    mock_data_dict = {
        consts["FIELD_USER_ID"]: torch.LongTensor(torch.randint(low=0, high=6040, size=[batch_size])),
        consts["FIELD_TARGET_ID"]: torch.LongTensor(torch.randint(low=0, high=3706, size=[batch_size])),
        consts["FIELD_CLK_SEQUENCE"]: torch.LongTensor(torch.randint(low=0, high=3706, size=[batch_size, seq_lens])),
        consts["FIELD_LABEL"]: torch.LongTensor(torch.randint(low=0, high=1, size=[batch_size])),
        consts["FIELD_GROUP_ID"]: torch.LongTensor(torch.randint(low=0, high=4, size=[batch_size]))
    }
    return mock_data_dict


def onnx_test(mock_data_func, onnx_export_path=None, onnx_model_name=None):
    '''
    func: get the result of onnx model
    mock_data_func: the func is used in onnx export
    onnx_export_path: onnx model path
    onnx_model_name: onnx model name
    '''
    data = mock_data_func()
    data = dict((k, to_numpy(v)) for k, v in data.items())
    input_data = []
    for i, (k, v) in enumerate(data.items()):
        input_data.append(np.array(v))
    onnx_model_name = onnx_model_name if onnx_model_name else "model_00.onnx"
    model_path = os.path.join(onnx_export_path, onnx_model_name)
    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs = {
        'input.1': data[consts["FIELD_TARGET_ID"]],
        'input.5': data[consts["FIELD_CLK_SEQUENCE"]]
    }
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)


if __name__ == '__main__':
    onnx_test(mock_data, '/mnt4/lzq/mobilem6/onnx_test', 'onnx_model.onnx')
