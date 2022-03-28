# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
import os
import onnxruntime
import sys
sys.path.append('/mnt2/songxiao/device-cloud-modelhub/')
from backend.utils import to_numpy

def mock_data(num_samples=2,data_names=None,data_sizes=None,data_nums=None):
    '''
    num_samples: batch_size
    data_names:
    data_types: int or float
    data_sizes: the size of one sample
    data_nums: dict, the max num of k
    '''
    mock_data={}
    data_names = data_names.split(",")
    data_sizes = data_sizes.split(",")
    assert len(data_names)==len(data_sizes)
    for name,size in zip(data_names,data_sizes):
        if name=="label":
            v = torch.randint(low=0, high=2, size=[num_samples], dtype=torch.int64)
        else:
            high_num = data_nums.get(name, 2) if data_nums else 2
            if int(size) == 1:
                v = torch.randint(low=1, high=high_num, size=[num_samples], dtype=torch.int64)
            else:
                v = torch.randint(low=1, high=high_num, size=[num_samples, int(size)], dtype=torch.int64)
        mock_data[name]=v
    return mock_data

def mock_edge_data():
    data_names = "user_id,item_id,cate_id,edge_item_seq,edge_cate_seq,seq_len,label"
    data_sizes = "1,1,1,100,100,1,1"
    data_nums = {"user_id": 1000,
                 "item_id": 1000,
                 "cate_id": 100,
                 "edge_item_seq": 1000,
                 "edge_cate_seq": 100,
                 "seq_len": 100,
                 "label": 2
                 }
    data = mock_data(2, data_names, data_sizes, data_nums)
    return data

def mock_cloud_data():
    data_names = "user_id,item_id,cate_id,edge_item_seq,edge_cate_seq,seq_len,label,item_seq,cate_seq"
    data_sizes = "1,1,1,100,100,1,1,100,100"
    data_nums = {"user_id": 1000,
                 "item_id": 1000,
                 "cate_id": 100,
                 "edge_item_seq": 1000,
                 "edge_cate_seq": 100,
                 "seq_len": 100,
                 "label": 2,
                 "item_seq": 1000,
                 "cate_seq": 100
                 }
    data = mock_data(2, data_names, data_sizes, data_nums)
    return data

def onnx_test(mock_data_func,onnx_export_path=None, onnx_model_name=None):
    '''
    func: get the result of onnx model
    mock_data_func: the func is used in onnx export
    onnx_export_path: onnx model path
    onnx_model_name: onnx model name
    '''
    data = mock_data_func()
    data = dict((k, to_numpy(v)) for k, v in data.items())
    input_data = []
    for i,(k, v) in enumerate(data.items()):
        input_data.append(v)
    onnx_model_name = onnx_model_name if onnx_model_name else "model_00.onnx"
    model_path = os.path.join(onnx_export_path, onnx_model_name)
    # model_path = "./output/model_00.onnx"
    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs={}
    for s_input in ort_session.get_inputs():
        if str(s_input.name).split(".")[0] == 'input':
            k = str(s_input.name).split(".")[-1]
            in_da = input_data[int(k) - 1]
            ort_inputs[s_input.name] = in_da
    ort_inputs['5'] = data['seq_len']
    ort_inputs['6'] = data['label']

    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)

if __name__ == '__main__':
    onnx_test(mock_edge_data, '/mnt2/songxiao/ecrec/test', 'onnx_model.onnx')