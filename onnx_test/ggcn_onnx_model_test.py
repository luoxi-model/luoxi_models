# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
from backend.dataset_hub.ggcn_datasets import GGCNDatasetLocal

def real_cigar_data():
    data_type = {'userid': "int",
                 'cms_segid': "int",
                 'cms_group_id': "int",
                 'final_gender_code': "int",
                 'age_level': "int",
                 'pvalue_level': "int",
                 'shopping_level': "int",
                 'occupation': "int",
                 'new_user_class_level': "int",
                 'adgroup_id': "int",
                 'seq_length': "int",
                 'item_embedding': "float",
                 'mean_embedding': "float",
                 'gnn_output': "float",
                 'group_score': "float",
                 'label': "float",
                 }

    data = {'userid': [257942, 352928, 296768, 325475, 544880],
            'cms_segid': [8, 79, 1, 1, 1],
            'cms_group_id': [3, 11, 5, 5, 5],
            'final_gender_code': [2, 1, 2, 2, 2],
            'age_level': [3, 5, 5, 5, 5],
            'pvalue_level': [3, 2, 1, 1, 3],
            'shopping_level': [3, 3, 1, 2, 3],
            'occupation': [1, 1, 1, 1, 1],
            'new_user_class_level': [3, 5, 1, 1, 1],
            'adgroup_id': [777487, 9781, 664671, 743740, 844239],
            'seq_length': [1., 0., 28., 2., 3.],
            'item_embedding': [
                [-0.1642, 0.0191, -0.0332, 0.0666, 0.0226, 0.0220, 0.1183, -0.1284, 0.1393, 0.2170, -0.1647,
                 -0.0316, -0.2213, -0.1487, 0.0263, 0.2481, 0.0220, 0.0649, 0.2545, 0.2207, 0.1615, -0.0936,
                 -0.0373, 0.2057, -0.0980, -0.0574, 0.3132, 0.0471, -0.1437, 0.1355, -0.0809, -0.0405, -0.0300,
                 -0.0126, 0.0446, -0.0285, 0.0164, 0.0605, -0.0186, -0.0894],
                [-0.6926, 0.0021, -0.2531, 0.1085, 0.0345, 0.2508, 0.2773, -0.3272, 0.0986, 0.2033, 0.0233, 0.0342,
                 -0.1619, -0.0700, -0.0821, 0.2925, -0.2639, 0.2183, -0.1321, -0.3341, 0.3004, 0.4415, -0.2169,
                 0.4756, -0.1051, 0.0158, 0.1106, 0.0990, -0.0461, -0.0206, -0.1337, 0.0419, 0.0617, 0.1032, 0.0272,
                 -0.0244, 0.1607, 0.0892, -0.0447, -0.0411],
                [-0.0928, 0.0016, -0.1122, 0.1218, -0.0621, -0.0516, 0.0155, -0.0864, 0.1379, 0.1051, -0.0403,
                 -0.0604, -0.1258, -0.2051, 0.0125, 0.0997, 0.0265, 0.0449, 0.0467, -0.0373, -0.0020, 0.0011,
                 0.1486, -0.0105, -0.2552, 0.0563, -0.0886, -0.1216, 0.0752, 0.0040, -0.0218, -0.0609, -0.0300,
                 -0.0126, 0.0446, -0.0285, 0.0164, 0.0605, -0.0186, -0.0894],
                [-0.1795, 0.0693, 0.0743, -0.0031, -0.0856, 0.0287, 0.1785, -0.0536, 0.1304, 0.1885, -0.0549,
                 -0.0532, -0.1828, -0.0833, 0.0737, 0.3236, -0.1278, 0.0281, 0.0973, -0.1154, 0.1054, 0.0590,
                 0.1405, 0.0663, -0.1268, 0.0774, -0.0258, -0.0322, -0.0032, -0.1190, -0.1028, 0.0612, -0.0199,
                 0.0594, -0.0132, -0.0351, 0.1327, 0.1379, -0.2662, -0.0991],
                [-0.0861, 0.0548, 0.0904, 0.0602, -0.0635, 0.0743, 0.1105, 0.0074, 0.0930, 0.1092, 0.0534, 0.0575,
                 -0.2001, -0.1564, -0.0505, 0.0641, -0.1482, 0.1948, -0.2620, -0.3430, 0.2245, 0.2375, 0.1858,
                 0.2586, -0.1011, -0.0609, 0.0232, 0.0289, 0.1379, -0.0487, -0.0158, 0.0918, 0.1813, 0.0669,
                 -0.0605, -0.1415, -0.0501, 0.0040, 0.0030, -0.0952]],
            'mean_embedding': [
                [4.4465e-02, 1.0445e-01, -2.7625e-01, -2.0569e-01, -6.8474e-02, -5.2394e-02, 1.0456e-01, 1.7022e-01,
                 1.3791e-01, 1.0512e-01, -4.0317e-02, -6.0382e-02, -1.2581e-01, -2.0511e-01, 1.2495e-02, 9.9727e-02,
                 -1.2358e-01, 2.7788e-02, -8.0873e-02, -1.1753e-01, 1.0135e-01, 5.6316e-03, -2.0245e-01,
                 -1.0968e-01, 7.0365e-02, -3.8328e-02, -1.5932e-01, -1.1318e-01, -1.4678e-01, -8.6553e-02,
                 6.5388e-02, -1.0718e-01, -1.8357e-02, -1.0338e-01, 8.6135e-02, 3.4204e-02, 7.0397e-02, -2.0644e-02,
                 1.0845e-01, 1.4099e-02],
                [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                [4.0462e-02, -8.6966e-02, -1.2895e-01, 6.5544e-02, -3.4426e-02, 1.1975e-01, -1.2570e-01, 2.4530e-02,
                 1.4350e-01, 1.1492e-01, -4.2107e-02, -4.2388e-02, -1.4527e-01, -1.9226e-01, 4.5437e-03, 1.1377e-01,
                 -1.1355e-02, -1.8809e-02, 4.4197e-02, 8.3866e-02, -6.0360e-02, -3.1770e-02, -2.8437e-02,
                 1.8450e-01, -1.8495e-01, -8.3106e-02, -3.4705e-04, 7.0822e-02, -9.0474e-02, 8.2293e-02,
                 -6.0735e-02, 1.1667e-01, 2.2143e-02, -1.3329e-02, -2.7356e-02, -7.3730e-02, 8.0943e-02, 9.9228e-02,
                 -7.0119e-02, -1.0540e-01],
                [2.7340e-02, 1.1924e-01, -5.9607e-02, 1.5895e-01, 2.1646e-02, -7.5258e-02, -7.0717e-02, 7.2195e-03,
                 1.5034e-01, 1.6499e-01, -4.7621e-02, -1.4737e-02, -1.7842e-01, -1.0894e-01, 1.5183e-02, 2.2416e-01,
                 -9.1066e-02, 1.4574e-01, 6.9024e-02, -5.4564e-02, 1.3869e-01, 7.7455e-02, -1.3641e-01, 1.2936e-01,
                 -1.0755e-01, 4.2002e-02, 6.5072e-02, 1.4974e-02, -7.0939e-03, -8.7588e-02, -5.9885e-02, 4.7056e-02,
                 1.7133e-02, -2.1070e-02, -2.4853e-02, -4.2562e-02, 6.3640e-02, 1.0982e-01, -6.8471e-02,
                 -8.8208e-02],
                [-1.2780e-01, 5.0805e-02, -1.1612e-01, 7.9578e-02, 1.0335e-01, 1.4314e-01, 3.3847e-03, -1.7688e-01,
                 1.3929e-01, 1.6951e-01, -6.5954e-02, -4.5142e-02, -1.4295e-01, -1.6744e-01, -2.4190e-02,
                 1.8433e-01, -3.6416e-02, 2.4685e-01, -3.6875e-02, -5.3090e-02, 8.7769e-02, 1.5820e-01, -1.7054e-01,
                 3.4149e-01, -3.6339e-01, -5.0575e-02, 4.6724e-02, 1.6422e-01, -1.8536e-01, -3.9412e-02,
                 -2.6019e-01, 1.9385e-01, -2.7335e-02, -1.8533e-02, -1.7009e-02, -3.2307e-02, 2.7097e-03,
                 3.9065e-02, -8.3710e-03, -5.8574e-02]],
            'gnn_output': [
                [-0.0719, -0.1300, -0.0700, 0.1248, 0.0943, -0.1491, -0.1248, -0.0546, 0.0144, 0.1653, -0.1943,
                 -0.0897, 0.1093, -0.2755, 0.2900, 0.0476, 0.3434, -0.1982, -0.2286, 0.1124, -0.3105, -0.3145,
                 0.0516, -0.0458, 0.0021, 0.0970, 0.1089, -0.0205, 0.0430, 0.0854, 0.2716, -0.0453, 0.2313, -0.0869,
                 0.3106, 0.0992, -0.0270, 0.0260, 0.0730, -0.2721],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0357, -0.2169, -0.2358, -0.1980, -0.0763, -0.0323, 0.0960, -0.3070, 0.0956, 0.0345, -0.0965,
                 -0.0754, -0.0028, -0.2407, 0.1383, 0.1988, 0.1502, 0.0690, -0.2667, -0.0154, -0.2668, -0.2654,
                 -0.1630, 0.1116, 0.0551, -0.0283, -0.1580, -0.0829, 0.0186, 0.1518, 0.3620, 0.1359, 0.3065,
                 -0.0371, 0.3105, -0.0218, -0.0081, -0.1666, 0.2048, -0.0792],
                [0.0136, 0.0489, -0.2396, 0.0582, 0.0072, 0.0309, 0.0462, -0.1681, -0.0556, 0.2680, 0.0539, -0.0192,
                 -0.0064, -0.3069, 0.2251, -0.0020, 0.3111, 0.1067, -0.1319, 0.1522, -0.2446, -0.3086, -0.0063,
                 -0.0076, 0.0004, -0.0174, -0.0301, -0.0181, 0.0073, 0.0129, 0.2869, 0.1295, 0.1268, 0.0299, 0.3440,
                 -0.0242, -0.1371, 0.0263, -0.0792, -0.2131],
                [0.0159, 0.0400, -0.2300, 0.1695, 0.1073, 0.0497, 0.0163, -0.1813, -0.0745, 0.3303, 0.0953, -0.0304,
                 -0.0085, -0.2792, 0.2309, 0.0410, 0.3900, 0.1206, -0.1116, 0.1467, -0.2310, -0.2889, -0.1278,
                 -0.0606, -0.0874, 0.0693, 0.0017, 0.0307, 0.0867, -0.0641, 0.3025, 0.0540, 0.2110, 0.0215, 0.3501,
                 0.1108, -0.1022, 0.0526, -0.0510, -0.1610]],
            'group_score': [0.0406, 0.0049, 0.0396, 0.0399, 0.0755],
            'label': [0., 0., 0., 0., 1.]}
    re_data = {}
    for k, v in data.items():
        if data_type.get(k) == "int":
            re_data[k] = torch.LongTensor(v)
        else:
            re_data[k] = torch.FloatTensor(v)
    return data_type, re_data

def mock_data(num_samples=1,data_names=None,data_types=None,data_sizes=None,data_nums=None):
    '''
    num_samples: batch_size
    data_names:
    data_types: int or float
    data_sizes: the size of one sample
    data_nums: dict, the max num of k
    '''
    mock_data={}
    data_names = data_names.split(",")
    data_types = data_types.split(",")
    data_sizes = data_sizes.split(",")
    assert len(data_names)==len(data_types)==len(data_sizes)
    for name,type,size in zip(data_names,data_types,data_sizes):
        if type=="int":
            high_num = data_nums.get(name,2) if data_nums else 2
            v=torch.randint(low=1,high=high_num,size=[num_samples],dtype=torch.int64)
        else:
            if name=="label":
                v=torch.ones([num_samples],dtype=torch.float32,requires_grad=True)
            else:
                v = torch.randn([num_samples,int(size)],dtype=torch.float32, requires_grad=True)
        mock_data[name]=v
    return mock_data

def mock_gnn_dataTable(args):
    data_names = "userid,cms_segid,cms_group_id,final_gender_code,age_level,pvalue_level,shopping_level,occupation,new_user_class_level,adgroup_id,seq_length,item_embedding,mean_embedding,gnn_output,group_score,label"
    data_types = "int,int,int,int,int,int,int,int,int,int,int,float,float,float,float,float"
    data_sizes = "1,1,1,1,1,1,1,1,1,1,1,40,40,40,1,1"
    data_nums = {"user_id": 1150000,
                 "cms_segid": 100,
                 "cms_group_id": 15,
                 "final_gender_code": 5,
                 "age_level": 10,
                 "pvalue_level": 5,
                 "shopping_level": 5,
                 "occupation": 5,
                 "new_user_class_level": 10,
                 "adgroup_id": 850000}
    data = mock_data(2, data_names, data_types, data_sizes, data_nums)
    return data

def mock_no_gnn_data(args):
    data_names = "userid,cms_segid,cms_group_id,final_gender_code,age_level,pvalue_level,shopping_level,occupation,new_user_class_level,adgroup_id,seq_length,item_embedding,mean_embedding,group_score,label"
    data_types = "int,int,int,int,int,int,int,int,int,int,int,float,float,float,float"
    data_sizes = "1,1,1,1,1,1,1,1,1,1,1,40,40,1,1"
    data_nums = {"user_id": 1150000,
                 "cms_segid": 100,
                 "cms_group_id": 15,
                 "final_gender_code": 5,
                 "age_level": 10,
                 "pvalue_level": 5,
                 "shopping_level": 5,
                 "occupation": 5,
                 "new_user_class_level": 10,
                 "adgroup_id": 850000}
    data = mock_data(2, data_names, data_types, data_sizes, data_nums)
    return data

def mock_gnn_data(args):
    input_tables = args.tables.split(",")
    dataset = GGCNDatasetLocal(args, input_tables[1], traintype='eval')
    data = next(iter(dataset))


    return data

def onnx_test(mock_data_func, label_col_name = 'label', onnx_export_path=None, onnx_model_name=None):
    '''
    func: get the result of onnx model
    **********
    Note that the order of the keys in the returned dict from mock_data_func must be the same as the order of the parameters in the forward method of your model
    **********
    mock_data_func: the func is used in onnx export
    onnx_export_path: onnx model path
    onnx_model_name: onnx model name
    '''

    def to_numpy(tensor, data_type=None):
        tensor = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        if data_type:
            if data_type == "int":
                data_type = np.int64
            else:
                data_type = np.float32
            tensor = tensor.astype(data_type)
        return tensor

    data = mock_data_func()
    assert isinstance(data, OrderedDict), "returned data from mock_data_func must be an OrderDict!"

    data = dict((k, to_numpy(v)) for k, v in data.items())

    y = data[label_col_name].astype(np.float32)
    input_data = []
    for i,(k, v) in enumerate(data.items()):
        input_data.append(v)
    input_data.append(y)
    onnx_model_name = onnx_model_name if onnx_model_name else "model_00.onnx"
    model_path = os.path.join(onnx_export_path, onnx_model_name)
    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs={}
    for s_input in ort_session.get_inputs():
        k = str(s_input.name).split(".")[-1]
        in_da = input_data[int(k)]
        ort_inputs[s_input.name] = in_da
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)
if __name__ == '__main__':
    onnx_test(mock_gnn_data)

