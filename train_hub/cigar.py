# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.dataset_hub.cigar_datasets import CIGARDatasetLocal
from backend.model_hub.cigar_model import *
from backend.task_backbone import task_dispatcher
import warnings
from onnx_test.cigar_onnx_model_test import mock_gnn_data, mock_no_gnn_data

warnings.filterwarnings('ignore')

def model_provider(args):
    """
    Build the model func.
    input schema:
         args: user defined arguments dictionary
    output schema:
         model: user defined model that implements the torch.nn.Module interface
    """
    if args.model == 'CIGAR':
        model = CIGAR(args=args, device = torch.cuda.current_device())
    elif args.model == 'CIGAR_WO_CDGNN':
        model = CIGAR_WO_CDGNN(args=args, device = torch.cuda.current_device())
    elif args.model == 'CIGAR_WO_PN':
        model = CIGAR_WO_PN(args=args, device = torch.cuda.current_device())
    elif args.model == 'PNN':
        model = PNN(args=args, device = torch.cuda.current_device())

    return model

def get_batch(data_iterator, args):
    """
    Generate a batch func.
    input schema:
        data_iterator: data iterator that implements the torch.utils.data.DataLoader interface
        args: user defined arguments dictionary
    output schema：
        dictionary (python dict()): a dictionary that contains all data used in the model forward step
    """
    data = next(data_iterator)

    cuda_device = torch.cuda.current_device()
    data = dict((k, v.to(cuda_device)) for k, v in data.items())

    #####################################################################
    return data
    #####################################################################

def get_inference_batch(data_iterator, args):
    data = next(data_iterator)

    cuda_device = torch.cuda.current_device()
    data = dict((k, v.to(cuda_device)) for k, v in data.items())

    return data

def forward_func(data_iterator, model, args):
    """
    Forward step.
    input schema:
        data_iterator: data iterator that implements the torch.utils.data.DataLoader interface
        model: a model that implements the torch.nn.Module interface and defined in the model_provider func
        args: user defined arguments dictionary
    output schema:
        loss: a one-dimensional loss vector that contains every sample's loss
    """
    if args.task_type == 'train':
        data = get_batch(data_iterator, args)
        loss, *stats = model(data, data["label"])
        return loss, stats

    # elif args.task_type == 'eval':
    #     data = get_batch(data_iterator, args)
    #     loss, label, score = model(data, args)
    #     return loss, label, score

    else:
        data = get_batch(data_iterator, args)
        infer_res = model(data, data["label"])
        return infer_res
        # (unique_id, input_ids, position_ids, token_type_ids, attention_mask, eos_indices,
        #  gen_label_ids, gen_label_masks, cls_label_ids, cls_label_masks) = get_inference_batch(data_iterator, args)

        # infer_res = model(
        #     input_ids, position_ids, token_type_ids, attention_mask, eos_indices,
        #     gen_label_ids, gen_label_masks, cls_label_ids, cls_label_masks, unique_id=unique_id
        # )
        # infer_res = list(infer_res)

def train_eval_datasets_provider(args):
    """
    Build train, valid, and test datasets.
    input schema:
        tokenizer: input sentence samples tokenizer
        args: user defined arguments dictionary
    output schema:
        train_dataset, valid_dataset, test_dataset: dataset that implements the torch.utils.data.Dataset interface
    """
    # Build the dataset.
    input_tables = args.tables.split(",")

    #run on local
    dataset = CIGARDatasetLocal(args, input_tables[0])
    print('train_eval_datasets_provider')

    if len(input_tables) > 1:
        #run on local
        eval_dataset = CIGARDatasetLocal(args, input_tables[1], is_test=True)
    else:
        eval_dataset = None

    return dataset, eval_dataset

def personalized_args_provider(parser):
    def add_model_config_args(parser):
        """Model arguments"""

        group = parser.add_argument_group('model', 'model configuration')

        group.add_argument("--model", type=str, default='cigar', help="model")
        group.add_argument("--kv_dimension", type=int, default=8, help="dimension of each feature field")
        group.add_argument("--mem_dimension", type=int, default=40, help="dimension of memory")
        group.add_argument("--gnn_layers", type=str, default='40', help="dimension of GNN layer")
        group.add_argument("--dim_hidden", type=str, default='128,64,1', help="dimension of prediction layer")
        group.add_argument("--prototype_num", type=int, default=5, help="prototype_num")
        group.add_argument("--seq_length", type=int, default=100, help="length of sequence")
        group.add_argument("--column_len", type=int, default=29, help="length of column")
        group.add_argument("--user_fea_name", type=str,
                           default='cms_segid,cms_group_id,final_gender_code,age_level,pvalue_level,shopping_level,occupation,new_user_class_level',
                           help="user_fea_name")
        group.add_argument("--user_fea_col_id", type=str, default='1,2,3,4,5,6,7,8', help="user_fea_col_id")
        group.add_argument("--item_fea_name", type=str, default='adgroup_id,cate_id,campaign_id,customer,brand', help="item_fea_name")
        group.add_argument("--item_fea_col_id", type=str, default='20,21,22,23,24', help="item_fea_col_id")
        group.add_argument("--seq_col_id", type=str, default='13,14,15,16,17', help="seq_col_id")
        group.add_argument("--table_size", type=str, default='1150000,100,15,5,10,5,5,5,10,850000,13000,425000,260000,461500', help="embedding table size of uid, user fea and item fea")
        group.add_argument("--uid_graph_label_col_id", type=str, default='0,9,28', help="column ID of user, neighbors and label")

        group.add_argument("--onnx_step", type=bool, default=False, help="if onnx_step")

        return parser

    return add_model_config_args(parser)

def inference_dataset_provider(args):

    input_tables = args.tables.split(",")

    dataset = CIGARDatasetLocal(args, input_tables[1], is_test=True)

    return dataset

def onnx_model_export(args):
    '''
    :param model: the trained model
    :param args:  user defined arguments dictionary
    :return:  None
    '''
    def mock_data_provider(args):
        """
        Build the model func.
        input schema:
             args: user defined arguments dictionary
        output schema:
             model: user defined model that implements the torch.nn.Module interface
        """
        if args.model == 'CIGAR':
            data = mock_gnn_data()
        elif args.model == 'CIGAR_WO_CDGNN':
            data = mock_no_gnn_data()
        elif args.model == 'CIGAR_WO_PN':
            data = mock_gnn_data()
        elif args.model == 'PNN':
            data = mock_no_gnn_data()

        return data

    print("=====start onnx export======")
    import torch.onnx as onnx
    from backend.utils import load_model_state_only

    data = mock_data_provider(args)
    cuda_device = torch.cuda.current_device()
    data = dict((k, v.to(cuda_device)) for k, v in data.items())
    y = data["label"]
    args.onnx_step = True
    args.task_type = "inference"
    model = model_provider(args)
    model.cuda(torch.cuda.current_device())
    print(' >export onnx model number of parameters on rank{}'.format(sum([p.nelement() for p in model.parameters()])), flush=True)
    load_model_state_only(model, args, remove_prefix=None, remap_prefix=None,
                          load_checkpoint_name=os.path.join(args.load,"rank_00_model_states.pt"))
    model.eval()
    model_path = os.path.join(args.save,"onnx_model_00.onnx")

    onnx.export(model, (data, y), model_path, export_params=True, verbose=False, opset_version=12)
    print("success save to:", model_path)

if __name__ == "__main__":
    #running what task depend on args.task_type's value(train or inference or onnx_export)
    task_dispatcher(train_eval_dataset_provider=train_eval_datasets_provider,
                    inference_dataset_provider=inference_dataset_provider,
                    model_provider=model_provider,
                    forward_func=forward_func,
                    personalized_args_provider=personalized_args_provider,
                    onnx_model_export_func=onnx_model_export)