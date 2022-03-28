# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.dataset_hub.ggcn_datasets import GGCNDatasetLocal
from backend.model_hub.ggcn_model import *
import torch
from backend.task_backbone import task_dispatcher
import numpy as np
from sklearn.metrics import f1_score
from onnx_test.ggcn_onnx_model_test import mock_gnn_data


def model_provider(args):
    """
    Build the model func.
    input schema:
         args: user defined arguments dictionary
    output schema:
         model: user defined model that implements the torch.nn.Module interface
    """
    model =  GGCN(nfeat=args.nfeat,
                    nlayers=args.layer,
                    nhidden=args.hidden,
                    nclass=args.nclass,
                    dropout=args.dropout,
                    lamda = args.lamda,
                    alpha=args.alpha,
                    variant=args.variant,args=args,onnx_step=args.onnx_step).to(torch.cuda.current_device())

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

 
    device = torch.cuda.current_device()

    feat, adj, labels, nodes,traintype = next(data_iterator)

    nodes = nodes.to(device)
    labels = labels[0].to(device)
    feat = feat[0].to(device)#.to_dense()
    adj = adj[0].to(device)
    return feat, adj, labels, nodes, traintype[0]

def get_inference_batch(data_iterator, args):
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
    if args.task_type == "inference":
        try:
            feat, adj, labels, nodes, traintype = get_batch(data_iterator, args)
        except:
            return [torch.tensor([0,0])]
    else:
        feat, adj, labels, nodes,traintype = get_batch(data_iterator, args)

    if traintype == 'train':

        output = model(feat, adj)
        lossfn = torch.nn.BCELoss()
        loss = lossfn(output[:nodes], labels[:nodes])
        predict = np.where(output[:nodes].data.cpu().numpy() > 0.5, 1, 0)
        score = f1_score(labels[:nodes].data.cpu().numpy(), predict, average='micro')
        score = torch.tensor(score).to(torch.cuda.current_device())

        return loss, [score]
    elif traintype == 'eval':
        lossfn = torch.nn.BCELoss()
        loss,score = model.module.evalModel()
        return loss,score
    elif traintype == 'test':
        loss,score = model.module.inferModel()
        infer_res = [torch.tensor([loss,loss])]
        return infer_res
    else:
        return 0,0

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
    dataset = GGCNDatasetLocal(args, input_tables[0],traintype='train')
    print('train_eval_datasets_provider')

    if len(input_tables) > 1:
        #run on local
        eval_dataset = GGCNDatasetLocal(args, input_tables[1],traintype='eval')
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
        group.add_argument("--epochs", type=int, default=8000, help='Number of epochs to train.')
        group.add_argument("--wd", type=float, default=0, help='Weight decay (L2 loss on parameters).')
        group.add_argument("--layer", type=int, default=9, help='Number of hidden layers.')
        group.add_argument("--hidden", type=int, default=2048, help='Number of hidden layers.')
        group.add_argument("--nfeat", type=int, default=50, help='Number of feature .')
        group.add_argument("--nclass", type=int, default=121, help='Number of classes.')
        group.add_argument("--dropout", type=float, default=0.2, help='Dropout rate (1 - keep probability).')
        group.add_argument("--patience", type=int, default=2000, help='Patience')
        group.add_argument("--data", default='ppi', help='dateset')
        group.add_argument("--dev", type=int, default=0, help='device id')
        group.add_argument("--alpha", type=float, default=0.5, help='alpha_l')
        group.add_argument("--lamda", type=float, default=1, help='lamda.')
        group.add_argument("--variant", action='store_true', default=False, help='GCN* model.')
        group.add_argument("--test", action='store_true', default=False, help='evaluation on test set.')

        group.add_argument("--onnx_step", type=bool, default=False, help="if onnx_step")

        group.add_argument("--load_model_path", type=str, default='',help="load_model_path")
        group.add_argument("--is_gpu", type=bool,default=True,help="if gpu")
        group.add_argument("--final_saved_iteration", type=int, default=0, help="if gpu")
        return parser

    return add_model_config_args(parser)

def inference_dataset_provider(args):
    """
    Build train, valid, and test datasets for inference.
    input schema:
        tokenizer: input sentence samples tokenizer
        args: user defined arguments dictionary
    output schema:
        train_dataset, valid_dataset, test_dataset: dataset that implements the torch.utils.data.Dataset interface
    """
    input_tables = args.tables.split(",")

    dataset = GGCNDatasetLocal(args, input_tables[1],traintype='test')

    return dataset

def onnx_model_export(args):
    '''
    :param model: the trained model
    :param args:  user defined arguments dictionary
    :return:  None
    '''
    print("=====start onnx export======")
    import torch.onnx as onnx
    from backend.utils import load_model_state_only
    load_checkpoint_name =os.path.join(args.load_model_path,"rank_00_model_states.pt")

    data = mock_gnn_data(args)
    cuda_device = torch.cuda.current_device()
    data = dict((k, v.to(cuda_device)) for k, v in data.items())
    y = data["label"]
    args.onnx_step = True
    args.task_type = "inference"
    model = model_provider(args)
    print(' >export onnx model number of parameters on rank{}'.format(sum([p.nelement() for p in model.parameters()])), flush=True)
    if args.is_gpu:
        model.cuda(torch.cuda.current_device())

    load_model_state_only(model, args, remove_prefix=None, remap_prefix=None, load_checkpoint_name=load_checkpoint_name)
    model.eval()
    model_path = os.path.join(args.onnx_export_path, args.onnx_model_name)

    onnx.export(model, (data, y), model_path, export_params=True, verbose=False, opset_version=12)
    print("success save to:", model_path)

if __name__ == "__main__":
    task_dispatcher(train_eval_dataset_provider=train_eval_datasets_provider,
                    inference_dataset_provider=inference_dataset_provider,
                    model_provider=model_provider,
                    forward_func=forward_func,
                    personalized_args_provider=personalized_args_provider,
                    onnx_model_export_func=onnx_model_export)