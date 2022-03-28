# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.dataset_hub.mcrec_dataset import CRecDataset, ORecDataset, MCRecDataset
from backend.model_hub.mcrec_model import CRec, ORec, Controller
import torch
from backend.task_backbone import task_dispatcher
from onnx_test.mcrec_onnx_model_test import mock_orec_data

def model_provider(args):
    '''
    Build the model func.
    :param args: user defined arguments dictionary
    :return:
        model: user defined model that implements the torch.nn.Module interface
    '''

    if args.model_type == 'crec':
    	model = CRec(args)
    elif args.model_type == 'orec':
    	args.dim = int(args.dim / 4) # make the on-device model smaller
    	model = ORec(args)
    elif args.model_type == 'mcrec':
    	model = Controller(args)
    else:
    	model = None

    return model

def get_batch(data_iterator, args):

    data = next(data_iterator)

    cuda_device = torch.cuda.current_device()
    data = dict((k, v.to(cuda_device)) for k, v in data.items())
    
    if args.model_type == 'crec':
    	hist_seq = data['hist_seq'].long()
    	cand = data['cand'].long()
    	label = data['label'].long()
    	
    	return (hist_seq, cand, label)

    elif args.model_type == 'orec':
    	hist_seq = data['hist_seq'].long()
    	cand = data['cand'].long()
    	prior_score = data['prior_score'].long()
    	label = data['label'].long()
    	
    	return (hist_seq, cand, prior_score, label)

    elif args.model_type == 'mcrec':
    	hist_seq = data['hist_seq'].long()
    	label = data['label'].long()
    	
    	return (hist_seq, label)

    else:
    	return None

def get_inference_batch(data_iterator, args):

    data = next(data_iterator)

    cuda_device = torch.cuda.current_device()
    data = dict((k, v.to(cuda_device)) for k, v in data.items())
    
    if args.model_type == 'crec':
    	hist_seq = data['hist_seq'].long()
    	cand = data['cand'].long()
    	label = data['label'].long()
    	
    	return (hist_seq, cand, label)

    elif args.model_type == 'orec':
    	hist_seq = data['hist_seq'].long()
    	cand = data['cand'].long()
    	prior_score = data['prior_score'].long()
    	label = data['label'].long()
    	
    	return (hist_seq, cand, prior_score, label)

    elif args.model_type == 'mcrec':
    	hist_seq = data['hist_seq'].long()
    	label = data['label'].long()
    	
    	return (hist_seq, label)

    else:
    	return None

def forward_func(data_iterator, model, args):

    if args.task_type == 'train':
        if args.model_type == 'crec':
        	hist_seq, cand, label = get_batch(data_iterator, args)

        	loss = model(hist_seq, cand, label)
        elif args.model_type == 'orec':
        	hist_seq, cand, prior_score, label = get_batch(data_iterator, args)

        	loss = model(hist_seq, cand, prior_score, label)
        elif args.model_type == 'mcrec':
        	hist_seq, label = get_batch(data_iterator, args)

        	loss = model(hist_seq, label)
        else:
        	loss = None

        return loss, []
    else:
        if args.model_type == 'crec':
        	hist_seq, cand, label = get_inference_batch(data_iterator, args)

        	pred = model(hist_seq, cand, label)
        elif args.model_type == 'orec':
        	hist_seq, cand, prior_score, label = get_inference_batch(data_iterator, args)

        	pred = model(hist_seq, cand, prior_score, label)
        elif args.model_type == 'mcrec':
        	hist_seq, label = get_inference_batch(data_iterator, args)

        	pred = model(hist_seq, label)
        else:
        	pred, label = None, None

        return pred, label

def train_eval_datasets_provider(args):
    '''
    Build train, valid, and test datasets for training job.
    :param args: user defined arguments dictionary
    :return:
        train_dataset, valid_dataset : dataset that implements the torch.utils.data.Dataset interface
    '''

    # Build the dataset.
    input_files = args.tables.split(",")

    if args.model_type == 'crec':
        dataset, eval_dataset = CRecDataset, CRecDataset
    elif args.model_type == 'orec':
        dataset, eval_dataset = ORecDataset, ORecDataset
    elif args.model_type == 'mcrec':
        dataset, eval_dataset = MCRecDataset, MCRecDataset
    else:
        dataset, eval_dataset = None, None
 
    dataset = dataset(args, input_files[0])

    if len(input_files) > 1:
        eval_dataset = eval_dataset(args, input_files[1], is_test=True)

    else:
        eval_dataset = None

    return dataset, eval_dataset

def inference_dataset_provider(args):
    '''
    Build train, valid, and test datasets for inference job.
    :param args: user defined arguments dictionary
    :return:
        train_dataset, valid_dataset : dataset that implements the torch.utils.data.Dataset interface
    '''
    input_files = args.tables.split(",")
    if args.model_type == 'crec':
        dataset, eval_dataset = CRecDataset, CRecDataset
    elif args.model_type == 'orec':
        dataset, eval_dataset = ORecDataset, ORecDataset
    elif args.model_type == 'mcrec':
        dataset, eval_dataset = MCRecDataset, MCRecDataset
    else:
        dataset, eval_dataset = None, None

    dataset = eval_dataset(args, input_files[1], is_test=True)

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
    def mock_data_provider(args):
        if args.model_type == 'orec':
            data = mock_orec_data()
        else:
            data = None
        return data

    data = mock_data_provider(args)
    cuda_device = torch.cuda.current_device()
    data = dict((k, v.to(cuda_device)) for k, v in data.items())
    model = model_provider(args)
    print(' >export onnx model number of parameters on rank{}'.format(sum([p.nelement() for p in model.parameters()])), flush=True)
    model.cuda(torch.cuda.current_device())

    load_model_state_only(model, args, remove_prefix=None, remap_prefix=None)
    model.eval()
    model_path = args.onnx_export_path

    onnx.export(model, (data["hist_seq"], data["cand"], data["prior_score"], data["label"]), model_path, export_params=True, verbose=False, opset_version=12)
    print("success save to:", model_path)

def personalized_args_provider(parser):
    '''
    User-defined parameters function
    :param parser: parser，the object of argparse.ArgumentParser
    :return: a python method where user defines parameters in it
    '''
    def add_model_config_args(parser):
        """Model arguments"""
        group = parser.add_argument_group('model', 'model configuration')

        group.add_argument('--dim', type=int, default=64, help='embedding dim')
        group.add_argument('--num', type=int, default=50000,
                           help='vocab size for items')
        group.add_argument('--cpu-optimizer', action='store_true',
                           help='Run optimizer on CPU')
        group.add_argument('--model_type', type=str, help='model type')

        return parser

    return add_model_config_args(parser)

if __name__ == "__main__":
    task_dispatcher(train_eval_dataset_provider=train_eval_datasets_provider,
                    inference_dataset_provider=inference_dataset_provider,
                    model_provider=model_provider,
                    forward_func=forward_func,
                    personalized_args_provider=personalized_args_provider,
                    onnx_model_export_func=onnx_model_export)
