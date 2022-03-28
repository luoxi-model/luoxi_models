# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.dataset_hub.ecrec_datasets import ECRecDatasetLocal
from backend.model_hub.ecrec_model import ECRec
import torch
from backend.task_backbone import task_dispatcher
from onnx_test.ecrec_onnx_test import mock_cloud_data

def model_provider(args):
    """
    Build the model func.
    input schema:
         args: user defined arguments dictionary
    output schema:
         model: user defined model that implements the torch.nn.Module interface
    """

    model = ECRec(args=args, device = torch.cuda.current_device())

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

    return data

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
        loss, score_avg = model(data)

        return loss, [loss]
    else:
        data = get_batch(data_iterator, args)
        loss, score_avg = model(data)
        return (score_avg, data['label'])

def train_eval_datasets_provider(args):
    """
    Build train, valid, and test datasets.
    input schema:
        tokenizer: input sentence samples tokenizer
        args: user defined arguments dictionary
    output schema:
        train_dataset, valid_dataset, test_dataset: dataset that implements the torch.utils.data.Dataset interface
    """

    input_tables = args.tables.split(",")
    print('eval table: ', input_tables[1])
    shuffle_buffer_size = 1

    dataset = ECRecDatasetLocal(args, input_tables[0], shuffle_buffer_size)

    if len(input_tables) > 1:
        eval_dataset = ECRecDatasetLocal(args, input_tables[1], shuffle_buffer_size, is_test=True)
    else:
        eval_dataset = None

    return dataset, eval_dataset

def personalized_args_provider(parser):
    def add_model_config_args(parser):
        """Model arguments"""

        group = parser.add_argument_group('model', 'model configuration')

        # group.add_argument("--model", type=str, default='model', help="model")

        group.add_argument("--num_item", type=int, default=300437, help="number of item vocabulary. ml: 42876")

        group.add_argument("--num_cat", type=int, default=1921, help="number of category vocabulary. ml: 1505")

        group.add_argument("--num_user", type=int, default=47227, help="number of user vocabulary. ml: 9716")

        group.add_argument("--num_head", type=int, default=4, help="number of heads")

        group.add_argument("--d_model", type=int, default=16, help="model dimension")

        group.add_argument("--d_memory", type=int, default=16, help="memory dimension")

        group.add_argument("--length", type=int, default=100, help="length of sequence")
        group.add_argument("--seq_length", type=int, default=100, help="length of sequence")
        # group.add_argument("--dropout", type=float, default=0.1, help="dropout ratio")
        group.add_argument("--drop_rate", type=float, default=0.3, help="drop rate")
        # group.add_argument("--temp", type=float, default=1.0, help="temperature")
        group.add_argument("--K", type=int, default=4, help="K")
        group.add_argument("--l", type = float, default = 10.0, help="lambda value")
        group.add_argument("--gate", type=float, default=0.5, help="lambda value")

        group.add_argument("--column_length", type=int, default=9, help="length of column")
        group.add_argument("--sequence_length", type=int, default=100, help="length of sequence")

        group.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
        group.add_argument("--with_bn", type=bool, default=False, help="whether use batch norm")
        group.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
        group.add_argument("--patience", type=int, default=3, help="patience")
        group.add_argument("--update_after_train", type=bool, default=True, help="update memory immediately after training")

        group.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
        group.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

        group.add_argument("--test_dataset", type=str, default=None, help="test set for evaluate train set")

        group.add_argument("--infer_table", type=str, default='', help="inference data")


        return parser

    return add_model_config_args(parser)

def inference_dataset_provider(args):

    input_table = args.infer_table

    shuffle_buffer_size = 1
    dataset = ECRecDatasetLocal(args, input_table, shuffle_buffer_size, is_test=True)

    return dataset

def training_post_processing_func(model, args):
    '''
    :param model: the trained model
    :param args:  user defined arguments dictionary
    :return:  None
    '''
    from backend.utils import save_checkpoint, print_rank_0
    from backend.task_backbone import make_inference_data_loader
    infer_dataset, _ = train_eval_datasets_provider(args)
    def update_dataset_provider(args):
        return infer_dataset
    infer_data = make_inference_data_loader(args, update_dataset_provider)
    infer_data_iterator = iter(infer_data)
    model.train()
    model.task_type = 'inference'
    iter_index = 0
    num_workers = torch.distributed.get_world_size()
    infer_iters = args.train_iters // args.num_epochs + num_workers

    with torch.no_grad():
        while iter_index < infer_iters:
            forward_func(infer_data_iterator, model, args)
            iter_index += 1
        print_rank_0('memory is updated!')
    model.task_type = 'train'
    save_checkpoint(args.iteration + 1, model, None, args)

def onnx_model_export(args):
    '''
    :param args:  user defined arguments dictionary
    :return:  None
    '''
    print("*****running task : export ONNX model*****")
    import torch.onnx as onnx
    from backend.utils import load_model_state_only

    def mock_data_provider():
        '''
        :param model: the trained model
        :param args:  user defined arguments dictionary
        :return:  the mock data for testing the ONNX model
        '''
        data = mock_cloud_data()
        return data

    data = mock_data_provider()
    if args.cuda:
        cuda_device = torch.cuda.current_device()
    data = dict((k, v.to(cuda_device)) for k, v in data.items())
    args.onnx_step = True
    args.task_type = 'inference'
    model = model_provider(args)
    print(' >export onnx model number of parameters on rank{}'.format(sum([p.nelement() for p in model.parameters()])), flush=True)
    if args.cuda:
        model.cuda(torch.cuda.current_device())

    load_model_state_only(model, args, remove_prefix=None, remap_prefix=None)
    model.eval()
    model_path = os.path.join(args.onnx_export_path, args.onnx_model_name)

    onnx.export(model, data, model_path, export_params=True, verbose=False, opset_version=12)
    print('success save to:', model_path)

if __name__ == "__main__":
    task_dispatcher(train_eval_dataset_provider=train_eval_datasets_provider,
                    inference_dataset_provider=inference_dataset_provider,
                    model_provider=model_provider,
                    forward_func=forward_func,
                    personalized_args_provider=personalized_args_provider,
                    training_post_processing_func=training_post_processing_func,
                    onnx_model_export_func=onnx_model_export)