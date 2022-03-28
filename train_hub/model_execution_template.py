# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.task_backbone import task_dispatcher
from torch.utils.data import Dataset

def model_provider(args):
    '''
    Build the model func.
    :param args: user defined arguments dictionary
    :return:
        model: user defined model that implements the torch.nn.Module interface
    '''
    model = None

    return model


def get_batch(data_iterator, args):
    '''
    Batch data processing method for training job.
    :param data_iterator: data iterator that implements the torch.utils.data.DataLoader interface
    :param args: user defined arguments dictionary
    :return:
        dictionary (python dict()): a dictionary that contains all data used in the model forward step
    '''
    ret_data = []

    return ret_data

def get_inference_batch(data_iterator, args):
    '''
    Batch data processing method for inference job.
    :param data_iterator: data iterator that implements the torch.utils.data.DataLoader interface
    :param args: user defined arguments dictionary
    :return:
        dictionary (python dict()): a dictionary that contains all data used in the model forward step
    '''
    ret_data = []

    return ret_data

def forward_func(data_iterator, model, args):
    '''
    Model forward step.
    :param data_iterator: data iterator that implements the torch.utils.data.DataLoader interface
    :param model: a model that implements the torch.nn.Module interface and defined in the model_provider func
    :param args: user defined arguments dictionary
    :return:
        if task_type = 'train':
            then return loss: a one-dimensional loss vector that contains every sample's loss
                        stats: other results which need print on terminal
        else(aka task_type = 'inference'):
            then return  infer_res： results list that output to files
    '''
    if args.task_type == 'train':
        ret_data = get_batch(data_iterator, args)


        loss, *stats = model(ret_data)

        return loss, stats
    else:
        ret_data = get_inference_batch(data_iterator, args)

        infer_res = model(ret_data)

        infer_res = list(infer_res)
        return infer_res


def train_eval_datasets_provider(args):
    '''
    Build train, valid, and test datasets for training job.
    :param args: user defined arguments dictionary
    :return:
        train_dataset, valid_dataset : dataset that implements the torch.utils.data.Dataset interface
    '''

    # Build the dataset.
    input_tables = args.tables.split(",")

    #this just a example, you can build a personal Dataset class that inherits the torch.utils.data.Dataset interface
    dataset =  Dataset(input_tables[0])

    if len(input_tables) > 1:
        eval_dataset = Dataset(input_tables[1])
        #run on local
        # eval_dataset = SeqToSeqDatasetLocal(args, input_tables[1], text_preprocessor, is_test=True)
    else:
        eval_dataset = None

    return dataset, eval_dataset

def personalized_args_provider(parser):
    '''
    User-defined parameters function
    :param parser: parser，the object of argparse.ArgumentParser
    :return: a python method where user defines parameters in it
    '''
    def add_model_config_args(parser):
        """Model arguments"""
        group = parser.add_argument_group('model', 'model configuration')

        #add some exclusive parameters that your model use
        group.add_argument('--mock', type=float, default=0.1, help='just a example')

        return parser

    return add_model_config_args(parser)

def inference_dataset_provider(args):
    '''
    Build train, valid, and test datasets for inference job.
    :param args: user defined arguments dictionary
    :return:
        train_dataset, valid_dataset : dataset that implements the torch.utils.data.Dataset interface
    '''
    input_tables = args.tables.split(",")
    # this just a example, you can build a personal Dataset class that inherits the torch.utils.data.Dataset interface
    dataset = Dataset(input_tables)

    return dataset

def training_post_processing_func(model, args):
    '''
    :param model: the trained model
    :param args:  user defined arguments dictionary
    :return:  None
    '''
    pass

if __name__ == "__main__":
    task_dispatcher(train_eval_dataset_provider=train_eval_datasets_provider,
                    inference_dataset_provider=inference_dataset_provider,
                    model_provider=model_provider,
                    forward_func=forward_func,
                    personalized_args_provider=personalized_args_provider,
                    training_post_processing_func=training_post_processing_func)