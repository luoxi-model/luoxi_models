# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.dataset_hub.pin_datasets import PINDatasetLocal
from backend.model_hub import pinrec_model
import torch
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel
from backend.task_backbone import task_dispatcher
from backend.utils import parse_arch_config_from_args
from onnx_test.pinrec_onnx_model_test import mock_data

trunk_layer_set = set()

def model_provider(args):
    """
    Build the model func.
    input schema:
         args: user defined arguments dictionary
    output schema:
         model: user defined model that implements the torch.nn.Module interface
    """
    model_plugin = pinrec_model.get_model_meta(args.model)
    model_plugin_conf, raw_model_plugin_conf = parse_arch_config_from_args(model_plugin, args)  # type: dict
    model = model_plugin.model_builder(model_conf=model_plugin_conf, group_num=args.group_num)
    for name, parms in model.named_parameters():
        if "plugin" not in name:
            trunk_layer_set.add(name)
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

    user_id = data[args.consts["FIELD_USER_ID"]].long()
    target_id = data[args.consts["FIELD_TARGET_ID"]].long()
    clk_seq = data[args.consts["FIELD_CLK_SEQUENCE"]].long()
    label = data[args.consts["FIELD_LABEL"]].long()
    group_id = data[args.consts["FIELD_GROUP_ID"]].long()

    data = {
        args.consts["FIELD_USER_ID"]: user_id,
        args.consts["FIELD_TARGET_ID"]: target_id,
        args.consts["FIELD_CLK_SEQUENCE"]: clk_seq,
        args.consts["FIELD_LABEL"]: label,
        args.consts["FIELD_GROUP_ID"]: group_id
    }
    return data

def get_inference_batch(data_iterator, args):
    data = next(data_iterator)

    user_id = data[args.consts["FIELD_USER_ID"]].long()
    target_id = data[args.consts["FIELD_TARGET_ID"]].long()
    clk_seq = data[args.consts["FIELD_CLK_SEQUENCE"]].long()
    label = data[args.consts["FIELD_LABEL"]].long()
    group_id = data[args.consts["FIELD_GROUP_ID"]].long()

    data = {
        args.consts["FIELD_USER_ID"]: user_id,
        args.consts["FIELD_TARGET_ID"]: target_id,
        args.consts["FIELD_CLK_SEQUENCE"]: clk_seq,
        args.consts["FIELD_LABEL"]: label,
        args.consts["FIELD_GROUP_ID"]: group_id
    }

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
    device = torch.cuda.current_device()
    criterion = nn.BCEWithLogitsLoss()
    if isinstance(model, DistributedDataParallel):
        model = model.module

    if args.task_type == 'train':
        # Calculate iters so that switch stage1 to stage2
        args.stage_switch_iters = int(args.stage_switch_epoch / args.num_epochs * args.train_iters)
        # stage1->trunk model   stage2->add plugin model
        plugin = True if (args.iteration > args.stage_switch_iters) else False
        # Reduce learning rate at stage2
        if args.iteration - args.stage_switch_iters == 1:
            for group in args.optimizer_runtime.param_groups:
                group['lr'] = group['lr'] / 10
        batch_data = get_batch(data_iterator, args)
        loss = 0.0
        stats = []
        device = torch.cuda.current_device()
        gradient_dict = {}
        for name, parms in model.named_parameters():
            gradient_dict[name] = torch.zeros((args.group_num, parms.view(-1).size()[0])).to(device)

        for group_index in range(args.group_num):
            group_index_tensor = torch.LongTensor([group_index]).repeat(batch_data[args.consts["FIELD_GROUP_ID"]].size())

            if len(batch_data[args.consts["FIELD_LABEL"]][torch.where(batch_data[args.consts["FIELD_GROUP_ID"]] == group_index_tensor)]) == 0:
                continue
            # set plugin module index according to group index
            model.set_plugin_index(group_index)

            logits = model({
                key: value[torch.where(batch_data[args.consts["FIELD_GROUP_ID"]] == group_index_tensor)].to(device)
                for key, value in batch_data.items()
                if key not in {args.consts["FIELD_USER_ID"], args.consts["FIELD_LABEL"], args.consts["FIELD_GROUP_ID"]}
            }, plugin=plugin)

            loss_item = criterion(logits, batch_data[args.consts["FIELD_LABEL"]][torch.where(batch_data[args.consts["FIELD_GROUP_ID"]] == group_index_tensor)].float().view(-1, 1).to(device))
            data_lens = batch_data[args.consts["FIELD_LABEL"]][torch.where(batch_data[args.consts["FIELD_GROUP_ID"]] == group_index_tensor)].size()[0]
            loss += loss_item * data_lens

            loss_item.backward()

            for name, parms in model.named_parameters():
                # Record the gradient of the trunk model calculated from each group
                if name in trunk_layer_set:
                    gradient_dict[name][group_index] = parms.grad.view(-1)

            stats.append(loss_item)
        # Aggregate the gradient of the trunk model,
        # you can also customize other aggregation methods
        for name, parms in model.named_parameters():
            if name in trunk_layer_set:
                parms.grad = torch.mean(gradient_dict[name], 0).reshape(parms.grad.size())

        loss = loss / args.batch_size  # calculate total loss
        return loss, stats
    else:
        infer_res_list = []
        batch_data = get_inference_batch(data_iterator, args)

        for group_index in range(args.group_num):
            group_index_tensor = torch.LongTensor([group_index]).repeat(batch_data[args.consts["FIELD_GROUP_ID"]].size())
            if len(batch_data[args.consts["FIELD_LABEL"]][torch.where(batch_data[args.consts["FIELD_GROUP_ID"]] == group_index_tensor)]) == 0:
                continue
            # set plugin module index according to group index
            model.set_plugin_index(group_index)

            infer_res = model({
                key: value[torch.where(batch_data[args.consts["FIELD_GROUP_ID"]] == group_index_tensor)].to(device)
                for key, value in batch_data.items()
                if key not in {args.consts["FIELD_USER_ID"], args.consts["FIELD_LABEL"], args.consts["FIELD_GROUP_ID"]}
            }, plugin=True)  # Plugin is true during inference
            infer_res = torch.sigmoid(infer_res)

            infer_res_list.extend(infer_res)

        return infer_res_list

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
    args.consts = pinrec_model.consts
    # run on local
    dataset = PINDatasetLocal(args, input_tables[0], is_test=False)
    eval_dataset = None
    return dataset, eval_dataset

def personalized_args_provider(parser):
    def add_model_config_args(parser):
        """Model arguments"""

        group = parser.add_argument_group('model', 'model configuration')

        parser.add_argument("--model", type=str, help="Model type")
        parser.add_argument("--group_num", type=int, default=5, help="Number of user groups")
        parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
        parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")
        parser.add_argument('--stage_switch_epoch',
                            type=int, default=2,
                            help='Number of training epochs (stage1)')
        return parser

    return add_model_config_args(parser)

def inference_dataset_provider(args):
    input_tables = args.tables.split(",")
    args.consts = pinrec_model.consts
    dataset = PINDatasetLocal(args, input_tables[1], is_test=True)
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

    with open(os.path.join(args.load, 'latest_iteration.txt')) as f:
        for line in f:
            folder = line.strip()
            break
    load_checkpoint_name = os.path.join(args.load, folder, "rank_00_model_states.pt")
    data = mock_data()
    cuda_device = torch.cuda.current_device()

    data = dict((k, v.to(cuda_device)) for k, v in data.items())
    plugin = torch.Tensor([1])
    args.onnx_step = True
    args.task_type = "inference"
    model = model_provider(args)
    print(' >export onnx model number of parameters on rank{}'.format(sum([p.nelement() for p in model.parameters()])), flush=True)
    model.cuda(torch.cuda.current_device())

    load_model_state_only(model, args, remove_prefix=None, remap_prefix=None, load_checkpoint_name=load_checkpoint_name)
    model.eval()
    model_path = os.path.join(args.onnx_export_path, args.onnx_model_name)

    onnx.export(model, (data, plugin), model_path, export_params=True, verbose=False, opset_version=12)
    print("success save to:", model_path)

if __name__ == "__main__":
    task_dispatcher(train_eval_dataset_provider=train_eval_datasets_provider,
                    inference_dataset_provider=inference_dataset_provider,
                    model_provider=model_provider,
                    forward_func=forward_func,
                    personalized_args_provider=personalized_args_provider,
                    onnx_model_export_func=onnx_model_export)