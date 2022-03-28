# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
from .arguments import get_args
import os
import time
from backend.utils import print_args, load_model_state_only, initialize_distribution_env, set_random_seed, Timer
from backend.utils import save_checkpoint, print_rank_0, make_local_writer
from torch.nn.parallel.distributed import DistributedDataParallel
from torch import nn
from datetime import datetime
from torch.utils.data.distributed import DistributedSampler

def make_inference_data_loader(args, inference_dataset_provider):
    print('make inference data loaders')

    num_workers = torch.distributed.get_world_size()

    test_data_set = inference_dataset_provider(args)

    total_samples = test_data_set.get_total_row_count()
    samples_per_iter = args.batch_size * num_workers
    args.infer_iters = (total_samples + samples_per_iter - 1) // samples_per_iter
    print('[inference]> samples=%d workers=%d batch=%d samples_per_iter=%d infer_iters=%d' % (
            total_samples, num_workers, args.batch_size, samples_per_iter, args.infer_iters))

    batch_sampler = DistributedSampler(dataset=test_data_set)
    infer_data_loader = torch.utils.data.DataLoader(test_data_set,
                                                    batch_size=args.batch_size,
                                                    sampler=batch_sampler,
                                                    drop_last=False)

    return infer_data_loader

def make_data_loader(args, train_eval_data_provider):
    print('make data loaders')

    num_workers = torch.distributed.get_world_size()

    train_data_set, eval_data_set = train_eval_data_provider(args)

    if args.num_epochs > 0:

        total_samples = train_data_set.get_total_row_count()
        samples_per_iter = args.batch_size * num_workers
        args.train_iters = (args.num_epochs * total_samples + samples_per_iter - 1) // samples_per_iter

        if eval_data_set is not None:
            eval_samples = eval_data_set.get_total_row_count()
            args.eval_iters = (eval_samples + args.batch_size - 1) // args.batch_size
        else:
            args.eval_iters = 0

        print('to', args.train_iters, 'due to args.num_epochs=', args.num_epochs)
        print('***num_epochs=%d samples=%d workers=%d batch=%d samples_per_iter=%d train_iters=%d eval_iters=%d' % (
            args.num_epochs, total_samples, num_workers, args.batch_size, samples_per_iter, args.train_iters, args.eval_iters))

    batch_sampler = DistributedSampler(dataset=train_data_set)
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=args.batch_size,
                                                    sampler=batch_sampler,
                                                    drop_last=True)

    if eval_data_set is not None:
        eval_data_loader = torch.utils.data.DataLoader(eval_data_set,
                                                       num_workers=0,
                                                       pin_memory=True,
                                                       batch_size=args.batch_size)
    else:
        eval_data_loader = None

    return train_data_loader, eval_data_loader

def get_model(args,
              model_provider,
              is_gpu=True):

    model = model_provider(args)

    print_rank_0('number of parameters on rank : {}'.format(
        sum([p.nelement() for p in model.parameters()])))

    if is_gpu:
        model.cuda(torch.cuda.current_device())

        from torch.nn.parallel.distributed import DistributedDataParallel
        i = torch.cuda.current_device()
        model = DistributedDataParallel(model, device_ids=[i], output_device=i, find_unused_parameters=args.find_unused_parameters)
    else:
        raise NotImplementedError

    return model

def get_params_for_weight_decay(module):
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params

def get_optimizer_param_groups(model):
    while isinstance(model, (DistributedDataParallel)):
        model = model.module
    param_groups = get_params_for_weight_decay(model)

    return param_groups

def get_optimizer(param_groups, args):
    assert args.optimizer in ('adam', 'adamw'), 'optimizer must be adam or adamw!'

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    args.optimizer_runtime = optimizer

    return optimizer

def setup_model_and_optimizer(args, model):
    param_groups = get_optimizer_param_groups(model)
    optimizer = get_optimizer(param_groups, args)

    return optimizer


def load_model_setup_optimizer(args, model, need_optimizer = True):
    optimizer = None
    if args.load is not None:
        local_device_id = torch.distributed.get_rank() % args.gpus_per_node
        if local_device_id == 0:
            os.makedirs('tmp/')
        else:
            while not os.path.exists('tmp/done_{}.txt'.format(local_device_id - 1)):
                time.sleep(1)

        load_model_state_only(model, args, remove_prefix=None, remap_prefix=None)

        if need_optimizer:
            optimizer = setup_model_and_optimizer(args, model=model)

        fout = open('tmp/done_{}.txt'.format(local_device_id), 'w')
        fout.close()
        if local_device_id < (args.gpus_per_node - 1):
            while not os.path.exists('tmp/done_{}.txt'.format(args.gpus_per_node - 1)):
                time.sleep(1)
    elif need_optimizer:
        optimizer = setup_model_and_optimizer(args, model=model)

    torch.distributed.barrier()

    return model, optimizer

def train_step(forward_step, data_iterator, model, optimizer, args):
    stats_reduced = loss_reduced = None
    for i in range(1):
        if i == 0:
            optimizer.zero_grad()

        loss, stats = forward_step(data_iterator, model, args)
        stats = torch.stack([loss.detach()] + [s.detach() for s in stats])

        if not args.backward_step_contains_in_forward_step:
            loss.backward()

        stats_reduced = stats.clone()

        torch.distributed.all_reduce(stats_reduced.data)
        stats_reduced.data = stats_reduced.data / args.world_size
        loss_reduced = stats_reduced[0]

        if args.clip_grad > 0:
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()

    return loss_reduced, stats_reduced

def evaluate(forward_step, model, data_iterator, args):
    model.eval()

    with torch.no_grad():

        sum_loss = 0
        sum_samples = 0
        iteration = 0

        while iteration < args.eval_iters:
            try:
                loss, stats = forward_step(data_iterator, model, args)
            except StopIteration:
                break

            loss = loss.item()

            sum_loss += loss

            sum_samples += 1

            iteration += 1

        torch.distributed.barrier()

        result = torch.LongTensor([sum_loss, sum_samples]).to(torch.cuda.current_device())
        torch.distributed.all_reduce(result)
        sum_loss = result[0].item()
        sum_samples = result[1].item()

        if args.local_rank == 0:
            print('[evaluation]global eval loss = %.4f (%d / %d)' % (sum_loss / sum_samples, args.eval_iters, sum_samples))

def train(forward_step_func, model, optimizer,
          train_data_iterator, valid_data_set, args):
    model.train()

    # Tracking loss.
    sum_iter = sum_loss = 0
    sum_stats = None

    timer = Timer()
    timer.reset()
    timer.start()
    while args.iteration < args.train_iters:
        loss, stats = train_step(forward_step_func, train_data_iterator, model, optimizer, args)

        loss = loss.item()
        stats = stats.data.detach().tolist()

        args.iteration = args.iteration + 1

        sum_iter += 1
        sum_loss += loss
        if sum_stats is None:
            sum_stats = [0.0] * len(stats)
        sum_stats = [a + b for a, b in zip(sum_stats, stats)]

        if args.iteration % args.log_interval == 0:
            use_time = timer.stop()
            per_iteration_use_time = use_time * 1000.0 / sum_iter

            if args.local_rank == 0:
                report_iteration_metrics(per_iteration_use_time, sum_loss / sum_iter, args.iteration, args.train_iters)
                print('stats: [', ', '.join(['%.6f' % (x / sum_iter) for x in sum_stats]), ']')

            sum_iter = sum_loss = 0
            sum_stats = None

        if args.iteration % args.save_interval == 0:
            args.final_saved_iteration=args.iteration
            save_checkpoint(args.iteration, model, optimizer, args)

        if valid_data_set is not None and args.iteration % args.eval_interval == 0:
            evaluate(forward_step_func, model, iter(valid_data_set), args)
            model.train()

        if args.iteration % args.log_interval == 0:
            timer.reset()
            timer.start()

def parse(torch_tensor):
    arr = torch_tensor.cpu().numpy()

    if len(arr.shape) == 0:
        arr = [arr]

    return ','.join(list([str(item) for item in arr]))

def inference(forward_step_func, model, writer,
              infer_data_iterator, args):
    model.eval()

    iter_index = 0
    samples = 0

    with torch.no_grad():
        while iter_index < args.infer_iters:
            stats = forward_step_func(infer_data_iterator, model, args)
            samples += stats[0].shape[0]

            iter_index += 1
            indices = [idx for idx in range(len(stats))]
            for j in range(stats[0].shape[0]):
                ret = [parse(item[j]) for item in stats]

                writer.write('\t'.join(ret) + '\n')

            if iter_index % args.log_interval == 0:
                print_rank_0("%d samples inference finished!" %(samples))
                samples = 0

        writer.close()

def report_iteration_metrics(per_iteration_use_time, loss, step, total_step):
    log_string = '\n'
    log_string += str(datetime.now())
    log_string += ' iteration %d(all:%d) ||' % (step, total_step)
    log_string += ' loss %.6f ||' % loss
    log_string += ' time per iteration (ms) %.1f |' % per_iteration_use_time

    print(log_string)

def task_dispatcher(train_eval_dataset_provider = None,
                    inference_dataset_provider = None,
                    model_provider = None,
                    forward_func = None,
                    personalized_args_provider = None,
                    training_post_processing_func = None,
                    onnx_model_export_func = None):
    '''
    task dispatcher, now support training task and inference task
    :param train_eval_dataset_provider:  a training used function that input args and return train&valid datasets
    :param inference_dataset_provider: a inference used function that input args and return inference dataset
    :param model_provider: a function that input args and return user-defined model
    :param forward_func: a function that input data iterator and args, and return
                            1) a list containing loss and other metrics that need to be printed  if args.task_type = train
                            2) a list containing prediction data that need to be printed to the local files if args.task_type = inference
    :param personalized_args_provider: a function that provide for users to define model specific parameters
    :param training_post_processing_func: a function that provide for users to do sth after training task finished
    :param onnx_model_export_func: onnx model export func
    :return:
    '''

    args = get_args(personalized_args_provider)

    # Pytorch distributed.
    initialize_distribution_env(args)

    # Random seeds for reproducibility.
    set_random_seed(seed=args.seed)

    assert args.task_type == 'train' or args.task_type == 'inference' or args.task_type == 'onnx_export', 'task type must be train or inference or onnx_export'
    if args.task_type == 'train':
        print_rank_0('*****running task : train*****')
        assert train_eval_dataset_provider is not None and model_provider is not None and forward_func is not None, \
            '[train_eval_dataset_provider, model_provider, forward_func] cannot be None when training task'
        training_backbone(train_eval_dataset_provider, model_provider, forward_func, training_post_processing_func, args)
    elif args.task_type == 'inference':
        print_rank_0('*****running task : inference*****')
        assert inference_dataset_provider is not None and model_provider is not None and forward_func is not None, \
            '[inference_dataset_provider, model_provider, forward_func] cannot be None when inference task'
        inference_backbone(inference_dataset_provider, model_provider, forward_func, args)
    elif args.task_type == 'onnx_export':
        onnx_export_backbone(onnx_model_export_func, args)
        pass

def onnx_export_backbone(onnx_model_export_func
                         ,args):
    '''
    onnx export backbone
    :param onnx_model_export_func: onnx model export func
    :param args: a arguments dictionary
    :return:
    '''
    onnx_model_export_func(args)

def inference_backbone(inference_dataset_provider,
                       model_provider,
                       forward_func,
                       args):
    '''
    inference task backbone
    :param inference_dataset_provider: a inference used function that input args and return inference dataset
    :param model_provider: a function that input args and return user-defined model
    :param forward_func: a function that input data iterator and args, and return a list containing prediction data that need to be printed to the local files
    :param args: a arguments dictionary
    :return:
    '''
    test_data = make_inference_data_loader(args, inference_dataset_provider)
    print('Inference data preparation done.')

    model = get_model(args, model_provider)

    model, _ = load_model_setup_optimizer(args, model)

    writer = make_local_writer(args)

    inference(forward_func, model, writer, iter(test_data), args)

def training_backbone(train_eval_dataset_provider,
                    model_provider,
                    forward_func,
                    training_post_processing_func,
                    args):
    '''
    training task backbone
    :param train_eval_dataset_provider: a training used function that input args and return train&valid dataset
    :param model_provider: a function that input args and return user-defined model
    :param forward_func: a function that input data iterator and args, and return a dict containing the loss and other metrics that need to be printed
    :param args: a arguments dictionary
    :return:
    '''
    model = get_model(args, model_provider)

    train_data, eval_data = make_data_loader(args, train_eval_dataset_provider)
    print('Data preparation done.')

    args.iteration = 0
    train_data_iterator = iter(train_data)

    if torch.distributed.get_rank() == 0:
        print_args(args)

    model, optimizer = load_model_setup_optimizer(args, model)

    train(forward_func, model, optimizer, train_data_iterator, eval_data, args)

    if training_post_processing_func is not None:
        training_post_processing_func(model, args)