# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
import os
from torch.nn.parallel.distributed import DistributedDataParallel
import random
import numpy as np
import time
import json

class Timer:
    def __init__(self):
        self.interval = 0.0
        self.is_start = False
        self.start_time = None

    def start(self):
        assert not self.is_start, 'timer has been started'
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.is_start = True

    def stop(self):
        assert self.is_start, 'timer is not started'
        torch.cuda.synchronize()
        self.interval += (time.time() - self.start_time)

        return self.interval

    def reset(self):
        self.interval = 0.0
        self.is_start = False

def print_args(args):
    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '-' * (20 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)

def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def initialize_distribution_env(args):
    assert torch.cuda.is_available(), 'requires CUDA.'

    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    print("device id: {}".format(device))

    torch.cuda.set_device(device)

    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)
    print('args.world_size =', args.world_size, ', args.rank =', args.rank, ', args.local_rank =', args.local_rank)
    assert args.rank == torch.distributed.get_rank()

def set_random_seed(seed=None):
    if seed is None:
        seed = int(time.time())
    print('set_random_seed', seed)
    seed = int(seed + torch.distributed.get_rank())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, 'latest_iteration.txt')

def get_saved_iteration(args):

    tracker_filename = get_checkpoint_tracker_filename(args.load)
    if not os.path.isfile(tracker_filename):
        print_rank_0('could not find {}  and will start from random'.format(tracker_filename))
        return 0
    iteration = 0
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            print_rank_0('the first row of {} must be an integer'.format(tracker_filename))
            exit()

    return iteration

def get_checkpoint_name(checkpoints_path, iteration):
    d = '{:d}'.format(iteration)

    return os.path.join(checkpoints_path, d, 'rank_{:02d}_model_states.pt'.format(0))

def load_model_state_only(model, args, remove_prefix=None, remap_prefix=None, force_remap=False, load_checkpoint_name=None):
    if load_checkpoint_name is None:
        iteration = get_saved_iteration(args)
        checkpoint_name = get_checkpoint_name(args.load, iteration)
    else:
        iteration = 0
        checkpoint_name = load_checkpoint_name
    # Load the checkpoint.
    sd = torch.load(checkpoint_name, map_location='cpu')

    if isinstance(model, DistributedDataParallel):
        model = model.module
    model_state = sd['module'] if 'module' in sd else sd

    if remove_prefix:
        for load_prefix in remove_prefix:
            keys = list(model_state.keys())
            for k in keys:
                if k.startswith(load_prefix):
                    print('Skip loading %s in the checkpoint.' % k)
                    del model_state[k]

    if remap_prefix:
        for var_prefix, load_prefix in remap_prefix.items():
            keys = list(model_state.keys())
            for k in keys:
                if k.startswith(load_prefix):
                    new_k = k.replace(load_prefix, var_prefix)
                    if new_k in model_state:
                        print('WARN: param %s already in the checkpoint.' % new_k)
                    if (new_k not in model_state) or force_remap:
                        print('Load param %s from %s in the checkpoint.' % (new_k, k))
                        model_state[new_k] = model_state[k]

    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as e:
        print(e)
        print('> strict load failed, try non-strict load instead')
        keys = model.load_state_dict(model_state, strict=False)
        print('> non-strict load done')
        print(keys)
    return iteration

def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def save_checkpoint(iteration, model, optimizer, args):
    if isinstance(model, DistributedDataParallel):
        model = model.module

    if torch.distributed.get_rank() == 0:
        checkpoint_name = get_checkpoint_name(args.save, iteration)
        print('rank {} is saving checkpoint at iteration {:7d} to {}'. \
                format(torch.distributed.get_rank(), iteration, checkpoint_name))

        sd = {}
        sd['iteration'] = iteration
        sd['module'] = model.state_dict()

        # Optimizer stuff.
        if optimizer is not None:
            sd['optimizer'] = optimizer.state_dict()

        ensure_directory_exists(checkpoint_name)
        torch.save(sd, checkpoint_name)
        print('Successfully saved {}'.format(checkpoint_name))


    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))

    torch.distributed.barrier()

def make_local_writer(args):
    file = open(args.outputs, 'w')

    return file

def to_numpy(tensor,data_type=None):
    tensor = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    if data_type:
        if data_type=="int":
            data_type=np.int64
        else:
            data_type = np.float32
        tensor=tensor.astype(data_type)
    return tensor

def parse_arch_config_from_args(model_meta, args):
    """
    Read or parse arch config
    :param model_meta:
    :param args:
    :return:
    """
    if args.arch_config is not None:
        with open(args.arch_config) as jsonfile:
            raw_arch_config = json.load(jsonfile)
    elif args.arch_config_path is not None:
        with open(args.arch_config_path, "rt") as reader:
            raw_arch_config = json.load(reader)
    else:
        raise KeyError("Model configuration not found")

    return model_meta.arch_config_parser(raw_arch_config), raw_arch_config