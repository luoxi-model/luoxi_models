# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import argparse
import os
import torch


def add_training_args(parser):
    group = parser.add_argument_group('train', 'training')

    #optimizer args
    group.add_argument('--batch-size', type=int, default=128,
                       help='data batch size')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--num-epochs', type=int, default=2,
                       help='num-epochs')
    group.add_argument('--train-iters', type=int, default=1000000,
                       help='total number of iterations to train')
    group.add_argument('--log-interval', type=int, default=100,
                       help='report metrics interval')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after this many new iterations.')
    group.add_argument('--seed', type=int, default=1234,
                       help='random seed')
    group.add_argument('--optimizer', default='adamw',
                       help='optimizer, One of [adamw, adam]')
    group.add_argument("--final-saved-iteration", type=int, default=0, help="if gpu")

    #ddp params
    group.add_argument('--find-unused-parameters', action='store_true', help='find_unused_parameters setting in DDP ')

    #backward args
    group.add_argument('--backward-step-contains-in-forward-step', action='store_true', help='backward operations contains in forward step')

    # Learning rate.
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='learning rate')
    group.add_argument('--weight-decay', type=float, default=0.0,
                       help='weight decay coefficient for L2 regularization')

    # model checkpointing args
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', type=int, default=5000,
                       help='number of iterations between saves')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--task-type', type=str, default='train',
                       help='task type: train, inference')

    # distributed args
    group.add_argument('--distributed-backend', default='nccl',
                       help='which backend to use for distributed '
                            'train_hub. One of [gloo, nccl]')
    group.add_argument('--local_rank', type=int, default=0,
                       help='local rank passed from distributed launcher')
    group.add_argument('--worker-cnt', type=int, default=1,
                       help='number of workers')
    group.add_argument('--gpus-per-node', type=int, default=1,
                       help='number of gpus per node')
    group.add_argument('--entry', type=str, default='pretrain_gpt2.py')


    return parser


def add_data_args(parser):
    group = parser.add_argument_group('data', 'data configurations')

    # add arguments of input and output
    group.add_argument('--tables', type=str, default='', help='input table(data) name (train, valid, test)')
    group.add_argument('--outputs', type=str, default='')

    #onnx model
    group.add_argument("--onnx_export_path", type=str, default='onnx model export path', help="onnx_export_path")
    group.add_argument("--onnx_model_name", type=str, default='onnx_model.onnx', help="ONNX model name")

    return parser

def add_validation_args(parser):
    group = parser.add_argument_group(title='validation')

    group.add_argument('--eval-iters', type=int, default=100,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')

    return parser

def get_args(add_personalized_args_fn=None):

    parser = argparse.ArgumentParser(description='..')
    parser = add_training_args(parser)
    parser = add_data_args(parser)
    parser = add_validation_args(parser)

    if add_personalized_args_fn is not None:
        parser = add_personalized_args_fn(parser)

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv('WORLD_SIZE', '1'))

    return args
