# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
from backend.dataset_hub.base_datasets import *

class CRecDataset(BaseDataset):
    def __init__(self, args, file_name, shuffle_buffer_size=8096, is_test=False):
        """
        function: read the dataset for the cloud-based recommendation model.
        for more details, please refer to the parent class
        """
        super().__init__(args, file_name, shuffle_buffer_size, is_test)

    def get_total_row_count(self):
        cnt = 0
        for _ in self.reader:
            if len(_) > 0:
                cnt += 1
        self.reader.seek(0)

        return cnt

    def _new_reader(self):
        if self.reader is not None:
            self.reader.close()
        print('self.file_name', self.table_name)
        reader = open(self.table_name, "r")

        return reader

    def _read_record(self):
        try:
            column_l = self.reader.readline().strip().split("\t")
            assert len(column_l) == 3, "len(column_l) must be 3, now is %d" % len(column_l)
        except:
            self.reader.seek(0)
            column_l = self.reader.readline().strip().split("\t")

        return column_l	

    def _parse_item(self, sample):
        # user feature
        hist_seq = sample[0].strip().split(',')
        hist_seq = list(map(int, hist_seq))

        # Item feature
        cand = int(sample[1])

        # Label
        label = float(sample[2])

        output = {
            "hist_seq" : torch.LongTensor(hist_seq),
            "cand" : cand,
            "label": label}
        return output

    def __del__(self):
        if self.reader is not None:
            self.reader.close()

class ORecDataset(BaseDataset):
    def __init__(self, args, file_name, shuffle_buffer_size=8096, is_test=False):
        """
        function: read the dataset for the on-device recommendation model.
        for more details, please refer to the parent class
        """
        super().__init__(args, file_name, shuffle_buffer_size, is_test)

    def get_total_row_count(self):
        cnt = 0
        for _ in self.reader:
            if len(_) > 0:
                cnt += 1
        self.reader.seek(0)

        return cnt

    def _new_reader(self):
        if self.reader is not None:
            self.reader.close()
        print('self.table_name', self.table_name)
        reader = open(self.table_name, "r")

        return reader

    def _read_record(self):
        try:
            column_l = self.reader.readline().strip().split("\t")
            assert len(column_l) == 4, "len(column_l) must be 4, now is %d" % len(column_l)
        except:
            self.reader.seek(0)
            column_l = self.reader.readline().strip().split("\t")

        return column_l	

    def _parse_item(self, sample):
        # user feature
        hist_seq = sample[0].strip().split(',')
        hist_seq = list(map(int, hist_seq))

        # Item feature
        cand = int(sample[1])

        # prior score
        prior_score = float(sample[2])

        # Label
        label = float(sample[3])

        output = {
            "hist_seq" : torch.LongTensor(hist_seq),
            "cand" : cand,
            "prior_score": prior_score,
            "label": label}
        return output

    def __del__(self):
        if self.reader is not None:
            self.reader.close()

class MCRecDataset(BaseDataset):
    def __init__(self, args, file_name, shuffle_buffer_size=8096, is_test=False):
        """
        function: read the counterfactual dataset for the meta controller.
        for more details, please refer to the parent class
        """
        super().__init__(args, file_name, shuffle_buffer_size, is_test)

    def get_total_row_count(self):
        cnt = 0
        for _ in self.reader:
            if len(_) > 0:
                cnt += 1
        self.reader.seek(0)

        return cnt

    def _new_reader(self):
        if self.reader is not None:
            self.reader.close()
        print('self.table_name', self.table_name)
        reader = open(self.table_name, "r")

        return reader

    def _read_record(self):
        try:
            column_l = self.reader.readline().strip().split("\t")
            assert len(column_l) == 2, "len(column_l) must be 2, now is %d" % len(column_l)
        except:
            self.reader.seek(0)
            column_l = self.reader.readline().strip().split("\t")

        return column_l	

    def _parse_item(self, sample):
        # user feature
        hist_seq = sample[0].strip().split(',')
        hist_seq = list(map(int, hist_seq))

        # Label
        label = float(sample[1])

        output = {
            "hist_seq" : torch.LongTensor(hist_seq),
            "label": label}
        return output

    def __del__(self):
        if self.reader is not None:
            self.reader.close()
