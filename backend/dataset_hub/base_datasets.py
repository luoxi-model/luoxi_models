# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from torch.utils.data import Dataset
import random

class BaseDataset(Dataset):
    def __init__(self,
                 args,
                 table_name,
                 shuffle_buffer_size,
                 is_test):
        self.args = args
        self.table_name = table_name

        self.reader = None
        self.reader = self._new_reader()
        self.fetch_iter = 0
        self.num_to_fetch = self.get_total_row_count()  # 单卡
        self.shuffle_buffer = []
        self.shuffle_size = shuffle_buffer_size
        self.is_test = is_test
        self._total_row_count = -1

    def get_total_row_count(self):
        raise NotImplementedError

    def _new_reader(self):
        raise NotImplementedError

    def _need_reload(self):
        return self.fetch_iter == self.num_to_fetch

    def _read_record(self):
        raise NotImplementedError

    def _read_item(self):
        if self._need_reload():
            self.fetch_iter = 0
            self.reader = self._new_reader()
        self.fetch_iter += 1
        column_l = self._read_record()
        column_l = [
            item.decode(encoding="utf8", errors="ignore") if type(item) == bytes else item
            for item in column_l
        ]
        return column_l

    def _parse_item(self, column_l):
        raise NotImplementedError

    def __getitem__(self, idx):
        if self.is_test:
            return self._parse_item(self._read_item())
        while ((not self._need_reload()) or (len(self.shuffle_buffer) == 0)) and (
                len(self.shuffle_buffer) < self.shuffle_size):
            self.shuffle_buffer.append(self._read_item())
        num_samples = len(self.shuffle_buffer)
        i = random.randint(0, num_samples - 1)
        if i != num_samples - 1:
            self.shuffle_buffer[i], self.shuffle_buffer[-1] = self.shuffle_buffer[-1], self.shuffle_buffer[i]
        ret_item = self.shuffle_buffer.pop(-1)
        ret_item = self._parse_item(ret_item)
        return ret_item

    def __len__(self):
        if self.is_test:
            return self.num_to_fetch
        return 2 ** 30  # fake