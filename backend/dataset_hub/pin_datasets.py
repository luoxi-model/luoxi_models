# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
from backend.dataset_hub.base_datasets import BaseDataset


class PINDatasetLocal(BaseDataset):
    def __init__(self,
                 args,
                 table_name,
                 shuffle_buffer_size=8194,
                 is_test=False):

        super().__init__(args, table_name, shuffle_buffer_size, is_test)
        self.column_length = len(args.consts.keys())

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
            column_l = self.reader.readline().strip().split(";")
            assert len(column_l) == self.column_length, "len(column_l) must be {}, now is {}}".format(self.column_length, len(column_l))
        except:
            self.reader.seek(0)
            column_l = self.reader.readline().strip().split(";")
        return column_l

    def __del__(self):
        if self.reader is not None:
            self.reader.close()

    def _parse_item(self, column_l):
        user_id = int(column_l[0])
        target_id = int(column_l[1])
        clk_sequence = torch.LongTensor(list(map(int, column_l[2].split(","))))
        label = int(column_l[3])
        group_id = int(column_l[4])
        ret = {
            "user_id": user_id,
            "target_id": target_id,
            "clk_sequence": clk_sequence,
            "label": label,
            "group_id": group_id
        }
        return ret