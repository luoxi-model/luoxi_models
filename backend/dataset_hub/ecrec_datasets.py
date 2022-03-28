# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
from backend.dataset_hub.base_datasets import BaseDataset


class ECRecDatasetLocal(BaseDataset):
    def __init__(self,
                 args,
                 table_name,
                 shuffle_buffer_size=8194,
                 is_test=False):

        super().__init__(args, table_name, shuffle_buffer_size, is_test)
        self.column_length = args.column_length
        self.maxlen = args.sequence_length

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
            assert len(column_l) == self.column_length, "len(column_l) must be {}, now is {}}".format(self.column_length, len(column_l))
        except:
            self.reader.seek(0)
            column_l = self.reader.readline().strip().split("\t")
        return column_l

    def __del__(self):
        if self.reader is not None:
            self.reader.close()

    def _parse_item(self, one_sample):
        '''parse items for each sample'''
        uid = int(one_sample[0])
        item_id = int(one_sample[1])
        cate_id = int(one_sample[2])
        label = int(one_sample[3])

        if one_sample[4] == '':
            hist_item = [0] * self.maxlen
            hist_cate = [0] * self.maxlen
        else:
            hist_item = one_sample[4].split(',')
            hist_item = hist_item[:self.maxlen] if len(hist_item)>=self.maxlen else hist_item + [0]*(self.maxlen-len(hist_item))
            hist_item = list(map(int, hist_item))
            hist_cate = one_sample[5].split(',')
            hist_cate = hist_cate[:self.maxlen] if len(hist_cate)>=self.maxlen else hist_cate + [0]*(self.maxlen-len(hist_cate))
            hist_cate = list(map(int, hist_cate))

        if one_sample[6] == '':
            edge_item = [0] * self.maxlen
            edge_cate = [0] * self.maxlen
        else:
            edge_item = one_sample[6].split(',')
            edge_item = edge_item[:self.maxlen] if len(edge_item) >= self.maxlen else edge_item + [0] * (
                                self.maxlen - len(edge_item))
            edge_item = list(map(int, edge_item))
            edge_cate = one_sample[7].split(',')
            edge_cate = edge_cate[:self.maxlen] if len(edge_cate) >= self.maxlen else edge_cate + [0] * (
                                self.maxlen - len(edge_cate))
            edge_cate = list(map(int, edge_cate))

        seq_len = int(one_sample[8])

        output = {'user_id': uid,
                    'item_id': item_id,
                    'cate_id': cate_id,
                    'label': label,
                    'item_seq': torch.LongTensor(hist_item),
                    'cate_seq': torch.LongTensor(hist_cate),
                    'edge_item_seq': torch.LongTensor(edge_item),
                    'edge_cate_seq': torch.LongTensor(edge_cate),
                    'seq_len': seq_len
                    }

        return output