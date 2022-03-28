# -*- coding: utf-8 -*-
# Copyright 2022 The Luoxi Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
from backend.dataset_hub.base_datasets import *

class CIGARDataset(BaseDataset):
    def __init__(self,
                 args,
                 table_name,
                 shuffle_buffer_size,
                 is_test=False,
                 max_len=-1,
                 max_neighbor=-1):
        super(CIGARDataset, self).__init__(args, table_name, shuffle_buffer_size, is_test)
        self.maxlen = max_len
        self.max_neighbor = max_neighbor

    def _parse_item(self, one_sample):
        output = {}
        #### User feature ####
        user_fea = [int(one_sample[int(x)])+1 for x in self.args.user_fea_col_id.split(',')]
        user_fea_name = self.args.user_fea_name.split(',')
        output.update(dict(zip(user_fea_name, user_fea)))
        #### Item feature ####
        item_fea = [int(one_sample[int(x)]) + 1 for x in self.args.item_fea_col_id.split(',')]
        item_fea_name = self.args.item_fea_name.split(',')
        output.update(dict(zip(item_fea_name, item_fea)))
        #### Seq feature ####
        seq_col_id = [int(x) for x in self.args.seq_col_id.split(',')]
        seq_fea_name = [x+'_seq' for x in self.args.item_fea_name.split(',')]
        # padding and truncation
        for i in range(len(seq_col_id)):
            if one_sample[seq_col_id[i]] == '':
                seq_fea = [0] * self.maxlen
            else:
                seq_fea = one_sample[seq_col_id[i]].split(',')
                seq_fea = seq_fea[-self.maxlen:] \
                    if len(seq_fea) >= self.maxlen \
                    else seq_fea + [0] * (self.maxlen - len(seq_fea))
                seq_fea = list(map(int, seq_fea))
            assert len(seq_fea) == self.maxlen
            output[seq_fea_name[i]] = torch.LongTensor(seq_fea)
        # uid, label
        uid, graph_id, label_id = [int(x) for x in self.args.uid_graph_label_col_id.split(',')]
        output['userid'] = int(one_sample[uid])
        output['label'] = float(one_sample[label_id])
        # Graph feature
        # padding and truncation
        if one_sample[graph_id] == '':
            neighbor_ids = [0] * self.max_neighbor
        else:
            neighbor_ids = one_sample[graph_id].split(',')
            neighbor_ids = neighbor_ids[:self.max_neighbor] \
                if len(neighbor_ids) >= self.max_neighbor \
                else neighbor_ids + [0] * (self.max_neighbor - len(neighbor_ids))
            neighbor_ids = list(map(int, neighbor_ids))
        assert len(neighbor_ids) == self.max_neighbor
        output['neighbor_ids'] = torch.LongTensor(neighbor_ids)
        return output

class CIGARDatasetLocal(CIGARDataset):
    def __init__(self,
                 args,
                 table_name,
                 shuffle_buffer_size=8194,
                 is_test=False,
                 max_len=100,
                 max_neighbor=10):

        super(CIGARDatasetLocal,self).__init__(args, table_name, shuffle_buffer_size, is_test, max_len, max_neighbor)

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
            # print(column_l)
            assert len(column_l) == self.args.column_len, "len(column_l) must be %d, now is %d" % (self.args.column_len, len(column_l))
        except:
            self.reader.seek(0)
            column_l = self.reader.readline().strip().split("\t")
        return column_l

    def __del__(self):
        if self.reader is not None:
            self.reader.close()
