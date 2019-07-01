# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging

import torch


LOGGER = logging.getLogger(__name__)


class FixedSizeDataLoader:
    def __init__(self, dataset, steps, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False,
                 sampler=None):
        sampler = InfiniteSampler(dataset, shuffle) if sampler is None else sampler
        self.batch_size = batch_size
        batch_size = 1 if batch_size is None else batch_size

        self.steps = steps
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _, data in zip(range(self.steps), self.dataloader):
            yield ([t[0] for t in data] if self.batch_size is None else data)


class InfiniteSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.data_source)
        while True:
            index_list = torch.randperm(n).tolist() if self.shuffle else list(range(n))
            for idx in index_list:
                yield idx

    def __len__(self):
        return len(self.data_source)
