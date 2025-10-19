# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

from abc import ABC
from torch.utils.data import DataLoader, Dataset


def _get_dataloader(dataset: Dataset, shuffle: bool, **kwargs) -> DataLoader:
    # 单卡场景下的标准 DataLoader
    return DataLoader(dataset, shuffle=shuffle, **kwargs)


class DataModule(ABC):
    """ Abstract DataModule. Children must define self.ds_{train | val | test}. """

    def __init__(self, **dataloader_kwargs):
        super().__init__()
        # 单卡直接准备数据
        self.prepare_data()

        self.dataloader_kwargs = {'pin_memory': True, 'persistent_workers': True, **dataloader_kwargs}
        self.ds_train, self.ds_val, self.ds_test = None, None, None

    def prepare_data(self):
        """ Method called only once per node. Put here any downloading or preprocessing """
        pass

    def train_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_train, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_val, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_test, shuffle=False, **self.dataloader_kwargs)
