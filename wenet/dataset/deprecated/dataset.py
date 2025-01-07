# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

import wenet.dataset.deprecated.processor as processor
from wenet.text.base_tokenizer import BaseTokenizer
from wenet.utils.file_utils import read_lists


class Processor(IterableDataset):

    # 初始化
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    # 设置数据源的epoch，Epoch是指在整个训练数据集上完整遍历一次的过程
    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    # 设置迭代器，实现了迭代器协议，返回一个处理后的迭代器。
    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    # 用于应用新的处理器函数，并返回一个新的 Processor 实例
    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


# 分布式采样
class DistributedSampler:

    # 初始化
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    # 该方法首先检查分布式环境是否可用，如果可用则获取当前的rank和world_size，否则设置默认值。
    # 然后通过 torch.utils.data.get_worker_info() 获取当前工作进程的信息，包括worker_id和num_workers。
    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    # 设置每个epoch的随机种子，以确保在分布式训练中每个epoch的数据打乱方式不同。
    def set_epoch(self, epoch):
        self.epoch = epoch

    # 根据 partition 和 shuffle 参数对输入数据进行采样。如果 partition 为True，则首先根据 shuffle 参数对数据进行随机打乱，然后根据rank和world_size对数据进行分区。
    # 最后根据worker_id和num_workers对数据进行进一步分区。
    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        # TODO(Binbin Zhang): fix this
        # We can not handle uneven data for CV on DDP, so we don't
        # sample data by rank, that means every GPU gets the same
        # and all the CV data
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


# 这个类的主要目的是在分布式训练环境中处理数据集的采样和迭代
class DataList(IterableDataset):

    # 初始化
    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        # 使用DistributedSampler来处理分布式采样，设置不同的epoch保证在分布式训练中，每个进程都能接收到不同的数据子集，从而避免数据重复和遗漏。
        self.sampler = DistributedSampler(shuffle, partition)

    # 设置epoch，Epoch是指在整个训练数据集上完整遍历一次的过程
    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    # 设置迭代器
    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            # yield dict(src=src)
            data = dict(src=self.lists[index])
            data.update(sampler_info)
            yield data


# 用于根据给定的参数构造数据集的函数。它主要处理从原始数据或分片数据到最终准备好用于训练的数据集的转换过程
def Dataset(data_type,
            data_list_file,
            tokenizer: BaseTokenizer,
            conf,
            partition=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            bpe_model(str): model for english bpe part
            partition(bool): whether to do data partition in terms of rank
    """
    # 验证数据类型是原始数据（'raw'）还是分片数据（'shard'），不是就不能输入，报错。
    assert data_type in ['raw', 'shard']
    # 从data_list_file中读取数据列表
    lists = read_lists(data_list_file)
    # 根据配置信息中的shuffle选项决定是否在开始时进行随机打乱
    shuffle = conf.get('shuffle', True)
    dataset = DataList(lists, shuffle=shuffle, partition=partition)
    # 如果是分片数据（'shard'），则依次使用URL打开器、tar文件处理和分组处理器
    if data_type == 'shard':
        dataset = Processor(dataset, processor.url_opener)
        dataset = Processor(dataset, processor.tar_file_and_group)
    # 如果是原始数据（'raw'），则使用解析原始数据的处理器
    else:
        dataset = Processor(dataset, processor.parse_raw)

    # 如果配置中有speaker_conf，则使用相应的说话人解析处理器
    speaker_conf = conf.get('speaker_conf', None)
    if speaker_conf is not None:
        dataset = Processor(dataset, processor.parse_speaker, **speaker_conf)

    # 使用给定的分词器进行分词处理。如果配置中有tokenize_conf，则使用相应的分词处理器
    dataset = Processor(dataset, processor.tokenize, tokenizer)
    # 根据配置中的filter_conf进行过滤处理
    filter_conf = conf.get('filter_conf', {})
    dataset = Processor(dataset, processor.filter, **filter_conf)

    # 重采样
    resample_conf = conf.get('resample_conf', {})
    dataset = Processor(dataset, processor.resample, **resample_conf)

    # 如果启用了速度扰动（speed_perturb），则应用速度扰动处理器。
    speed_perturb = conf.get('speed_perturb', False)
    if speed_perturb:
        dataset = Processor(dataset, processor.speed_perturb)

    # 根据配置选择特征类型（feats_type）
    feats_type = conf.get('feats_type', 'fbank')
    # 根据配置选择特征类型（feats_type），可以是FBank、MFCC或对数梅尔频谱图，并应用相应的计算处理器。
    assert feats_type in ['fbank', 'mfcc', 'log_mel_spectrogram']
    if feats_type == 'fbank':
        fbank_conf = conf.get('fbank_conf', {})
        dataset = Processor(dataset, processor.compute_fbank, **fbank_conf)
    elif feats_type == 'mfcc':
        mfcc_conf = conf.get('mfcc_conf', {})
        dataset = Processor(dataset, processor.compute_mfcc, **mfcc_conf)
    elif feats_type == 'log_mel_spectrogram':
        log_mel_spectrogram_conf = conf.get('log_mel_spectrogram_conf', {})
        dataset = Processor(dataset, processor.compute_log_mel_spectrogram,
                            **log_mel_spectrogram_conf)

    spec_aug = conf.get('spec_aug', True)
    spec_sub = conf.get('spec_sub', False)
    spec_trim = conf.get('spec_trim', False)
    # 如果启用了频谱增强（spec_aug）、频谱替换（spec_sub）或频谱修剪（spec_trim），则应用相应的处理器。
    if spec_aug:
        spec_aug_conf = conf.get('spec_aug_conf', {})
        dataset = Processor(dataset, processor.spec_aug, **spec_aug_conf)
    if spec_sub:
        spec_sub_conf = conf.get('spec_sub_conf', {})
        dataset = Processor(dataset, processor.spec_sub, **spec_sub_conf)
    if spec_trim:
        spec_trim_conf = conf.get('spec_trim_conf', {})
        dataset = Processor(dataset, processor.spec_trim, **spec_trim_conf)

    # 如果配置中启用了随机打乱（shuffle），则再次应用随机打乱处理器。
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)

    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = Processor(dataset, processor.sort, **sort_conf)

    # 如果配置中启用了排序（sort），则应用排序处理器。
    batch_conf = conf.get('batch_conf', {})
    dataset = Processor(dataset, processor.batch, **batch_conf)
    dataset = Processor(dataset, processor.padding)
    # 返回结果
    return dataset

# 总结：数据集处理