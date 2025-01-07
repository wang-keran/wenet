# Copyright (c) 2023 Wenet Community. (authors: Dinghao Zhou)
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

import collections
from collections.abc import Callable
import copy
import sys
import tarfile
import logging
from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data import datapipes
from torch.utils.data.datapipes.iter import Mapper
from torch.utils.data.datapipes.iter.sharding import (
    SHARDING_PRIORITIES, ShardingFilterIterDataPipe)
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn

from wenet.dataset.processor import parse_url


@functional_datapipe("map_ignore_error")
class MapperIgnoreErrorDataPipe(Mapper):

    # 初始化管道，初始化了可迭代的数据管道IterDataPipe,Callable对象,输入列输出列（可选），错误日志记录
    def __init__(self,
                 dataset: IterDataPipe,
                 fn: Callable,
                 input_col=None,
                 output_col=None,
                 log_error: bool = True) -> None:
        super().__init__(dataset, fn, input_col, output_col)
        self._iter = None
        self.log_error = log_error

    def __iter__(self):
        # 该方法首先检查self._iter是否为None。如果是，则初始化self._iter为iter(self.datapipe)，即创建一个迭代器对象。
        if self._iter is None:
            self._iter = iter(self.datapipe)

        # 如果self._iter已经存在，则直接返回它。
        while True:
            try:
                # 在每次循环中，尝试使用next(self._iter)获取下一个元素，并将其传递给self._apply_fn函数进行处理。
                elem = next(self._iter)
                # 如果成功获取到元素，则使用yield关键字返回处理后的元素。
                yield self._apply_fn(elem)
             # 如果遇到StopIteration异常，则将self._iter重置为None，并返回，结束迭代。
            except StopIteration:
                self._iter = None
                return
            except Exception as ex:
                if self.log_error:
                    logging.warning(str(ex))


# 是一个用于根据序列长度对数据进行分桶的迭代数据管道。
# 它通过 elem_length_func 函数计算每个元素的长度，并根据 bucket_boundaries 和 bucket_batch_sizes 将数据分桶。
# 每个桶的大小由 bucket_batch_sizes 决定，当一个桶达到其对应的大小时，整个桶将被填充并形成一个批次。
@functional_datapipe('bucket_by_sequence_length')
class BucketBySequenceLengthDataPipe(IterDataPipe):

    def __init__(
        self,
        dataset: IterDataPipe,
        elem_length_func,
        bucket_boundaries: List[int],
        bucket_batch_sizes: List[int],
        wrapper_class=None,
    ) -> None:
        super().__init__()
        _check_unpickable_fn(elem_length_func)
        # 首先检查 elem_length_func 是否可序列化，然后断言bucket_batch_sizes 的长度等于 bucket_boundaries 的长度加一。
        # 接着，将 bucket_boundaries 扩展到包含 sys.maxsize，以便处理超出边界的情况。
        # 最后，将 elem_length_func 和 bucket_batch_sizes 存储为类属性，并初始化 GroupByWindowDataPipe 对象。
        assert len(bucket_batch_sizes) == len(bucket_boundaries) + 1
        self.bucket_batch_sizes = bucket_batch_sizes
        self.bucket_boundaries = bucket_boundaries + [sys.maxsize]
        self.elem_length_func = elem_length_func

        self._group_dp = GroupByWindowDataPipe(dataset,
                                               self._element_to_bucket_id,
                                               self._window_size_func,
                                               wrapper_class=wrapper_class)

    # 该方法通过 yield from 语句返回 GroupByWindowDataPipe 对象的迭代结果。
    def __iter__(self):
        yield from self._group_dp

    # 该方法计算元素的长度，并根据 bucket_boundaries 确定元素所属的桶ID。
    def _element_to_bucket_id(self, elem):
        seq_len = self.elem_length_func(elem)
        bucket_id = 0
        for (i, b) in enumerate(self.bucket_boundaries):
            if seq_len < b:
                bucket_id = i
                break
        return bucket_id

    # 该方法根据桶ID返回对应的批次大小。
    def _window_size_func(self, bucket_id):
        return self.bucket_batch_sizes[bucket_id]


# GroupByWindowDataPipe 类继承自 datapipes.iter.Grouper，并实现了对数据集进行分组和窗口处理的功能。
@functional_datapipe("group_by_window")
class GroupByWindowDataPipe(datapipes.iter.Grouper):

    # 初始化方法 (__init__)：dataset（数据集）、key_func（用于生成键的函数）、window_size_func（用于确定窗口大小的函数）和 wrapper_class（可选，用于包装结果的类）。
    def __init__(
        self,
        dataset: IterDataPipe,
        key_func,
        window_size_func,
        wrapper_class=None,
    ):
        super().__init__(dataset,
                         key_func,
                         keep_key=False,
                         group_size=None,
                         drop_remaining=False)
        _check_unpickable_fn(window_size_func)
        self.dp = dataset
        self.window_size_func = window_size_func
        if wrapper_class is not None:
            _check_unpickable_fn(wrapper_class)
            del self.wrapper_class
            self.wrapper_class = wrapper_class

    # 迭代方法 (__iter__)：
    # 该方法遍历数据集中的每个元素，并根据 key_func 生成键。
    # 将元素按键分组存储在 buffer_elements 中，并更新当前缓冲区的大小。
    # 根据 window_size_func 计算窗口大小，并在窗口大小达到时，使用 wrapper_class 包装结果并生成输出。
    # 如果当前缓冲区大小达到最大缓冲区大小，则移除并生成最大的键对应的组。
    # 最后，处理剩余的缓冲区元素，生成并返回所有结果。
    def __iter__(self):
        for x in self.datapipe:
            key = self.group_key_fn(x)

            self.buffer_elements[key].append(x)
            self.curr_buffer_size += 1

            group_size = self.window_size_func(key)
            if group_size == len(self.buffer_elements[key]):
                result = self.wrapper_class(self.buffer_elements[key])
                yield result
                self.curr_buffer_size -= len(self.buffer_elements[key])
                del self.buffer_elements[key]

            if self.curr_buffer_size == self.max_buffer_size:
                result_to_yield = self._remove_biggest_key()
                if result_to_yield is not None:
                    result = self.wrapper_class(result_to_yield)
                    yield result

        for key in tuple(self.buffer_elements.keys()):
            result = self.wrapper_class(self.buffer_elements.pop(key))
            self.curr_buffer_size -= len(result)
            yield result


# SortDataPipe类继承自IterDataPipe，并实现了对数据进行排序的功能。
@functional_datapipe("sort")
class SortDataPipe(IterDataPipe):

    # 这是类的构造方法，用于初始化对象的属性。在这个方法中，dataset是输入的数据管道，buffer_size是缓冲区的大小，默认值为500。
    def __init__(self,
                 dataset: IterDataPipe,
                 buffer_size: int = 500,
                 key_func=None,
                 reverse=False) -> None:
        # 如果提供了key_func，则检查该函数是否可序列化（即是否可以被pickle）。
        if key_func is not None:
            _check_unpickable_fn(key_func)
        self.buffer_size = buffer_size
        super().__init__()
        self.dp = dataset
        self._buffer = []
        self.key_func = key_func
        self.reverse = reverse

    # 这个方法实现了数据管道的迭代功能。它从输入的数据管道self.dp中逐个获取元素，并将其添加到缓冲区self._buffer中。
    def __iter__(self):
        for elem in self.dp:
            self._buffer.append(elem)
            # 当缓冲区的大小达到或超过buffer_size时，缓冲区中的元素会被排序并逐个yield出来。
            if len(self._buffer) >= self.buffer_size:
                self._buffer.sort(key=self.key_func, reverse=self.reverse)
                for x in self._buffer:
                    yield x
                del self._buffer
                self._buffer = []
        # The sample left over在数据管道的所有元素都被处理完毕后，剩余的元素（如果有的话）也会被排序并yield出来。
        self._buffer.sort(key=self.key_func, reverse=self.reverse)
        for x in self._buffer:
            yield x
        del self._buffer
        self._buffer = []


# 动态地根据输入数据元素和当前缓冲区中的元素数量来创建批次（batch）
# 这个类的主要功能是根据 window_class 的判断条件，将数据分批处理，并使用 wrapper_class 将这些批次的数据包装起来。
@functional_datapipe("dynamic_batch")
class DynamicBatchDataPipe(IterDataPipe):

    def __init__(self, dataset: IterDataPipe, window_class,
                 wrapper_class) -> None:
        _check_unpickable_fn(window_class)
        _check_unpickable_fn(wrapper_class)
        super().__init__()
        self.dp = dataset
        # 用于判断是否应该将当前元素添加到缓冲区中。
        assert window_class is not None
        # 用于包装缓冲区中的数据。
        assert wrapper_class is not None
        self.window_class = window_class
        self._buffer = []
        self._wrappr_class = wrapper_class

    # 定义了如何逐个产生批次。
    def __iter__(self):
        # 它遍历输入的数据管道 self.dp。
        for elem in self.dp:
            # 如果 window_class 的实例返回 False，表示当前元素应该被添加到缓冲区中，因此它将其添加到 self._buffer。
            if not self.window_class(elem, len(self._buffer)):
                self._buffer.append(elem)
            # 如果返回 True，表示应该结束当前批次并开始一个新的批次。如果缓冲区不为空，它使用 wrapper_class 来包装缓冲区中的数据，并产生（yield）这个结果。
            # 然后，它清空缓冲区，并将当前元素作为新批次的第一个元素放入缓冲区。
            else:
                if len(self._buffer) > 0:
                    yield self._wrappr_class(self._buffer)
                del self._buffer
                self._buffer = [elem]
        # 在迭代完所有数据元素后，如果缓冲区中还有剩余的元素，它也会使用 wrapper_class 来包装这些元素，并产生结果。
        if len(self._buffer) > 0:
            yield self._wrappr_class(self._buffer)
        # 最后，它再次清空缓冲区
        del self._buffer
        self._buffer = []


# 实现了数据预取功能。它通过维护一个缓冲区来预先加载数据，以便在迭代时能够更快地提供数据。
# 具体来说，当缓冲区中的数据量低于一半时，它会从数据源中加载更多数据到缓冲区中，直到缓冲区达到其最大容量。
# 当缓冲区中的数据量超过一半时，它会从缓冲区中取出数据并逐个yield出来。
@functional_datapipe("prefetch")
class PrefetchDataPipe(IterDataPipe):
    """Performs prefetching"""

    def __init__(
        self,
        dataset: IterDataPipe,
        buffer_size: int = 500,
    ):
        # TODO(Mddct): support multiprocessing pool with shared-memory to
        #   prefetch
        # 首先调用父类的构造函数，保存输入的数据管道 dataset 到 self.dp。
        super().__init__()
        self.dp = dataset
        # 初始化一个迭代器 _iter 为 None，它将用于遍历输入的数据管道
        self._iter = None
        # 保存了预取缓冲区的大小
        self._prefetch_buffer_size = buffer_size
        # 是一个双端队列（collections.deque），它将被用作预取缓冲区。如果 buffer_size 大于 0，则创建一个具有最大长度的双端队列作为缓冲区。
        self._buffer = None
        if self._prefetch_buffer_size > 0:
            self._buffer = collections.deque(maxlen=self._prefetch_buffer_size)

    # 这个方法定义了如何逐个产生数据元素，同时实现预取逻辑。
    def __iter__(self):
        # 如果 buffer_size 大于 0，则进行预取。
        if self._prefetch_buffer_size > 0:
            # 如果 _iter 是 None，则初始化它为 self.dp 的迭代器。
            if self._iter is None:
                self._iter = iter(self.dp)
            # 确保 _buffer 不是 None（在初始化方法中已经处理了这个情况）。
            assert self._buffer is not None

            while True:
                # 使用一个外层 while True 循环来不断产生数据元素，直到没有更多的数据可以产生。
                if len(self._buffer) <= self._prefetch_buffer_size // 2:
                    while len(self._buffer) < self._prefetch_buffer_size:
                        try:
                            self._buffer.append(next(self._iter))
                        # 如果在填充缓冲区时遇到 StopIteration，并且缓冲区不为空，则逐个产生缓冲区中的元素，直到缓冲区为空。然后，将 _iter 设置为 None 并返回，表示迭代结束。
                        except StopIteration:
                            if len(self._buffer) != 0:
                                while len(self._buffer) > 0:
                                    yield self._buffer.popleft()
                            self._iter = None
                            return
                # 当缓冲区中的元素数量大于缓冲区大小的一半时，逐个产生缓冲区中的元素（从队列的左端开始），以便为新的数据元素腾出空间。
                while len(self._buffer) > self._prefetch_buffer_size // 2:
                    elem = self._buffer.popleft()
                    yield elem

        # 如果 buffer_size 不大于 0，则不进行预取，而是直接逐个产生输入数据管道中的元素。
        else:
            yield from self.dp


# 用于重复迭代一个数据集，可以控制数据集的迭代次数。如果设置为 -1，则表示无限次重复。
@functional_datapipe("repeat")
class RepeatDatapipe(IterDataPipe):

    def __init__(self, dataset: IterDataPipe, count: int = -1):
        super().__init__()
        self.dp = dataset
        self.count = count

    def __iter__(self):
        if self.count == 1:
            yield from self.dp
            return
        i = 0
        while self.count < 0 or i < self.count:
            for elem in self.dp:
                new_elem = copy.copy(elem)
                yield new_elem
            i += 1


# 用于数据分片的自定义数据管道，目的是将输入的数据管道 dataset 分成多个部分（或“分片”），以便在分布式数据并行（DDP）或其他并行处理环境中使用。
# 通过分片，可以确保每个处理单元（如GPU）接收到不同的数据子集，从而避免数据重复和竞争条件。
@functional_datapipe("shard")
class ShardDataPipe(ShardingFilterIterDataPipe):

    def __init__(self, dataset: IterDataPipe, partition: bool = False):
        # 输入的数据管道，应该是一个 IterDataPipe 对象。
        super().__init__(dataset, None)
        # 一个布尔值，指示是否启用分区模式。在分区模式下，数据将根据实例ID和实例总数进行分片。
        # 如果为 False，则使用一种不同的分片策略，其中每个GPU都获得相同的数据集（但可能以不同的顺序或批次大小）。
        self.partition = partition
        # 然后，保存 partition 标志和 dataset 到实例变量中。
        self.dp = dataset

    # 这个方法定义了如何根据给定的参数对数据进行分片。
    def apply_sharding(self, num_of_instances: int, instance_id: int,
                       sharding_group: SHARDING_PRIORITIES):
        # 如果 partition 为 True，则调用父类的 apply_sharding 方法来执行分片逻辑。
        if self.partition:
            return super().apply_sharding(num_of_instances, instance_id,
                                          sharding_group)
        # 否则：首先，尝试获取工作器信息（torch.utils.data.get_worker_info()），这在DataLoader的多进程数据加载上下文中是可用的。
        else:
            # We can not handle uneven data for CV on DDP, so we don't
            # sample data by rank, that means every GPU gets the same
            # and all the CV data
            info = torch.utils.data.get_worker_info()
            # 如果没有工作器信息（即不在多进程环境中），则假设只有一个实例，并设置 self.num_of_instances 为 1 和 self.instance_id 为 0。
            if info is None:
                self.num_of_instances = 1
                self.instance_id = 0
            # 如果有工作器信息，则根据工作器的数量和当前工作器的ID来设置 self.num_of_instances 和 self.instance_id。
            else:
                n_workers_per_device = info.num_workers
                self.num_of_instances = n_workers_per_device
                self.instance_id = info.id


# 定义了一个名为 InterleaveDataPipe 的数据管道类，用于从多个数据管道中按权重随机选择数据。
@functional_datapipe("interleave")
class InterlaveDataPipe(IterDataPipe):

    def __init__(
        self,
        source_datapipes: List[IterDataPipe],
        weights: Optional[List[float]] = None,
        seed=2027,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.source_datapipes = source_datapipes
        self.weights = weights
        if weights is None:
            self.weights = [1 / len(self.source_datapipes)] * len(
                self.source_datapipes)
        else:
            self.weights = [weight / sum(weights) for weight in weights]
        self.iters = None

    def __iter__(self):
        weights = copy.deepcopy(self.weights)
        exhausted = len(self.source_datapipes) * [False]
        if self.iters is None:
            self.iters = [(i, iter(d))
                          for i, d in enumerate(self.source_datapipes)]
        while True:
            # TODO(Mddct): rng
            index_iter = self.rng.choice(self.iters, p=weights)
            i, ite = index_iter
            try:
                elem = next(ite)
                yield elem
            except StopIteration:
                weights[i] = 0.
                exhausted[i] = True
                if all(exhausted):
                    return
                weights = [weight / sum(weights) for weight in weights]


# 实现了文件的逐行读取功能。这个类的主要作用是将多个文件中的文本行逐行读取并生成一个包含文件名和行内容的字典。
class TextLineDataPipe(IterDataPipe):
    """ Streamming Text line
    """

    def __init__(self, filenames, mode='r'):
        super().__init__()
        _dp = datapipes.iter.FileLister(filenames)
        _dp = datapipes.iter.FileOpener(_dp, mode=mode)
        self.dp = _dp

    # 遍历 self.dp 中的每个文件和文件流。
    def __iter__(self):
        for fname, stream in self.dp:
            for line in stream:
                # 对于每个文件，逐行读取文件内容，并去除每行末尾的换行符。
                line = line.strip('\n')
                # 将文件名和行内容打包成一个字典 {"file_name": fname, "line": line}，并通过 yield 关键字返回。
                yield {"file_name": fname, "line": line}
            # 最后，关闭当前文件流。
            stream.close()


# 用于处理 Wenet 框架中的 tar 文件。
# 这个类能够迭代地解码 tar 文件中的内容，并根据文件扩展名提取出文本（.txt）和音频（由 AUDIO_FORMAT_SETS 指定的格式）数据，然后将这些数据以字典的形式产出。
@functional_datapipe("tar_file_and_group")
class TarsDataPipe(IterDataPipe):
    """ Decode wenet's tar , yield {'txt': "...", "raw": "..."}
    """

    # 初始化
    def __init__(self, dataset: IterDataPipe) -> None:
        super().__init__()
        self.dp = dataset

    # 如何迭代 TarsDataPipe 实例以产生数据。
    def __iter__(self):
        # 从 wenet.dataset.processor 导入 AUDIO_FORMAT_SETS，这是一个包含有效音频文件扩展名的集合。
        from wenet.dataset.processor import AUDIO_FORMAT_SETS
        for sample in self.dp:
            # 断言样本字典中包含 'file_name', 'line', 和 'stream' 键。
            assert 'file_name' in sample
            assert 'line' in sample
            assert 'stream' in sample
            try:
                # 尝试使用 tarfile.open 打开 'stream' 指向的 tar 文件（这里使用 fileobj=sample['stream'] 参数将文件流对象直接传递给 tarfile.open）。
                with tarfile.open(fileobj=sample['stream'],
                                  mode="r:*") as stream:
                    prev_prefix = None
                    example = {
                        'file_name': sample['file_name'],
                        'tar_file_name': sample['line']
                    }
                    valid = True
                    # 遍历 tar 文件中的每个条目（tarinfo 对象）：
                    for tarinfo in stream:
                        # 提取文件名（name），并找到最后一个点（.）的位置来分割前缀（prefix）和后缀（postfix）。
                        name = tarinfo.name
                        pos = name.rfind('.')
                        assert pos > 0
                        prefix, postfix = name[:pos], name[pos + 1:]
                        # 如果当前条目的前缀与前一个条目的前缀不同，且前一个条目有效，则产出前一个条目的字典（example），并开始一个新的字典。
                        if prev_prefix is not None and prefix != prev_prefix:
                            example['key'] = prev_prefix
                            if valid:
                                yield example
                            example = {
                                'file_name': sample['file_name'],
                                'tar_file_name': sample['line']
                            }
                            valid = True
                        # 根据文件后缀处理文件内容：
                        with stream.extractfile(tarinfo) as file_obj:
                            try:
                                if postfix == 'txt':
                                    example['txt'] = file_obj.read().decode(
                                        'utf8').strip()
                                elif postfix in AUDIO_FORMAT_SETS:
                                    example['wav'] = file_obj.read()
                                else:
                                    example[postfix] = file_obj.read()
                            # 如果在处理文件内容时发生异常，则记录警告，并将 valid 标志设置为 False。
                            except Exception as ex:
                                valid = False
                                logging.warning(
                                    'error to parse {}'.format(name))
                            prev_prefix = prefix
                    # 在处理完 tar 文件中的所有条目后，如果最后一个条目的前缀有效，则产出该条目的字典。
                    if prev_prefix is not None:
                        example['key'] = prev_prefix
                        yield example
            # 如果在处理 tar 文件时发生异常，则记录警告。
            except Exception as ex:
                msg = 'In tar_file_and_group: {} when processing {}'.format(
                    ex, sample['line'])
                logging.warning(msg)
            # 最后，如果样本字典中包含 'process' 键（这在这个类的上下文中并不明确，可能是用于处理 tar 文件之前的某个步骤的子进程），
            # 则调用 communicate() 方法来等待子进程完成，并关闭 'stream'。
            finally:
                if 'process' in sample:
                    sample['process'].communicate()
                sample['stream'].close()


# WenetRawDatasetSource类是一个用于从文件中读取文本行，并提供一个迭代器来遍历这些行的数据管道源。
# 它通过预先读取和分片技术来提高数据处理的效率。
# 这样的类在处理大规模文本数据，特别是在语音识别（如Wenet项目）等需要高效数据处理的场景中非常有用。
class WenetRawDatasetSource(IterDataPipe):

    def __init__(self,
                 filenames: str,
                 prefetch: int = 500,
                 partition: bool = True,
                 shuffle: bool = False,
                 shuffle_size: int = 10000,
                 cycle: int = 1) -> None:
        super().__init__()
        self.dp = TextLineDataPipe(filenames)
        if shuffle:
            self.dp = self.dp.shuffle(buffer_size=shuffle_size)
        self.dp = self.dp.repeat(cycle).prefetch(prefetch)
        self.dp = self.dp.shard(partition)

    def __iter__(self):
        for d in self.dp:
            yield d


# 处理存储在TAR归档文件中的文本数据，并允许用户通过迭代来访问这些数据。
class WenetTarShardDatasetSource(IterDataPipe):

    def __init__(self,
                 filenames: str,
                 prefetch: int = 500,
                 partition: bool = True,
                 shuffle: bool = False,
                 shuffle_size: int = 10000,
                 cycle: int = 1) -> None:
        super().__init__()
        self.dp = TextLineDataPipe(filenames)
        if shuffle:
            self.dp = self.dp.shuffle(buffer_size=shuffle_size)
        self.dp = self.dp.repeat(cycle)
        self.dp = self.dp.shard(partition).map_ignore_error(
            parse_url).tar_file_and_group().prefetch(prefetch)

    def __iter__(self):
        for d in self.dp:
            yield d


# 总结：数据处理和管道管理。