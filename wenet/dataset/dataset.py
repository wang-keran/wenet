# Copyright (c) 2021 Wenet Community. (authors: Binbin Zhang)
#               2023 Wenet Community. (authors: Dinghao Zhou)
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

from functools import partial
import sys
from typing import Optional
from wenet.dataset import processor
from wenet.dataset.datapipes import (WenetRawDatasetSource,
                                     WenetTarShardDatasetSource)
from wenet.text.base_tokenizer import BaseTokenizer
from wenet.utils.file_utils import read_symbol_table


def Dataset(data_type,
            data_list_file,
            tokenizer: Optional[BaseTokenizer] = None,
            conf=None,
            partition=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            tokenizer (BaseTokenizer or None): tokenizer to tokenize
            partition(bool): whether to do data partition in terms of rank
    """
    assert conf is not None
    assert data_type in ['raw', 'shard']
    # 根据 data_type，使用 WenetRawDatasetSource 或 WenetTarShardDatasetSource 来加载数据。
    # cycle dataset，循环次数
    cycle = conf.get('cycle', 1)
    # stage1 shuffle: source，将列表全部打乱，让数据有随机性，避免模型过拟合
    list_shuffle = conf.get('list_shuffle', True)
    list_shuffle_size = sys.maxsize
    if list_shuffle:
        list_shuffle_conf = conf.get('list_shuffle_conf', {})
        list_shuffle_size = list_shuffle_conf.get('shuffle_size',
                                                  list_shuffle_size)
    # 根据 data_type，使用 WenetRawDatasetSource 或 WenetTarShardDatasetSource 来加载数据。
    if data_type == 'raw':
        dataset = WenetRawDatasetSource(data_list_file,
                                        partition=partition,
                                        shuffle=list_shuffle,
                                        shuffle_size=list_shuffle_size,
                                        cycle=cycle)
        # 数据集首先通过 map 方法应用 processor.parse_json 函数，对数据进行解析。
        dataset = dataset.map(processor.parse_json)
    else:
        dataset = WenetTarShardDatasetSource(data_list_file,
                                             partition=partition,
                                             shuffle=list_shuffle,
                                             shuffle_size=list_shuffle_size,
                                             cycle=cycle)
    # 然后通过 map_ignore_error 方法应用 processor.decode_wav 函数，解码音频文件。
    dataset = dataset.map_ignore_error(processor.decode_wav)

    # 对数据集中的每个元素应用一个处理函数 processor.singal_channel，并传递特定的配置参数 singal_channel_conf。
    singal_channel_conf = conf.get('singal_channel_conf', {})
    dataset = dataset.map(
        partial(processor.singal_channel, **singal_channel_conf))

    # 如果配置中包含 speaker_conf，则读取 speaker_table_path 并应用 processor.parse_speaker 函数。
    speaker_conf = conf.get('speaker_conf', None)
    if speaker_conf is not None:
        assert 'speaker_table_path' in speaker_conf
        speaker_table = read_symbol_table(speaker_conf['speaker_table_path'])
        dataset = dataset.map(
            partial(processor.parse_speaker, speaker_dict=speaker_table))

    # 如果 tokenizer 不为空，则应用 processor.tokenize 函数对数据进行分词处理。
    if tokenizer is not None:
        dataset = dataset.map(partial(processor.tokenize, tokenizer=tokenizer))

    # 根据配置中的 filter_conf，应用 processor.filter 函数对数据进行过滤。
    filter_conf = conf.get('filter_conf', {})
    dataset = dataset.filter(partial(processor.filter, **filter_conf))

    # 根据配置中的 resample_conf，应用 processor.resample 函数对数据进行重采样
    resample_conf = conf.get('resample_conf', {})
    dataset = dataset.map(partial(processor.resample, **resample_conf))

    # 如果配置中的 speed_perturb 为 True，则应用 processor.speed_perturb 函数对数据进行速度扰动。
    speed_perturb = conf.get('speed_perturb', False)
    if speed_perturb:
        dataset = dataset.map(partial(processor.speed_perturb))

    # 根据配置中的 feats_type，选择不同的特征提取方法（如 'fbank', 'mfcc', 'log_mel_spectrogram'）。
    feats_type = conf.get('feats_type', 'fbank')
    assert feats_type in ['fbank', 'mfcc', 'log_mel_spectrogram']
    # 特征提取
    if feats_type == 'fbank':
        fbank_conf = conf.get('fbank_conf', {})
        dataset = dataset.map(partial(processor.compute_fbank, **fbank_conf))
    elif feats_type == 'mfcc':
        mfcc_conf = conf.get('mfcc_conf', {})
        dataset = dataset.map(partial(processor.compute_mfcc, **mfcc_conf))
    elif feats_type == 'log_mel_spectrogram':
        log_mel_spectrogram_conf = conf.get('log_mel_spectrogram_conf', {})
        dataset = dataset.map(
            partial(processor.compute_log_mel_spectrogram,
                    **log_mel_spectrogram_conf))
    spec_aug = conf.get('spec_aug', True)
    spec_sub = conf.get('spec_sub', False)
    spec_trim = conf.get('spec_trim', False)
    # 如果启用了频谱增强，则使用 processor.spec_aug 进行增强。
    if spec_aug:
        spec_aug_conf = conf.get('spec_aug_conf', {})
        dataset = dataset.map(partial(processor.spec_aug, **spec_aug_conf))
    # 可选地进行频谱替换（spec_sub）和频谱修剪（spec_trim）。
    if spec_sub:
        spec_sub_conf = conf.get('spec_sub_conf', {})
        dataset = dataset.map(partial(processor.spec_sub, **spec_sub_conf))
    # 可选地进行频谱替换（spec_sub）和频谱修剪（spec_trim）。
    if spec_trim:
        spec_trim_conf = conf.get('spec_trim_conf', {})
        dataset = dataset.map(partial(processor.spec_trim, **spec_trim_conf))

    # 使用 processor.detect_language 检测文本的语言。（可选）
    language_conf = conf.get('language_conf', {"limited_langs": ['zh', 'en']})
    dataset = dataset.map(partial(processor.detect_language, **language_conf))
    # 使用 processor.detect_task 检测任务类型（如语音识别、关键词检测等）。（可选）
    dataset = dataset.map(processor.detect_task)

    # 如果启用了打乱，则使用 dataset.shuffle 随机打乱数据。（可选）
    shuffle = conf.get('shuffle', True)
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = dataset.shuffle(buffer_size=shuffle_conf['shuffle_size'])

    # 如果启用了排序，则根据特征对数据进行排序。（可选）
    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = dataset.sort(buffer_size=sort_conf['sort_size'],
                               key_func=processor.sort_by_feats)

    # 根据配置进行静态批量处理、桶批量处理或动态批量处理。
    batch_conf = conf.get('batch_conf', {})
    batch_type = batch_conf.get('batch_type', 'static')
    assert batch_type in ['static', 'bucket', 'dynamic']
    if batch_type == 'static':
        assert 'batch_size' in batch_conf
        batch_size = batch_conf.get('batch_size', 16)
        dataset = dataset.batch(batch_size, wrapper_class=processor.padding)
    elif batch_type == 'bucket':
        assert 'bucket_boundaries' in batch_conf
        assert 'bucket_batch_sizes' in batch_conf
        dataset = dataset.bucket_by_sequence_length(
            processor.feats_length_fn,
            batch_conf['bucket_boundaries'],
            batch_conf['bucket_batch_sizes'],
            wrapper_class=processor.padding)
    else:
        max_frames_in_batch = batch_conf.get('max_frames_in_batch', 12000)
        dataset = dataset.dynamic_batch(
            processor.DynamicBatchWindow(max_frames_in_batch),
            wrapper_class=processor.padding,
        )

    # 返回处理后的数据集对象。
    return dataset

# 这个函数展示了在语音识别任务中，从原始数据到最终用于模型训练的数据集，需要经过的一系列复杂而精细的预处理步骤。
# 通过配置字典 conf，可以灵活地调整每个步骤的参数，以适应不同的数据和任务需求。