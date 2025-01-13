# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple

import numpy as np

import torch
import torchaudio.functional as F


# 实现了一些处理 CTC（Connectionist Temporal Classification）模型输出的功能，包括去除重复元素、生成时间戳、强制对齐等

# 去除输入序列中的重复元素和空白元素。
def remove_duplicates_and_blank(hyp: List[int],
                                blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


# 将重复元素替换为一个空白元素。
def replace_duplicates_with_blank(hyp: List[int],
                                  blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        new_hyp.append(hyp[cur])
        prev = cur
        cur += 1
        while cur < len(
                hyp) and hyp[cur] == hyp[prev] and hyp[cur] != blank_id:
            new_hyp.append(blank_id)
            cur += 1
    return new_hyp


# 生成 CTC 解码过程中非空白符号的时间点。
def gen_ctc_peak_time(hyp: List[int], blank_id: int = 0) -> List[int]:
    times = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            times.append(cur)
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return times


# 生成 CTC 输出中非空白元素的时间戳。
def gen_timestamps_from_peak(
    peaks: List[int],
    max_duration: float,
    frame_rate: float = 0.04,
    max_token_duration: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Args:
        peaks: ctc peaks time stamp
        max_duration: max_duration of the sentence
        frame_rate: frame rate of every time stamp, in seconds
        max_token_duration: max duration of the token, in seconds
    Returns:
        list(start, end) of each token
    """
    times = []
    half_max = max_token_duration / 2
    for i in range(len(peaks)):
        if i == 0:
            start = max(0, peaks[0] * frame_rate - half_max)
        else:
            start = max((peaks[i - 1] + peaks[i]) / 2 * frame_rate,
                        peaks[i] * frame_rate - half_max)

        if i == len(peaks) - 1:
            end = min(max_duration, peaks[-1] * frame_rate + half_max)
        else:
            end = min((peaks[i] + peaks[i + 1]) / 2 * frame_rate,
                      peaks[i] * frame_rate + half_max)
        times.append((start, end))
    return times


# 在每两个标签 token 之间插入空白 token。
def insert_blank(label, blank_id=0):
    """Insert blank token between every two label token."""
    label = np.expand_dims(label, 1)
    blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
    label = np.concatenate([blanks, label], axis=1)
    label = label.reshape(-1)
    label = np.append(label, label[0])
    return label


# 实现 CTC 强制对齐。
def force_align(ctc_probs: torch.Tensor, y: torch.Tensor, blank_id=0) -> list:
    """ctc forced alignment.

    Args:
        torch.Tensor ctc_probs: hidden state sequence, 2d tensor (T, D)
        torch.Tensor y: id sequence tensor 1d tensor (L)
        int blank_id: blank symbol index
    Returns:
        torch.Tensor: alignment result
    """
    ctc_probs = ctc_probs[None].cpu()
    y = y[None].cpu()
    alignments, _ = F.forced_align(ctc_probs, y, blank=blank_id)
    return alignments[0]


# 从配置和符号表中获取空白 ID。
# 该函数的目的是从符号表中提取 <blank> 对应的 ID 并确保配置中正确保存了 ctc_blank_id，
# 如果符号表中没有 <blank>，则依赖 configs['ctc_conf']['ctc_blank_id'] 中的预设值。
# 如果符号表和配置中都有 <blank>，它们的值必须保持一致，否则程序会抛出异常。
def get_blank_id(configs, symbol_table):
    if 'ctc_conf' not in configs:
        configs['ctc_conf'] = {}

    if '<blank>' in symbol_table:
        if 'ctc_blank_id' in configs['ctc_conf']:
            assert configs['ctc_conf']['ctc_blank_id'] == symbol_table[
                '<blank>']
        else:
            configs['ctc_conf']['ctc_blank_id'] = symbol_table['<blank>']
    else:
        assert 'ctc_blank_id' in configs[
            'ctc_conf'], "PLZ set ctc_blank_id in yaml"

    return configs, configs['ctc_conf']['ctc_blank_id']

# 总结：实现了一些处理 CTC（Connectionist Temporal Classification）模型输出的功能，包括去除重复元素、生成时间戳、强制对齐等