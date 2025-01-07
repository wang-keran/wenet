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
#已废弃

import copy
import librosa
import logging
import json
import random
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from wenet.text.base_tokenizer import BaseTokenizer

torchaudio.utils.sox_utils.set_buffer_size(16500)

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


# 按照url打开网络文件
def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'wget -q -O - {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


# 函数会读取每个 tar 文件中的内容，并根据文件名的前缀进行分组，然后将每个分组的内容输出。
def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = None
        try:
            stream = tarfile.open(fileobj=sample['stream'], mode="r:*")
            prev_prefix = None
            example = {}
            valid = True
            for tarinfo in stream:
                name = tarinfo.name
                pos = name.rfind('.')
                assert pos > 0
                prefix, postfix = name[:pos], name[pos + 1:]
                if prev_prefix is not None and prefix != prev_prefix:
                    example['key'] = prev_prefix
                    if valid:
                        yield example
                    example = {}
                    valid = True
                with stream.extractfile(tarinfo) as file_obj:
                    try:
                        if postfix == 'txt':
                            example['txt'] = file_obj.read().decode(
                                'utf8').strip()
                        elif postfix in AUDIO_FORMAT_SETS:
                            waveform, sample_rate = torchaudio.load(file_obj)
                            example['wav'] = waveform
                            example['sample_rate'] = sample_rate
                        else:
                            example[postfix] = file_obj.read()
                    except Exception as ex:
                        valid = False
                        logging.warning('error to parse {}'.format(name))
                prev_prefix = prefix
            if prev_prefix is not None:
                example['key'] = prev_prefix
                yield example
        except Exception as ex:
            logging.warning(
                'In tar_file_and_group: {} when processing {}'.format(
                    ex, sample['src']))
        finally:
            if stream is not None:
                stream.close()
            if 'process' in sample:
                sample['process'].communicate()
            sample['stream'].close()


# 从 JSON 数据中解析出音频文件的路径、文本内容以及采样率，并返回一个包含这些信息的字典。
def parse_raw(data):
    """ Parse key/wav/txt from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/txt

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'wav' in obj
        assert 'txt' in obj
        key = obj['key']
        wav_file = obj['wav']
        txt = obj['txt']
        try:
            if 'start' in obj:
                assert 'end' in obj
                sample_rate = torchaudio.info(wav_file).sample_rate
                start_frame = int(obj['start'] * sample_rate)
                end_frame = int(obj['end'] * sample_rate)
                waveform, _ = torchaudio.load(filepath=wav_file,
                                              num_frames=end_frame -
                                              start_frame,
                                              frame_offset=start_frame)
            else:
                waveform, sample_rate = torchaudio.load(wav_file)
            example = copy.deepcopy(obj)  # copy and keep all the fields
            example['wav'] = waveform  # overwrite wav
            example['sample_rate'] = sample_rate
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))


# parse_speaker 函数的主要功能是读取一个包含说话人信息的文件，并将这些信息映射到一个字典中。
# 然后，该函数遍历输入的数据集，将每个样本中的说话人信息替换为对应的整数值。
def parse_speaker(data, speaker_table_path):
    speaker_dict = {}
    with open(speaker_table_path, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            speaker_dict[arr[0]] = int(arr[1])
    for sample in data:
        assert 'speaker' in sample
        speaker = sample['speaker']
        sample['speaker'] = speaker_dict.get(speaker, 0)
        yield sample


# 过滤器，特征长度小于min_length（以10ms为单位），大于max_length（以10ms为单位），丢弃
# 标签长度小于token_min_length，则丢弃该样本。标签长度大于token_max_length，则丢弃该样本。
def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'label' in sample
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        if len(sample['label']) < token_min_length:
            continue
        if len(sample['label']) > token_max_length:
            continue
        if num_frames != 0:
            if len(sample['label']) / num_frames < min_output_input_ratio:
                continue
            if len(sample['label']) / num_frames > max_output_input_ratio:
                continue
        yield sample


# 将音频数据的采样率重置为目标采样率
def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


# 在给定的代码中，speed_perturb 函数用于对音频数据进行速度扰动。
# 该函数通过调整音频信号的播放速度来增加数据的多样性，从而提高语音识别模型的鲁棒性。
# 具体来说，函数会随机选择一个速度因子（如0.9、1.0或1.1），并使用 torchaudio.sox_effects.apply_effects_tensor 方法来调整音频信号的速度。
# 如果选择的速度因子不是1.0，则会重新采样音频信号以匹配原始采样率。
def speed_perturb(data, speeds=None):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            speeds(List[float]): optional speed

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed = random.choice(speeds)
        if speed != 1.0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
            sample['wav'] = wav

        yield sample


# 计算fbank,首先遍历 data 中的每个样本,对于每个样本，函数会检查是否包含 'sample_rate'、'wav'、'key' 和 'label' 这四个键。
# 然后，函数将音频波形数据 waveform 乘以 (1 << 15)，这是将音频数据从 int16 格式转换为 float32 格式的标准做法。
# 接着，函数使用 kaldi.fbank 函数提取 FBank 特征，并将结果存储在 mat 中。
# 最后，函数将提取的特征 mat 添加到样本字典中，并通过 yield 返回更新后的样本。
def compute_fbank(data,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          energy_floor=0.0,
                          sample_frequency=sample_rate)
        sample['feat'] = mat
        yield sample


# 计算从音频数据中提取梅尔频率倒谱系数（MFCC）
def compute_mfcc(data,
                 num_mel_bins=23,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.0,
                 num_ceps=40,
                 high_freq=0.0,
                 low_freq=20.0):
    """ Extract mfcc

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        # 对输入数据进行验证，确保每个样本包含 sample_rate, wav, key, label 这四个键。
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        # 波形处理: 将音频波形乘以 (1 << 15)，这一步可能是为了调整波形的幅度。
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        # MFCC计算
        mat = kaldi.mfcc(waveform,
                         num_mel_bins=num_mel_bins,
                         frame_length=frame_length,
                         frame_shift=frame_shift,
                         dither=dither,
                         num_ceps=num_ceps,
                         high_freq=high_freq,
                         low_freq=low_freq,
                         sample_frequency=sample_rate)
        sample['feat'] = mat
        # 过 yield 返回处理后的样本。
        yield sample


# 从音频数据中计算对数梅尔频谱图（Log Mel Spectrogram）。
def compute_log_mel_spectrogram(data,
                                n_fft=400,
                                hop_length=160,
                                num_mel_bins=80,
                                padding=0):
    """ Extract log mel spectrogram, modified from openai-whisper, see:
        - https://github.com/openai/whisper/blob/main/whisper/audio.py
        - https://github.com/wenet-e2e/wenet/pull/2141#issuecomment-1811765040

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        # 输入数据验证
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        # 波形数据处理
        waveform = sample['wav'].squeeze(0)  # (channel=1, sample) -> (sample,)
        # 如果设置了 padding，则使用 torch.nn.functional.pad 对波形数据进行填充。填充的方式是将波形数据在末尾添加 padding 个零。
        if padding > 0:
            waveform = F.pad(waveform, (0, padding))
        window = torch.hann_window(n_fft)
        # 短时傅里叶变换（STFT）
        stft = torch.stft(waveform,
                          n_fft,
                          hop_length,
                          window=window,
                          return_complex=True)
        magnitudes = stft[..., :-1].abs()**2

        # Mel频谱图计算
        filters = torch.from_numpy(
            librosa.filters.mel(sr=sample_rate,
                                n_fft=n_fft,
                                n_mels=num_mel_bins))
        mel_spec = filters @ magnitudes

        # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
        # 对数变换
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        # 最大值限制在 log_spec.max() - 8.0，然后进行归一化处理，使得最大值为 4.0。
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        # 最后，将归一化后的频谱图加上 4.0 并除以 4.0，得到最终的对数梅尔频谱图。
        log_spec = (log_spec + 4.0) / 4.0
        # 将计算得到的对数梅尔频谱图 log_spec 转置为 (n_frames, num_mel_bins) 的形状，并将其存储在样本的 feat 字段中。
        sample['feat'] = log_spec.transpose(0, 1)
        # 最后，函数返回处理后的样本。
        yield sample


# 文本数据进行分词处理，并将分词结果和标签添加到原始数据中
def tokenize(data, tokenizer: BaseTokenizer):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    for sample in data:
        # 首先检查 'txt' 字段是否存在。
        assert 'txt' in sample
        # 调用 tokenizer.tokenize(sample['txt']) 对文本进行分词，返回分词结果 tokens 和标签 label。
        tokens, label = tokenizer.tokenize(sample['txt'])
        # 将分词结果 tokens 和标签 label 分别存储在样本的 'tokens' 和 'label' 字段中。
        sample['tokens'] = tokens
        sample['label'] = label
        # 使用 yield 关键字返回处理后的样本。
        yield sample


# 实现了SpecAugment技术，用于对音频数据进行增强。
def spec_aug(data, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        # 首先检查是否包含feat字段，并确保其为torch.Tensor类型。
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        # 创建一个与输入特征相同形状的副本y，并进行掩码操作。
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask时间掩码：随机选择起始帧和掩码长度，将该区间内的所有帧设置为0。
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask频率掩码：随机选择起始频率和掩码长度，将该区间内的所有频率设置为0。
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        # 最后，将处理后的特征替换回原始样本，并返回处理后的样本。
        sample['feat'] = y
        yield sample


# 对数据进行特殊替换操作。具体来说，它会对每个样本的特征（feat）进行随机的时间替换
def spec_sub(data, max_t=20, num_t_sub=3):
    """ Do spec substitute
        Inplace operation
        ref: U2++, section 3.2.3 [https://arxiv.org/abs/2106.05642]

        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        # 首先检查样本中是否包含特征（feat），并且特征必须是 torch.Tensor 类型。
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        # 创建特征的副本，这样可以确保在不共享内存的情况下进行操作，并且不会影响原始数据。
        y = x.clone().detach()
        # 计算特征的最大帧数（max_frames）
        max_frames = y.size(0)
        for i in range(num_t_sub):
            # 对于每个时间替换操作，随机选择一个起始位置（start）和一个长度（length），并计算结束位置（end）。
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            # only substitute the earlier time chosen randomly for current time
            # 随机选择一个位置（pos），将特征从起始位置到结束位置的值替换为从该位置向前偏移 pos 的值。
            pos = random.randint(0, start)
            # 将修改后的特征赋值回样本的特征字段，并生成修改后的样本。
            y[start:end, :] = x[start - pos:end - pos, :]
        sample['feat'] = y
        yield sample


# 用于对数据进行尾部帧的修剪操作。具体来说，它会随机选择一个长度（从1到 max_t），然后从数据的尾部截取相应长度的帧，并将剩余部分保留。截取的单位是帧
def spec_trim(data, max_t=20):
    """ Trim tailing frames. Inplace operation.
        ref: TrimTail [https://arxiv.org/abs/2211.00522]

        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of length trimming

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        # 首先检查样本中是否包含 feat 键，并且 feat 是一个 torch.Tensor 类型。
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        # 使用 random.randint(1, max_t) 随机生成一个整数 length，这个整数表示要修剪的尾部帧数。
        max_frames = x.size(0)
        # 如果生成的 length 小于 max_frames 的一半，则执行修剪操作：创建一个新的张量 y，它是原特征张量去掉尾部 length 帧后的结果。
        length = random.randint(1, max_t)
        if length < max_frames / 2:
            y = x.clone().detach()[:max_frames - length]
            sample['feat'] = y
        # 将修剪后的特征张量 y 赋值回样本的 feat 键。
        yield sample


# 对输入的数据进行局部洗牌。
# 具体来说，它会将数据分成多个缓冲区（buffer），每个缓冲区的大小由 shuffle_size 参数决定。当
# 缓冲区达到指定大小时，函数会使用 random.shuffle 方法对缓冲区中的数据进行随机打乱，然后逐个输出缓冲区中的数据。
# 最后，函数还会对剩余的数据进行一次随机打乱并输出。
def shuffle(data, shuffle_size=10000):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    # 代码首先创建一个空列表 buf 用于存储数据。
    buf = []
    for sample in data:
        buf.append(sample)
        # 对于输入数据中的每个样本，将其添加到 buf 中。
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    # 当 buf 的长度达到 shuffle_size 时，使用 random.shuffle 对 buf 进行随机打乱，然后逐个输出 buf 中的数据，并清空 buf。
    random.shuffle(buf)
    for x in buf:
        yield x


# 具体是根据每个样本的特征长度（feat）进行排序。
# 使用了yield关键字来逐个返回排序后的样本，而不是一次性返回所有样本。这种设计使得函数可以处理大数据集时不会占用过多内存，因为每次只处理并返回一个样本。
def sort(data, sort_size=500):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            # 当缓冲区大小达到sort_size时，对缓冲区内的样本按特征长度进行排序
            buf.sort(key=lambda x: x['feat'].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['feat'].size(0))
    for x in buf:
        yield x


# 将数据进行固定大小分批
def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


# 动态分批，直到每批数据的总帧数达到 max_frames_in_batch。
# 它不是简单地按照样本数量来分批，而是根据批次中所有样本的特征（feat）帧总数来分批，直到达到或超过指定的最大帧数max_frames_in_batch。
def dynamic_batch(data, max_frames_in_batch=12000):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    # 初始化变量，用于存储当前批次的数据
    buf = []
    # 记录当前批次中最长的帧数
    longest_frames = 0
    # 遍历数据
    for sample in data:
        # 对于每个样本，首先检查样本中是否包含 'feat' 键，并且 'feat' 的值是一个 torch.Tensor。
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        # 计算当前样本的帧数 new_sample_frames。
        new_sample_frames = sample['feat'].size(0)
        # 更新 longest_frames，使其保持为当前批次中最长的帧数。
        longest_frames = max(longest_frames, new_sample_frames)
        # 计算如果将当前样本添加到 buf 后的总帧数 frames_after_padding。
        frames_after_padding = longest_frames * (len(buf) + 1)
        # 如果 frames_after_padding 超过了 max_frames_in_batch，则生成当前批次 buf，并将 buf 重置为包含当前样本的新批次，同时更新 longest_frames。
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        # 否则，将当前样本添加到 buf 中。
        else:
            buf.append(sample)
    # 循环结束后，如果 buf 中还有数据，则生成最后一个批次。
    if len(buf) > 0:
        yield buf


# 选择不同分批方式进行处理
def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000):
    # batch_type：分批处理的类型，可以是 'static'（静态）或 'dynamic'（动态），默认为 'static'。
    # batch_size：当 batch_type 为 'static' 时，每个批次的大小，默认为 16。
    # max_frames_in_batch：当 batch_type 为 'dynamic' 时，每个批次中的最大帧数，默认为 12000。
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_frames_in_batch)
    else:
        # 如果 batch_type 不是 'static' 或 'dynamic' 中的任何一个，则记录一条致命日志（logging.fatal），指出不支持的批处理类型。
        logging.fatal('Unsupported batch type {}'.format(batch_type))


# 将不同长度的序列数据填充成固定长度的批次数据，以便于训练模型。
def padding(data):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        # 首先，函数遍历输入的 data，确保每个样本都是一个列表。
        assert isinstance(sample, list)
        # 计算每个样本中特征（feat）的长度，并将其存储在 feats_length 中。
        feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                    dtype=torch.int32)
        # 使用 torch.argsort 对特征长度进行降序排序，得到排序后的索引 order。
        order = torch.argsort(feats_length, descending=True)
        # 根据排序后的索引 order，重新排列特征、标签、键和音频数据。
        feats_lengths = torch.tensor(
            [sample[i]['feat'].size(0) for i in order], dtype=torch.int32)
        sorted_feats = [sample[i]['feat'] for i in order]
        sorted_keys = [sample[i]['key'] for i in order]
        sorted_labels = [
            torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order
        ]
        sorted_wavs = [sample[i]['wav'].squeeze(0) for i in order]
        # 计算排序后标签和音频数据的长度，并将其存储在 label_lengths 和 wav_lengths 中。
        label_lengths = torch.tensor([x.size(0) for x in sorted_labels],
                                     dtype=torch.int32)
        wav_lengths = torch.tensor([x.size(0) for x in sorted_wavs],
                                   dtype=torch.int32)

        # 使用 pad_sequence 函数对特征、标签和音频数据进行填充，填充值分别为 0 和 -1。
        # 填充后的数据分别存储在 padded_feats、padding_labels 和 padded_wavs 中。
        padded_feats = pad_sequence(sorted_feats,
                                    batch_first=True,
                                    padding_value=0)
        padding_labels = pad_sequence(sorted_labels,
                                      batch_first=True,
                                      padding_value=-1)
        padded_wavs = pad_sequence(sorted_wavs,
                                   batch_first=True,
                                   padding_value=0)
        # 将填充后的数据和相应的长度信息存储在一个字典 batch 中，返回该字典。
        batch = {
            "keys": sorted_keys,
            "feats": padded_feats,
            "target": padding_labels,
            "feats_lengths": feats_lengths,
            "target_lengths": label_lengths,
            "pcm": padded_wavs,
            "pcm_length": wav_lengths,
        }
        if 'speaker' in sample[0]:
            speaker = torch.tensor([sample[i]['speaker'] for i in order],
                                   dtype=torch.int32)
            batch['speaker'] = speaker
        yield batch
