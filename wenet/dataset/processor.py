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

import io
import json
from subprocess import PIPE, Popen
from urllib.parse import urlparse
from langid.langid import LanguageIdentifier, model
import logging
import librosa
import random

import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
from wenet.text.base_tokenizer import BaseTokenizer

torchaudio.utils.sox_utils.set_buffer_size(16500)

#设置音频种类
AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

# 创建一个语言标识器对象 lid，该对象使用 model 模型进行语言识别，并且启用了归一化概率（norm_probs=True）。
# 这意味着在识别语言时，返回的概率值会被归一化处理，使得所有语言的概率总和为1。
lid = LanguageIdentifier.from_modelstring(model, norm_probs=True)

# 设置名为 'langid' 的日志记录器（Logger）的日志级别为 INFO，低于INFO的日志不读取
logging.getLogger('langid').setLevel(logging.INFO)

import os
try:
    cpu_info = os.popen("lscpu | grep 'Vendor ID'").read()
    # 0x48 --> HiSilicon
    if (cpu_info.rstrip().split(" ")[-1] == "0x48"):
        # NOTE (MengqingCao): set number of threads in the subprocesses to 1
        # Why? There may be some operators ultilizing multi-threads in processor,
        # causing possibly deadlock in Kunpeng.
        # Similar issue in PyTorch: https://github.com/pytorch/pytorch/issues/45198
        torch.set_num_threads(1)
except Exception as ex:
    logging.warning('Failed to set number of thread in Kunpeng, \
        this may cause segmentfault while dataloading, \
        ignore this warning if you are not using Kunpeng')


# 定义异常类，返回错误信息
class UrlOpenError(Exception):

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self.err_msg = msg

    def __str__(self) -> str:
        return self.err_msg


# 确保了JSON字符串被正确解析为Python字典，并且额外的信息（如文件名）被添加到字典中。
def parse_json(elem):
    line = elem['line']
    obj = json.loads(line)
    obj['file_name'] = elem['file_name']
    return dict(obj)


# 解析一个 URL 并返回一个包含文件流的字典。
def parse_url(elem):
    assert 'file_name' in elem
    assert 'line' in elem
    assert isinstance(elem, dict)
    url = elem['line']
    try:
        pr = urlparse(url)
        # local file
        if pr.scheme == '' or pr.scheme == 'file':
            stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
        else:
            cmd = f'wget -q -O - {url}'
            process = Popen(cmd, shell=True, stdout=PIPE)
            elem.update(process=process)
            stream = process.stdout
        elem.update(stream=stream)
        return elem
    except Exception as ex:
        err_msg = 'Failed to open {}'.format(url)
        raise UrlOpenError(err_msg) from ex


# 解析一个包含 speaker（说话人） 键的字典，并将其值替换为 speaker_dict 中对应的值。如果 speaker 键在 speaker_dict 中不存在，则将其值设置为 0。
def parse_speaker(sample, speaker_dict):
    assert 'speaker' in sample
    speaker = sample['speaker']
    sample['speaker'] = speaker_dict.get(speaker, 0)
    return sample


# 识别语言，将检测结果添加到样本字典中。
def detect_language(sample, limited_langs):
    assert 'txt' in sample
    # NOTE(xcsong): Because language classification may not be very accurate
    #   (for example, Chinese being classified as Japanese), our workaround,
    #   given we know for certain that the training data only consists of
    #   Chinese and English, is to limit the classification results to reduce
    #   the impact of misclassification.
    lid.set_languages(limited_langs)
    # i.e., ('zh', 0.9999999909903544)
    sample['lang'] = lid.classify(sample['txt'])[0]
    return sample


# 将传入样本 sample 的任务硬编码为 'transcribe'
def detect_task(sample):
    # TODO(xcsong): Currently, the task is hard-coded to 'transcribe'.
    #   In the future, we could dynamically determine the task based on
    #   the contents of sample. For instance, if a sample contains both
    #   'txt_en' and 'txt_zh', the task should be set to 'translate'.
    sample['task'] = "transcribe"
    return sample


# 从 JSON 格式的行字符串中解析出键（key）、音频文件（wav）和文本（txt），然后加载音频文件，并根据需要裁剪音频
def decode_wav(sample):
    """ Parse key/wav/txt from json line

        Args:
            sample: str, str is a json line has key/wav

        Returns:
            {key, wav, sample_rate, ...}
    """
    assert 'key' in sample
    assert 'wav' in sample
    wav_file = sample['wav']
    if isinstance(wav_file, str):
        with open(wav_file, 'rb') as f:
            wav_file = f.read()
    if 'start' in sample:
        assert 'end' in sample
        sample_rate = torchaudio.info(wav_file).sample_rate
        start_frame = int(sample['start'] * sample_rate)
        end_frame = int(sample['end'] * sample_rate)
        with io.BytesIO(wav_file) as file_obj:
            waveform, _ = torchaudio.load(file_obj,
                                          num_frames=end_frame - start_frame,
                                          frame_offset=start_frame)
    else:
        with io.BytesIO(wav_file) as file_obj:
            waveform, sample_rate = torchaudio.load(file_obj)
    # del wav_file
    del sample['wav']
    sample['wav'] = waveform  # overwrite wav
    sample['sample_rate'] = sample_rate
    return sample

# 选择音频样本中的特定通道。通过检查样本中的音频数据，获取通道数量，并选择目标通道，函数对输入的音频样本进行原地操作，并返回处理后的样本。
def singal_channel(sample, channel=0):
    """ Choose a channel of sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            channel: target channel index

        Returns:
            {key, wav, label, sample_rate}
    """
    assert 'wav' in sample
    waveform = sample['wav']
    channel_nums = waveform.size(0)
    assert channel < channel_nums
    if channel_nums != 1:
        waveform = waveform[channel, :].unsqueeze(0)
    sample['wav'] = waveform
    return sample


# 重采样给定的音频样本
def resample(sample, resample_rate=16000):
    """ Resample sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            resample_rate: target resample rate

        Returns:
            {key, wav, label, sample_rate}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    if sample_rate != resample_rate:
        sample['sample_rate'] = resample_rate
        sample['wav'] = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    return sample


# 在给定的代码中，speed_perturb 函数用于对音频数据进行速度扰动。
# 该函数通过调整音频信号的播放速度来增加数据的多样性，从而提高语音识别模型的鲁棒性。
# 具体来说，函数会随机选择一个速度因子（如0.9、1.0或1.1），并使用 torchaudio.sox_effects.apply_effects_tensor 方法来调整音频信号的速度。
# 如果选择的速度因子不是1.0，则会重新采样音频信号以匹配原始采样率。
def speed_perturb(sample, speeds=None):
    """ Apply speed perturb to the sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            speeds(List[float]): optional speed

        Returns:
            key, wav, label, sample_rate}
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
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

    return sample


# 计算fbank,首先遍历 data 中的每个样本,对于每个样本，函数会检查是否包含 'sample_rate'、'wav'、'key' 和 'label' 这四个键。
# 然后，函数将音频波形数据 waveform 乘以 (1 << 15)，这是将音频数据从 int16 格式转换为 float32 格式的标准做法。
# 接着，函数使用 kaldi.fbank 函数提取 FBank 特征，并将结果存储在 mat 中。
# 最后，函数将提取的特征 mat 添加到样本字典中，并通过 yield 返回更新后的样本。
# fbank:梅尔频率倒谱系数（FBank）。
def compute_fbank(sample,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0,
                  window_type="povey"):
    """ Extract fbank

        Args:
            sample: {key, wav, sample_rate, ...}

        Returns:
            {key, feat, wav, sample_rate, ...}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    assert 'key' in sample
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
                      sample_frequency=sample_rate,
                      window_type=window_type)
    sample['feat'] = mat
    return sample


# 提取预训练的 w2vbert 模型所需的特征（fbank），并对这些特征进行标准化处理。
def compute_w2vbert_fbank(sample,
                          num_mel_bins=23,
                          frame_length=25,
                          frame_shift=10,
                          dither=0.0):
    """ Extract Pretrain w2vbert(4.5M hours) fbank
    """
    # 首先，函数调用 compute_fbank 函数来计算输入样本的 fbank 特征。
    sample = compute_fbank(sample, num_mel_bins, frame_length, frame_shift,
                           dither)
    # 对生成的 fbank 特征矩阵进行标准化处理
    mat = sample['feat']
    std, mean = torch.std_mean(mat, dim=0)
    mat = mat.subtract(mean).divide(std)
    sample['feat'] = mat
    return sample


# 检查输入的 sample 是否包含 'feat' 键
def sort_by_feats(sample):
    assert 'feat' in sample
    assert isinstance(sample['feat'], torch.Tensor)
    return sample['feat'].size(0)


# 返回这些特征的长度
def feats_length_fn(sample) -> int:
    assert 'feat' in sample
    return sample['feat'].size(0)


# 计算从音频数据中提取梅尔频率倒谱系数（MFCC）
def compute_mfcc(sample,
                 num_mel_bins=23,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.0,
                 num_ceps=40,
                 high_freq=0.0,
                 low_freq=20.0):
    """ Extract mfcc

        Args:
            sample: {key, wav, sample_rate, ...}

        Returns:
            {key, wav, feat, sample_rate, ...}
    """
    assert 'wav' in sample
    assert 'key' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    waveform = waveform * (1 << 15)
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
    return sample


# 从音频数据中计算对数梅尔频谱图（Log Mel Spectrogram）。
def compute_log_mel_spectrogram(sample,
                                n_fft=400,
                                hop_length=160,
                                num_mel_bins=80,
                                padding=0,
                                pad_or_trim: bool = False,
                                max_duration: int = 30):
    """ Extract log mel spectrogram, modified from openai-whisper, see:
        - https://github.com/openai/whisper/blob/main/whisper/audio.py
        - https://github.com/wenet-e2e/wenet/pull/2141#issuecomment-1811765040

        Args:
            sample: {key, wav, sample_rate, ...}
            max_duration: valid when pad_or_trim is True (orign whisper style)

        Returns:
            {key, feat, wav, sample_rate, ...}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    assert 'key' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav'].squeeze(0)  # (channel=1, sample) -> (sample,)
    if padding > 0:
        waveform = F.pad(waveform, (0, padding))
    if pad_or_trim:
        length = max_duration * sample_rate
        if waveform.size(0) >= length:
            waveform = waveform[:length]
        else:
            waveform = F.pad(waveform, (0, length - waveform.size(0)))

    window = torch.hann_window(n_fft)
    stft = torch.stft(waveform,
                      n_fft,
                      hop_length,
                      window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs()**2

    filters = torch.from_numpy(
        librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mel_bins))
    mel_spec = filters @ magnitudes

    # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    sample['feat'] = log_spec.transpose(0, 1)
    return sample


# 文本数据进行分词处理，并将分词结果和标签添加到原始数据中
def tokenize(sample, tokenizer: BaseTokenizer):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            sample: {key, wav, txt, sample_rate, ...}

        Returns:
            {key, wav, txt, tokens, label, sample_rate, ...}
    """
    assert 'txt' in sample
    tokens, label = tokenizer.tokenize(sample['txt'])
    sample['tokens'] = tokens
    sample['label'] = label
    return sample


# 它会检查样本的音频帧数是否在指定的最小和最大长度之间，并且如果存在标签，则还会检查标签的长度是否在指定的最小和最大长度之间，
# 以及标签与特征的比率是否在指定的最小和最大比率之间。如果所有条件都满足，则返回 True，否则返回 False。
def filter(sample,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            sample: {key, wav, label, sample_rate, ...}]
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
            bool: True to keep, False to filter
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    # sample['wav'] is torch.Tensor, we have 100 frames every second
    num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
    if num_frames < min_length:
        return False
    if num_frames > max_length:
        return False

    if 'label' in sample:
        if len(sample['label']) < token_min_length:
            return False
        if len(sample['label']) > token_max_length:
            return False
        if num_frames != 0:
            if len(sample['label']) / num_frames < min_output_input_ratio:
                return False
            if len(sample['label']) / num_frames > max_output_input_ratio:
                return False
    return True


# 实现了SpecAugment技术，用于对音频数据进行增强。
def spec_aug(sample, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            sample: {key, feat, ...}
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            {key, feat, ....}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    y = x.clone().detach()
    max_frames = y.size(0)
    max_freq = y.size(1)
    # time mask
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0
    # freq mask
    for _ in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    sample['feat'] = y
    return sample


# 对数据进行特殊替换操作。具体来说，它会对每个样本的特征（feat）进行随机的时间替换
def spec_sub(sample, max_t=20, num_t_sub=3):
    """ Do spec substitute
        Inplace operation
        ref: U2++, section 3.2.3 [https://arxiv.org/abs/2106.05642]

        Args:
            sample: Iterable{key, feat, ...}
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns
            {key, feat, ...}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    y = x.clone().detach()
    max_frames = y.size(0)
    for _ in range(num_t_sub):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        # only substitute the earlier time chosen randomly for current time
        pos = random.randint(0, start)
        y[start:end, :] = x[start - pos:end - pos, :]
    sample['feat'] = y
    return sample


# 用于对数据进行尾部帧的修剪操作。具体来说，它会随机选择一个长度（从1到 max_t），然后从数据的尾部截取相应长度的帧，并将剩余部分保留。截取的单位是帧
def spec_trim(sample, max_t=20):
    """ Trim tailing frames. Inplace operation.
        ref: TrimTail [https://arxiv.org/abs/2211.00522]

        Args:
            sample: {key, feat, label}
            max_t: max width of length trimming

        Returns:
            {key, feat, label}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    max_frames = x.size(0)
    length = random.randint(1, max_t)
    if length < max_frames / 2:
        y = x.clone().detach()[:max_frames - length]
        sample['feat'] = y
    return sample


# 将不同长度的序列数据填充成固定长度的批次数据，以便于训练模型。
def padding(data):
    """ Padding the data into training data

        Args:
            data: List[{key, feat, label}

        Returns:
            Tuple(keys, feats, labels, feats lengths, label lengths)
    """
    sample = data
    assert isinstance(sample, list)
    feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                dtype=torch.int32)
    order = torch.argsort(feats_length, descending=True)
    feats_lengths = torch.tensor([sample[i]['feat'].size(0) for i in order],
                                 dtype=torch.int32)
    sorted_feats = [sample[i]['feat'] for i in order]
    sorted_keys = [sample[i]['key'] for i in order]
    sorted_labels = [
        torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order
    ]
    sorted_wavs = [sample[i]['wav'].squeeze(0) for i in order]
    langs = [sample[i]['lang'] for i in order]
    tasks = [sample[i]['task'] for i in order]
    label_lengths = torch.tensor([x.size(0) for x in sorted_labels],
                                 dtype=torch.int32)
    wav_lengths = torch.tensor([x.size(0) for x in sorted_wavs],
                               dtype=torch.int32)
    padded_feats = pad_sequence(sorted_feats,
                                batch_first=True,
                                padding_value=0)
    padding_labels = pad_sequence(sorted_labels,
                                  batch_first=True,
                                  padding_value=-1)
    padded_wavs = pad_sequence(sorted_wavs, batch_first=True, padding_value=0)

    batch = {
        "keys": sorted_keys,
        "feats": padded_feats,
        "target": padding_labels,
        "feats_lengths": feats_lengths,
        "target_lengths": label_lengths,
        "pcm": padded_wavs,
        "pcm_length": wav_lengths,
        "langs": langs,
        "tasks": tasks,
    }
    if 'speaker' in sample[0]:
        speaker = torch.tensor([sample[i]['speaker'] for i in order],
                               dtype=torch.int32)
        batch['speaker'] = speaker
    return batch


# 动态调整批处理窗口的大小，以适应不同的数据样本
# 体来说，它通过维护一个最长帧数的记录，并根据这个记录来决定是否需要调整批处理窗口的大小。
# 如果当前样本的帧数超过了预设的最大帧数限制，那么批处理窗口的大小就会被调整。
class DynamicBatchWindow:

    def __init__(self, max_frames_in_batch=12000):
        self.longest_frames = 0
        self.max_frames_in_batch = max_frames_in_batch

    def __call__(self, sample, buffer_size):
        assert isinstance(sample, dict)
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        # 它计算当前样本的帧数，并更新 longest_frames 为当前最长帧数和新样本帧数的最大值。
        new_sample_frames = sample['feat'].size(0)
        self.longest_frames = max(self.longest_frames, new_sample_frames)
        # 接着，它计算在填充后的帧数，并检查是否超过了最大帧数限制。
        frames_after_padding = self.longest_frames * (buffer_size + 1)
        # 如果超过了，则重置 longest_frames 为当前样本的帧数，并返回 True，表示需要调整批处理窗口的大小。
        if frames_after_padding > self.max_frames_in_batch:
            self.longest_frames = new_sample_frames
            return True
        # 否则，返回 False，表示不需要调整。
        return False

# 总结：音频的处理，计算帧数，分块等数据。处理数据集。具体来说，它负责将原始音频数据转换为适合模型训练的格式，并进行必要的预处理操作。