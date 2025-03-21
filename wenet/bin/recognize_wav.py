# Revised from WeNet repo
from __future__ import print_function
import time
import numpy as np
import argparse
import copy
import logging
import os
import io

import torch
import yaml
# from torch.utils.data import DataLoader
import torchaudio
import torchaudio.compliance.kaldi as kaldi

import sys
sys.path.insert(0, '/home/wangkeran/桌面/WENET/wenet')
import time # 用来计时

# from my_dataset import Dataset
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.context_graph import ContextGraph
from wenet.utils.ctc_utils import get_blank_id
# from wenet.dataset.processor import compute_fbank
from torch.nn.utils.rnn import pad_sequence

# 调试代码打断点
import pdb

def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config',  help='config file'
                        #,default="/home/wangkeran/桌面/WENET/aishell_u2pp_conformer_exp/20210601_u2++_conformer_exp_aishell/train.yaml"
                        ,default="/home/wangkeran/桌面/WENET/librispeech_u2pp_conformer_exp/20210610_u2pp_conformer_exp_librispeech/train.yaml"
                        )
    parser.add_argument('--test_data', help='test data file'
                        ,default = "inference/data.list"
                        )
    # 没用
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    # 与recognize一样
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    # 与recognize一样
    parser.add_argument('--checkpoint',  help='checkpoint model'
                        #,default="/home/wangkeran/桌面/WENET/aishell_u2pp_conformer_exp/20210601_u2++_conformer_exp_aishell/final.pt"
                        ,default="/home/wangkeran/桌面/WENET/librispeech_u2pp_conformer_exp/20210610_u2pp_conformer_exp_librispeech/final.pt"
                        )
    # 这里和decode用，与recognize一样
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    # 与recognize一样用在decode里
    parser.add_argument('--length_penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    # 与recognize一样用在decode里
    parser.add_argument('--blank_penalty',
                        type=float,
                        default=0.0,
                        help='blank penalty')
    # 默认输出到当前文件夹下新创建的result文件夹
    parser.add_argument('--result_dir',  help='asr result file'
                        ,default="result"
                        )
    # 不在recoginze里面，不在dataloader里
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    # 支持多种模式解码：python script.py --modes attention ctc_greedy_search
    parser.add_argument('--modes',
                        nargs='+',
                        help="""decoding mode, support the following:
                             attention
                             ctc_greedy_search
                             ctc_prefix_beam_search
                             attention_rescoring
                             rnnt_greedy_search
                             rnnt_beam_search
                             rnnt_beam_attn_rescoring
                             ctc_beam_td_attn_rescoring
                             hlg_onebest
                             hlg_rescore
                             paraformer_greedy_search
                             paraformer_beam_search""",
                             default =["ctc_prefix_beam_search"]        # 原本的：ctc_prefix_beam_search
                             )
    # 与recognize一样
    parser.add_argument('--search_ctc_weight',
                        type=float,
                        default=1.0,
                        help='ctc weight for nbest generation')
    # 与recognize一样
    parser.add_argument('--search_transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for nbest generation')
    # 与recognize一样
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for rescoring weight in \
                                  attention rescoring decode mode \
                              ctc weight for rescoring weight in \
                                  transducer attention rescore decode mode')

    # 与recognize一样
    parser.add_argument('--transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for rescoring weight in '
                        'transducer attention rescore mode')
    # 与recognize一样
    parser.add_argument('--attn_weight',
                        type=float,
                        default=0.0,
                        help='attention weight for rescoring weight in '
                        'transducer attention rescore mode')
    # 与recognize一样
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    # 与recognize一样
    # 丢弃在三分之一的时候损失最大，后面损失块多了输出就正常了，即12：4损失最大
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    # 与recognize一样
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    # 与recognize一样
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    # 没有用在override_config里面，没用
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    # 与recognize一样
    parser.add_argument('--word',
                        default='',
                        type=str,
                        help='word file, only used for hlg decode')
    # 与recognize一样
    parser.add_argument('--hlg',
                        default='',
                        type=str,
                        help='hlg file, only used for hlg decode')
    # 与recognize一样
    parser.add_argument('--lm_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    # 与recognize一样
    parser.add_argument('--decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    # 与recognize一样
    parser.add_argument('--r_decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')

    parser.add_argument(
        '--context_bias_mode',
        type=str,
        default='decoding-graph',
        help='''Context bias mode, selectable from the following
                                option: decoding-graph, deep-biasing''')
    parser.add_argument('--context_list_path',
                        type=str,
                        # default='/home/wangkeran/桌面/WENET/aishell_u2pp_conformer_exp/20210601_u2++_conformer_exp_aishell/units.txt',
                        default="/home/wangkeran/桌面/WENET/librispeech_u2pp_conformer_exp/20210610_u2pp_conformer_exp_librispeech/words.txt",
                        help='Context list path')
    parser.add_argument('--context_graph_score',
                        type=float,
                        default=3.0,
                        help='''The higher the score, the greater the degree of
                                bias using decoding-graph for biasing''')
    args = parser.parse_args()
    return args

def load_data(wav_file):
    sample = {}
    #wav_file = '/home/wangkeran/桌面/WENET/wenet/runtime/gpu/client/test_wavs/mid.wav'
    #wav_file='/home/wangkeran/桌面/WENET/wenet/runtime/gpu/client/test_wavs/common_voice_en_1044.wav'
    # with io.BytesIO(wav_file) as file_obj:
    waveform, sample_rate = torchaudio.load(wav_file)
            # del wav_file
    # 作为样本的唯一标识
    sample['key'] = "自定义keys"
    # 表示模型的目标输出
    sample['txt'] = "语音label"
    sample['wav'] = waveform
    sample['sample_rate'] = sample_rate
    # print("****","sample_rate is ",sample_rate,"***")
    return sample
# 分词处理，将结果存储在sample字典中
def tokenize(sample, tokenizer):#: BaseTokenizer):
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
# 特征提取
def compute_fbank(sample,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
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
    # print(waveform,'\n',waveform.shape)
    waveform = waveform * (1 << 15)
    # print(waveform,'\n',waveform.shape)
    # Only keep key, feat, label
    # 使用fbank提取特征
    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      energy_floor=0.0,
                      sample_frequency=sample_rate)
    sample['feat'] = mat
    # print(mat.shape,"********")
    return sample  
# 重采样，采样率设置为16000
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
        # print("****","resample_","***")
    return sample
# 进行填充和排序，更好进行推理和训练
def padding(data):
    """ Padding the data into training data

        Args:
            data: List[{key, feat, label}

        Returns:
            Tuple(keys, feats, labels, feats lengths, label lengths)
    """
    # data = list(data)
    # print(data)
    # sample = list(data.items())
    sample =[]
    sample.append(data)
    # print(sample)
    assert isinstance(sample, list)
    feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                dtype=torch.int32)
    order = torch.argsort(feats_length, descending=True)
    # 生成一个包含每个音频特征序列有效长度的张量 feats_lengths。
    # 具体来说，它通过遍历 order 列表中的索引，获取每个样本的特征矩阵的长度，并将这些长度组合成一个张量。
    feats_lengths = torch.tensor([sample[i]['feat'].size(0) for i in order],
                                 dtype=torch.int32)
    # 特征是输入模型的主要数据，排序后可以将较长的特征放在前面，有助于减少填充，并且在后续的处理（如RNN计算）中更高效。
    sorted_feats = [sample[i]['feat'] for i in order]
    # 这些键通常用于标识每个样本。虽然对这些键进行排序的直接作用可能不明显，但保持与其他排序元素一致性是必要的（降序），确保键、特征和标签之间的对应关系不变。
    sorted_keys = [sample[i]['key'] for i in order]
    # 标签是模型训练的目标。通过排序，确保对应的标签与输入特征匹配（降序），并且同样减少填充数量。
    sorted_labels = [
        torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order
    ]
    
    # 减少填充的数量
    #填充的目的：在处理变长序列时，填充是为了保证所有序列具有相同的长度，这样才能批量输入到模型中进行训练。在填充过程中，会将短于最长序列的部分用特定的填充值（如0或-1）填充。
    #降序排列的效果：将特征值按降序排列后，序列长度较长的样本会被优先处理。这意味着在批处理的过程中，当处理到长度较短的序列时，它们只需要填充到当前批次中的最大长度，
    # 而不是整个训练集的最大长度。这样，减少了无效填充，降低了计算资源的浪费。
    
    #提高批处理效率
    #GPU并行计算：在深度学习中，利用GPU进行批量训练时，通常希望每个批次中的序列长度尽可能相近，以提高GPU的利用率。
    # 降序排列确保每个批次的序列长度差异较小，避免了因长度差异过大而导致的额外填充。
    #降低填充计算的负担：通过降序排列，模型在处理时只需关注有效的输入部分，从而可以跳过填充部分的计算，节省计算时间和资源。
    
    #动态计算的灵活性
    #动态计算图：在使用动态计算图（如PyTorch的计算方式）时，模型会根据实际输入动态构建计算图。
    # 降序排列的特征值可以使得模型更有效地利用动态计算，因为它可以根据当前批次中最长的序列进行调整，避免无谓的填充和计算。
    
    #优化内存使用
    #内存布局：当序列长度差异较小时，内存的使用效率更高。因为内存中的数据可以被连续访问，从而减少缓存失效和内存访问延迟。
    
    # 提升模型训练效率
    #特征传递：在训练过程中，长序列（包含更多特征信息）在前，模型能更快捕捉到输入序列中的重要特征。通过有效的填充和计算安排，模型的学习效率和收敛速度可以提高。
    
    # 波形数据也是变长的，排序有助于后续的处理，确保输入与输出之间的一致性（降序），尤其是在处理音频数据时。
    sorted_wavs = [sample[i]['wav'].squeeze(0) for i in order]
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
    }
    if 'speaker' in sample[0]:
        speaker = torch.tensor([sample[i]['speaker'] for i in order],
                               dtype=torch.int32)
        batch['speaker'] = speaker
    #print("******","padding","*******")
    return batch
def load_data_test():
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
    args = get_args()
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
        # print("配置文件是：")
        # print(configs)
        # print("配置文件打印结束")
    #wav_file = '/home/wangkeran/桌面/WENET/wenet/runtime/gpu/client/test_wavs/long.wav'
    wav_file='/home/wangkeran/桌面/WENET/wenet/runtime/gpu/client/test_wavs/common_voice_en_1044.wav'
    start_read_wav = time.perf_counter()
    sample = load_data(wav_file)
    # 在init_tokenizer的tokenizer_conf中有词表的路径，就是同路径下的units.txt
    tokenizer = init_tokenizer(configs)
    # 分词器不能删除，会报错：Traceback (most recent call last):
    #   File "/home/wangkeran/桌面/WENET/wenet/wenet/bin/recognize_wav.py", line 479, in <module>
    #     load_data_test()
    #   File "/home/wangkeran/桌面/WENET/wenet/wenet/bin/recognize_wav.py", line 396, in load_data_test
    #     sample = padding(sample)
    #              ^^^^^^^^^^^^^^^
    #   File "/home/wangkeran/桌面/WENET/wenet/wenet/bin/recognize_wav.py", line 319, in padding
    #     torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order
    #                  ~~~~~~~~~^^^^^^^^^
    # KeyError: 'label'
    sample  = tokenize(sample,tokenizer)
    sample = resample(sample, resample_rate=16000)
    
    
    # print(sample['wav'].shape) # torch.Size([2, 65388])
    # print(sample['sample_rate'])
    sample = compute_fbank(sample,
                  num_mel_bins=80, ###23
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0)
    
    sample = padding(sample)
    # print(sample['feats'],sample['feats'].shape)
    # np.save("tensor_data.npy", sample['feat'])
    end_read_wav = time.perf_counter()
    runTime_read_wav = end_read_wav - start_read_wav
    
   
    # 这里进行了编码器，解码器，CTC的初始化
    args.jit = False
    model, configs = init_model(args, configs)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()

    context_graph = None
    # 热词功能，没有整个生成上下文的功能
    if 'decoding-graph' in args.context_bias_mode:
        context_graph = ContextGraph(args.context_list_path,
                                     tokenizer.symbol_table,
                                     configs['tokenizer_conf']['bpe_path'],
                                     args.context_graph_score)
    _, blank_id = get_blank_id(configs, tokenizer.symbol_table)
    #logging.info("blank_id is {}".format(blank_id))
    
    files = {}
    for mode in args.modes: #mode:解码模式
        dir_name = os.path.join(args.result_dir, mode)
        os.makedirs(dir_name, exist_ok=True)
        file_name = os.path.join(dir_name, 'text')
        files[mode] = open(file_name, 'w')
    max_format_len = max([len(mode) for mode in args.modes])
    start = time.perf_counter()
    with torch.no_grad():
        batch= sample
        keys = batch["keys"] 
        feats = batch["feats"].to(device)# [1,798,80],批次，帧数（批次大小），一帧的维度
        #print("main process \n",feats,feats.shape)
        target = batch["target"].to(device) # token的数字,和units文字对应
        feats_lengths = batch["feats_lengths"].to(device) #798
        #print("feats_lengths:",feats_lengths)
        # 这里打个断点
        # pdb.set_trace() 
        #print("断点调试测试")
        target_lengths = batch["target_lengths"].to(device) # 16
        results = model.decode(
            args.modes,
            feats,
            feats_lengths,
            args.beam_size,
            decoding_chunk_size=args.decoding_chunk_size,
            num_decoding_left_chunks=args.num_decoding_left_chunks,
            ctc_weight=args.ctc_weight,
            simulate_streaming=args.simulate_streaming,
            reverse_weight=args.reverse_weight,
            context_graph=context_graph,
            blank_id=blank_id,
            blank_penalty=args.blank_penalty,
            length_penalty=args.length_penalty)
        #print("返回结果是：",results.items())
        for i, key in enumerate(keys):
            for mode, hyps in results.items(): #dict:[keys，value]
                tokens = hyps[i].tokens
                #time.sleep(3)      # 输出调试的
                #解码结果，输出的是keys和语音识别文本
                #line = '{} {}'.format(key, tokenizer.detokenize(tokens)[0]) 
                line =  tokenizer.detokenize(tokens)[0]  # 只保留识别结果文本
                #print("一行数据是",line)
                logging.info('{} {}'.format(mode.ljust(max_format_len), #解码方法
                                           line))
                #logging.info('{} {}'.format('result:',line))
                # line中是真正的最终输出结果
                # print(line)
                # logging.info('{} {}'.format('result:',line))
                files[mode].write(line + '\n')
    end = time.perf_counter()
    runTime = end - start + runTime_read_wav
    print("更精确的运行时间为：", runTime, "秒")
    for mode, f in files.items():
        f.close()
    return sample
    


if __name__ == '__main__':
    start = time.perf_counter()
    load_data_test()
    end = time.perf_counter()
    runTime = end - start
    runTime_ms = runTime * 1000
    rtf=runTime/8
   # print("运行时间：", runTime, "秒")
   # print("运行时间：", runTime_ms, "毫秒")
   # print("RTF为：",rtf)
