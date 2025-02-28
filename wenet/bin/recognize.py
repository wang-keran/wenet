# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
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

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '/home/wangkeran/桌面/WENET/wenet')
from wenet.dataset.dataset import Dataset
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.context_graph import ContextGraph
from wenet.utils.ctc_utils import get_blank_id
from wenet.utils.common import TORCH_NPU_AVAILABLE  # noqa just ensure to check torch-npu


# 添加参数
def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', help='config file'
)
    parser.add_argument('--test_data',  help='test data file',
                        default = "inference/data.list")
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--device',
                        type=str,
                        default="cpu",
                        choices=["cpu", "npu", "cuda"],
                        help='accelerator to use')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp32',
                        choices=['fp16', 'fp32', 'bf16'],
                        help='model\'s dtype')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    # 在init_model文件里选择这个参数
    parser.add_argument('--checkpoint', help='checkpoint model'
)
    # 在context_graph.py中选择这个参数
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    # 用在asr_model.py文件中用来解码，长度惩罚
    parser.add_argument('--length_penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    # 用在asr_model.py文件中用来解码，空白惩罚
    parser.add_argument('--blank_penalty',
                        type=float,
                        default=0.0,
                        help='blank penalty')
    # 结果存储路径
    parser.add_argument('--result_dir',  help='asr result file'
                        ,default="result")
    # 批次大小，dataset用了，根据配置进行静态批量处理、桶批量处理或动态批量处理。
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    # 选择解码模式，在transformer/asr_model.py里
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
                             paraformer_beam_search""")
    # 放到transformer/asr_model.py里
    parser.add_argument('--search_ctc_weight',
                        type=float,
                        default=1.0,
                        help='ctc weight for nbest generation')
    # 放到transformer/asr_model.py里
    parser.add_argument('--search_transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for nbest generation')
    # 解码设置的CTC权重，在transformer/asr_model.py里
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for rescoring weight in \
                                  attention rescoring decode mode \
                              ctc weight for rescoring weight in \
                                  transducer attention rescore decode mode')

    # 放到transformer/asr_model.py里
    parser.add_argument('--transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for rescoring weight in '
                        'transducer attention rescore mode')
    # 在transformer/asr_model.py里
    parser.add_argument('--attn_weight',
                        type=float,
                        default=0.0,
                        help='attention weight for rescoring weight in '
                        'transducer attention rescore mode')
    # 用在解码中，在transformer/asr_model.py里
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    # 用在解码中，在transformer/asr_model.py里
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    # 用在解码中，在transformer/asr_model.py里
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    # 用在解码中，在transformer/asr_model.py里
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    # 用在wenet/utils/config.py里
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    # decode全在transformer/asr_model.py里
    parser.add_argument('--word',
                        default='',
                        type=str,
                        help='word file, only used for hlg decode')
    parser.add_argument('--hlg',
                        default='',
                        type=str,
                        help='hlg file, only used for hlg decode')
    parser.add_argument('--lm_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--r_decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')

    # 参数，上下文图和得分，给context_graph用
    parser.add_argument(
        '--context_bias_mode',
        type=str,
        default='',
        help='''Context bias mode, selectable from the following
                                option: decoding-graph, deep-biasing''')
    parser.add_argument('--context_list_path',
                        type=str,
                        default='',
                        help='Context list path')
    parser.add_argument('--context_graph_score',
                        type=float,
                        default=0.0,
                        help='''The higher the score, the greater the degree of
                                bias using decoding-graph for biasing''')

    parser.add_argument('--use_lora',
                        type=bool,
                        default=False,
                        help='''Whether to use lora for biasing''')
    parser.add_argument("--lora_ckpt_path",
                        default=None,
                        type=str,
                        help="lora checkpoint path.")
    args = parser.parse_args()
    print(args)
    return args


def main():
    # 接收输入参数和日志
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # 先判断有没有GPU
    if args.gpu != -1:
        # remain the original usage of gpu
        args.device = "cuda"
    if "cuda" in args.device:
        # 设置当前 Python 环境中可见的 GPU 设备
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 读取配置文件
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        # wenet/utils/config.py
        # 这段代码的功能是更新配置字典，允许通过一系列指定的覆盖项来修改原有的配置。
        # 读入原始字典，返回新字典，将后面字典的参数覆盖前面的参数
        configs = override_config(configs, args.override_config)

    # 读取数据集的配置文件，复制一份测试集的配置文件
    test_conf = copy.deepcopy(configs['dataset_conf'])

    # 设置测试集的配置文件，设置成102400，因为GPU版本显存保留1GB（1024000000）为了方便运算，
    # 这些设置为false的参数全是用来训练的，在推理时反而会拖后腿
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    # 禁用速度扰动、频谱增强、频谱剪裁、频谱替换、数据集洗牌、数据集排序、数据集循环、数据集列表洗牌
    # 速度扰动在训练时增强鲁棒性，在识别时会拖后腿，所以识别禁用
    test_conf['speed_perturb'] = False
    # 对频谱图进行随即扭曲遮挡增强范化能力，识别时没用还会拖后腿，所以识别禁用
    test_conf['spec_aug'] = False
    # 训练时裁减来学习特征，识别时没必要
    test_conf['spec_sub'] = False
    # 频谱替换是另一种数据增强技术，通过将频谱图中的某些部分替换为其他内容。
    # 在推理或评估阶段禁用它，可以确保输入数据的真实性和一致性。
    test_conf['spec_trim'] = False
    # 数据集洗牌会随机打乱数据顺序。在训练阶段，这有助于打破数据的顺序依赖性，
    # 但在推理或评估阶段，禁用洗牌可以确保每次运行时数据的顺序一致，从而提高结果的可重复性。
    test_conf['shuffle'] = False
    # 排序通常用于按某种标准（如长度）对数据进行排列，以便更高效地处理。
    # 在推理或评估阶段禁用排序，可以确保数据按照原始顺序处理，避免因排序引入的偏差。
    test_conf['sort'] = False
    # 在训练阶段，数据集可能会被多次循环使用以增加训练样本量。
    # 但在推理或评估阶段，通常只需要遍历一次数据集即可。设置 cycle = 1 确保了这一点，避免了不必要的重复处理。
    test_conf['cycle'] = 1
    # 禁用列表洗牌，确保每次运行时数据的顺序一致，从而提高结果的可重复性。
    test_conf['list_shuffle'] = False
    # 禁用抖动，因为识别时没必要，抖动是在训练时添加细小噪音的训练方法，有助于训练时提高鲁棒性，识别时不需要
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    # 设置批量处理类型为静态,静态批量处理意味着每个批次的大小是固定的，这有助于简化推理过程并提高效率。
    test_conf['batch_conf']['batch_type'] = "static"
    # 设置批量大小,根据命令行参数 args.batch_size 设置批量大小。
    # 批量大小可以根据硬件资源或性能需求灵活调整。通过从命令行参数中读取批量大小，可以在不同环境下优化推理性能。
    test_conf['batch_conf']['batch_size'] = args.batch_size

    # 初始化这个分词器，根据config配置文件来的
    # 传入配置字典，返回一个分词器
    tokenizer = init_tokenizer(configs)
    # 从原始数据变成数据集返回
    # 传入数据类型，原数据集，分词器，训练配置，返回一个可以被模型理解的数据集字典，这里面有特征提取，分词，重采样，速度扰动，裁剪学习特征，批处理等等
    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           tokenizer,
                           test_conf,
                           partition=False)

    # 该 DataLoader 主要用于测试阶段，逐一加载测试样本，并将它们输入到模型中进行推理或评估。batch_size 设置为 None，它不会进行批量处理。
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=None,
                                  num_workers=args.num_workers)

    # Init asr model from configs
    args.jit = False
    # 输入模型信息和配置文件，返回初始化完成的模型和配置文件,这里encode编码了
    model, configs = init_model(args, configs)

    device = torch.device(args.device)
    # 将模型移动到指定的设备上（如CPU或GPU）
    model = model.to(device)
    # 将模型从训练模式切换到评估模式
    model.eval()
    # 用于设置 PyTorch 中张量的数据类型（dtype）为float32型
    dtype = torch.float32
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    logging.info("compute dtype is {}".format(dtype))

    context_graph = None
    if 'decoding-graph' in args.context_bias_mode:
        # 返回得分和上下文图
        context_graph = ContextGraph(args.context_list_path,
                                     tokenizer.symbol_table,
                                     configs['tokenizer_conf']['bpe_path'],
                                     args.context_graph_score)

    # 从配置和符号表中获取空白 ID
    _, blank_id = get_blank_id(configs, tokenizer.symbol_table)
    logging.info("blank_id is {}".format(blank_id))

    # TODO(Dinghao Zhou): Support RNN-T related decoding
    # TODO(Lv Xiang): Support k2 related decoding
    # TODO(Kaixun Huang): Support context graph
    # 开始准备写文件,根据不同的解码方式写入不同的文件
    files = {}
    for mode in args.modes:
        # 使用 os.path.join 将 args.result_dir（结果目录）与当前模式 mode 连接起来，生成一个新的目录路径 dir_name。
        dir_name = os.path.join(args.result_dir, mode)
        # 创建这个目录，如果目录已经存在则不会抛出错误。这是为了确保后续的文件可以正确写入。
        os.makedirs(dir_name, exist_ok=True)
        # 文件名text
        file_name = os.path.join(dir_name, 'text')
        # 开始写入：使用 open(file_name, 'w') 以写入模式打开文件，并将文件对象存储在 files 字典中，以 mode 为键。这意味着每个模式都有一个对应的文件用于记录相关的数据。
        files[mode] = open(file_name, 'w')
    # 计算最大模式长度：计算每种模式名称的长度，并使用 max() 函数获取最长的模式名称的长度，存储在 max_format_len 变量中。这通常用于后续日志格式化时，使输出更加整齐。
    max_format_len = max([len(mode) for mode in args.modes])

    # 这段代码主要用于在PyTorch中进行模型推理，利用了混合精度计算和上下文管理。
    # 启用混合精度计算；enabled=True：表示开启自动混合精度。；dtype=dtype：将之前定义的 dtype（通常是 torch.float32）作为数据类型传入，确保模型计算时使用该数据类型。；cache_enabled=False：不使用缓存，可以减少内存占用。
    with torch.cuda.amp.autocast(enabled=True,
                                 dtype=dtype,
                                 cache_enabled=False):
        # 无梯度计算上下文：此上下文管理器表示在其范围内不会计算梯度，适合于模型推理阶段。
        # 这样可以降低内存使用，并加快计算速度，因为不需要存储用于反向传播的梯度信息。
        with torch.no_grad():
            # 通过 enumerate(test_data_loader) 遍历测试数据加载器 test_data_loader，
            # 每次迭代会返回一个批次的数据 batch 和当前的批次索引 batch_idx。
            for batch_idx, batch in enumerate(test_data_loader):
                # 提取数据
                keys = batch["keys"]
                feats = batch["feats"].to(device)
                target = batch["target"].to(device)
                feats_lengths = batch["feats_lengths"].to(device)
                target_lengths = batch["target_lengths"].to(device)
                infos = {"tasks": batch["tasks"], "langs": batch["langs"]}
                # 调用模型的 decode 方法进行解码,在transformer/asr_model.py里
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
                    length_penalty=args.length_penalty,
                    infos=infos)
                print("返回的结果是：", results)
                # decode 函数返回的是一个字典，包含所请求解码方法的名称及其对应的解码结果。
                for i, key in enumerate(keys):
                    for mode, hyps in results.items():
                        # 遍历解码结果 results，提取每个模式下的假设（hypothesis）。
                        tokens = hyps[i].tokens
                        # 根据ID列表重建原始文本，返回组合好的文字和帧
                        line = '{} {}'.format(key,
                                              tokenizer.detokenize(tokens)[0])
                        print("解析出的一句话是：",line)
                        # 记录日志信息，并写入文件
                        logging.info('{} {}'.format(mode.ljust(max_format_len),
                                                    line))
                        # 这是一个字典访问操作，files 是一个字典，其中的每个键（mode）对应一个打开的文件对象。每种模式（例如，训练、验证、测试等）都在之前的代码中被映射到一个文件对象上。
                        # 按照解码模式的顺序对应写入语音识别解码后组合完成的句子
                        files[mode].write(line + '\n')
        # 关闭文件读写
        for mode, f in files.items():
            f.close()


# 运行主函数
if __name__ == '__main__':
    main()

# 总结：初始化和配置一个用于测试的语音识别模型