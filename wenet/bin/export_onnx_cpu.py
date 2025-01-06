# Copyright (c) 2022, Xingchen Song (sxc19@mails.tsinghua.edu.cn)
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
import logging
import os
import copy
import sys

import torch
import yaml
import numpy as np

from wenet.utils.init_model import init_model

try:
    import onnx
    import onnxruntime
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print('Please install onnx and onnxruntime!')
    sys.exit(1)


# 获取输入信息的
def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--chunk_size',
                        required=True,
                        type=int,
                        help='decoding chunk size')
    parser.add_argument('--num_decoding_left_chunks',
                        required=True,
                        type=int,
                        help='cache chunks')
    parser.add_argument('--reverse_weight',
                        default=0.5,
                        type=float,
                        help='reverse_weight in attention_rescoing')
    args = parser.parse_args()
    return args


# 转换成numpy列表
def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


# 打印输入输出信息的
def print_input_output_info(onnx_model, name, prefix="\t\t"):
    input_names = [node.name for node in onnx_model.graph.input]
    input_shapes = [[d.dim_value for d in node.type.tensor_type.shape.dim]
                    for node in onnx_model.graph.input]
    output_names = [node.name for node in onnx_model.graph.output]
    output_shapes = [[d.dim_value for d in node.type.tensor_type.shape.dim]
                     for node in onnx_model.graph.output]
    print("{}{} inputs : {}".format(prefix, name, input_names))
    print("{}{} input shapes : {}".format(prefix, name, input_shapes))
    print("{}{} outputs: {}".format(prefix, name, output_names))
    print("{}{} output shapes : {}".format(prefix, name, output_shapes))


# 导出编码器
def export_encoder(asr_model, args):
    print("Stage-1: export encoder")
    # 编码器和前向函数直接用wenet里现有的，输出路径用输入的
    encoder = asr_model.encoder
    encoder.forward = encoder.forward_chunk
    encoder_outpath = os.path.join(args['output_dir'], 'encoder.onnx')

    # 准备转换编码器的输入
    print("\tStage-1.1: prepare inputs for encoder")
    chunk = torch.randn(
        (args['batch'], args['decoding_window'], args['feature_size']))
    offset = 0
    # NOTE(xcsong): The uncertainty of `next_cache_start` only appears
    #   in the first few chunks, this is caused by dynamic att_cache shape, i,e
    #   (0, 0, 0, 0) for 1st chunk and (elayers, head, ?, d_k*2) for subsequent
    #   chunks. One way to ease the ONNX export is to keep `next_cache_start`
    #   as a fixed value. To do this, for the **first** chunk, if
    #   left_chunks > 0, we feed real cache & real mask to the model, otherwise
    #   fake cache & fake mask. In this way, we get:
    #   1. 16/-1 mode: next_cache_start == 0 for all chunks
    #   2. 16/4  mode: next_cache_start == chunk_size for all chunks
    #   3. 16/0  mode: next_cache_start == chunk_size for all chunks
    #   4. -1/-1 mode: next_cache_start == 0 for all chunks
    #   NO MORE DYNAMIC CHANGES!!
    #
    # NOTE(Mddct): We retain the current design for the convenience of supporting some
    #   inference frameworks without dynamic shapes. If you're interested in all-in-one
    #   model that supports different chunks please see:
    #   https://github.com/wenet-e2e/wenet/pull/1174

    # 剩余块数大于0,说明是流式模型
    if args['left_chunks'] > 0:  # 16/4
        required_cache_size = args['chunk_size'] * args['left_chunks']
        # 指定偏移量（从某个基准点到目标位置的偏移量）
        offset = required_cache_size
        # Real cache，真实缓存，流式模型需要真实的缓存和掩码处理连续的数据块，以便在识别中保持上下文信息
        # 生成全是0的张量，形状为(args['num_blocks'], args['head'], required_cache_size, args['output_size'] // args['head'] * 2)
        att_cache = torch.zeros(
            (args['num_blocks'], args['head'], required_cache_size,
             args['output_size'] // args['head'] * 2))
        # Real mask，真实掩码，全是1
        att_mask = torch.ones(
            (args['batch'], 1, required_cache_size + args['chunk_size']),
            dtype=torch.bool)
        # 选取所有批次样本，选取整个第二维，选取从0到required_cache_size的所有数据赋值为0
        att_mask[:, :, :required_cache_size] = 0
    # 剩余块数小于0,非流式模型
    elif args['left_chunks'] <= 0:  # 16/-1, -1/-1, 16/0
        required_cache_size = -1 if args['left_chunks'] < 0 else 0
        # Fake cache
        att_cache = torch.zeros((args['num_blocks'], args['head'], 0,
                                 args['output_size'] // args['head'] * 2))
        # Fake mask
        att_mask = torch.ones((0, 0, 0), dtype=torch.bool)
    # 卷积神经网络缓存
    cnn_cache = torch.zeros(
        (args['num_blocks'], args['batch'], args['output_size'],
         args['cnn_module_kernel'] - 1))
    # 输入形状
    inputs = (chunk, offset, required_cache_size, att_cache, cnn_cache,
              att_mask)
    # 打印输入规定
    print("\t\tchunk.size(): {}\n".format(chunk.size()),
          "\t\toffset: {}\n".format(offset),
          "\t\trequired_cache: {}\n".format(required_cache_size),
          "\t\tatt_cache.size(): {}\n".format(att_cache.size()),
          "\t\tcnn_cache.size(): {}\n".format(cnn_cache.size()),
          "\t\tatt_mask.size(): {}\n".format(att_mask.size()))

    # 准备开始存储编码器
    print("\tStage-1.2: torch.onnx.export")
    # 动态轴，可以浮动的数据
    dynamic_axes = {
        # 块大小
        'chunk': {
            1: 'T'
        },
        # 注意力缓存
        'att_cache': {
            2: 'T_CACHE'
        },
        # 注意力掩码，遮蔽未来信息
        'att_mask': {
            2: 'T_ADD_T_CACHE'
        },
        # 输出
        'output': {
            1: 'T'
        },
        # 反向注意力卷积神经网络缓存
        'r_att_cache': {
            2: 'T_CACHE'
        },
    }
    # NOTE(xcsong): We keep dynamic axes even if in 16/4 mode, this is
    #   to avoid padding the last chunk (which usually contains less
    #   frames than required). For users who want static axes, just pop
    #   out specific axis.
    # if args['chunk_size'] > 0:  # 16/4, 16/-1, 16/0
    #     dynamic_axes.pop('chunk')
    #     dynamic_axes.pop('output')
    # if args['left_chunks'] >= 0:  # 16/4, 16/0
    #     # NOTE(xsong): since we feed real cache & real mask into the
    #     #   model when left_chunks > 0, the shape of cache will never
    #     #   be changed.
    #     dynamic_axes.pop('att_cache')
    #     dynamic_axes.pop('r_att_cache')
    # 导出onnx编码器
    torch.onnx.export(encoder,
                      inputs,
                      encoder_outpath,
                      opset_version=13,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=[
                          'chunk', 'offset', 'required_cache_size',
                          'att_cache', 'cnn_cache', 'att_mask'
                      ],
                      output_names=['output', 'r_att_cache', 'r_cnn_cache'],
                      dynamic_axes=dynamic_axes,
                      verbose=False)
    # 加载onnx编码器模型
    onnx_encoder = onnx.load(encoder_outpath)
    # 在导出 ONNX 模型时，将模型的配置信息（如训练参数、超参数等）添加到模型的元数据中。
    for (k, v) in args.items():
        meta = onnx_encoder.metadata_props.add()
        meta.key, meta.value = str(k), str(v)
    # 导出或加载 ONNX 模型后，验证模型的有效性并打印模型的图结构。
    onnx.checker.check_model(onnx_encoder)
    onnx.helper.printable_graph(onnx_encoder.graph)
    # NOTE(xcsong): to add those metadatas we need to reopen
    #   the file and resave it.为了保存元数据，我们需要重新打开文件并重新保存。
    # 将一个 ONNX 模型对象保存到指定的文件路径。
    onnx.save(onnx_encoder, encoder_outpath)
    print_input_output_info(onnx_encoder, "onnx_encoder")
    # Dynamic quantization，动态量化，减少计算位数，提高计算速度
    model_fp32 = encoder_outpath
    model_quant = os.path.join(args['output_dir'], 'encoder.quant.onnx')
    quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
    print('\t\tExport onnx_encoder, done! see {}'.format(encoder_outpath))

    # 检查onnx编码器和torch编码器
    print("\tStage-1.3: check onnx_encoder and torch_encoder")
    # 检查torch编码器
    torch_output = []
    torch_chunk = copy.deepcopy(chunk)
    torch_offset = copy.deepcopy(offset)
    torch_required_cache_size = copy.deepcopy(required_cache_size)
    torch_att_cache = copy.deepcopy(att_cache)
    torch_cnn_cache = copy.deepcopy(cnn_cache)
    torch_att_mask = copy.deepcopy(att_mask)
    for i in range(10):
        print("\t\ttorch chunk-{}: {}, offset: {}, att_cache: {},"
              " cnn_cache: {}, att_mask: {}".format(
                  i, list(torch_chunk.size()), torch_offset,
                  list(torch_att_cache.size()), list(torch_cnn_cache.size()),
                  list(torch_att_mask.size())))
        # NOTE(xsong): att_mask of the first few batches need changes if
        #   we use 16/4 mode.流式的情况
        if args['left_chunks'] > 0:  # 16/4
            torch_att_mask[:, :, -(args['chunk_size'] * (i + 1)):] = 1
        out, torch_att_cache, torch_cnn_cache = encoder(
            torch_chunk, torch_offset, torch_required_cache_size,
            torch_att_cache, torch_cnn_cache, torch_att_mask)
        torch_output.append(out)
        torch_offset += out.size(1)
    torch_output = torch.cat(torch_output, dim=1)

    # onnx编码器
    onnx_output = []
    onnx_chunk = to_numpy(chunk)
    onnx_offset = np.array((offset)).astype(np.int64)
    onnx_required_cache_size = np.array((required_cache_size)).astype(np.int64)
    onnx_att_cache = to_numpy(att_cache)
    onnx_cnn_cache = to_numpy(cnn_cache)
    onnx_att_mask = to_numpy(att_mask)
    # 用于在推理过程中加载和运行 ONNX 模型。通过创建推理会话，可以在指定的硬件上运行模型，并获取推理结果
    ort_session = onnxruntime.InferenceSession(
        encoder_outpath, providers=['CPUExecutionProvider'])
    # 规定输入项的名称
    input_names = [node.name for node in onnx_encoder.graph.input]
    for i in range(10):
        print("\t\tonnx  chunk-{}: {}, offset: {}, att_cache: {},"
              " cnn_cache: {}, att_mask: {}".format(i, onnx_chunk.shape,
                                                    onnx_offset,
                                                    onnx_att_cache.shape,
                                                    onnx_cnn_cache.shape,
                                                    onnx_att_mask.shape))
        # NOTE(xsong): att_mask of the first few batches need changes if
        #   we use 16/4 mode.
        # 流式的情况
        if args['left_chunks'] > 0:  # 16/4
            onnx_att_mask[:, :, -(args['chunk_size'] * (i + 1)):] = 1
        ort_inputs = {
            'chunk': onnx_chunk,
            'offset': onnx_offset,
            'required_cache_size': onnx_required_cache_size,
            'att_cache': onnx_att_cache,
            'cnn_cache': onnx_cnn_cache,
            'att_mask': onnx_att_mask
        }
        # NOTE(xcsong): If we use 16/-1, -1/-1 or 16/0 mode, `next_cache_start`
        #   will be hardcoded to 0 or chunk_size by ONNX, thus
        #   required_cache_size and att_mask are no more needed and they will
        #   be removed by ONNX automatically.
        # 去掉不需要的数据
        for k in list(ort_inputs):
            if k not in input_names:
                ort_inputs.pop(k)
        # 使用 ONNX Runtime 推理会话运行模型，并处理模型的输出
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_att_cache, onnx_cnn_cache = ort_outs[1], ort_outs[2]
        onnx_output.append(ort_outs[0])
        onnx_offset += ort_outs[0].shape[1]
    # 拼接 ONNX 模型的输出
    onnx_output = np.concatenate(onnx_output, axis=1)

    # 比较 PyTorch 和 ONNX 模型的输出
    # to_numpy(torch_output) 将 PyTorch 模型的输出转换为 NumPy 数组。
    # onnx_output 是拼接后的 ONNX 模型的输出。
    # rtol=1e-03 和 atol=1e-05 分别表示相对误差和绝对误差的容忍度。如果两个数组在这些误差范围内相等，则测试通过；否则，抛出异常。
    np.testing.assert_allclose(to_numpy(torch_output),
                               onnx_output,
                               rtol=1e-03,
                               atol=1e-05)
    # 获取 ONNX 模型的元数据
    meta = ort_session.get_modelmeta()
    # 打印模型的自定义元数据映射
    print("\t\tcustom_metadata_map={}".format(meta.custom_metadata_map))
    print("\t\tCheck onnx_encoder, pass!")


# 导出ctc，这是一个单独的onnx，与GPU版本的不同
def export_ctc(asr_model, args):
    print("Stage-2: export ctc")
    ctc = asr_model.ctc
    ctc.forward = ctc.log_softmax
    ctc_outpath = os.path.join(args['output_dir'], 'ctc.onnx')

    # 给ctc模型准备输入
    print("\tStage-2.1: prepare inputs for ctc")
    hidden = torch.randn(
        (args['batch'], args['chunk_size'] if args['chunk_size'] > 0 else 16,
         args['output_size']))

    # 导出onnx_ctc
    print("\tStage-2.2: torch.onnx.export")
    dynamic_axes = {'hidden': {1: 'T'}, 'probs': {1: 'T'}}
    torch.onnx.export(ctc,
                      hidden,
                      ctc_outpath,
                      opset_version=13,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['hidden'],
                      output_names=['probs'],
                      dynamic_axes=dynamic_axes,
                      verbose=False)
    # 加载onnx_ctc
    onnx_ctc = onnx.load(ctc_outpath)
    for (k, v) in args.items():
        meta = onnx_ctc.metadata_props.add()
        meta.key, meta.value = str(k), str(v)
    onnx.checker.check_model(onnx_ctc)
    onnx.helper.printable_graph(onnx_ctc.graph)
    onnx.save(onnx_ctc, ctc_outpath)
    print_input_output_info(onnx_ctc, "onnx_ctc")
    # Dynamic quantization，动态量化，减少计算位数，提高计算速度
    model_fp32 = ctc_outpath
    model_quant = os.path.join(args['output_dir'], 'ctc.quant.onnx')
    quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
    print('\t\tExport onnx_ctc, done! see {}'.format(ctc_outpath))

    # 检查onnx_ctc和torch_ctc的差别，差别不大可以导出，差别大报错
    print("\tStage-2.3: check onnx_ctc and torch_ctc")
    torch_output = ctc(hidden)
    ort_session = onnxruntime.InferenceSession(
        ctc_outpath, providers=['CPUExecutionProvider'])
    onnx_output = ort_session.run(None, {'hidden': to_numpy(hidden)})

    np.testing.assert_allclose(to_numpy(torch_output),
                               onnx_output[0],
                               rtol=1e-03,
                               atol=1e-05)
    print("\t\tCheck onnx_ctc, pass!")


# 导出解码器
def export_decoder(asr_model, args):
    print("Stage-3: export decoder")
    # 用的wenet里的解码器和前向函数，输出路径用输入的
    decoder = asr_model
    # NOTE(lzhin): parameters of encoder will be automatically removed
    #   since they are not used during rescoring.
    # 注意(lzhin)：编码器的参数将会被自动移除
    #   因为在重新评分过程中它们并没有被使用。
    # 指定前向函数和输出路径
    decoder.forward = decoder.forward_attention_decoder
    decoder_outpath = os.path.join(args['output_dir'], 'decoder.onnx')

    # 准备转换解码器的输入
    print("\tStage-3.1: prepare inputs for decoder")
    # hardcode time->200 nbest->10 len->20, they are dynamic axes.
    encoder_out = torch.randn((1, 200, args['output_size']))
    hyps = torch.randint(low=0, high=args['vocab_size'], size=[10, 20])
    hyps[:, 0] = args['vocab_size'] - 1  # <sos>
    hyps_lens = torch.randint(low=15, high=21, size=[10])

    # 导出解码器
    print("\tStage-3.2: torch.onnx.export")
    # 可以浮动的数据
    dynamic_axes = {
        'hyps': {
            0: 'NBEST',
            1: 'L'
        },
        'hyps_lens': {
            0: 'NBEST'
        },
        'encoder_out': {
            1: 'T'
        },
        'score': {
            0: 'NBEST',
            1: 'L'
        },
        'r_score': {
            0: 'NBEST',
            1: 'L'
        }
    }
    # 指定输入项
    inputs = (hyps, hyps_lens, encoder_out, args['reverse_weight'])
    # 导出onnx解码器
    torch.onnx.export(
        decoder,
        inputs,
        decoder_outpath,
        opset_version=13,
        export_params=True,
        do_constant_folding=True,
        input_names=['hyps', 'hyps_lens', 'encoder_out', 'reverse_weight'],
        output_names=['score', 'r_score'],
        dynamic_axes=dynamic_axes,
        verbose=False)
    # 加载onnx解码器并加入metadata来保存参数
    onnx_decoder = onnx.load(decoder_outpath)
    for (k, v) in args.items():
        meta = onnx_decoder.metadata_props.add()
        meta.key, meta.value = str(k), str(v)
    # 检测模型的有效性并保存
    onnx.checker.check_model(onnx_decoder)
    onnx.helper.printable_graph(onnx_decoder.graph)
    onnx.save(onnx_decoder, decoder_outpath)
    # 打印输入信息
    print_input_output_info(onnx_decoder, "onnx_decoder")
    model_fp32 = decoder_outpath
    model_quant = os.path.join(args['output_dir'], 'decoder.quant.onnx')
    # 想量化模型，减少计算量
    quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
    print('\t\tExport onnx_decoder, done! see {}'.format(decoder_outpath))

    # 检测onnx和torch解码器的差别，差别小就结束差别大就报错
    print("\tStage-3.3: check onnx_decoder and torch_decoder")
    torch_score, torch_r_score = decoder(hyps, hyps_lens, encoder_out,
                                         args['reverse_weight'])
    ort_session = onnxruntime.InferenceSession(
        decoder_outpath, providers=['CPUExecutionProvider'])
    input_names = [node.name for node in onnx_decoder.graph.input]
    ort_inputs = {
        'hyps': to_numpy(hyps),
        'hyps_lens': to_numpy(hyps_lens),
        'encoder_out': to_numpy(encoder_out),
        'reverse_weight': np.array((args['reverse_weight'])),
    }
    for k in list(ort_inputs):
        if k not in input_names:
            ort_inputs.pop(k)
    onnx_output = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_score),
                               onnx_output[0],
                               rtol=1e-03,
                               atol=1e-05)
    if args['is_bidirectional_decoder'] and args['reverse_weight'] > 0.0:
        np.testing.assert_allclose(to_numpy(torch_r_score),
                                   onnx_output[1],
                                   rtol=1e-03,
                                   atol=1e-05)
    print("\t\tCheck onnx_decoder, pass!")


def main():
    # 设置随机数生成器种子，保证每次运行生成的随机数都一样，可以复现
    torch.manual_seed(777)
    # 获取输入信息
    args = get_args()
    # 设置日志级别为DEBUG
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # 获取输出路径
    output_dir = args.output_dir
    # 创建输出路径文件夹
    os.system("mkdir -p " + output_dir)
    # 禁用GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # 打开配置文件获取配置
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # 初始化pt模型
    model, configs = init_model(args, configs)
    # 将模型转换为评估模式
    model.eval()
    # 打印模型
    print(model)

    # 将所有输入数据存在arguments字典中
    arguments = {}
    arguments['output_dir'] = output_dir
    arguments['batch'] = 1
    arguments['chunk_size'] = args.chunk_size
    arguments['left_chunks'] = args.num_decoding_left_chunks
    arguments['reverse_weight'] = args.reverse_weight
    arguments['output_size'] = configs['encoder_conf']['output_size']
    arguments['num_blocks'] = configs['encoder_conf']['num_blocks']
    arguments['cnn_module_kernel'] = configs['encoder_conf'].get(
        'cnn_module_kernel', 1)
    arguments['head'] = configs['encoder_conf']['attention_heads']
    arguments['feature_size'] = configs['input_dim']
    arguments['vocab_size'] = configs['output_dim']
    # NOTE(xcsong): if chunk_size == -1, hardcode to 67
    arguments['decoding_window'] = (args.chunk_size - 1) * \
        model.encoder.embed.subsampling_rate + \
        model.encoder.embed.right_context + 1 if args.chunk_size > 0 else 67
    arguments['encoder'] = configs['encoder']
    arguments['decoder'] = configs['decoder']
    arguments['subsampling_rate'] = model.subsampling_rate()
    arguments['right_context'] = model.right_context()
    arguments['sos_symbol'] = model.sos_symbol()
    arguments['eos_symbol'] = model.eos_symbol()
    # 判断模型的解码器是否是双向的，如果是则设置为 1，否则设置为 0。
    arguments['is_bidirectional_decoder'] = 1 \
        if model.is_bidirectional_decoder() else 0

    # NOTE(xcsong): Please note that -1/-1 means non-streaming model! It is
    #   not a [16/4 16/-1 16/0] all-in-one model and it should not be used in
    #   streaming mode (i.e., setting chunk_size=16 in `decoder_main`). If you
    #   want to use 16/-1 or any other streaming mode in `decoder_main`,
    #   please export onnx in the same config.
    # 注意(xcsong)：请注意，-1/-1 表示非流式模型！它不是一个 [16/4 16/-1 16/0] 一体化模型，且不应该在流式模式下使用
    #   （即在 `decoder_main` 中设置 chunk_size=16）。如果你想在 `decoder_main` 中使用 16/-1 或任何其他流式模式，
    #   请在相同的配置下导出 ONNX 模型。
    # 流式的话要确定块尺寸大于0
    if arguments['left_chunks'] > 0:
        assert arguments['chunk_size'] > 0  # -1/4 not supported

    # 导出编码器.onnx,ctc.onnx,解码器.onnx
    export_encoder(model, arguments)
    export_ctc(model, arguments)
    export_decoder(model, arguments)


# 运行脚本
if __name__ == '__main__':
    main()
