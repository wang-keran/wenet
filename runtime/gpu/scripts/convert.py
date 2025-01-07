#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import yaml
import os
import subprocess
import onnx

if __name__ == "__main__":
    # 创建一个解析器对象，描述为“生成 model_repo 的 config.pbtxt”。
    parser = argparse.ArgumentParser(
        description='generate config.pbtxt for model_repo')
    # 添加了多个命令行参数，包括 --config、--vocab、--model_repo、--onnx_model_dir 和 --lm_path。
    # 其中 --config（配置文件路径）、--vocab（音频路径） 和 --model_repo(模型仓库路径) 是必需的，而 --onnx_model_dir（onnx模型路径） 和 --lm_path （大模型路径）是可选的。
    # 给parse添加config文件路径：--config=$onnx_model_dir/train.yaml，必需参数
    parser.add_argument('--config', required=True, help='config file')
    # 给parse添加字表映射文件--vocab=$onnx_model_dir/words.txt ，必需参数
    parser.add_argument('--vocab',
                        required=True,
                        help='vocabulary file, units.txt')
    # 添加模型仓库路径，必需参数
    parser.add_argument('--model_repo',
                        required=True,
                        help='model repo directory')
    # 添加onnx模型仓库路径，必需参数
    parser.add_argument('--onnx_model_dir',
                        default=True,
                        type=str,
                        required=False,
                        help="onnx model path")
    # 添加大模型路径，非必需参数
    parser.add_argument('--lm_path',
                        default=None,
                        type=str,
                        required=False,
                        help="the additional language model path")
    # 使用 parse_args() 方法解析命令行参数，并将结果存储在 args 对象中。
    args = parser.parse_args()
    # 打开并读取 args.config 指定的配置文件onnx_model_dir/train.yaml训练配置文件，并使用 yaml.load 加载其内容到 configs 字典中。
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # 打开并读取 args.config 指定的配置文件onnx_model_dir/config.yaml 模型配置文件,并使用 yaml.load 加载其内容到 onnx_configs 字典中。
    with open(os.path.join(args.onnx_model_dir, 'config.yaml'), 'r') as fin:
        onnx_configs = yaml.load(fin, Loader=yaml.FullLoader)

    # 初始化一个包含默认值的模型参数字典 model_params，如果没有输入的话可以按默认的字典运行。
    params = [("#beam_size", 10), ("#num_mel_bins", 80), ("#frame_shift", 10),
              ("#frame_length", 25), ("#sample_rate", 16000),
              ("#output_size", 256), ("#lm_path", ""), ("#bidecoder", 0),
              ("#vocabulary_path", ""), ("#DTYPE", "FP32")]
    model_params = dict(params)
    # fill values,填写数值,将train.yaml训练配置文件的数据写到初始化的字典里
    model_params["#beam_size"] = onnx_configs["beam_size"]
    # 需要使用fp16格式的模型的话将DTYPE赋值为FP16,否则默认FP32
    if onnx_configs["fp16"]:
        model_params["#DTYPE"] = "FP16"
    feature_conf = configs["dataset_conf"]["fbank_conf"]
    #下面三条是dataset_conf下的fbank_conf里的数据   {dataset_conf[fbank_conf(num_mel_bins)]}
    model_params["#num_mel_bins"] = feature_conf["num_mel_bins"]
    model_params["#frame_shift"] = feature_conf["frame_shift"]
    model_params["#frame_length"] = feature_conf["frame_length"]
    dataset_conf = configs["dataset_conf"]["resample_conf"]
    # 下面一条是dataset_conf下的resample_rate里的数据 
    # resample_rate 是一个术语，通常用于音频处理领域，指的是重新采样率或重采样过程中的目标采样率。
    # 在数字音频中，采样率是指每秒从连续模拟信号中捕获的样本数量，以赫兹（Hz）为单位。
    # 采样率赋值为train.yaml中的重新采样率
    model_params["#sample_rate"] = dataset_conf["resample_rate"]
    # 获取输出的大小,这两个值相同
    model_params["#output_size"] = configs["encoder_conf"]["output_size"]   
    model_params["#encoder_output_size"] = model_params["#output_size"]     # 注意力维度
    # 赋值大模型路径
    model_params["#lm_path"] = args.lm_path
    if configs["decoder"].startswith("bi"):
        model_params["#bidecoder"] = 1
        # 获取音频路径
    model_params["#vocabulary_path"] = args.vocab
    # 获取train.yaml里的词汇表大小
    model_params["#vocab_size"] = configs["output_dim"]

    # 检查onnx_config数组表（读取的onnx_model文件夹里的config.yaml配置文件）是否有decoding_window的文件，如果有就是1或true
    streaming = "decoding_window" in onnx_configs
    if streaming:
        # add streaming model parameters，填写流式
        chunk_size = onnx_configs["decoding_chunk_size"]
        num_left_chunks = onnx_configs["num_decoding_left_chunks"]
        cache_size = chunk_size * num_left_chunks
        model_params["#cache_size"] = cache_size
        subsampling_rate = onnx_configs["subsampling_rate"]
        # 从上面train.yaml获取的
        frame_shift = model_params["#frame_shift"]
        chunk_seconds = (chunk_size * subsampling_rate * frame_shift) / 1000
        model_params["#chunk_size_in_seconds"] = chunk_seconds
        # 从train.yaml中获取的
        model_params["#num_layers"] = configs["encoder_conf"]["num_blocks"]
        model_params["#context"] = onnx_configs["context"]
        model_params["#cnn_module_cache"] = onnx_configs[
            "cnn_module_kernel_cache"]
        model_params["#decoding_window"] = onnx_configs["decoding_window"]
        # 从train.yaml中获取的
        head = configs["encoder_conf"]["attention_heads"]
        model_params["#num_head"] = head
        # 从train.yaml中获取的
        d_k = configs["encoder_conf"]["output_size"] // head
        model_params["#att_cache_output_size"] = d_k * 2

    # 模型类型判断：根据不同类型使用不同的pbtxt模板,model是模型仓库/ws/model_repo里的模型
    for model in os.listdir(args.model_repo):
        template = "config_template.pbtxt"
        # non u2++ decoder不是bi开头的情况，单向解码器非双向，使用template2.pbtxt模板
        if "decoder" == model and model_params["#bidecoder"] == 0:
            template = "config_template2.pbtxt"
        # streaming transformer encoder流式transformer编码器的情况下用模板2
        if "encoder" == model and model_params.get("#cnn_module_cache",
                                                   -1) == 0:
            template = "config_template2.pbtxt"

        # 在模型仓库输出新的config.pbtxt，此时没有值，都是“#对象元素名”的形式
        model_dir = os.path.join(args.model_repo, model)
        out = os.path.join(model_dir, "config.pbtxt")
        out = open(out, "w")

        # 模型文件复制根据是否为fp16
        if model in ("decoder", "encoder"):
            if onnx_configs["fp16"]:
                model_name = model + "_fp16.onnx"
            else:
                model_name = model + ".onnx"
            # 目前的onnx模型的位置和文件名称
            source_model = os.path.join(args.onnx_model_dir, model_name)
            # 要传入的onnx模型的位置和新文件名称
            target_model = os.path.join(model_dir, "1", model + ".onnx")
            # 通过系统接口拷贝文件并且不用shell避免注入
            res = subprocess.call(["cp", source_model, target_model],
                                  shell=False)
            # encoder模型时
            if model == "encoder":
                # currently, with torch 1.10, the
                # exported conformer encoder output size is -1
                # Solution: Please upgrade your torch version
                # torch version >= 1.11.0 should fix this issue
                # 加载源模型
                model = onnx.load(source_model)
                # 如果是流式的编码，编码器的输出是图的第三个输出。在onnx手册中或者模型输出规定中
                if streaming:
                    encoder_out = model.graph.output[2]
                    # 否则是第一个输出
                else:
                    encoder_out = model.graph.output[0]
                    # 设置输出词汇表的大小
                output_dim = encoder_out.type.tensor_type.shape.dim[
                    2].dim_param
                # 编码输出大小变成-1注意力
                if output_dim.startswith("Add"):
                    model_params["#encoder_output_size"] = -1

        # 代码会读取模板文件 config_template.pbtxt 或 config_template2.pbtxt，并根据模型参数生成最终的配置文件 config.pbtxt。
        with open(os.path.join(model_dir, template), "r",
                  encoding="utf-8") as f:
            for line in f:
                # 跳过注释
                if line.startswith("#"):
                    continue
                # 每一行都遍历一遍param键值对并对同名的进行替换生成新的config.pbtxt
                for key, value in model_params.items():
                    line = line.replace(key, str(value))
                # 输出到config.pbtxt中
                out.write(line)
        # 关闭out
        out.close()

#1.if streaming:
#                    encoder_out = model.graph.output[2]里面的model.graph.output[2]从模型的计算图中获取特定的输出节点

#总结，读取train.yaml和模型的config.yaml并依靠config模板pbtxt输出config.pbtxt配置文件，然后输出新的onnx到model_repo，最后给config.pbtxt赋值
