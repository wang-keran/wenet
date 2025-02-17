# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
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

import os
import torch

from wenet.finetune.lora.utils import (inject_lora_to_model,
                                       mark_only_lora_as_trainable)
from wenet.k2.model import K2Model
from wenet.paraformer.cif import Cif
from wenet.paraformer.layers import SanmDecoder, SanmEncoder
from wenet.paraformer.paraformer import Paraformer, Predictor
from wenet.ssl.init_model import WENET_SSL_MODEL_CLASS
from wenet.transducer.joint import TransducerJoint
from wenet.transducer.predictor import (ConvPredictor, EmbeddingPredictor,
                                        RNNPredictor)
from wenet.transducer.transducer import Transducer
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.encoder import TransformerEncoder, ConformerEncoder
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.branchformer.encoder import BranchformerEncoder
from wenet.e_branchformer.encoder import EBranchformerEncoder
from wenet.squeezeformer.encoder import SqueezeformerEncoder
from wenet.efficient_conformer.encoder import EfficientConformerEncoder
from wenet.ctl_model.encoder import DualTransformerEncoder, DualConformerEncoder
from wenet.ctl_model.asr_model_ctl import CTLModel
from wenet.whisper.whisper import Whisper
from wenet.utils.cmvn import load_cmvn
from wenet.utils.checkpoint import load_checkpoint, load_trained_modules

WENET_ENCODER_CLASSES = {
    "transformer": TransformerEncoder,
    "conformer": ConformerEncoder,
    "squeezeformer": SqueezeformerEncoder,
    "efficientConformer": EfficientConformerEncoder,
    "branchformer": BranchformerEncoder,
    "e_branchformer": EBranchformerEncoder,
    "dual_transformer": DualTransformerEncoder,
    "dual_conformer": DualConformerEncoder,
    'sanm_encoder': SanmEncoder,
}

WENET_DECODER_CLASSES = {
    "transformer": TransformerDecoder,
    "bitransformer": BiTransformerDecoder,
    "sanm_decoder": SanmDecoder,
}

WENET_CTC_CLASSES = {
    "ctc": CTC,
}

WENET_PREDICTOR_CLASSES = {
    "rnn": RNNPredictor,
    "embedding": EmbeddingPredictor,
    "conv": ConvPredictor,
    "cif_predictor": Cif,
    "paraformer_predictor": Predictor,
}

WENET_JOINT_CLASSES = {
    "transducer_joint": TransducerJoint,
}

WENET_MODEL_CLASSES = {
    "asr_model": ASRModel,
    "ctl_model": CTLModel,
    "whisper": Whisper,
    "k2_model": K2Model,
    "transducer": Transducer,
    'paraformer': Paraformer,
}


# 根据给定的 args 和 configs 初始化模型。此函数会选择适当的编码器、解码器、CTC 模块，设置一些配置并返回一个完整的模型实例和更新的配置。
def init_speech_model(args, configs):
    # TODO(xcsong): Forcefully read the 'cmvn' attribute.
    # 配置 CMVN（均值方差归一化）
    if configs.get('cmvn', None) == 'global_cmvn':
        mean, istd = load_cmvn(configs['cmvn_conf']['cmvn_file'],
                               configs['cmvn_conf']['is_json_cmvn'])    #原本这样在aishell里可以运行，aishell2不能运行configs['cmvn_conf']['is_json_cmvn']
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    # input_dim 通常用于表示输入数据的维度。例如，在神经网络中，如果输入是一个向量或矩阵，input_dim 可能表示输入向量的特征数或矩阵的列数。
    input_dim = configs['input_dim']
    # vocab_size 通常用于表示输出空间的大小。它通常在自然语言处理（NLP）任务中使用，表示词汇表的大小，例如在一个语言模型或分类任务中，模型的输出维度就是词汇表的大小，或者是类别的数量。
    vocab_size = configs['output_dim']

    # 编码器、解码器、CTC 的初始化
    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')
    ctc_type = configs.get('ctc', 'ctc')

    # 先到最上面的类中，然后进入对应名称的类中进行初始化，都是拿torch实现的
    encoder = WENET_ENCODER_CLASSES[encoder_type](
        input_dim,
        global_cmvn=global_cmvn,
        **configs['encoder_conf'],
        **configs['encoder_conf']['efficient_conf']
        if 'efficient_conf' in configs['encoder_conf'] else {})

    # 先到最上面的类中，然后进入对应名称的类中进行初始化
    # 到decoder.py中，使用BiTransformerDecoder类，再到TransformerDecoder类，都是拿torch实现的
    decoder = WENET_DECODER_CLASSES[decoder_type](vocab_size,
                                                  encoder.output_size(),
                                                  **configs['decoder_conf'])

    # 先到最上面的类中，然后进入对应名称的类中进行初始化，这里也拿torch实现
    ctc = WENET_CTC_CLASSES[ctc_type](
        vocab_size,
        encoder.output_size(),
        blank_id=configs['ctc_conf']['ctc_blank_id']
        if 'ctc_conf' in configs else 0)

    # 特定模型类型的处理
    # 获取模型类型。默认模型是 asr_model。
    model_type = configs.get('model', 'asr_model')
    # 对于 "transducer"：
    if model_type == "transducer":
        # 获取预测器类型和联合模块类型（predictor_type 和 joint_type）。
        predictor_type = configs.get('predictor', 'rnn')
        joint_type = configs.get('joint', 'transducer_joint')
        # 从 WENET_PREDICTOR_CLASSES 和 WENET_JOINT_CLASSES 字典中获取相应的类，并传入配置信息初始化 predictor 和 joint 对象。
        predictor = WENET_PREDICTOR_CLASSES[predictor_type](
            vocab_size, **configs['predictor_conf'])
        joint = WENET_JOINT_CLASSES[joint_type](vocab_size,
                                                **configs['joint_conf'])
        # 然后创建 transducer 类型的模型实例。
        model = WENET_MODEL_CLASSES[model_type](
            vocab_size=vocab_size,
            blank=0,
            predictor=predictor,
            encoder=encoder,
            attention_decoder=decoder,
            joint=joint,
            ctc=ctc,
            special_tokens=configs.get('tokenizer_conf',
                                       {}).get('special_tokens', None),
            **configs['model_conf'])
    # 对于 "paraformer"：
    elif model_type == 'paraformer':
        # 获取预测器类型并从 WENET_PREDICTOR_CLASSES 中获取类实例。
        predictor_type = configs.get('predictor', 'cif')
        predictor = WENET_PREDICTOR_CLASSES[predictor_type](
            **configs['predictor_conf'])
        # 使用 paraformer 的配置和组件创建模型实例。
        model = WENET_MODEL_CLASSES[model_type](
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            ctc=ctc,
            **configs['model_conf'],
            special_tokens=configs.get('tokenizer_conf',
                                       {}).get('special_tokens', None),
        )
    # 根据模型类型初始化 SSL（自监督学习）模型。
    elif model_type in WENET_SSL_MODEL_CLASS.keys():
        from wenet.ssl.init_model import init_model as init_ssl_model
        model = init_ssl_model(configs, encoder)
    # 否则，使用默认的 asr_model 配置创建模型，从上面的transformer得到的编码解码器和CTC损失，最底层都用了torch.nn系列的方法。
    else:
        model = WENET_MODEL_CLASSES[model_type](
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            special_tokens=configs.get('tokenizer_conf',
                                       {}).get('special_tokens', None),
            **configs['model_conf'])
    return model, configs


# 函数 init_model 根据配置初始化模型，并进行一些额外的设置，如加载检查点、注入 LoRA、绑定权重等。
def init_model(args, configs):

    # 从配置中获取模型类型，如果未指定则默认为 'asr_model'。
    model_type = configs.get('model', 'asr_model')
    # 将模型类型写回配置。
    configs['model'] = model_type
    model, configs = init_speech_model(args, configs)

    # 如果参数中指定了使用 LoRA，则调用 inject_lora_to_model 函数将 LoRA 注入到模型中。Lora作用是什么
    if hasattr(args, 'use_lora') and args.use_lora:
        inject_lora_to_model(model, configs['lora_conf'])

    # If specify checkpoint, load some info from checkpoint
    # 如果参数中指定了检查点，则调用 load_checkpoint 函数加载检查点信息。
    if hasattr(args, 'checkpoint') and args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    # 如果参数中指定了编码器初始化，则调用 load_trained_modules 函数加载训练好的模块。
    elif hasattr(args, 'enc_init') and args.enc_init is not None:
        infos = load_trained_modules(model, args)
    # 否则，初始化一个空的字典。
    else:
        infos = {}
    # 将加载的信息写入配置。
    configs["init_infos"] = infos

    # 如果参数中指定了使用 LoRA，并且指定了 LoRA 检查点路径，则调用 load_checkpoint 函数加载 LoRA 检查点。
    if hasattr(args, 'use_lora') and args.use_lora:
        if hasattr(args, 'lora_ckpt_path') and args.lora_ckpt_path:
            load_checkpoint(model, args.lora_ckpt_path)

    # Trye to tie some weights
    # 如果模型具有 tie_or_clone_weights 方法，则调用该方法绑定或克隆权重。
    if hasattr(model, 'tie_or_clone_weights'):
        # 如果参数中未指定 jit，则将其设置为 True。
        if not hasattr(args, 'jit'):
            jit = True  # i.e. export onnx/jit/ipex
        else:
            jit = False
        model.tie_or_clone_weights(jit)

    # 仅优化 LoRA：如果参数中指定了仅优化 LoRA，则调用 mark_only_lora_as_trainable 函数将模型中仅 LoRA 部分设置为可训练。
    if hasattr(args, 'only_optimize_lora') and args.only_optimize_lora:
        mark_only_lora_as_trainable(model, bias='lora_only')

    # 打印配置（仅 rank 0）：如果当前进程的 rank 为 0，则打印配置。
    if int(os.environ.get('RANK', 0)) == 0:
        print(configs)

    # 返回模型和配置
    return model, configs

# 总结：初始化模型的脚本
# RANK 是一个环境变量，用于标识当前进程在所有进程中的唯一编号。
# RANK=0 表示当前进程是主进程，通常负责一些全局性的操作。
# 在分布式计算中，通过检查 RANK 的值，可以控制某些操作仅在特定进程中执行，以避免重复和冲突。