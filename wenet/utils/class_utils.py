#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
import torch
from torch.nn import BatchNorm1d, LayerNorm
from wenet.paraformer.embedding import ParaformerPositinoalEncoding
from wenet.transformer.norm import RMSNorm
from wenet.transformer.positionwise_feed_forward import (
    GatedVariantsMLP, MoEFFNLayer, PositionwiseFeedForward)

from wenet.transformer.swish import Swish
from wenet.transformer.subsampling import (
    LinearNoSubsampling,
    EmbedinigNoSubsampling,
    Conv1dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    StackNFramesSubsampling,
)
from wenet.efficient_conformer.subsampling import Conv2dSubsampling2
from wenet.squeezeformer.subsampling import DepthwiseConv2dSubsampling4
from wenet.transformer.embedding import (PositionalEncoding,
                                         RelPositionalEncoding,
                                         RopePositionalEncoding,
                                         WhisperPositionalEncoding,
                                         LearnablePositionalEncoding,
                                         NoPositionalEncoding)
from wenet.transformer.attention import (MultiHeadedAttention,
                                         MultiHeadedCrossAttention,
                                         RelPositionMultiHeadedAttention,
                                         RopeMultiHeadedAttention,
                                         ShawRelPositionMultiHeadedAttention)
from wenet.efficient_conformer.attention import (
    GroupedRelPositionMultiHeadedAttention)

# 这段代码定义了一组用于构建深度学习模型的类字典，这些字典包含了不同类型的激活函数、RNN 模块、子采样层、嵌入层、注意力机制、前馈网络和归一化层。
# 激活函数字典：将常用的激活函数与字符串标签关联，便于根据名称动态获取相应的激活函数类。
WENET_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}

# RNN 模块字典：提供不同类型的循环神经网络（RNN）模型。
WENET_RNN_CLASSES = {
    "rnn": torch.nn.RNN,
    "lstm": torch.nn.LSTM,
    "gru": torch.nn.GRU,
}

# 子采样层字典：将不同的子采样方法与标签关联。
WENET_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d2": Conv2dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "dwconv2d4": DepthwiseConv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    'paraformer_dummy': torch.nn.Identity,
    'stack_n_frames': StackNFramesSubsampling,
}

# 嵌入层字典：提供不同的嵌入方式，特别是在序列建模中使用的位置信息。
WENET_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
    "abs_pos_paraformer": ParaformerPositinoalEncoding,
    'rope_pos': RopePositionalEncoding,
}

# 注意力机制字典：提供不同的注意力机制。
WENET_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
    "grouped_rel_selfattn": GroupedRelPositionMultiHeadedAttention,
    "crossattn": MultiHeadedCrossAttention,
    'shaw_rel_selfattn': ShawRelPositionMultiHeadedAttention,
    'rope_abs_selfattn': RopeMultiHeadedAttention,
}

# 前馈神经网络字典：提供不同类型的前馈网络。
WENET_MLP_CLASSES = {
    'position_wise_feed_forward': PositionwiseFeedForward,
    'moe': MoEFFNLayer,
    'gated': GatedVariantsMLP
}

# 归一化层字典：提供不同的归一化方法。
WENET_NORM_CLASSES = {
    'layer_norm': LayerNorm,
    'batch_norm': BatchNorm1d,
    'rms_norm': RMSNorm
}

# 总结：这些字典通过将不同类型的模块（如激活函数、RNN、子采样、嵌入、注意力机制、前馈网络和归一化层）与其对应的字符串标签关联，为构建深度学习模型提供了灵活性。