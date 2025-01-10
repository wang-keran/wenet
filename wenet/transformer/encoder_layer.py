# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder self-attention layer definition."""

from functools import partial
from typing import Optional, Tuple

import torch
from torch import nn
from wenet.transformer.attention import T_CACHE

from wenet.utils.class_utils import WENET_NORM_CLASSES


# 表示一个 Transformer 编码器层。
class TransformerEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
    """

    # 初始化
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        rms_norm_offset: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        assert layer_norm_type in ['layer_norm', 'rms_norm']

        norm_class = WENET_NORM_CLASSES[layer_norm_type]
        if layer_norm_type == "rms_norm":
            norm_class = partial(
                norm_class,
                add_unit_offset=rms_norm_offset,
            )
        self.norm1 = norm_class(size, eps=norm_eps)
        self.norm2 = norm_class(size, eps=norm_eps)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    # 计算编码特征。
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: T_CACHE = (torch.zeros(
            (0, 0, 0, 0)), torch.zeros((0, 0, 0, 0))),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): just for interface compatibility
                to ConformerEncoderLayer
            mask_pad (torch.Tensor): does not used in transformer layer,
                just for unified api with conformer.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2), not used here, it's for interface
                compatibility to ConformerEncoderLayer.
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch=1, size, cache_t2).

        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        x_att, new_att_cache = self.self_attn(x,
                                              x,
                                              x,
                                              mask,
                                              pos_emb,
                                              cache=att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        fake_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        return x, mask, new_att_cache, fake_cnn_cache


# 表示一个 Conformer 编码器层，结合了卷积模块和自注意力机制。
class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """

    # 初始化各个模块和归一化层，使用与 TransformerEncoderLayer 相同的逻辑。
    # 处理 feed_forward_macaron 和 conv_module，创建相应的归一化层。
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        # WENET_NORM_CLASSES除了RMSNorm以外全是torch.nn自带的类
        self.norm_ff = WENET_NORM_CLASSES[layer_norm_type](
            size, eps=norm_eps)  # for the FNN module
        self.norm_mha = WENET_NORM_CLASSES[layer_norm_type](
            size, eps=norm_eps)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = WENET_NORM_CLASSES[layer_norm_type](
                size, eps=norm_eps)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = WENET_NORM_CLASSES[layer_norm_type](
                size, eps=norm_eps)  # for the CNN module
            self.norm_final = WENET_NORM_CLASSES[layer_norm_type](
                size, eps=norm_eps)  # for the final output of the block
        # 这是 PyTorch 中的 Dropout 层，用于对神经网络的输出进行随机失活。
        # dropout_rate 是失活的概率，表示在每次训练时，有多少比例的神经元输出会被设置为零。
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    # 计算编码特征，支持 Conformer 特有的模块。
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: T_CACHE = (torch.zeros(
            (0, 0, 0, 0)), torch.zeros((0, 0, 0, 0))),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

        # whether to use macaron style
        # 这段代码实现了马卡龙风格的前馈神经网络模块，该模块可以选择性地在前馈计算前或计算后进行归一化，并使用残差连接来增强模型的稳定性和学习能力。
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        # 实现了多头自注意力模块
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb,
                                              att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                # 调用了convolution.py里的forward函数
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache

# TransformerEncoderLayer 和 ConformerEncoderLayer 是构建 Transformer 和 Conformer 模型的基础组件。
# 它们实现了多头自注意力机制、前馈网络和可选的卷积模块。
# 两个类都允许在每个子模块之前或之后应用层归一化，提供了灵活性以适应不同的网络架构需求。