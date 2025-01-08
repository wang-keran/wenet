# Copyright (c) 2022 Ximalaya Inc. (authors: Yuguang Yang)
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
# Modified from Squeezeformer(https://github.com/kssteven418/Squeezeformer)
#               Squeezeformer(https://github.com/upskyy/Squeezeformer)
#               NeMo(https://github.com/NVIDIA/NeMo)
"""DepthwiseConv2dSubsampling4 and TimeReductionLayer definition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from wenet.transformer.subsampling import BaseSubsampling
from typing import Tuple
from wenet.squeezeformer.conv2d import Conv2dValid


# 该类实现了通过深度卷积进行 2D 子采样，将输入的特征长度缩减到原来的 1/4，通过减少特征的时间维度来降低计算复杂性
class DepthwiseConv2dSubsampling4(BaseSubsampling):
    """Depthwise Convolutional 2D subsampling (to 1/4 length).

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            pos_enc_class (nn.Module): position encoding class.
            dw_stride (int): Whether do depthwise convolution.
            input_size (int): filter bank dimension.

        """

    # 初始化方法
    def __init__(self,
                 idim: int,
                 odim: int,
                 pos_enc_class: torch.nn.Module,
                 dw_stride: bool = False,
                 input_size: int = 80,
                 input_dropout_rate: float = 0.1,
                 init_weights: bool = True):
        super(DepthwiseConv2dSubsampling4, self).__init__()
        self.idim = idim
        self.odim = odim
        self.pw_conv = nn.Conv2d(in_channels=idim,
                                 out_channels=odim,
                                 kernel_size=3,
                                 stride=2)
        self.act1 = nn.ReLU()
        self.dw_conv = nn.Conv2d(in_channels=odim,
                                 out_channels=odim,
                                 kernel_size=3,
                                 stride=2,
                                 groups=odim if dw_stride else 1)
        self.act2 = nn.ReLU()
        self.pos_enc = pos_enc_class
        self.input_proj = nn.Sequential(
            nn.Linear(odim * (((input_size - 1) // 2 - 1) // 2), odim),
            nn.Dropout(p=input_dropout_rate),
        )
        if init_weights:
            linear_max = (odim * input_size / 4)**-0.5
            torch.nn.init.uniform_(self.input_proj.state_dict()['0.weight'],
                                   -linear_max, linear_max)
            torch.nn.init.uniform_(self.input_proj.state_dict()['0.bias'],
                                   -linear_max, linear_max)
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    # 前向传播方法
    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.pw_conv(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        x = self.act2(x)
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(b, t, c * f)
        x, pos_emb = self.pos_enc(x, offset)
        x = self.input_proj(x)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


# 主要用于对音频信号在时间维度上进行降采样。该类是基于 NeMo 框架的 Squeezeformer 模型中提取的时间降采样层的实现。
# 实现了通过深度卷积和逐点卷积对输入音频信号进行时间降采样的过程。降采样可以减少输入数据的时间长度，从而提高后续处理的效率。
class TimeReductionLayer1D(nn.Module):
    """
    Modified NeMo,
    Squeezeformer Time Reduction procedure.
    Downsamples the audio by `stride` in the time dimension.
    Args:
        channel (int): input dimension of
                       MultiheadAttentionMechanism and PositionwiseFeedForward
        out_dim (int): Output dimension of the module.
        kernel_size (int): Conv kernel size for
                           depthwise convolution in convolution module
        stride (int): Downsampling factor in time dimension.
    """

    # 初始化方法
    def __init__(self,
                 channel: int,
                 out_dim: int,
                 kernel_size: int = 5,
                 stride: int = 2):
        super(TimeReductionLayer1D, self).__init__()

        self.channel = channel
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = max(0, self.kernel_size - self.stride)

        self.dw_conv = nn.Conv1d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            groups=channel,
        )

        self.pw_conv = nn.Conv1d(
            in_channels=channel,
            out_channels=out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )

        self.init_weights()

    # 权重初始化方法：该方法使用均匀分布对深度卷积和逐点卷积的权重和偏置进行初始化，确保初始化时权重不会过大或过小，以便模型能够有效学习。
    def init_weights(self):
        dw_max = self.kernel_size**-0.5
        pw_max = self.channel**-0.5
        torch.nn.init.uniform_(self.dw_conv.weight, -dw_max, dw_max)
        torch.nn.init.uniform_(self.dw_conv.bias, -dw_max, dw_max)
        torch.nn.init.uniform_(self.pw_conv.weight, -pw_max, pw_max)
        torch.nn.init.uniform_(self.pw_conv.bias, -pw_max, pw_max)

    # 前向传播方法
    def forward(
            self,
            xs,
            xs_lens: torch.Tensor,
            mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ):
        xs = xs.transpose(1, 2)  # [B, C, T]
        xs = xs.masked_fill(mask_pad.eq(0), 0.0)

        xs = self.dw_conv(xs)
        xs = self.pw_conv(xs)

        xs = xs.transpose(1, 2)  # [B, T, C]

        B, T, D = xs.size()
        mask = mask[:, ::self.stride, ::self.stride]
        mask_pad = mask_pad[:, :, ::self.stride]
        L = mask_pad.size(-1)
        # For JIT exporting, we remove F.pad operator.
        if L - T < 0:
            xs = xs[:, :L - T, :].contiguous()
        else:
            dummy_pad = torch.zeros(B, L - T, D, device=xs.device)
            xs = torch.cat([xs, dummy_pad], dim=1)

        xs_lens = torch.div(xs_lens + 1, 2, rounding_mode='trunc')
        return xs, xs_lens, mask, mask_pad


# 该类主要通过深度卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution）来实现 2D 时间降采样，适用于处理时频特征图
class TimeReductionLayer2D(nn.Module):

    # 初始化方法
    def __init__(self,
                 kernel_size: int = 5,
                 stride: int = 2,
                 encoder_dim: int = 256):
        super(TimeReductionLayer2D, self).__init__()
        self.encoder_dim = encoder_dim
        self.kernel_size = kernel_size
        self.dw_conv = Conv2dValid(in_channels=encoder_dim,
                                   out_channels=encoder_dim,
                                   kernel_size=(kernel_size, 1),
                                   stride=stride,
                                   valid_trigy=True)
        self.pw_conv = Conv2dValid(
            in_channels=encoder_dim,
            out_channels=encoder_dim,
            kernel_size=1,
            stride=1,
            valid_trigx=False,
            valid_trigy=False,
        )

        self.kernel_size = kernel_size
        self.stride = stride
        self.init_weights()

    # 权重初始化方法
    def init_weights(self):
        dw_max = self.kernel_size**-0.5
        pw_max = self.encoder_dim**-0.5
        torch.nn.init.uniform_(self.dw_conv.weight, -dw_max, dw_max)
        torch.nn.init.uniform_(self.dw_conv.bias, -dw_max, dw_max)
        torch.nn.init.uniform_(self.pw_conv.weight, -pw_max, pw_max)
        torch.nn.init.uniform_(self.pw_conv.bias, -pw_max, pw_max)

    # 前向传播方法
    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xs = xs.masked_fill(mask_pad.transpose(1, 2).eq(0), 0.0)
        xs = xs.unsqueeze(2)
        padding1 = self.kernel_size - self.stride
        xs = F.pad(xs, (0, 0, 0, 0, 0, padding1, 0, 0),
                   mode='constant',
                   value=0.)
        xs = self.dw_conv(xs.permute(0, 3, 1, 2))
        xs = self.pw_conv(xs).permute(0, 3, 2, 1).squeeze(1).contiguous()
        tmp_length = xs.size(1)
        xs_lens = torch.div(xs_lens + 1, 2, rounding_mode='trunc')
        padding2 = max(0, (xs_lens.max() - tmp_length).data.item())
        batch_size, hidden = xs.size(0), xs.size(-1)
        dummy_pad = torch.zeros(batch_size, padding2, hidden, device=xs.device)
        xs = torch.cat([xs, dummy_pad], dim=1)
        mask = mask[:, ::2, ::2]
        mask_pad = mask_pad[:, :, ::2]
        return xs, xs_lens, mask, mask_pad


# 该类实现了 Squeezeformer 的时间降采样功能，主要用于处理音频信号。
# 通过使用深度卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution），在时间维度上降低输入的长度。
class TimeReductionLayerStream(nn.Module):
    """
    Squeezeformer Time Reduction procedure.
    Downsamples the audio by `stride` in the time dimension.
    Args:
        channel (int): input dimension of
            MultiheadAttentionMechanism and PositionwiseFeedForward
        out_dim (int): Output dimension of the module.
        kernel_size (int): Conv kernel size for
            depthwise convolution in convolution module
        stride (int): Downsampling factor in time dimension.
    """

    # 初始化方法
    def __init__(self,
                 channel: int,
                 out_dim: int,
                 kernel_size: int = 1,
                 stride: int = 2):
        super(TimeReductionLayerStream, self).__init__()

        self.channel = channel
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride

        self.dw_conv = nn.Conv1d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=channel,
        )

        self.pw_conv = nn.Conv1d(
            in_channels=channel,
            out_channels=out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )

        self.init_weights()

    # 权重初始化方法
    def init_weights(self):
        dw_max = self.kernel_size**-0.5
        pw_max = self.channel**-0.5
        torch.nn.init.uniform_(self.dw_conv.weight, -dw_max, dw_max)
        torch.nn.init.uniform_(self.dw_conv.bias, -dw_max, dw_max)
        torch.nn.init.uniform_(self.pw_conv.weight, -pw_max, pw_max)
        torch.nn.init.uniform_(self.pw_conv.bias, -pw_max, pw_max)

    # 前向传播方法
    def forward(
            self,
            xs,
            xs_lens: torch.Tensor,
            mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ):
        xs = xs.transpose(1, 2)  # [B, C, T]
        xs = xs.masked_fill(mask_pad.eq(0), 0.0)

        xs = self.dw_conv(xs)
        xs = self.pw_conv(xs)

        xs = xs.transpose(1, 2)  # [B, T, C]

        B, T, D = xs.size()
        mask = mask[:, ::self.stride, ::self.stride]
        mask_pad = mask_pad[:, :, ::self.stride]
        L = mask_pad.size(-1)
        # For JIT exporting, we remove F.pad operator.
        if L - T < 0:
            xs = xs[:, :L - T, :].contiguous()
        else:
            dummy_pad = torch.zeros(B, L - T, D, device=xs.device)
            xs = torch.cat([xs, dummy_pad], dim=1)

        xs_lens = torch.div(xs_lens + 1, 2, rounding_mode='trunc')
        return xs, xs_lens, mask, mask_pad

# 总结：实现了基于squeezeformer的下采样机制。