# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#               2022 Ximalaya Inc. (authors: Yuguang Yang)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""ConvolutionModule definition."""

from typing import Tuple

import torch
from torch import nn


# 实现了一个卷积模块，常用于 Conformer 模型。该模块包含多个卷积层和归一化层，并支持因果卷积以及可自适应缩放等功能。
class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    # 初始化方法
    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True,
                 adaptive_scale: bool = False,
                 init_weights: bool = False):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()
        self.bias = bias
        self.channels = channels
        self.kernel_size = kernel_size
        self.adaptive_scale = adaptive_scale
        self.ada_scale = torch.nn.Parameter(torch.ones([1, 1, channels]),
                                            requires_grad=adaptive_scale)
        self.ada_bias = torch.nn.Parameter(torch.zeros([1, 1, channels]),
                                           requires_grad=adaptive_scale)

        # 第一层点卷积：通过 Conv1d 创建第一层点卷积，输出通道为 2 * channels。
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        # 因果卷积配置：根据是否为因果卷积设置 padding 和 lorder。
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        # 深度卷积：创建深度卷积层，使用 groups 将通道分开进行卷积。
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        # 归一化层：根据 norm 参数创建对应的归一化层。
        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        # 第二层点卷积：创建第二层点卷积。
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation
        # 激活函数：保存激活函数，若需要初始化权重，则调用 init_weights 方法。
        if init_weights:
            self.init_weights()

    # 初始化卷积层的权重，使用均匀分布进行初始化。
    def init_weights(self):
        pw_max = self.channels**-0.5
        dw_max = self.kernel_size**-0.5
        torch.nn.init.uniform_(self.pointwise_conv1.weight.data, -pw_max,
                               pw_max)
        if self.bias:
            torch.nn.init.uniform_(self.pointwise_conv1.bias.data, -pw_max,
                                   pw_max)
        torch.nn.init.uniform_(self.depthwise_conv.weight.data, -dw_max,
                               dw_max)
        if self.bias:
            torch.nn.init.uniform_(self.depthwise_conv.bias.data, -dw_max,
                                   dw_max)
        torch.nn.init.uniform_(self.pointwise_conv2.weight.data, -pw_max,
                               pw_max)
        if self.bias:
            torch.nn.init.uniform_(self.pointwise_conv2.bias.data, -pw_max,
                                   pw_max)

    # 前向传播方法
    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # 自适应缩放：如果启用自适应缩放，则对输入进行缩放和偏置调整。
        if self.adaptive_scale:
            x = self.ada_scale * x + self.ada_bias
        # 维度交换：将时间维度和特征维度进行交换，以便进行卷积操作。
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)
        # mask batch padding
        # 掩码填充：使用掩码填充输入，以避免处理无效的时间步。
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        # 因果卷积处理：如果是因果卷积，则处理上下文缓存并进行适当的填充。
        if self.lorder > 0:
            if cache.size(2) == 0:  # cache_t == 0
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)  # equal batch
                assert cache.size(1) == x.size(1)  # equal channel
                x = torch.cat((cache, x), dim=2)
            assert (x.size(2) > self.lorder)
            new_cache = x[:, :, -self.lorder:]
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # GLU mechanism
        # GLU机制：应用第一层点卷积并使用门控线性单元（GLU）机制。
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        # 实现了一维深度卷积的处理，以及对卷积输出的归一化和激活。
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache

# 总结：实现了一个强大的卷积模块，结合了因果卷积、深度卷积、GLU 机制和归一化，可以有效处理序列数据，特别适合用于语音和音频信号处理的任务。