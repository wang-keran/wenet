# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
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

from wenet.utils.class_utils import WENET_NORM_CLASSES


# 专门用于 Conformer 模型中的处理输入特征并卷积操作。
class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    # 初始化
    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: nn.Module = nn.ReLU(),
        norm: str = "batch_norm",
        causal: bool = False,
        bias: bool = True,
        norm_eps: float = 1e-5,
    ):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()

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
        # 判断是否为因果卷积（causal convolution）。
        # 因果卷积与普通卷积的区别在于，因果卷积的输出仅依赖于当前及之前的输入数据（即不依赖未来的输入），通常用于时间序列或语言模型中。
        if causal:
            # padding = 0：对于因果卷积，输出依赖于当前和之前的时间步，因此无需在输入的右侧添加填充（padding）。填充值设置为 0。
            padding = 0
            # self.lorder = kernel_size - 1：lorder 表示卷积的左侧填充（也叫做“历史步数”）。对于因果卷积，lorder 等于卷积核的大小减去 1（即保证每个输出仅依赖于当前及之前的输入）。
            self.lorder = kernel_size - 1
        # 如果不是因果卷积，即是对称卷积（symmetric convolution）：
        else:
            # kernel_size should be an odd number for none causal convolution
            # assert (kernel_size - 1) % 2 == 0：对称卷积的卷积核大小必须是奇数，因此 kernel_size - 1 必须是偶数。
            assert (kernel_size - 1) % 2 == 0
            # 对称卷积会在输入的左右两侧各加上相同数量的填充（padding）。填充的大小为 (kernel_size - 1) // 2，这样卷积操作可以保持输入和输出的长度一致。
            padding = (kernel_size - 1) // 2
            # 对称卷积不需要考虑因果性，所以设置 lorder 为 0。
            self.lorder = 0
        # 深度可分卷积：深度可分卷积的特点是每个输入通道分别用自己的卷积核进行卷积操作，而不像普通卷积那样进行通道间的组合。
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        # 这行代码检查归一化类型是否为 batch_norm、layer_norm 或 rms_norm 中的一个。
        assert norm in ['batch_norm', 'layer_norm', 'rms_norm']
        # 如果选择了批归一化（batch normalization）：
        if norm == "batch_norm":
            # 设置 use_layer_norm 为 False，表示不使用层归一化。
            self.use_layer_norm = False
            # 使用批归一化层，channels 是通道数，norm_eps 是用于计算标准差时添加的小常数，防止除零错误。也是torch.nn内置的
            self.norm = WENET_NORM_CLASSES['batch_norm'](channels,
                                                         eps=norm_eps)
        # 如果选择的是层归一化（layer normalization）或 RMS 归一化（RMS normalization）：
        else:
            # 设置 use_layer_norm 为 True，表示使用层归一化。
            self.use_layer_norm = True
            # 根据配置选择不同的归一化层。
            self.norm = WENET_NORM_CLASSES[norm](channels, eps=norm_eps)

        # 定义了一个点卷积（pointwise convolution），它是一个 1x1 卷积，用于调整通道数。在这里，它的作用是将卷积后的输出映射到期望的通道数。
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            # 卷积核大小为 1，表示它只是对每个通道进行独立的线性变换。
            kernel_size=1,
            stride=1,
            # 没有填充。
            padding=0,
            bias=bias,
        )
        # 最后，代码将激活函数（如 ReLU、GELU 等）赋值给 self.activation，激活函数将在后续的前向传播过程中使用。
        self.activation = activation

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
        # 交换输入的维度，将时间维度和通道维度互换，变为 (batch, channels, time)。
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)

        # 应用填充掩码，将掩码为 False 的位置填充为 0。
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        # 处理因果卷积的缓存
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

        # GLU 和卷积操作
        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 应用深度卷积、归一化和激活函数，然后通过第二个点卷积处理。
        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            # 交换维度
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        # 再次应用填充掩码，并将输出的维度恢复为 (batch, time, channels)。最终返回输出张量和新的缓存。
        return x.transpose(1, 2), new_cache

# 总结：该类实现了 Conformer 模型中的卷积模块，结合了因果卷积、深度卷积、归一化和激活机制，适用于序列数据处理。
# 通过支持因果卷积，可以处理时序数据时有效地考虑上下文信息。