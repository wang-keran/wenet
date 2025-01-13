# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
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
"""Subsampling layer definition."""

from typing import Tuple, Union

import torch

from wenet.utils.mask import make_pad_mask


# 定义了一系列用于处理输入数据的类，主要通过不同的子采样技术来减少输入数据的时间维度。
# 所有子采样类的基类
class BaseSubsampling(torch.nn.Module):

    # 初始化
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    # 用于生成位置编码
    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


# 该类用于对输入进行嵌入，不进行子采样。
class EmbedinigNoSubsampling(BaseSubsampling):
    """Embedding input without subsampling
    """

    # 构造方法中使用 torch.nn.Embedding 创建嵌入层，并保存位置编码类的实例。
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        super().__init__()
        self.embed = torch.nn.Embedding(idim, odim)
        self.pos_enc = pos_enc_class

    # 将输入 x 通过嵌入层转换。
    # 获取位置编码并返回转换后的张量、位置编码和输入掩码。
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.embed(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


# 该类进行线性变换，同样不进行子采样。
class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    # 使用 torch.nn.Sequential 创建一个线性层，层归一化和丢弃层。
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    # forward 方法功能类似于 EmbedinigNoSubsampling，将输入通过线性层转换，并返回转换后的张量和掩码。
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


# 该类通过一维卷积实现子采样，输出的长度为原长度的一半。
# 使用两个卷积层进行特征提取，最后返回经过子采样后的张量和位置编码。
class Conv1dSubsampling2(BaseSubsampling):
    """Convolutional 1D subsampling (to 1/2 length).
       It is designed for Whisper, ref:
       https://github.com/openai/whisper/blob/main/whisper/model.py

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv1dSubsampling2 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
        )
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 2
        # 4 = (3 - 1) * 1 + (3 - 1) * 1
        self.right_context = 4

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
            torch.Tensor: positional encoding

        """
        time = x.size(1)
        x = x.transpose(1, 2)  # (b, f, t)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (b, t, f)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, (time + 1) % 2::2]


# 使用两个二维卷积层进行处理，最终返回的时间维度为原始的1/4。
# 右侧上下文为6。
class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2]


# 这些类都是通过2D卷积实现更高的子采样率（分别为1/4, 1/6, 1/8）
# 该类具有更复杂的卷积结构（一个3x3和一个5x5卷积），使得时间维度变为1/6。
# 右侧上下文为10。
class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3),
                                      odim)
        self.pos_enc = pos_enc_class
        # 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 4::3]


# 使用三个2D卷积层，进一步缩小时间维度至1/8。
# 右侧上下文为14。
class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(
            odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2][:, :, 2::2]


# 实现了堆叠多帧的下采样方法。
class StackNFramesSubsampling(BaseSubsampling):

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: torch.nn.Module,
                 stride: int = 2):

        super().__init__()
        del dropout_rate
        self.pos_enc_class = pos_enc_class
        self.stride = stride
        self.idim = idim

        self.norm = torch.nn.LayerNorm(idim * stride, eps=1e-5)
        self.out = torch.nn.Linear(idim * stride, odim)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // stride.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // stride.
            torch.Tensor: positional encoding
        """
        with torch.no_grad():
            b, s, _ = x.size()

            seq_len = x_mask.sum(-1).view(b)
            r = s % self.stride
            s -= r
            x = x[:, :s, :]
            seq_len = torch.where(seq_len > s, s, seq_len)
            seq_len = seq_len // self.stride
            new_mask = ~make_pad_mask(seq_len, max_len=s // self.stride)
            x = x.view(b, s // self.stride, self.idim * self.stride)
            _, pos_emb = self.pos_enc_class(x, offset)
        x = self.norm(x)
        x = self.out(x)
        return x, pos_emb, new_mask.unsqueeze(1)

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return self.pos_enc_class.position_encoding(offset, size)

# 总结：
# 总体来看，这段代码展示了如何使用不同的子采样策略对输入进行处理，以便于模型的计算和表示。
# 不同的子采样方法会对输入数据的时间维度进行不同程度的压缩，同时保持或增强其特征表示能力。
# 各类中都涉及到位置编码的使用，以帮助模型捕捉输入序列中位置信息。