# Copyright (c) 2020 Johns Hopkins University (Shinji Watanabe)
#               2020 Northwestern Polytechnical University (Pengcheng Guo)
#               2020 Mobvoi Inc (Binbin Zhang)
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
"""Swish() activation function for Conformer."""

import torch


# 该类表示 Swish 激活函数。
class Swish(torch.nn.Module):
    """Construct an Swish object."""

    # 这是一个实现前向传播的方法，接收输入 x，类型为 torch.Tensor，并返回一个 torch.Tensor 作为输出。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swish activation function."""
        # 这行代码实现了 Swish 激活函数的核心计算。
        return x * torch.sigmoid(x)
    
# 总结：Swish 类定义了一个简单的激活函数，它结合了线性和非线性特性，通过使用 Sigmoid 函数来调节激活的输出。
