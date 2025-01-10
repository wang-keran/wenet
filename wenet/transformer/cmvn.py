# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
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

import torch


# 实现了一个全局均值方差归一化（Global CMVN）的 PyTorch 模块。
# 主要用于对输入特征进行归一化处理。
class GlobalCMVN(torch.nn.Module):

    # 初始化方法
    def __init__(self,
                 mean: torch.Tensor,
                 istd: torch.Tensor,
                 norm_var: bool = True):
        """
        Args:
            mean (torch.Tensor): mean stats
            istd (torch.Tensor): inverse std, std which is 1.0 / std
        """
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        # 这段代码的作用是将张量 mean 和 istd 注册为 PyTorch 模型中的缓冲区（buffer），并为它们命名为 "mean" 和 "istd"。
        # 通过这种方式，mean 和 istd 可以作为模型的一部分进行保存和加载，但它们不会参与梯度计算。
        # The buffer can be accessed from this module using self.mean
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    # 前向传播方法
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (batch, max_len, feat_dim)

        Returns:
            (torch.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x

# 总结：该类实现了全局均值方差归一化的功能，可以对输入特征进行均值和标准差的归一化处理。