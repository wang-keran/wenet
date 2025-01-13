# Copyright (c) 2019 Shigeki Karita
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
"""Positionwise feed forward layer definition."""

import torch


# PositionwiseFeedForward 实现了一个位置逐步的前馈层。该层对序列中的每个位置应用相同的前馈网络，输出维度与输入维度相同，也是拿torch初始化实现的。
class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        *dummy_args,
        **dummy_kwargs,
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    # 前向传播方法：对输入 xs 进行前向传播，经过两个线性变换和一个激活函数，输出张量的形状与输入相同，即 (B, L, D)。
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


# MoEFFNLayer 实现了一个混合专家（Mixture of Experts）层，结合了 PositionwiseFeedForward。
# 该层允许为每个输入令牌选择多个专家，从而动态地调整计算资源。
class MoEFFNLayer(torch.nn.Module):
    """
    Mixture of expert with Positionwise feed forward layer
    See also figure 1 in https://arxiv.org/pdf/2305.15663.pdf
    The output dim is same with the input dim.

    Modified from https://github.com/Lightning-AI/lit-gpt/pull/823
                  https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
    Args:
        n_expert: number of expert.
        n_expert_activated: The actual number of experts used for each frame
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    # 初始化
    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = False,
        n_expert: int = 8,
        n_expert_activated: int = 2,
    ):
        super(MoEFFNLayer, self).__init__()
        self.gate = torch.nn.Linear(idim, n_expert, bias=False)
        self.experts = torch.nn.ModuleList(
            PositionwiseFeedForward(
                idim, hidden_units, dropout_rate, activation, bias=bias)
            for _ in range(n_expert))
        self.n_expert = n_expert
        self.n_expert_activated = n_expert_activated

    # 前向传播方法：将输入张量 xs 形状调整为 (B*L, D)，使得每个令牌可以独立处理；
    # 计算门控（gate）输出，以确定每个专家的激活。
    # 使用 torch.topk 获取激活值最大的专家索引，并计算相应的权重。
    # 对每个专家进行前向传播，并根据权重加权组合输出。
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Foward function.
        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)

        """
        B, L, D = xs.size(
        )  # batch size, sequence length, embedding dimension (idim)
        xs = xs.view(-1, D)  # (B*L, D)
        router = self.gate(xs)  # (B*L, n_expert)
        logits, selected_experts = torch.topk(
            router, self.n_expert_activated
        )  # probs:(B*L, n_expert_activated), selected_exp: (B*L, n_expert_activated)
        weights = torch.nn.functional.softmax(
            logits, dim=1,
            dtype=torch.float).to(dtype=xs.dtype)  # (B*L, n_expert_activated)
        output = torch.zeros_like(xs)  # (B*L, D)
        for i, expert in enumerate(self.experts):
            mask = selected_experts == i
            token_ids, ith_expert = torch.where(mask)
            output[token_ids] += weights[token_ids, ith_expert, None] * expert(
                xs[token_ids])
        return output.view(B, L, D)


# GatedVariantsMLP 是一个带有门控机制的多层感知机（MLP），它通过门控机制增强输入特征的选择性。
class GatedVariantsMLP(torch.nn.Module):
    """ https://arxiv.org/pdf/2002.05202.pdf
    """

    # 设计类似于 PositionwiseFeedForward，但增加了门控机制。
    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.GELU(),
        bias: bool = True,
        *dummy_args,
        **dummy_kwargs,
    ):
        """Construct a PositionwiseFeedForward object."""
        super(GatedVariantsMLP, self).__init__()
        self.gate = torch.nn.Linear(idim, hidden_units, bias=False)
        self.activation = activation
        # w_1 as up proj
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.dropout = torch.nn.Dropout(dropout_rate)
        # w_2 as down proj
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    # 计算门控值，并与输入 x 进行逐元素相乘（fuse），形成加权的上升投影。最后通过一个线性层（w_2）生成最终输出。最后通过一个线性层（w_2）生成最终输出。
    def forward(self, x) -> torch.Tensor:
        """Foward function.
        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)

        """
        gate = self.activation(self.gate(x))
        up = self.w_1(x)
        fuse = gate * up
        return self.w_2(self.dropout(fuse))

# 总结：PositionwiseFeedForward: 一个简单的前馈网络，对每个位置独立处理。
# MoEFFNLayer: 结合混合专家机制的前馈网络，可以选择不同的专家对输入进行处理，提高模型的灵活性和表达能力。
# GatedVariantsMLP: 通过门控机制增强特征的选择性，从而提高模型的表现。
# 这三个模块可以结合使用，构建复杂的神经网络架构，特别是在需要处理序列数据的任务中，如自然语言处理和时间序列预测等。