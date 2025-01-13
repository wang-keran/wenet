import torch


# 定义了一个名为 RMSNorm 的 PyTorch 模块，该模块实现了一种归一化方法。
class RMSNorm(torch.nn.Module):
    """ https://arxiv.org/pdf/1910.07467.pdf
    """

    # 初始化方法
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.add_unit_offset = add_unit_offset

    # 计算输入张量 x 的 RMS（均方根）归一化。
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # 执行前向传播，返回归一化后的输出。
    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            return x * (1 + self.weight)
        else:
            return x * self.weight

# 总结： RMSNorm 实现了一种均方根归一化方法，能够在训练过程中动态地调整输入的尺度，从而增强模型的表现和稳定性。