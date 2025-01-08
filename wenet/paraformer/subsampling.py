from typing import Tuple, Union
import torch
from wenet.transformer.subsampling import BaseSubsampling


# 这个类是一个基于Paraformer的下采样实现，主要用于对输入张量进行处理，以减少数据量并进行位置编码。
class IdentitySubsampling(BaseSubsampling):
    """ Paraformer subsampling
    """

    # 初始化方法
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        super().__init__()
        _, _ = idim, odim
        self.right_context = 6
        self.subsampling_rate = 6
        self.pos_enc = pos_enc_class

    # 前向传播方法，执行下采样操作
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[torch.Tensor, int] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time
            torch.Tensor: positional encoding

        """
        # NOTE(Mddct): Paraformer starts from 1
        if isinstance(offset, torch.Tensor):
            offset = torch.add(offset, 1)
        else:
            offset = offset + 1
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask

    # 位置编码方法：用于生成指定大小的位置信息编码。
    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset + 1, size)

# IdentitySubsampling 类实现了基于Paraformer的下采样机制。它通过继承 BaseSubsampling 类，重写了前向传播方法以实现对输入张量的下采样和位置编码。