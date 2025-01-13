import torch


# 预计算频率的复数表示（cis）。
# copy from:https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L84
def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


# 将旋转嵌入应用于查询和键张量。
# modified from:
#     https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L95
def google_apply_rotary_emb(x: torch.Tensor,
                            freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1)
    return x_out


# 将旋转嵌入应用于查询和键张量（LLaMA 版本）。
def llama_apply_rotary_emb(x: torch.Tensor,
                           freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


# 定义一个字典，将旋转嵌入函数与其对应的名称关联。
WENET_APPLY_ROTARY_EMB = {
    'google': google_apply_rotary_emb,
    'llama': llama_apply_rotary_emb,
}

# 总结：该文件定义了用于预计算频率复数表示和应用旋转嵌入的函数。
# 预计算频率复数表示函数 precompute_freqs_cis。
# 两个旋转嵌入应用函数 google_apply_rotary_emb 和 llama_apply_rotary_emb。
# 一个字典 WENET_APPLY_ROTARY_EMB 将旋转嵌入函数与其对应的名称关联。