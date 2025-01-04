# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#转onnx模型脚本,输出config.yaml,encoder.onnx,encoder_fp16.onnx,decoder.onnx,decoder_fp16.onnx
from __future__ import print_function

import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F
import yaml
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.init_model import init_model
from wenet.utils.mask import make_pad_mask

try:
    import onnxruntime
except ImportError:
    print("Please install onnxruntime-gpu!")
    sys.exit(1)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


# 根据torch生成了一个编码器
class Encoder(torch.nn.Module):

    # 这是直接调用的wenet/tranformer/encoder.py中的BaseEncoder类,CTC是wenet/transformer/ctc.py中的CTC类，都不用自己实现，直接赋值了
    def __init__(self, encoder: BaseEncoder, ctc: CTC, beam_size: int = 10):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc
        self.beam_size = beam_size

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ):
        """Encoder
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        Returns:
            encoder_out: B x T x F
            encoder_out_lens: B
            ctc_log_probs: B x T x V
            beam_log_probs: B x T x beam_size
            beam_log_probs_idx: B x T x beam_size
        """
        # 进入encoder的forward函数中，获取输出和掩码
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths, -1,
                                                 -1)
        # 计算编码器输出的长度，张量结构为(B,1,T),先去掉1,再对T进行求和表示每个批次中有效时间步数的总和，最后是（B,）的结构
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        # 计算CTC的对数概率（torch自带的方法）归一化
        ctc_log_probs = self.ctc.log_softmax(encoder_out)
        encoder_out_lens = encoder_out_lens.int()
        # 沿着指定维度提取前k个最大值及其索引
        beam_log_probs, beam_log_probs_idx = torch.topk(ctc_log_probs,
                                                        self.beam_size,
                                                        dim=2)
        return (
            encoder_out,
            encoder_out_lens,
            ctc_log_probs,
            beam_log_probs,
            beam_log_probs_idx,
        )


# 定义了流式编码器
class StreamingEncoder(torch.nn.Module):

    def __init__(
        self,
        model,
        required_cache_size,
        beam_size,
        transformer=False,
        return_ctc_logprobs=False,
    ):
        super().__init__()
        # 用的model都是wenet/transformer/encoder.py中的BaseEncoder类和decoder.py,ctc.py拼凑出来的
        self.ctc = model.ctc
        self.subsampling_rate = model.encoder.embed.subsampling_rate
        self.embed = model.encoder.embed
        self.global_cmvn = model.encoder.global_cmvn
        self.required_cache_size = required_cache_size
        self.beam_size = beam_size
        self.encoder = model.encoder
        self.transformer = transformer
        self.return_ctc_logprobs = return_ctc_logprobs

    def forward(self, chunk_xs, chunk_lens, offset, att_cache, cnn_cache,
                cache_mask):
        """Streaming Encoder
        Args:
            xs (torch.Tensor): chunk input, with shape (b, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (torch.Tensor): offset with shape (b, 1)
                        1 is retained for triton deployment
            required_cache_size (int): cache size required for next chunk
                compuation
                > 0: actual cache size
                <= 0: not allowed in streaming gpu encoder                   `
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (b, elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (b, elayers, b, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
            cache_mask: (torch.Tensor): cache mask with shape (b, required_cache_size)
                 in a batch of request, each request may have different
                 history cache. Cache mask is used to indidate the effective
                 cache for each request
        Returns:
            torch.Tensor: log probabilities of ctc output and cutoff by beam size
                with shape (b, chunk_size, beam)
            torch.Tensor: index of top beam size probabilities for each timestep
                with shape (b, chunk_size, beam)
            torch.Tensor: output of current input xs,
                with shape (b, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                same shape (b, elayers, head, cache_t1, d_k * 2)
                as the original att_cache
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
            torch.Tensor: new cache mask, with same shape as the original
                cache mask
        """
        # 数据预处理
        # 压缩偏移量的维度
        offset = offset.squeeze(1)
        # 获取时如数据块的时间维度大小
        T = chunk_xs.size(1)
        # 生成掩码，标记填充部分
        chunk_mask = ~make_pad_mask(chunk_lens, T).unsqueeze(1)
        # B X 1 X T
        # 将掩码转换为与输入数据相同的数据类型
        chunk_mask = chunk_mask.to(chunk_xs.dtype)
        # transpose batch & num_layers dim 转置注意力缓存和卷积缓存。
        att_cache = torch.transpose(att_cache, 0, 1)
        cnn_cache = torch.transpose(cnn_cache, 0, 1)

        # rewrite encoder.forward_chunk 重写按一块前向传播
        # <---------forward_chunk START--------->开始
        # 对输入数据进行全局均值方差归一化。
        xs = self.global_cmvn(chunk_xs)
        # chunk mask is important for batch inferencing since
        # different sequence in a batch has different length
        # 块掩码对于批量推理很重要，因为批量中的不同序列具有不同的长度
        xs, pos_emb, chunk_mask = self.embed(xs, chunk_mask, offset)
        cache_size = att_cache.size(3)  # required cache size，需要的cache缓存大小
        # 拼接缓存掩码和数据块掩码。
        masks = torch.cat((cache_mask, chunk_mask), dim=2)
        index = offset - cache_size

        # 生成位置编码并转换为dtype格式。
        pos_emb = self.embed.position_encoding(index, cache_size + xs.size(1))
        pos_emb = pos_emb.to(dtype=xs.dtype)

        # 计算下一个缓存的起始位置和缓存掩码。
        next_cache_start = -self.required_cache_size
        r_cache_mask = masks[:, :, next_cache_start:]

        # 初始化注意力缓存和卷积缓存。
        r_att_cache = []
        r_cnn_cache = []
        # 遍历每一层编码器，进行前向传播，并更新注意力缓存和卷积缓存。
        for i, layer in enumerate(self.encoder.encoders):
            i_kv_cache = att_cache[i]
            size = att_cache.size(-1) // 2
            kv_cache = (i_kv_cache[:, :, :, :size], i_kv_cache[:, :, :, size:])
            xs, _, new_kv_cache, new_cnn_cache = layer(
                xs,
                masks,
                pos_emb,
                att_cache=kv_cache,
                cnn_cache=cnn_cache[i],
            )
            #   shape(new_att_cache) is (B, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (B, hidden-dim, cache_t2)
            new_att_cache = torch.cat(new_kv_cache, dim=-1)
            r_att_cache.append(
                new_att_cache[:, :, next_cache_start:, :].unsqueeze(1))
            if not self.transformer:
                r_cnn_cache.append(new_cnn_cache.unsqueeze(1))
        if self.encoder.normalize_before:
            chunk_out = self.encoder.after_norm(xs)
        else:
            chunk_out = xs

        r_att_cache = torch.cat(r_att_cache, dim=1)  # concat on layers idx，在层索引上连接
        if not self.transformer:
            r_cnn_cache = torch.cat(r_cnn_cache, dim=1)  # concat on layers，在层上连接

        # <---------forward_chunk END--------->前向块传播结束（正向块推理结束）

        # 计算CTC输出的对数概率。
        log_ctc_probs = self.ctc.log_softmax(chunk_out)
        # 获取Top-K的CTC对数概率及其索引。
        log_probs, log_probs_idx = torch.topk(log_ctc_probs,
                                              self.beam_size,
                                              dim=2)
        # 将对数概率转换成与输入数据相同的数据类型。
        log_probs = log_probs.to(chunk_xs.dtype)

        # 计算新的右偏移量，作用是过更新偏移量，确保在处理连续的数据块时，模型能够正确地对齐和处理每个数据块。
        r_offset = offset + chunk_out.shape[1]
        # the below ops not supported in Tensorrt
        # chunk_out_lens = torch.div(chunk_lens, subsampling_rate,
        #                   rounding_mode='floor')
        # 计算输出数据块的长度。
        chunk_out_lens = chunk_lens // self.subsampling_rate
        # 给右偏移量增加一个维度
        r_offset = r_offset.unsqueeze(1)
        if self.return_ctc_logprobs:
            # 如果需要返回CTC对数概率，则返回CTC对数概率及其索引。
            return (
                log_ctc_probs,
                chunk_out,
                chunk_out_lens,
                r_offset,
                r_att_cache,
                r_cnn_cache,
                r_cache_mask,
            )
        else:
            # 如果不需要返回CTC对数概率，则返回Top-K的CTC对数概率及其索引。
            return (
                log_probs,
                log_probs_idx,
                chunk_out,
                chunk_out_lens,
                r_offset,
                r_att_cache,
                r_cnn_cache,
                r_cache_mask,
            )


# 定义流式 Squeezeformer 编码器模型
class StreamingSqueezeformerEncoder(torch.nn.Module):

    def __init__(self, model, required_cache_size, beam_size):
        super().__init__()
        self.ctc = model.ctc
        self.subsampling_rate = model.encoder.embed.subsampling_rate
        self.embed = model.encoder.embed
        self.global_cmvn = model.encoder.global_cmvn
        self.required_cache_size = required_cache_size
        self.beam_size = beam_size
        self.encoder = model.encoder
        self.reduce_idx = model.encoder.reduce_idx
        self.recover_idx = model.encoder.recover_idx
        if self.reduce_idx is None:
            self.time_reduce = None
        else:
            if self.recover_idx is None:
                self.time_reduce = "normal"  # no recovery at the end
            else:
                self.time_reduce = "recover"  # recovery at the end
                assert len(self.reduce_idx) == len(self.recover_idx)

    # 计算在第 i 层编码器层之后的下采样因子。下采样因子表示在经过若干层编码器之后，时间步的减少倍数，用来降低计算复杂度和内存消耗。
    def calculate_downsampling_factor(self, i: int) -> int:
        if self.reduce_idx is None:
            return 1
        else:
            reduce_exp, recover_exp = 0, 0
            for exp, rd_idx in enumerate(self.reduce_idx):
                if i >= rd_idx:
                    reduce_exp = exp + 1
            if self.recover_idx is not None:
                for exp, rc_idx in enumerate(self.recover_idx):
                    if i >= rc_idx:
                        recover_exp = exp + 1
            return int(2**(reduce_exp - recover_exp))

    def forward(self, chunk_xs, chunk_lens, offset, att_cache, cnn_cache,
                cache_mask):
        """Streaming Encoder
        Args:
            xs (torch.Tensor): chunk input, with shape (b, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (torch.Tensor): offset with shape (b, 1)
                        1 is retained for triton deployment
            required_cache_size (int): cache size required for next chunk
                compuation
                > 0: actual cache size
                <= 0: not allowed in streaming gpu encoder                   `
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (b, elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (b, elayers, b, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
            cache_mask: (torch.Tensor): cache mask with shape (b, required_cache_size)
                 in a batch of request, each request may have different
                 history cache. Cache mask is used to indidate the effective
                 cache for each request
        Returns:
            torch.Tensor: log probabilities of ctc output and cutoff by beam size
                with shape (b, chunk_size, beam)
            torch.Tensor: index of top beam size probabilities for each timestep
                with shape (b, chunk_size, beam)
            torch.Tensor: output of current input xs,
                with shape (b, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                same shape (b, elayers, head, cache_t1, d_k * 2)
                as the original att_cache
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
            torch.Tensor: new cache mask, with same shape as the original
                cache mask
        """
        # 减少偏移量一个维度
        offset = offset.squeeze(1)
        # 获取数据块的时间步大小
        T = chunk_xs.size(1)
        # 生成掩码，标记填充部分
        chunk_mask = ~make_pad_mask(chunk_lens, T).unsqueeze(1)
        # （B X 1 X T），这是做完掩码的数据维度结构
        chunk_mask = chunk_mask.to(chunk_xs.dtype)
        # transpose batch & num_layers dim，转置注意力缓存和卷积缓存，将第0维度和第1维度进行交换，即(a,b)转置为(b,a)。
        att_cache = torch.transpose(att_cache, 0, 1)
        cnn_cache = torch.transpose(cnn_cache, 0, 1)

        # rewrite encoder.forward_chunk,重写按一块前向传播方法，即按块推理方法
        # <---------forward_chunk START--------->向前推理开始
        # 获取全局均值方差归一化的数据（预处理的数据
        xs = self.global_cmvn(chunk_xs)
        # chunk mask is important for batch inferencing since
        # different sequence in a batch has different length
        # 块掩码对于批量推理非常重要，因为批次中的不同序列具有不同的长度。
        xs, pos_emb, chunk_mask = self.embed(xs, chunk_mask, offset)
        # 获取编码器层数和缓存大小
        elayers, cache_size = att_cache.size(0), att_cache.size(3)
        # 获取注意力掩码
        att_mask = torch.cat((cache_mask, chunk_mask), dim=2)
        # 获取索引
        index = offset - cache_size

        # 生成位置编码并转换为dtype格式。生成位置编码用的wenet/tranformer中项目自带的方法
        pos_emb = self.embed.position_encoding(index, cache_size + xs.size(1))
        pos_emb = pos_emb.to(dtype=xs.dtype)

        # 计算下一个缓存的起始位置和缓存掩码。
        next_cache_start = -self.required_cache_size
        r_cache_mask = att_mask[:, :, next_cache_start:]

        # 初始化注意力缓存和卷积神经网络缓存
        r_att_cache = []
        r_cnn_cache = []
        # 创建了一个全为 1 的张量，形状为 (1, xs.size(1))，
        # 数据类型为布尔型，并且与输入张量 xs 位于相同的设备上。xs.size(1) 获取了输入张量 xs 在第一个维度上的大小。
        mask_pad = torch.ones(1,
                              xs.size(1),
                              device=xs.device,
                              dtype=torch.bool)
        # 增加一个维度
        mask_pad = mask_pad.unsqueeze(1)
        # 初始化了一个名为 max_att_len 的整数变量，并将其值设置为 0。
        # 这个变量可能用于跟踪注意力机制中的最大长度。
        max_att_len: int = 0
        # 一个列表里四个torch.Tensor类型的元组，初始化空列表
        recover_activations: List[Tuple[torch.Tensor, torch.Tensor,
                                        torch.Tensor, torch.Tensor]] = []
        # 初始化索引为0
        index = 0
        # 创建xs_lens张量，用于记录xs的长度
        xs_lens = torch.tensor([xs.size(1)], device=xs.device, dtype=torch.int)
        # 预处理："pre-layer normalization"（预层归一化），
        # 这是一种在深度学习模型中常用的技术，用于在输入数据进入模型之前对其进行归一化处理，以提高模型的训练效果和稳定性。
        xs = self.encoder.preln(xs)
        # 遍历编码器的每一层，并获取当前层的索引 i 和层对象 layer。
        for i, layer in enumerate(self.encoder.encoders):
            # 检查是否需要进行时间步下采样
            # 检查 reduce_idx 是否为 None，如果不是，则继续检查 time_reduce 是否为 None 并且当前层索引 i 是否在 reduce_idx 中。
            if self.reduce_idx is not None:
                if self.time_reduce is not None and i in self.reduce_idx:
                    # 如果条件满足，将当前的 xs、att_mask、pos_emb 和 mask_pad 添加到 recover_activations 列表中。
                    recover_activations.append(
                        (xs, att_mask, pos_emb, mask_pad))
                    (
                        xs,
                        xs_lens,
                        att_mask,
                        mask_pad,
                        # 调用 self.encoder.time_reduction_layer 方法对 xs、xs_lens、att_mask 和 mask_pad 进行时间步下采样。
                    ) = self.encoder.time_reduction_layer(
                        xs, xs_lens, att_mask, mask_pad)
                    # 更新 pos_emb，将其时间步数减半。
                    pos_emb = pos_emb[:, ::2, :]
                    # 如果 self.encoder.pos_enc_layer_type 为 "rel_pos_repaired"，则进一步调整 pos_emb 的形状。
                    if self.encoder.pos_enc_layer_type == "rel_pos_repaired":
                        pos_emb = pos_emb[:, :xs.size(1) * 2 - 1, :]
                    # 增加 index 的值。
                    index += 1

            # 检查是否需要进行时间步恢复
            # 检查 recover_idx 是否为 None，
            if self.recover_idx is not None:
                # 如果不是，则继续检查 time_reduce 是否为 "recover" 并且当前层索引 i 是否在 recover_idx 中。
                if self.time_reduce == "recover" and i in self.recover_idx:
                    # 如果条件满足，减少 index 的值。
                    index -= 1
                    # 从 recover_activations 列表中取出对应index索引值的 
                    # recover_tensor、recover_att_mask、recover_pos_emb 和 recover_mask_pad。
                    (
                        recover_tensor,
                        recover_att_mask,
                        recover_pos_emb,
                        recover_mask_pad,
                    ) = recover_activations[index]
                    # recover output length for ctc decode，恢复CTC解码的输出长度，先通过 unsqueeze 和 repeat 操作将时间步数加倍
                    xs = xs.unsqueeze(2).repeat(1, 1, 2, 1).flatten(1, 2)
                    # 然后调用 self.encoder.time_recover_layer 方法进行恢复。
                    xs = self.encoder.time_recover_layer(xs)
                    # 获取恢复后的时间步数
                    recoverd_t = recover_tensor.size(1)
                    # 更新 xs，将恢复后的 xs 与 recover_tensor 相加。
                    xs = recover_tensor + xs[:, :recoverd_t, :].contiguous()
                    # 获取恢复后的注意力掩码、位置编码和掩码。mask_pad是用于表示输入序列中的填充位置。
                    att_mask = recover_att_mask
                    pos_emb = recover_pos_emb
                    mask_pad = recover_mask_pad

            # 调用 calculate_downsampling_factor 方法计算当前层的下采样因子 factor。
            factor = self.calculate_downsampling_factor(i)

            # 调用 layer 方法进行前向传播，并更新注意力缓存和卷积缓存。
            # 调用当前层 layer 的前向传播方法，传入 xs、att_mask、pos_emb、att_cache 和 cnn_cache。
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                # 根据 factor 对 att_cache 进行下采样。
                att_cache=att_cache[i][:, :, ::factor, :]
                # 获取新的注意力缓存 new_att_cache 和卷积神经网络缓存 new_cnn_cache。
                [:, :, :pos_emb.size(1) - xs.size(1), :]
                if elayers > 0 else att_cache[:, :, ::factor, :],
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache,
            )
            # 处理新的缓存
            # 对新的注意力缓存 new_att_cache 进行处理，获取 cached_att。
            cached_att = new_att_cache[:, :, next_cache_start // factor:, :]
            # 对新的卷积神经网络缓存 new_cnn_cache 进行处理，获取 cached_cnn。
            cached_cnn = new_cnn_cache.unsqueeze(1)
            cached_att = (cached_att.unsqueeze(3).repeat(1, 1, 1, factor,
                                                         1).flatten(2, 3))
            # 如果当前是第一个层，则记录 cached_att 的长度为 max_att_len。
            if i == 0:
                # record length for the first block as max length
                max_att_len = cached_att.size(2)
            # 将处理后的 cached_att 和 cached_cnn 分别添加到 r_att_cache 和 r_cnn_cache 列表中。
            r_att_cache.append(cached_att[:, :, :max_att_len, :].unsqueeze(1))
            r_cnn_cache.append(cached_cnn)

        chunk_out = xs
        r_att_cache = torch.cat(r_att_cache, dim=1)  # concat on layers idx 层索引连接
        r_cnn_cache = torch.cat(r_cnn_cache, dim=1)  # concat on layers     层连接

        # <---------forward_chunk END---------> 前向块推理结束

        # # 计算CTC输出的对数概率。
        log_ctc_probs = self.ctc.log_softmax(chunk_out)
        # 将对数概率转换成与输入数据相同的数据类型。
        log_probs, log_probs_idx = torch.topk(log_ctc_probs,
                                              self.beam_size,
                                              dim=2)
        # 将对数概率转换成与输入数据相同的数据类型。
        log_probs = log_probs.to(chunk_xs.dtype)

        # 计算新的右偏移量，作用是过更新偏移量，确保在处理连续的数据块时，模型能够正确地对齐和处理每个数据块。。
        r_offset = offset + chunk_out.shape[1]
        # the below ops not supported in Tensorrt
        # chunk_out_lens = torch.div(chunk_lens, subsampling_rate,
        #                   rounding_mode='floor')
        # 计算输出数据块的长度。
        chunk_out_lens = chunk_lens // self.subsampling_rate
        # 增加一个全是1的维度，从(B)变成(B,1)
        r_offset = r_offset.unsqueeze(1)

        # 返回结果
        return (
            log_probs,
            log_probs_idx,
            chunk_out,
            chunk_out_lens,
            r_offset,
            r_att_cache,
            r_cnn_cache,
            r_cache_mask,
        )


class StreamingEfficientConformerEncoder(torch.nn.Module):

    def __init__(self, model, required_cache_size, beam_size):
        super().__init__()
        self.ctc = model.ctc
        self.subsampling_rate = model.encoder.embed.subsampling_rate
        self.embed = model.encoder.embed
        self.global_cmvn = model.encoder.global_cmvn
        self.required_cache_size = required_cache_size
        self.beam_size = beam_size
        self.encoder = model.encoder

        # Efficient Conformer，高效的Conformer编码器的区别
        self.stride_layer_idx = model.encoder.stride_layer_idx
        self.stride = model.encoder.stride
        self.num_blocks = model.encoder.num_blocks
        self.cnn_module_kernel = model.encoder.cnn_module_kernel

    # 计算下采样因子
    def calculate_downsampling_factor(self, i: int) -> int:
        factor = 1
        for idx, stride_idx in enumerate(self.stride_layer_idx):
            if i > stride_idx:
                factor *= self.stride[idx]
        return factor

    def forward(self, chunk_xs, chunk_lens, offset, att_cache, cnn_cache,
                cache_mask):
        """Streaming Encoder
        Args:
            chunk_xs (torch.Tensor): chunk input, with shape (b, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            chunk_lens (torch.Tensor):
            offset (torch.Tensor): offset with shape (b, 1)
                        1 is retained for triton deployment
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (b, elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (b, elayers, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
            cache_mask: (torch.Tensor): cache mask with shape (b, required_cache_size)
                 in a batch of request, each request may have different
                 history cache. Cache mask is used to indidate the effective
                 cache for each request
        Returns:
            torch.Tensor: log probabilities of ctc output and cutoff by beam size
                with shape (b, chunk_size, beam)
            torch.Tensor: index of top beam size probabilities for each timestep
                with shape (b, chunk_size, beam)
            torch.Tensor: output of current input xs,
                with shape (b, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                same shape (b, elayers, head, cache_t1, d_k * 2)
                as the original att_cache
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
            torch.Tensor: new cache mask, with same shape as the original
                cache mask
        """
        # 处理偏移量和下采样
        offset = offset.squeeze(1)  # (b, )
        offset *= self.calculate_downsampling_factor(self.num_blocks + 1)

        # 生成掩码
        T = chunk_xs.size(1)
        chunk_mask = ~make_pad_mask(chunk_lens, T).unsqueeze(1)  # (b, 1, T)
        # B X 1 X T
        chunk_mask = chunk_mask.to(chunk_xs.dtype)
        # transpose batch & num_layers dim
        #   Shape(att_cache): (elayers, b, head, cache_t1, d_k * 2)
        #   Shape(cnn_cache): (elayers, b, outsize, cnn_kernel)
        # 将缓存的维度进行转置，确保它们的形状与后续计算兼容。
        att_cache = torch.transpose(att_cache, 0, 1)
        cnn_cache = torch.transpose(cnn_cache, 0, 1)

        # rewrite encoder.forward_chunk,重写向前传播的方法
        # <---------forward_chunk START---------> 开始块前向传播
        # 对输入进行全局均值方差归一化，并通过嵌入层和位置编码层获取嵌入表示和位置信息。
        xs = self.global_cmvn(chunk_xs)
        # chunk mask is important for batch inferencing since
        # different sequence in a batch has different length
        xs, pos_emb, chunk_mask = self.embed(xs, chunk_mask, offset)
        cache_size = att_cache.size(3)  # required cache size
        masks = torch.cat((cache_mask, chunk_mask), dim=2)
        att_mask = torch.cat((cache_mask, chunk_mask), dim=2)
        index = offset - cache_size

        pos_emb = self.embed.position_encoding(index, cache_size + xs.size(1))
        pos_emb = pos_emb.to(dtype=xs.dtype)

        next_cache_start = -self.required_cache_size
        r_cache_mask = masks[:, :, next_cache_start:]

        r_att_cache = []
        r_cnn_cache = []
        mask_pad = chunk_mask.to(torch.bool)
        max_att_len, max_cnn_len = (
            0,
            0,
        )  # for repeat_interleave of new_att_cache
        # 逐层通过 Conformer 编码器进行前向传播，处理每个层的输入，更新注意力缓存和 CNN 缓存。
        for i, layer in enumerate(self.encoder.encoders):
            factor = self.calculate_downsampling_factor(i)
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i:i + 1]) is (b, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            # shape(new_att_cache) = [ batch, head, time2, outdim//head * 2 ]
            att_cache_trunc = 0
            if xs.size(1) + att_cache.size(3) / factor > pos_emb.size(1):
                # The time step is not divisible by the downsampling multiple
                # We propose to double the chunk_size.
                att_cache_trunc = (xs.size(1) + att_cache.size(3) // factor -
                                   pos_emb.size(1) + 1)
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                mask_pad=mask_pad,
                att_cache=att_cache[i][:, :, ::factor, :][:, :,
                                                          att_cache_trunc:, :],
                cnn_cache=cnn_cache[i, :, :, :]
                if cnn_cache.size(0) > 0 else cnn_cache,
            )

            if i in self.stride_layer_idx:
                # compute time dimension for next block
                efficient_index = self.stride_layer_idx.index(i)
                att_mask = att_mask[:, ::self.stride[efficient_index], ::self.
                                    stride[efficient_index], ]
                mask_pad = mask_pad[:, ::self.stride[efficient_index], ::self.
                                    stride[efficient_index], ]
                pos_emb = pos_emb[:, ::self.stride[efficient_index], :]

            # shape(new_att_cache) = [batch, head, time2, outdim]
            new_att_cache = new_att_cache[:, :, next_cache_start // factor:, :]
            # shape(new_cnn_cache) = [batch, 1, outdim, cache_t2]
            new_cnn_cache = new_cnn_cache.unsqueeze(1)  # shape(1):layerID

            # use repeat_interleave to new_att_cache
            # new_att_cache = new_att_cache.repeat_interleave(repeats=factor, dim=2)
            new_att_cache = (new_att_cache.unsqueeze(3).repeat(
                1, 1, 1, factor, 1).flatten(2, 3))
            # padding new_cnn_cache to cnn.lorder for casual convolution
            new_cnn_cache = F.pad(
                new_cnn_cache,
                (self.cnn_module_kernel - 1 - new_cnn_cache.size(3), 0),
            )

            if i == 0:
                # record length for the first block as max length
                max_att_len = new_att_cache.size(2)
                max_cnn_len = new_cnn_cache.size(3)

            # update real shape of att_cache and cnn_cache
            # 将新的注意力缓存和 CNN 缓存添加到列表中。
            r_att_cache.append(new_att_cache[:, :,
                                             -max_att_len:, :].unsqueeze(1))
            r_cnn_cache.append(new_cnn_cache[:, :, :, -max_cnn_len:])

        # 对输出进行归一化（如果需要），然后计算 CTC 的对数概率。
        if self.encoder.normalize_before:
            chunk_out = self.encoder.after_norm(xs)
        else:
            chunk_out = xs

        # shape of r_att_cache: (b, elayers, head, time2, outdim)
        r_att_cache = torch.cat(r_att_cache, dim=1)  # concat on layers idx
        # shape of r_cnn_cache: (b, elayers, outdim, cache_t2)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=1)  # concat on layers

        # <---------forward_chunk END--------->

        # 计算 CTC 输出的对数概率，并根据 beam_size 获取前 beam_size 个最可能的候选。
        log_ctc_probs = self.ctc.log_softmax(chunk_out)
        log_probs, log_probs_idx = torch.topk(log_ctc_probs,
                                              self.beam_size,
                                              dim=2)
        log_probs = log_probs.to(chunk_xs.dtype)

        r_offset = offset + chunk_out.shape[1]
        # the below ops not supported in Tensorrt
        # chunk_out_lens = torch.div(chunk_lens, subsampling_rate,
        #                   rounding_mode='floor')
        chunk_out_lens = (
            chunk_lens // self.subsampling_rate //
            self.calculate_downsampling_factor(self.num_blocks + 1))
        chunk_out_lens += 1
        r_offset = r_offset.unsqueeze(1)

        return (
            log_probs,
            log_probs_idx,
            chunk_out,
            chunk_out_lens,
            r_offset,
            r_att_cache,
            r_cnn_cache,
            r_cache_mask,
        )


# 最基础的解码器
class Decoder(torch.nn.Module):

    def __init__(
        self,
        decoder: TransformerDecoder,
        ctc_weight: float = 0.5,
        reverse_weight: float = 0.0,
        beam_size: int = 10,
        decoder_fastertransformer: bool = False,
    ):
        super().__init__()
        self.decoder = decoder
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.beam_size = beam_size
        self.decoder_fastertransformer = decoder_fastertransformer

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_lens: torch.Tensor,
        hyps_pad_sos_eos: torch.Tensor,
        hyps_lens_sos: torch.Tensor,
        r_hyps_pad_sos_eos: torch.Tensor,
        ctc_score: torch.Tensor,
    ):
        """Encoder
        Args:
            encoder_out: B x T x F
            encoder_lens: B
            hyps_pad_sos_eos: B x beam x (T2+1),
                        hyps with sos & eos and padded by ignore id
            hyps_lens_sos: B x beam, length for each hyp with sos
            r_hyps_pad_sos_eos: B x beam x (T2+1),
                    reversed hyps with sos & eos and padded by ignore id
            ctc_score: B x beam, ctc score for each hyp
        Returns:
            decoder_out: B x beam x T2 x V
            r_decoder_out: B x beam x T2 x V
            best_index: B
        """
        B, T, F = encoder_out.shape
        bz = self.beam_size
        B2 = B * bz
        # encoder_out 和 encoder_mask 被调整为适应束搜索的批次大小（B * beam_size），通过重复（repeat）和视图变换（view）。
        # 处理 hyps_pad_sos_eos 和 r_hyps_pad_sos_eos，分别为正向和反向解码的假设序列，并拆分成 SOS 和 EOS 部分。
        encoder_out = encoder_out.repeat(1, bz, 1).view(B2, T, F)
        encoder_mask = ~make_pad_mask(encoder_lens, T).unsqueeze(1)
        encoder_mask = encoder_mask.repeat(1, bz, 1).view(B2, 1, T)
        T2 = hyps_pad_sos_eos.shape[2] - 1
        hyps_pad = hyps_pad_sos_eos.view(B2, T2 + 1)
        hyps_lens = hyps_lens_sos.view(B2, )
        hyps_pad_sos = hyps_pad[:, :-1].contiguous()
        hyps_pad_eos = hyps_pad[:, 1:].contiguous()

        r_hyps_pad = r_hyps_pad_sos_eos.view(B2, T2 + 1)
        r_hyps_pad_sos = r_hyps_pad[:, :-1].contiguous()
        r_hyps_pad_eos = r_hyps_pad[:, 1:].contiguous()

        # 正向解码器（self.decoder）处理 encoder_out 和 hyps_pad_sos，返回正向解码器输出 decoder_out 和反向解码器输出 r_decoder_out。
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out,
            encoder_mask,
            hyps_pad_sos,
            hyps_lens,
            r_hyps_pad_sos,
            self.reverse_weight,
        )
        # 对 decoder_out 进行 log_softmax 操作，得到对数概率。
        # 通过 make_pad_mask 创建一个掩码，忽略填充部分。
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        V = decoder_out.shape[-1]
        decoder_out = decoder_out.view(B2, T2, V)
        mask = ~make_pad_mask(hyps_lens, T2)  # B2 x T2
        # mask index, remove ignore id
        # 使用 gather 操作从 decoder_out 中选择对应的得分，并应用掩码，忽略填充部分。
        index = torch.unsqueeze(hyps_pad_eos * mask, 2)
        score = decoder_out.gather(2, index).squeeze(2)  # B2 X T2
        # mask padded part
        score = score * mask
        decoder_out = decoder_out.view(B, bz, T2, V)
        # 如果 reverse_weight 大于零，反向解码器的输出 r_decoder_out 被处理并与正向解码器的得分 score 加权平均。
        # 这里的 reverse_weight 控制正向和反向解码得分的权重。
        if self.reverse_weight > 0:
            r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out,
                                                            dim=-1)
            r_decoder_out = r_decoder_out.view(B2, T2, V)
            index = torch.unsqueeze(r_hyps_pad_eos * mask, 2)
            r_score = r_decoder_out.gather(2, index).squeeze(2)
            r_score = r_score * mask
            score = (score * (1 - self.reverse_weight) +
                     self.reverse_weight * r_score)
            r_decoder_out = r_decoder_out.view(B, bz, T2, V)
        # 最终的得分是所有时间步的得分的总和，并加上 CTC 得分（通过 ctc_weight 加权）。然后通过 torch.argmax 找到得分最高的假设。
        score = torch.sum(score, axis=1)  # B2
        score = torch.reshape(score, (B, bz)) + self.ctc_weight * ctc_score
        best_index = torch.argmax(score, dim=1)
        # 如果启用了 decoder_fastertransformer，则返回解码器输出和最佳索引；否则，仅返回最佳索引
        if self.decoder_fastertransformer:
            return decoder_out, best_index
        else:
            return best_index


# 将 PyTorch 张量（torch.Tensor）转换为 NumPy 数组。
def to_numpy(tensors):
    out = []
    if type(tensors) == torch.tensor:
        tensors = [tensors]
    for tensor in tensors:
        if tensor.requires_grad:
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = tensor.cpu().numpy()
        out.append(tensor)
    return out


# 逐一比较两个列表中的张量（xlist 和 blist）是否在给定的相对容忍度（rtol）和绝对容忍度（atol）下相等。
def test(xlist, blist, rtol=1e-3, atol=1e-5, tolerate_small_mismatch=True):
    for a, b in zip(xlist, blist):
        try:
            torch.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        except AssertionError as error:
            if tolerate_small_mismatch:
                print(error)
            else:
                raise


# 导出离线（非流式）编码器模型。
def export_offline_encoder(model, configs, args, logger, encoder_onnx_path):
    bz = 32
    seq_len = 100
    beam_size = args.beam_size
    feature_size = configs["input_dim"]

    # 生成一个随机的语音特征张量和语音长度张量。
    speech = torch.randn(bz, seq_len, feature_size, dtype=torch.float32)
    speech_lens = torch.randint(low=10,
                                high=seq_len,
                                size=(bz, ),
                                dtype=torch.int32)
    # 创建一个编码器对象，并将其设置为评估模式，把encoder和ctc一起放在新的encoder.onnx中了。
    encoder = Encoder(model.encoder, model.ctc, beam_size)
    encoder.eval()

    # 将pytorch模型导出为onnx模型
    torch.onnx.export(
        # 要导出的 PyTorch 模型。
        encoder,
        # 模型的输入张量，speech 表示输入的语音特征，speech_lens 表示语音特征的长度。
        (speech, speech_lens),
        # 导出的 ONNX 模型文件的路径。
        encoder_onnx_path,
        # 导出模型时包含模型参数。
        export_params=True,
        #指定 ONNX 的操作集版本。
        opset_version=13,
        # 在导出过程中执行常量折叠优化。
        do_constant_folding=True,
        # 指定输入张量的名称。
        input_names=["speech", "speech_lengths"],
        # 指定输出张量的名称，输出张量是模型前向传播的结果，前向传播相当于推理过程，这里指定了输出的格式。
        output_names=[
            "encoder_out",
            "encoder_out_lens",
            "ctc_log_probs",
            "beam_log_probs",
            "beam_log_probs_idx",
        ],
        # 指定动态轴，用于支持可变长度的输入和输出。动态轴指定的这些维度可以在推理时动态变化不固定
        dynamic_axes={
            # 语音
            "speech": {
                # 第 0 维表示批量大小（B），第 1 维表示时间步数（T）。
                0: "B",
                1: "T"
            },
            # 语音块长度
            "speech_lengths": {
                # 第 0 维表示批量大小（B）。
                0: "B"
            },
            # 编码器输出（因为输入音频块大小是浮动的）
            "encoder_out": {
                # 第 0 维表示批量大小（B），第 1 维表示输出时间步数（T_OUT）。
                0: "B",
                1: "T_OUT"
            },
            # 编码器输出长度（因为输入音频长度大小是浮动的）
            "encoder_out_lens": {
                # 第 0 维表示批量大小（B）。
                0: "B"
            },
            # CTC的对数概率（torch自带的方法）因为输入音频长度不同所以有变化
            "ctc_log_probs": {
                # 第 0 维表示批量大小（B），第 1 维表示输出时间步数（T_OUT）。
                0: "B",
                1: "T_OUT"
            },
            # 束搜索算法的输出概率，因为输入音频长度不同所以有变化
            "beam_log_probs": {
                # 第 0 维表示批量大小（B），第 1 维表示输出时间步数（T_OUT）。
                0: "B",
                1: "T_OUT"
            },
            # 束搜索的索引也浮动，因为输入长度是动态的，所以输出也是动态的
            "beam_log_probs_idx": {
                # 第 0 维表示批量大小（B），第 1 维表示输出时间步数（T_OUT）。
                0: "B",
                1: "T_OUT"
            },
        },
        # 禁用详细日志输出。
        verbose=False,
    )

    # 这段代码的作用是在不计算梯度的上下文中，使用模型 encoder 对输入 speech 和 speech_lens 进行前向传播，并获取多个输出
    with torch.no_grad():
        o0, o1, o2, o3, o4 = encoder(speech, speech_lens)

    # 这段代码的作用是使用 ONNX Runtime 创建一个推理会话，并指定使用 CUDA 作为执行提供程序
    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(encoder_onnx_path,
                                               providers=providers)
    # 将输入数据转换为Numpy数组，使用onnxruntime推理会话进行推理
    ort_inputs = {
        "speech": to_numpy(speech),
        "speech_lengths": to_numpy(speech_lens),
    }
    # 进行推理
    ort_outs = ort_session.run(None, ort_inputs)

    # check encoder output，检查编码器输出
    # 测试导出的onnx模型
    test(to_numpy([o0, o1, o2, o3, o4]), ort_outs)
    # 记录成功导出的信息
    logger.info("export offline onnx encoder succeed!")
    # 返回onnx配置
    onnx_config = {
        "beam_size": args.beam_size,
        "reverse_weight": args.reverse_weight,
        "ctc_weight": args.ctc_weight,
        "fp16": args.fp16,
    }
    return onnx_config


# 导出在线（流式）编码器模型。几个变量分别为：Pytorch模型、配置、命令行参数、日志、编码器ONNX的导出路径。
def export_online_encoder(model, configs, args, logger, encoder_onnx_path):
    # 解码块大小
    decoding_chunk_size = args.decoding_chunk_size
    # 下采样率
    subsampling = model.encoder.embed.subsampling_rate
    # 文本上下文图
    context = model.encoder.embed.right_context + 1
    # 解码窗口大小
    decoding_window = (decoding_chunk_size - 1) * subsampling + context
    # 输入数据形状
    # 批次大小，32个帧
    batch_size = 32
    # 音频长度
    audio_len = decoding_window
    # 特征维度
    feature_size = configs["input_dim"]
    # 输出大小
    output_size = configs["encoder_conf"]["output_size"]
    # 输出层数
    num_layers = configs["encoder_conf"]["num_blocks"]
    # in transformer the cnn module will not be available
    # 这段代码检查是否使用 Transformer 模型，如果 cnn_module_kernel 为 0，则使用 Transformer。
    transformer = False
    cnn_module_kernel = configs["encoder_conf"].get("cnn_module_kernel", 1) - 1
    if not cnn_module_kernel:
        transformer = True
    # 这些参数用于计算解码过程中所需的缓存大小。
    num_decoding_left_chunks = args.num_decoding_left_chunks
    required_cache_size = decoding_chunk_size * num_decoding_left_chunks
    # 根据配置选择不同类型的编码器，并将其设置为评估模式。
    if configs["encoder"] == "squeezeformer":
        encoder = StreamingSqueezeformerEncoder(model, required_cache_size,
                                                args.beam_size)
    elif configs["encoder"] == "efficientConformer":
        encoder = StreamingEfficientConformerEncoder(model,
                                                     required_cache_size,
                                                     args.beam_size)
    else:
        encoder = StreamingEncoder(
            model,
            required_cache_size,
            args.beam_size,
            transformer,
            args.return_ctc_logprobs,
        )
    encoder.eval()

    # begin to export encoder，开始导出编码器
    chunk_xs = torch.randn(batch_size,
                           audio_len,
                           feature_size,
                           dtype=torch.float32)
    # 创建了一个全为 1 的张量，形状为 (1, xs.size(1))，
    # 数据类型为布尔型，并且与输入张量 xs 位于相同的设备上。xs.size(1) 获取了输入张量 xs 在第一个维度上的大小。
    chunk_lens = torch.ones(batch_size, dtype=torch.int32) * audio_len

    # 这部分代码使用 PyTorch 的 arange 函数生成一个从 0 到 batch_size-1 的一维张量。然后使用 unsqueeze 函数在第二个维度上增加一个维度。
    offset = torch.arange(0, batch_size).unsqueeze(1)
    #  (elayers, b, head, cache_t1, d_k * 2)
    # 这行代码从配置字典 configs 中获取注意力头的数量。
    head = configs["encoder_conf"]["attention_heads"]
    # 这行代码从配置字典中获取编码器的输出大小，并计算每个注意力头的维度大小。
    d_k = configs["encoder_conf"]["output_size"] // head
    att_cache = torch.randn(
        batch_size,
        num_layers,
        head,
        required_cache_size,
        d_k * 2,
        dtype=torch.float32,
    )
    # 创建一个名为 cnn_cache 的张量，用于存储 CNN 模块的缓存。
    cnn_cache = torch.randn(
        batch_size,
        num_layers,
        output_size,
        cnn_module_kernel,
        dtype=torch.float32,
    )

    # 创建一个名为 cache_mask 的张量，用于存储缓存掩码。
    cache_mask = torch.ones(batch_size,
                            1,
                            required_cache_size,
                            dtype=torch.float32)
    # 确定输入和输出的名称。
    input_names = [
        "chunk_xs",
        "chunk_lens",
        "offset",
        "att_cache",
        "cnn_cache",
        "cache_mask",
    ]
    output_names = [
        "log_probs",
        "log_probs_idx",
        "chunk_out",
        "chunk_out_lens",
        "r_offset",
        "r_att_cache",
        "r_cnn_cache",
        "r_cache_mask",
    ]
    # 如果是返回 CTC 对数概率，就要变成这种输出。
    if args.return_ctc_logprobs:
        output_names = [
            "ctc_log_probs",
            "chunk_out",
            "chunk_out_lens",
            "r_offset",
            "r_att_cache",
            "r_cnn_cache",
            "r_cache_mask",
        ]
    # 这段代码的作用是将多个输入张量组合成一个元组 input_tensors，用于在后续的模型导出过程中作为模型的输入。
    input_tensors = (
        chunk_xs,
        chunk_lens,
        offset,
        att_cache,
        cnn_cache,
        cache_mask,
    )
    # 如果是 Transformer 模型，就要删除 cnn_cache。
    if transformer:
        assert (args.return_ctc_logprobs is
                False), "return_ctc_logprobs is not supported in transformer"
        output_names.pop(6)

    # 获取所有的输入输出名称
    all_names = input_names + output_names
    # 多种轴，决定输入输出的浮动形状
    dynamic_axes = {}
    for name in all_names:
        # only the first dimension is dynamic
        # all other dimension is fixed
        # 只有第一维度可以浮动，其他的不行
        dynamic_axes[name] = {0: "B"}

    # 写入onnx文件
    torch.onnx.export(
        encoder,
        input_tensors,
        encoder_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    # 这段代码的作用是在不计算梯度的上下文中，使用模型 encoder 对输入 speech 和 speech_lens 进行前向传播，并获取多个输出
    with torch.no_grad():
        torch_outs = encoder(chunk_xs, chunk_lens, offset, att_cache,
                             cnn_cache, cache_mask)
    if transformer:
        # 检查是否使用 Transformer 模型，如果是，则移除第 6 个输出张量。
        torch_outs = list(torch_outs).pop(6)
    # 使用 ONNX Runtime 创建一个推理会话，并指定使用 CUDA 作为执行提供程序。
    ort_session = onnxruntime.InferenceSession(
        encoder_onnx_path, providers=["CUDAExecutionProvider"])
    # 创建一个字典 ort_inputs，用于存储 ONNX 推理会话的输入。
    ort_inputs = {}
    # 将输入数据转换为Numpy数组，使用onnxruntime推理会话进行推理
    input_tensors = to_numpy(input_tensors)
    # 将索引和名称组合成一个字典，用于在后续的模型导出过程中作为模型的输入。
    for idx, name in enumerate(input_names):
        ort_inputs[name] = input_tensors[idx]
    # 如果是 Transformer 模型，就要删除 cnn_cache。
    if transformer:
        del ort_inputs["cnn_cache"]
    # 使用 ONNX Runtime 推理会话进行推理。
    ort_outs = ort_session.run(None, ort_inputs)
    # 检查编码器输出。
    test(to_numpy(torch_outs), ort_outs, rtol=1e-03, atol=1e-05)
    # 打印结果
    logger.info("export to onnx streaming encoder succeed!")
    # 规定onnx配置
    onnx_config = {
        "subsampling_rate": subsampling,
        "context": context,
        "decoding_chunk_size": decoding_chunk_size,
        "num_decoding_left_chunks": num_decoding_left_chunks,
        "beam_size": args.beam_size,
        "fp16": args.fp16,
        "feat_size": feature_size,
        "decoding_window": decoding_window,
        "cnn_module_kernel_cache": cnn_module_kernel,
        "return_ctc_logprobs": args.return_ctc_logprobs,
    }
    # 返回onnx配置
    return onnx_config


# 导出解码器模型。
def export_rescoring_decoder(model, configs, args, logger, decoder_onnx_path,
                             decoder_fastertransformer):
    # 规定使用之前的decoder
    bz, seq_len = 32, 100
    beam_size = args.beam_size
    decoder = Decoder(
        model.decoder,
        model.ctc_weight,
        model.reverse_weight,
        beam_size,
        decoder_fastertransformer,
    )
    # 转为评估模式
    decoder.eval()

    # 生成随机张量
    hyps_pad_sos_eos = torch.randint(low=3,
                                     high=1000,
                                     size=(bz, beam_size, seq_len))
    hyps_lens_sos = torch.randint(low=3,
                                  high=seq_len,
                                  size=(bz, beam_size),
                                  dtype=torch.int32)
    r_hyps_pad_sos_eos = torch.randint(low=3,
                                       high=1000,
                                       size=(bz, beam_size, seq_len))

    # 获取编码器输出，输出本身，尺寸，ctc得分，长度
    output_size = configs["encoder_conf"]["output_size"]
    encoder_out = torch.randn(bz, seq_len, output_size, dtype=torch.float32)
    encoder_out_lens = torch.randint(low=3,
                                     high=seq_len,
                                     size=(bz, ),
                                     dtype=torch.int32)
    ctc_score = torch.randn(bz, beam_size, dtype=torch.float32)

    # 规定输入输出名称
    input_names = [
        "encoder_out",
        "encoder_out_lens",
        "hyps_pad_sos_eos",
        "hyps_lens_sos",
        "r_hyps_pad_sos_eos",
        "ctc_score",
    ]
    output_names = ["best_index"]
    # 如果是返回decoder_out和best_index，就要加上decoder_out
    if decoder_fastertransformer:
        output_names.insert(0, "decoder_out")

    # 写入onnx文件
    torch.onnx.export(
        decoder,
        (
            encoder_out,
            encoder_out_lens,
            hyps_pad_sos_eos,
            hyps_lens_sos,
            r_hyps_pad_sos_eos,
            ctc_score,
        ),
        decoder_onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        # 这些是浮动可变的轴，决定输入输出的形状
        dynamic_axes={
            "encoder_out": {
                0: "B",
                1: "T"
            },
            "encoder_out_lens": {
                0: "B"
            },
            "hyps_pad_sos_eos": {
                0: "B",
                2: "T2"
            },
            "hyps_lens_sos": {
                0: "B"
            },
            "r_hyps_pad_sos_eos": {
                0: "B",
                2: "T2"
            },
            "ctc_score": {
                0: "B"
            },
            "best_index": {
                0: "B"
            },
        },
        verbose=False,
    )
    # 这段代码的作用是在不计算梯度的上下文中，使用模型 encoder 对输入 speech 和 speech_lens 进行前向传播，并获取多个输出
    with torch.no_grad():
        o0 = decoder(
            encoder_out,
            encoder_out_lens,
            hyps_pad_sos_eos,
            hyps_lens_sos,
            r_hyps_pad_sos_eos,
            ctc_score,
        )
    # 用显卡跑
    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(decoder_onnx_path,
                                               providers=providers)
    
    # 指定输入张量的名称。
    input_tensors = [
        encoder_out,
        encoder_out_lens,
        hyps_pad_sos_eos,
        hyps_lens_sos,
        r_hyps_pad_sos_eos,
        ctc_score,
    ]
    ort_inputs = {}
    input_tensors = to_numpy(input_tensors)
    #名称索引对应
    for idx, name in enumerate(input_names):
        ort_inputs[name] = input_tensors[idx]

    # if model.reverse weight == 0,
    # the r_hyps_pad will be removed
    # from the onnx decoder since it doen't play any role
    # 如果 reverse_weight 为 0，则删除输入字典 ort_inputs 中的 r_hyps_pad_sos_eos 键。
    # reverse_weight 是一个模型参数，通常用于控制反向解码的权重。如果其值为 0，表示反向解码不被使用，因此可以删除相关的输入。
    if model.reverse_weight == 0:
        del ort_inputs["r_hyps_pad_sos_eos"]
    ort_outs = ort_session.run(None, ort_inputs)

    # check decoder output
    # 测试转换后的onnx模型输出
    if decoder_fastertransformer:
        test(to_numpy(o0), ort_outs, rtol=1e-03, atol=1e-05)
    else:
        test(to_numpy([o0]), ort_outs, rtol=1e-03, atol=1e-05)
    logger.info("export to onnx decoder succeed!")


if __name__ == "__main__":
    # 指定各种参数
    parser = argparse.ArgumentParser(description="export x86_gpu model")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--checkpoint", required=True, help="checkpoint model")
    parser.add_argument(
        "--cmvn_file",
        required=False,
        default="",
        type=str,
        help="global_cmvn file, default path is in config file",
    )
    parser.add_argument(
        "--reverse_weight",
        default=-1.0,
        type=float,
        required=False,
        help="reverse weight for bitransformer," +
        "default value is in config file",
    )
    parser.add_argument(
        "--ctc_weight",
        default=-1.0,
        type=float,
        required=False,
        help="ctc weight, default value is in config file",
    )
    parser.add_argument(
        "--beam_size",
        default=10,
        type=int,
        required=False,
        help="beam size would be ctc output size",
    )
    parser.add_argument(
        "--output_onnx_dir",
        default="onnx_model",
        help="output onnx encoder and decoder directory",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="whether to export fp16 model, default false",
    )
    # arguments for streaming encoder
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="whether to export streaming encoder, default false",
    )
    parser.add_argument(
        "--decoding_chunk_size",
        default=16,
        type=int,
        required=False,
        help="the decoding chunk size, <=0 is not supported",
    )
    parser.add_argument(
        "--num_decoding_left_chunks",
        default=5,
        type=int,
        required=False,
        help="number of left chunks, <= 0 is not supported",
    )
    parser.add_argument(
        "--decoder_fastertransformer",
        action="store_true",
        help="return decoder_out and best_index for ft",
    )
    parser.add_argument(
        "--return_ctc_logprobs",
        action="store_true",
        help="return full ctc_log_probs for TLG streaming encoder",
    )
    args = parser.parse_args()

    # 设置随机数种子，这行代码使用 torch.manual_seed 函数设置 PyTorch 的随机数种子为 0。
    # 设置随机数种子可以确保每次运行代码时生成的随机数相同，从而使实验结果具有可重复性。
    torch.manual_seed(0)
    # 这行代码使用 torch.set_printoptions 函数设置 PyTorch 的打印选项，将打印精度设置为 10 位小数。
    # 这意味着在打印张量时，小数部分将显示 10 位数字，从而提供更高的精度。
    torch.set_printoptions(precision=10)

    # 读取配置文件和归一化文件
    with open(args.config, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if args.cmvn_file and os.path.exists(args.cmvn_file):
        if 'cmvn' not in configs:
            configs['cmvn'] = "global_cmvn"
            configs['cmvn_conf'] = {}
        else:
            assert configs['cmvn'] == "global_cmvn"
            assert configs['cmvn_conf'] is not None
        configs['cmvn_conf']["cmvn_file"] = args.cmvn_file
    if (args.reverse_weight != -1.0
            and "reverse_weight" in configs["model_conf"]):
        configs["model_conf"]["reverse_weight"] = args.reverse_weight
        print("Update reverse weight to", args.reverse_weight)
    if args.ctc_weight != -1:
        print("Update ctc weight to ", args.ctc_weight)
        configs["model_conf"]["ctc_weight"] = args.ctc_weight
    configs["encoder_conf"]["use_dynamic_chunk"] = False

    # 初始化模型并设置为评估模式
    model, configs = init_model(args, configs)
    model.eval()

    # 没有路径就创建路径
    if not os.path.exists(args.output_onnx_dir):
        os.mkdir(args.output_onnx_dir)
    encoder_onnx_path = os.path.join(args.output_onnx_dir, "encoder.onnx")
    export_enc_func = None
    # 流式就先保证块大小和剩余块数大于0，再引入流式编码器，没有剩余块那就是非流式的
    if args.streaming:
        assert args.decoding_chunk_size > 0
        assert args.num_decoding_left_chunks > 0
        export_enc_func = export_online_encoder
    # 非流式直接引入就行
    else:
        export_enc_func = export_offline_encoder

    onnx_config = export_enc_func(model, configs, args, logger,
                                  encoder_onnx_path)

    decoder_onnx_path = os.path.join(args.output_onnx_dir, "decoder.onnx")
    # 导出重打分的解码器
    export_rescoring_decoder(
        model,
        configs,
        args,
        logger,
        decoder_onnx_path,
        args.decoder_fastertransformer,
    )

    # 如果是 fp16，就要转换为fp16再保存
    if args.fp16:
        try:
            import onnxmltools
            from onnxmltools.utils.float16_converter import \
                convert_float_to_float16
        except ImportError:
            print("Please install onnxmltools!")
            sys.exit(1)
        encoder_onnx_model = onnxmltools.utils.load_model(encoder_onnx_path)
        encoder_onnx_model = convert_float_to_float16(encoder_onnx_model)
        encoder_onnx_path = os.path.join(args.output_onnx_dir,
                                         "encoder_fp16.onnx")
        onnxmltools.utils.save_model(encoder_onnx_model, encoder_onnx_path)
        decoder_onnx_model = onnxmltools.utils.load_model(decoder_onnx_path)
        decoder_onnx_model = convert_float_to_float16(decoder_onnx_model)
        decoder_onnx_path = os.path.join(args.output_onnx_dir,
                                         "decoder_fp16.onnx")
        onnxmltools.utils.save_model(decoder_onnx_model, decoder_onnx_path)
    # dump configurations

    # 保存配置文件
    config_dir = os.path.join(args.output_onnx_dir, "config.yaml")
    with open(config_dir, "w") as out:
        yaml.dump(onnx_config, out)

# 显卡版本导出模型完成