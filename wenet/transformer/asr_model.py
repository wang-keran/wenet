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

from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.transformer.search import (ctc_greedy_search,
                                      ctc_prefix_beam_search,
                                      attention_beam_search,
                                      attention_rescoring, DecodeResult)
from wenet.utils.mask import make_pad_mask
from wenet.utils.common import (IGNORE_ID, add_sos_eos, th_accuracy,
                                reverse_pad_list)
from wenet.utils.context_graph import ContextGraph


# 这是一个混合的CTC-注意力机制的语音识别模型类 ASRModel，基于PyTorch。
# 它使用了基于CTC (Connectionist Temporal Classification) 和注意力机制的解码器（Attention Decoder），适用于语音识别任务。
class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    #  类的结构与初始化
    def __init__(
        self,
        vocab_size: int,
        encoder: BaseEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        special_tokens: Optional[dict] = None,
        apply_non_blank_embedding: bool = False,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = (vocab_size - 1 if special_tokens is None else
                    special_tokens.get("<sos>", vocab_size - 1))
        self.eos = (vocab_size - 1 if special_tokens is None else
                    special_tokens.get("<eos>", vocab_size - 1))
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.apply_non_blank_embedding = apply_non_blank_embedding

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    # 前向传播函数执行整个ASR模型的核心计算
    @torch.jit.unused
    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss"""
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, ctc_probs = self.ctc(encoder_out, encoder_out_lens, text,
                                           text_lengths)
        else:
            loss_ctc, ctc_probs = None, None

        # 2b. Attention-decoder branch
        # use non blank (token level) embedding for decoder
        if self.apply_non_blank_embedding:
            assert self.ctc_weight != 0
            assert ctc_probs is not None
            encoder_out, encoder_mask = self.filter_blank_embedding(
                ctc_probs, encoder_out)
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(
                encoder_out, encoder_mask, text, text_lengths, {
                    "langs": batch["langs"],
                    "tasks": batch["tasks"]
                })
        else:
            loss_att = None
            acc_att = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
            
        print("非流式编码结果1：")
        print(loss)

        return {
            "loss": loss,
            "loss_att": loss_att,
            "loss_ctc": loss_ctc,
            "th_accuracy": acc_att,
        }

    # 这段代码定义了一个名为 tie_or_clone_weights 的方法，用于在当前对象的解码器中调用同名方法。
    # 通过传递 jit_mode 参数，可以控制解码器中的权重共享或复制行为
    def tie_or_clone_weights(self, jit_mode: bool = True):
        self.decoder.tie_or_clone_weights(jit_mode)

    # 计算CTC损失
    @torch.jit.unused
    def _forward_ctc(
            self, encoder_out: torch.Tensor, encoder_mask: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        loss_ctc, ctc_probs = self.ctc(encoder_out, encoder_out_lens, text,
                                       text_lengths)
        return loss_ctc, ctc_probs

    # 获得CTC的概率分布。
    #   ctc_probs 是一个张量，表示 CTC 的概率输出，
    #   encoder_out 是编码器的输出。
    def filter_blank_embedding(
            self, ctc_probs: torch.Tensor,
            encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 确定输入批次大小
        batch_size = encoder_out.size(0)
        # 获取最大长度 maxlen
        # encoder_out 的形状为 (batch_size, maxlen, feature_dim)，表示每个样本在时间维度上的特征。
        maxlen = encoder_out.size(1)
        # 计算 CTC 概率中的最大值索引 top1_index，选择每个时间步的最高概率的类别（即标记）。
        # top1_index 的形状为 (batch_size, maxlen)。
        top1_index = torch.argmax(ctc_probs, dim=2)
        # 初始化一个空列表 indices，用于存储每个样本中非空白标记的索引。
        indices = []
        # 对于每个样本 j，使用列表推导式生成一个包含非空白标记（即不等于0的标记）的索引列表，并将其转换为张量，添加到 indices 列表中。
        for j in range(batch_size):
            indices.append(
                torch.tensor(
                    [i for i in range(maxlen) if top1_index[j][i] != 0]))

        # 使用 torch.index_select 根据之前计算的 indices 从 encoder_out 中选择非空白标记对应的特征。
        select_encoder_out = [
            torch.index_select(encoder_out[i, :, :], 0,
                               indices[i].to(encoder_out.device))
            # 这里对每个样本 i 进行索引选择，提取对应的编码器输出特征，生成新的张量列表 select_encoder_out。
            for i in range(batch_size)
        ]
        # 使用 pad_sequence 函数将 select_encoder_out 中的张量填充成相同长度。
        # batch_first=True 表示将批次维度放在第一维，padding_value=0 指定填充值为0。
        select_encoder_out = pad_sequence(select_encoder_out,
                                          batch_first=True,
                                          padding_value=0).to(
                                              # 将填充后的结果移动到与 encoder_out 相同的设备上（CPU或GPU）。
                                              encoder_out.device)
        # 计算每个样本中非空白标记的数量，并将结果转换为张量 xs_lens，表示每个样本的实际长度，同样移动到与 encoder_out 相同的设备上。
        xs_lens = torch.tensor([len(indices[i]) for i in range(batch_size)
                                ]).to(encoder_out.device)
        # 获取填充后编码器输出的时间步长 T，即样本中的最大时间步长度。
        T = select_encoder_out.size(1)
        # 调用 make_pad_mask 函数，根据每个样本的长度 xs_lens 创建一个填充掩码。
        # 通过取反（~）来标记有效时间步，unsqueeze(1) 将其维度扩展为 (batch_size, 1, T)，以便与编码器输出进行广播操作。
        encoder_mask = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        # 将处理后的编码器输出赋值给 encoder_out。
        encoder_out = select_encoder_out
        return encoder_out, encoder_mask

    # 计算注意力机制的损失，包括正向解码和（可选的）反向解码。
    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        infos: Dict[str, List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     r_ys_in_pad,
                                                     self.reverse_weight)
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (
            1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att

    # 这个函数根据是否模拟流式处理来决定语音数据的编码方式。
    # 流式处理的好处是可以边接收边处理，适用于实时语音识别等场景；非流式处理则适用于离线场景。
    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        #print(simulate_streaming)           # 调试用
        #print("切块数量为：")                   # 调试用
        #print(decoding_chunk_size)          # 调试用
        #print("剩余块数为：")
        #print(num_decoding_left_chunks)
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    # The same interface just like whisper
    # see https://github.com/openai/whisper/blob/main/whisper/model.py#L287
    def embed_audio(
        self,
        mel: torch.Tensor,
        mel_len: torch.Tensor,
        chunk_size: int = -1,
    ) -> [torch.Tensor, torch.Tensor]:
        encoder_out, encoder_mask = self.encoder(mel, mel_len, chunk_size)
        return encoder_out, encoder_mask

    @torch.jit.unused
    def ctc_logprobs(self,
                     encoder_out: torch.Tensor,
                     blank_penalty: float = 0.0,
                     blank_id: int = 0):
        # 如果有空白惩罚
        if blank_penalty > 0.0:
            # 通过 CTC 层将编码器输出转化为 logits，使用torch自带的方法。
            # logits指的是模型输出的原始预测分数，这些分数通常是在应用激活函数（如 softmax 或 sigmoid）之前的结果。
            logits = self.ctc.ctc_lo(encoder_out)
            # 将空白标签的 logits 减去惩罚值，以便在计算时降低空白标签的概率。
            logits[:, :, blank_id] -= blank_penalty
            # 使用 log_softmax 方法计算对数概率,这个方法也是torch中的
            ctc_probs = logits.log_softmax(dim=2)
        # 如果没有空白惩罚，直接对编码器输出进行计算：ctc_probs = self.ctc.log_softmax(encoder_out)。
        else:
            ctc_probs = self.ctc.log_softmax(encoder_out)

        return ctc_probs

    # decode 函数为多种解码方法提供了统一的接口，允许根据需求选择不同的解码策略，包括CTC贪婪搜索、CTC前缀束搜索、注意力机制解码以及注意力重新评分
    def decode(
        self,
        methods: List[str],
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
        context_graph: ContextGraph = None,
        blank_id: int = 0,
        blank_penalty: float = 0.0,
        length_penalty: float = 0.0,
        infos: Dict[str, List[str]] = None,
    ) -> Dict[str, List[DecodeResult]]:
        """ Decode input speech

        Args:
            methods:(List[str]): list of decoding methods to use, which could
                could contain the following decoding methods, please refer paper:
                https://arxiv.org/pdf/2102.01547.pdf
                   * ctc_greedy_search
                   * ctc_prefix_beam_search
                   * atttention
                   * attention_rescoring
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns: dict results of all decoding methods
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        # _forward_encoder到encoder.py的forward到subsampling.py中Conv2dSubsampling4(BaseSubsampling)类的forward，到embeddinng.py中的RelPositionalEncoding(PositionalEncoding)类的forward
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)
        print("编码结果为2：")
        print(encoder_out.size(1))
        encoder_lens = encoder_mask.squeeze(1).sum(1)
        #print("编码后块数量为：")
        #print(encoder_lens)
        ctc_probs = self.ctc_logprobs(encoder_out, blank_penalty, blank_id)
        #print("对数概率为：")
        #print(ctc_probs)
        results = {}
        if 'attention' in methods:
            results['attention'] = attention_beam_search(
                self, encoder_out, encoder_mask, beam_size, length_penalty,
                infos)
        if 'ctc_greedy_search' in methods:
            results['ctc_greedy_search'] = ctc_greedy_search(
                ctc_probs, encoder_lens, blank_id)
        if 'ctc_prefix_beam_search' in methods:
            ctc_prefix_result = ctc_prefix_beam_search(ctc_probs, encoder_lens,
                                                       beam_size,
                                                       context_graph, blank_id)
            results['ctc_prefix_beam_search'] = ctc_prefix_result
        if 'attention_rescoring' in methods:
            # attention_rescoring depends on ctc_prefix_beam_search nbest
            if 'ctc_prefix_beam_search' in results:
                ctc_prefix_result = results['ctc_prefix_beam_search']
            else:
                ctc_prefix_result = ctc_prefix_beam_search(
                    ctc_probs, encoder_lens, beam_size, context_graph,
                    blank_id)
            if self.apply_non_blank_embedding:
                encoder_out, _ = self.filter_blank_embedding(
                    ctc_probs, encoder_out)
            results['attention_rescoring'] = attention_rescoring(
                self, ctc_prefix_result, encoder_out, encoder_lens, ctc_weight,
                reverse_weight, infos)
        return results


    # 这个方法的作用是提供模型中下采样率的接口，便于在 C++ 中调用，从而获取模型在处理序列输入时的下采样参数。
    @torch.jit.export
    def subsampling_rate(self) -> int:
        """ Export interface for c++ call, return subsampling_rate of the
            model
        """
        return self.encoder.embed.subsampling_rate

    # 提供一个接口，便于从模型中获取右上下文的长度信息，使得 C++ 中可以调用这个方法来了解模型在处理输入时所参考的右侧上下文长度。
    @torch.jit.export
    def right_context(self) -> int:
        """ Export interface for c++ call, return right_context of the model
        """
        return self.encoder.embed.right_context

    # 将模型的起始符号（SOS）的 ID 值导出为 C++ 接口，以便在 C++ 环境中调用并用于序列生成任务中的解码启动。
    @torch.jit.export
    def sos_symbol(self) -> int:
        """ Export interface for c++ call, return sos symbol id of the model
        """
        return self.sos

    # 通过 @torch.jit.export 装饰器，将模型的结束符号（EOS）的 ID 值导出为 C++ 环境可调用的接口。
    # 这个结束符号 ID 可以帮助模型在生成序列时，确定何时停止生成。
    @torch.jit.export
    def eos_symbol(self) -> int:
        """ Export interface for c++ call, return eos symbol id of the model
        """
        return self.eos

    # 通过接收音频片段及其相关缓存，执行片段级别的前向推理，并返回当前片段的输出以及更新后的缓存。适合流式推理
    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, give input chunk xs, and return
            output from time 0 to current chunk.

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          att_cache, cnn_cache)

    # 该方法简化了 CTC 解码的前处理步骤，确保模型输出适合进行 CTC 解码，同时通过导出为 C++ 接口，提升模型在实际应用中的推理速度
    @torch.jit.export
    def ctc_activation(self, xs: torch.Tensor) -> torch.Tensor:
        """ Export interface for c++ call, apply linear transform and log
            softmax before ctc
        Args:
            xs (torch.Tensor): encoder output

        Returns:
            torch.Tensor: activation before ctc

        """
        return self.ctc.log_softmax(xs)

    # 主要功能是检查解码器是否是双向解码器。
    @torch.jit.export
    def is_bidirectional_decoder(self) -> bool:
        """
        Returns:
            torch.Tensor: decoder output
        """
        if hasattr(self.decoder, 'right_decoder'):
            return True
        else:
            return False

    #hyps：从CTC前缀束搜索得到的假设序列，已经在开头填充了起始符号（sos）。
    # hyps_lens：每个假设序列的长度。
    # encoder_out：编码器的输出，表示输入音频的特征表示。
    # reverse_weight：控制是否使用反向解码的权重，大于0时会使用反向解码。
    # 该方法主要用于通过多个假设（来自 CTC 前缀束搜索）和一个编码器输出进行解码。
    @torch.jit.export
    def forward_attention_decoder(
        self,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        reverse_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output
            r_hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad eos at the begining which is used fo right to left decoder
            reverse_weight: used for verfing whether used right to left decoder,
            > 0 will use.

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        # 对输入的 encoder_out 进行重复，以适应假设的数量。
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        # 创建一个掩码（encoder_mask），用于指示哪些时间步是有效的。
        encoder_mask = torch.ones(num_hyps,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=encoder_out.device)

        # input for right to left decoder
        # this hyps_lens has count <sos> token, we need minus it.
        # 为反向解码准备数据：计算反向假设的长度和内容，并处理填充。
        r_hyps_lens = hyps_lens - 1
        # this hyps has included <sos> token, so it should be
        # convert the original hyps.
        r_hyps = hyps[:, 1:]
        #   >>> r_hyps
        #   >>> tensor([[ 1,  2,  3],
        #   >>>         [ 9,  8,  4],
        #   >>>         [ 2, -1, -1]])
        #   >>> r_hyps_lens
        #   >>> tensor([3, 3, 1])

        # NOTE(Mddct): `pad_sequence` is not supported by ONNX, it is used
        #   in `reverse_pad_list` thus we have to refine the below code.
        #   Issue: https://github.com/wenet-e2e/wenet/issues/1113
        # Equal to:
        #   >>> r_hyps = reverse_pad_list(r_hyps, r_hyps_lens, float(self.ignore_id))
        #   >>> r_hyps, _ = add_sos_eos(r_hyps, self.sos, self.eos, self.ignore_id)
        max_len = torch.max(r_hyps_lens)
        index_range = torch.arange(0, max_len, 1).to(encoder_out.device)
        seq_len_expand = r_hyps_lens.unsqueeze(1)
        seq_mask = seq_len_expand > index_range  # (beam, max_len)
        #   >>> seq_mask
        #   >>> tensor([[ True,  True,  True],
        #   >>>         [ True,  True,  True],
        #   >>>         [ True, False, False]])
        index = (seq_len_expand - 1) - index_range  # (beam, max_len)
        #   >>> index
        #   >>> tensor([[ 2,  1,  0],
        #   >>>         [ 2,  1,  0],
        #   >>>         [ 0, -1, -2]])
        index = index * seq_mask
        #   >>> index
        #   >>> tensor([[2, 1, 0],
        #   >>>         [2, 1, 0],
        #   >>>         [0, 0, 0]])
        # 通过 torch.gather 提取有效的反向假设。
        r_hyps = torch.gather(r_hyps, 1, index)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, 2, 2]])
        # 使用 torch.where 和 torch.cat 准备反向假设序列，以便在解码时使用。
        r_hyps = torch.where(seq_mask, r_hyps, self.eos)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, eos, eos]])
        r_hyps = torch.cat([hyps[:, 0:1], r_hyps], dim=1)
        #   >>> r_hyps
        #   >>> tensor([[sos, 3, 2, 1],
        #   >>>         [sos, 4, 8, 9],
        #   >>>         [sos, 2, eos, eos]])

        # 调用内部的 decoder 函数执行前向和反向解码，得到解码输出。
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
            reverse_weight)  # (num_hyps, max_hyps_len, vocab_size)
        # 使用 log_softmax 将输出转换为对数概率，以便进行后续的评分和选择。
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)

        # right to left decoder may be not used during decoding process,
        # which depends on reverse_weight param.
        # r_dccoder_out will be 0.0, if reverse_weight is 0.0
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        return decoder_out, r_decoder_out

# 总结：这个 ASRModel 类是一个高度灵活的语音识别模型，结合了CTC和注意力机制的优点，能够处理实时语音流的输入。