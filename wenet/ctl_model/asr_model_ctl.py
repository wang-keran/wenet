# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#               2023 NetEase Inc. (authors: Yuting Yang)
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
# Modified from ESPnet(https://github.com/espnet/espnet) and
# fairseq(https://github.com/facebookresearch/fairseq)

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.ctl_model.encoder import TransformerEncoder
from wenet.transformer.asr_model import ASRModel
from wenet.utils.common import IGNORE_ID


class CTLModel(ASRModel):
    """
        Implementation of Interspeecch 2023 paper:
        'Enhancing the Unified Streaming and Non-streaming Model
         with Contrastive Learning'
        https://arxiv.org/abs/2306.00755
    """

    # 初始化
    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        logit_temp: float = 0.1,
        n_negatives: int = 0,
        ctl_weight: float = 1,
        special_tokens: dict = None,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        super().__init__(vocab_size,
                         encoder,
                         decoder,
                         ctc,
                         ctc_weight,
                         ignore_id,
                         reverse_weight,
                         lsm_weight,
                         length_normalized_loss,
                         special_tokens=special_tokens)

        # For CTL Loss
        self.n_negatives = n_negatives
        self.ctl_weight = ctl_weight
        self.logit_temp = logit_temp

    # 告诉编译器这是python
    @torch.jit.unused
    # 前向传播函数
    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:

        # 数据准备
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)
        # 全序列前向传播
        loss_full, encoder_out_full, _, _ = self.forward_full(
            speech, speech_lengths, text, text_lengths)
        # 分块前向传播
        loss_chunk, encoder_out, lens_chunk, encoder_mask = self.forward_chunk(
            speech, speech_lengths, text, text_lengths)

        # 对比学习损失（CTL）
        ctl_loss = 0.0
        if self.ctl_weight > 0 and self.n_negatives > 0:
            num = encoder_out_full.size(1)
            targets = encoder_out_full
            src = encoder_out
            negs, negs_idxs = self.sample_negatives(targets,
                                                    targets.size(1),
                                                    speech_lengths=lens_chunk)
            ctl_loss = self.CTL(src, targets, negs, encoder_mask)

        # 总损失计算
        loss = loss_full + loss_chunk + self.ctl_weight * ctl_loss
        # 返回结果
        return {
            "loss": loss,
            "loss_full": loss_full,
            "loss_chunk": loss_chunk,
            "loss_ctl": ctl_loss
        }

    # 编码解码，计算损失
    # 返回一个元组，包含总损失loss、编码器输出encoder_out、编码器输出长度encoder_out_lens和编码器掩码encoder_mask。
    def forward_full(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        """Full context mode
        Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder.forward_full(
            speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths)
        else:
            loss_att = None

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc[0] + (1 -
                                                    self.ctc_weight) * loss_att
        return loss, encoder_out, encoder_out_lens, encoder_mask

    # 基于块的上下文模式的前向传播过程，编码器解码器，计算总损失，
    # 返回一个元组，包含总损失loss、编码器输出encoder_out、编码器输出长度encoder_out_lens和编码器掩码encoder_mask。
    def forward_chunk(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        """Chunk-based context mode
        Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths)
        else:
            loss_att = None

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc[0] + (1 -
                                                    self.ctc_weight) * loss_att
        return loss, encoder_out, encoder_out_lens, encoder_mask

    # 在给定的特征张量y中为每个样本随机抽取一定数量的负样本（或称为负例）。
    # 这在训练某些类型的模型时很有用，比如对比学习、噪声对比估计（NCE）等，其中模型需要学会区分正样本和负样本。
    # self: 指向当前对象实例的引用。
    # y: 输入的特征张量，形状为(batch_size, time_steps, feature_size)。
    # num: 要为每个样本抽取的负样本总数。
    # padding_count: 可选参数，指定要从时间步长中减去的填充数。默认为0。
    # speech_lengths: 可选参数，包含每个样本的实际语音长度（不考虑填充）。如果提供，则用于确保负样本索引不会超出实际语音内容。
    def sample_negatives(self, y, num, padding_count=0, speech_lengths=None):
        if self.n_negatives == 0:
            return y.new(0)
        bsz, tsz, fsz = y.shape
        y = y.reshape(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        # 计算有效的时间步长high
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (torch.arange(num).unsqueeze(-1).expand(
                    -1, self.n_negatives).flatten())
                # 抽取负样本索引，如果提供了speech_lengths，则根据每个样本的实际长度在有效范围内随机抽取负样本索引。
                if speech_lengths is not None:
                    neg_idxs = [
                        torch.randint(low=0,
                                      high=speech_lengths[i].item() - 1,
                                      size=(1, self.n_negatives * tsz))
                        for i in range(len(speech_lengths))
                    ]
                    neg_idxs = torch.cat(neg_idxs).reshape(
                        bsz, self.n_negatives * tsz)
                # 如果没有提供speech_lengths，则在总的样本数（不考虑填充）范围内随机抽取负样本索引。
                else:
                    neg_idxs = torch.randint(low=0,
                                             high=num - 1,
                                             size=(bsz,
                                                   self.n_negatives * tsz))
                neg_idxs[neg_idxs >= tszs] += 1

        # 调整索引以适应批次大小
        if self.n_negatives > 0:
            neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)

        # 获取负样本特征
        negs = y[neg_idxs.view(-1)]
        negs = negs.contiguous().view(bsz, num, self.n_negatives,
                                      fsz).permute(2, 0, 1, 3)  # to NxBxTxC
        return negs, neg_idxs

    #  函数的主要目的是计算输入向量 x 与目标向量 y 和负样本向量 negatives 之间的余弦相似度，并根据某些条件对结果进行调整
    def compute_preds(self, x, y, negatives):
        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)
        logits = logits / self.logit_temp
        logits = logits.type_as(x)

        if neg_is_pos.any():
            if not hasattr(self, "_inftensor"):
                self._inftensor = float("-inf")
            # logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)
            logits[1:][neg_is_pos] = self._inftensor
        logits = logits.transpose(0, 2)
        logits = logits.transpose(0, 1)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    # 实现了一个质心三元组损失，用于计算损失值。这个损失函数主要用于图像特征的聚合和优化，以避免误导性的训练信号，并保持模型输入大小的灵活性。
    def CTL(self, x, y, negs, mask=None):
        # Step1: compute cosine similarity, shape [B*T, n_negatives+1]
        logits = self.compute_preds(x, y, negs)

        # Step2: target shape [B*T]
        target = x.new_zeros(x.size(0) * x.size(1), dtype=torch.long)

        # Step3: compute CTL loss
        if mask is not None:
            normalize_length = mask.sum()
            bz, sz = mask.size(0), mask.size(-1)
            mask = mask.squeeze(1).reshape(bz * sz).eq(0)
            ce = F.cross_entropy(logits, target, reduction='none')
            loss = ce.masked_fill(mask, 0).sum() / normalize_length
        else:
            loss = F.cross_entropy(logits, target)

        return loss

# 总结：控制和管理端到端语音识别模型的运行