# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
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

import math
from collections import defaultdict
from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence

from wenet.utils.common import (add_sos_eos, log_add, add_whisper_tokens,
                                mask_to_bias)
from wenet.utils.ctc_utils import remove_duplicates_and_blank
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)
from wenet.utils.context_graph import ContextGraph, ContextState


# 实现了语音识别中解码结果的处理，包括了多种解码算法（CTC 贪心搜索、CTC 前缀束搜索、注意力机制束搜索以及注意力重评分）。
# DecodeResult 类用于存储解码结果。它的构造函数接收多个参数，提供了完整的解码结果信息。
class DecodeResult:

    def __init__(self,
                 tokens: List[int],
                 score: float = 0.0,
                 confidence: float = 0.0,
                 tokens_confidence: List[float] = None,
                 times: List[int] = None,
                 nbest: List[List[int]] = None,
                 nbest_scores: List[float] = None,
                 nbest_times: List[List[int]] = None):
        """
        Args:
            tokens: decode token list
            score: the total decode score of this result
            confidence: the total confidence of this result, it's in 0~1
            tokens_confidence: confidence of each token
            times: timestamp of each token, list of (start, end)
            nbest: nbest result
            nbest_scores: score of each nbest
            nbest_times:
        """
        self.tokens = tokens
        self.score = score
        self.confidence = confidence
        self.tokens_confidence = tokens_confidence
        self.times = times
        self.nbest = nbest
        self.nbest_scores = nbest_scores
        self.nbest_times = nbest_times


# PrefixScore 类用于 CTC（连接时序分类）前缀束搜索。它主要用于存储当前前缀的得分和状态。
class PrefixScore:
    """ For CTC prefix beam search """

    def __init__(self,
                 s: float = float('-inf'),
                 ns: float = float('-inf'),
                 v_s: float = float('-inf'),
                 v_ns: float = float('-inf'),
                 context_state: ContextState = None,
                 context_score: float = 0.0):
        self.s = s  # blank_ending_score
        self.ns = ns  # none_blank_ending_score
        self.v_s = v_s  # viterbi blank ending score
        self.v_ns = v_ns  # viterbi none blank ending score
        self.cur_token_prob = float('-inf')  # prob of current token
        self.times_s = []  # times of viterbi blank path
        self.times_ns = []  # times of viterbi none blank path
        self.context_state = context_state
        self.context_score = context_score
        self.has_context = False

    # 返回当前得分。
    def score(self):
        return log_add(self.s, self.ns)

    # 返回 Viterbi 的得分。
    def viterbi_score(self):
        return self.v_s if self.v_s > self.v_ns else self.v_ns

    # 返回时间戳。
    def times(self):
        return self.times_s if self.v_s > self.v_ns else self.times_ns

    # 返回总得分。
    def total_score(self):
        return self.score() + self.context_score

    # 复制上下文。
    def copy_context(self, prefix_score):
        self.context_score = prefix_score.context_score
        self.context_state = prefix_score.context_state

    # 更新上下文。
    def update_context(self, context_graph, prefix_score, word_id):
        self.copy_context(prefix_score)
        (score, context_state) = context_graph.forward_one_step(
            prefix_score.context_state, word_id)
        self.context_score += score
        self.context_state = context_state


# 没有用decoder
# CTC 贪心搜索算法。找到每个样本的最高概率的 token，返回解码结果的列表。
# ctc_probs：形状为 (B, maxlen, vocab_size) 的张量，表示每个时间步的 CTC 概率，其中 B 是批次大小，maxlen 是序列的最大长度，vocab_size 是词汇表的大小。
# ctc_lens：形状为 (B,) 的张量，表示每个输入序列的真实长度。
# blank_id：空白符的 ID，通常在 CTC 中用于表示未产生的输出，默认为 0。
def ctc_greedy_search(ctc_probs: torch.Tensor,
                      ctc_lens: torch.Tensor,
                      blank_id: int = 0) -> List[DecodeResult]:
    # 获取批次大小 batch_size 和最大序列长度 maxlen。
    print("进入ctc_greedy_search方法")
    batch_size = ctc_probs.shape[0]
    maxlen = ctc_probs.size(1)
    # 使用 ctc_probs.topk(1, dim=2) 获取每个时间步中概率最高的索引，返回值为 topk_prob 和 topk_index。
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    # topk_index 形状为 (B, maxlen, 1)，通过 view 方法将其转换为 (B, maxlen) 的形状。
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    # 调用 make_pad_mask(ctc_lens, maxlen) 生成一个掩码，用于标记填充的时间步。
    mask = make_pad_mask(ctc_lens, maxlen)  # (B, maxlen)
    # 利用 masked_fill_ 方法对 topk_index 张量进行处理。mask 是一个布尔张量，其形状与 topk_index 相同，
    # 指示哪些位置需要被替换。blank_id 是填充的值，通常用于指示无效的或被屏蔽的索引。
    topk_index = topk_index.masked_fill_(mask, blank_id)  # (B, maxlen)
    # 这行代码将 topk_index 中的每个元素（通常是解码的假设）转换为 Python 列表。
    hyps = [hyp.tolist() for hyp in topk_index]
    # 这行代码用于计算 topk_prob 张量中每一行的最大概率值。
    scores = topk_prob.max(1)
    # 初始化返回的result列表
    results = []
    # 对于每个假设 hyp，调用 remove_duplicates_and_blank(hyp, blank_id) 函数，该函数可能用于移除重复的元素和填充的空值（blank_id），
    # 以确保每个解码结果是唯一的并且有效。
    for hyp in hyps:
        r = DecodeResult(remove_duplicates_and_blank(hyp, blank_id))
        results.append(r)
    return results
# 小结：CTC给每个时间步给出概率然后贪心搜索概率最高的结果，给每帧音频打上概率，获取每个时间步上最大的概率标签


# 没有用decoder
# 实现了 CTC 前缀束搜索。逐步进行 CTC 束搜索，返回每个样本的 n-best 解码结果。
# 输入的几个字符：
#   ctc_probs: CTC 模型输出的概率分布，形状为 (batch_size, max_time, vocab_size)。
#   ctc_lens: 每个输入序列的有效长度，形状为 (batch_size,)。
#   beam_size: 束搜索的大小，即每个时间步保留的最佳候选数。
#   context_graph: 可选的上下文图，可能用于增强解码的上下文信息。
#   blank_id: 用于指示 CTC 解码中的空白符的 ID，默认为 0。
# DecodeResult 包含以下几个字段：
# tokens：最佳解码路径，表示解码得到的最终结果，通常是一个词或音素的序列。
# score：解码路径的总得分，用于评估解码路径的好坏。
# times：每个解码时间点，表示在输入的语音序列中每个解码出的 token 对应的时间索引。
# nbest：一个包含 n 个候选解码结果的列表（即 N-best 结果）。
# nbest_scores：每个候选解码结果的得分。
# nbest_times：每个候选解码结果的时间点列表。
def ctc_prefix_beam_search(
    ctc_probs: torch.Tensor,
    ctc_lens: torch.Tensor,
    beam_size: int,
    context_graph: ContextGraph = None,
    blank_id: int = 0,
) -> List[DecodeResult]:
    """
        Returns:
            List[List[List[int]]]: nbest result for each utterance
    """
    print("进入ctc_prefix_beam_search方法")
    # 获取输入张量 ctc_probs 的批量大小
    batch_size = ctc_probs.shape[0]
    # 初始化输出结果列表
    results = []
    # CTC prefix beam search can not be paralleled, so search one by one
    # 对每个样本进行单独处理，获取当前样本的概率分布和长度，并初始化当前假设 cur_hyps 为包含一个空前缀和初始得分的列表。
    for i in range(batch_size):
        # 获取当前步数CTC 模型输出的概率分布
        ctc_prob = ctc_probs[i]
        print(ctc_prob)
        # 获取当前输入序列的有效长度，形状为 (batch_size,)。
        num_t = ctc_lens[i]
        # 初始化当前假设 cur_hyps 为包含一个空前缀和初始得分的列表。
        cur_hyps = [(tuple(),   # 创建一个空元组，表示当前的前缀是空的。
                     # 存储当前前缀得分和状态
                     PrefixScore(s=0.0,
                                 ns=-float('inf'),
                                 v_s=0.0,
                                 v_ns=0.0,
                                 context_state=None if context_graph is None
                                 else context_graph.root,
                                 context_score=0.0))]
        # 2. CTC beam search step by step
        for t in range(0, num_t):
            # 遍历每个时间步 t，提取当前时间步的概率分布 logp。
            logp = ctc_prob[t]  # (vocab_size,)
            # key: prefix, value: PrefixScore
            # 使用 defaultdict 创建 next_hyps，用于存储下一步的假设及其得分。
            next_hyps = defaultdict(lambda: PrefixScore())
            # 2.1 First beam prune: select topk best
            # 获取当前时间步的前 beam_size 个最高概率的词汇及其索引。
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            # 这行代码遍历 top_k_index 中的每个高概率词汇 u，即在当前时间步上，可能性最大的 beam_size 个词汇。
            for u in top_k_index:
                # 更新假设得分
                # u 是一个单独的词汇索引。由于 top_k_index 中的元素为张量类型，通过调用 .item() 方法将其转换为 Python 的标量数据类型。
                u = u.item()
                # 使用 .item() 方法将 logp[u] 转换为 Python 标量，从而得到该词汇 u 的对数概率值 prob。
                prob = logp[u].item()
                # cur_hyps 包含每个前缀 prefix 及其对应的得分 prefix_score。
                # 每个 prefix 是一个表示已解码词汇序列的元组，prefix_score 存储与该前缀相关的各种得分信息。
                for prefix, prefix_score in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    # 当 u 为空白符时，不改变 prefix 结构，而是仅更新对应的得分。
                    if u == blank_id:  # blank
                        next_score = next_hyps[prefix]
                        # 使用 log_add 累加当前前缀得分 prefix_score.score() 和概率 prob，结果存入 next_score.s，用于计算后续的累积得分。
                        next_score.s = log_add(next_score.s,
                                               prefix_score.score() + prob)
                        next_score.v_s = prefix_score.viterbi_score() + prob
                        next_score.times_s = prefix_score.times().copy()
                        # perfix not changed, copy the context from prefix
                        # 若启用 context_graph 并且上下文尚未被设置，则拷贝 prefix_score 的上下文状态。
                        if context_graph and not next_score.has_context:
                            next_score.copy_context(prefix_score)
                            next_score.has_context = True
                    # 处理当前词汇为前缀的最后一个词（重复符号）：
                    elif u == last:
                        #  Update *uu -> *u;
                        # 更新 next_score1.ns 以包含非空白符累加得分；若此得分更高，替换 next_score1.v_ns。
                        next_score1 = next_hyps[prefix]
                        # 更新当前词汇出现时间 times_ns。
                        next_score1.ns = log_add(next_score1.ns,
                                                 prefix_score.ns + prob)
                        # prefix_score.v_ns + prob 是当前前缀在加上当前概率 prob 后的新非空白符得分。
                        # 若该值比 next_score1.v_ns 的已有得分更高，则更新 v_ns 为此值，以记录路径上最高的非空白符得分。
                        if next_score1.v_ns < prefix_score.v_ns + prob:
                            next_score1.v_ns = prefix_score.v_ns + prob
                            # 若当前概率 prob 高于 cur_token_prob，
                            if next_score1.cur_token_prob < prob:
                                # 则将 cur_token_prob 更新为 prob。
                                next_score1.cur_token_prob = prob
                                # 复制并更新时间戳 times_ns：保留 prefix_score 的时间记录，且更新当前词的时间戳 t，以跟踪每个符号的出现时间。
                                next_score1.times_ns = prefix_score.times_ns.copy(
                                )
                                next_score1.times_ns[-1] = t
                        # 若启用 context_graph 且 next_score1 尚无上下文，
                        if context_graph and not next_score1.has_context:
                            # 则复制 prefix_score 的上下文状态。
                            next_score1.copy_context(prefix_score)
                            # 标记 has_context 为 True，表明已完成上下文状态的拷贝。
                            next_score1.has_context = True

                        # 如果 u 不为空白符也不重复，扩展当前前缀为 n_prefix。
                        # Update *u-u -> *uu, - is for blank
                        n_prefix = prefix + (u, )
                        # 累加总得分，记录 times_ns 及上下文，保证最佳路径得分及其上下文状态的正确性。
                        next_score2 = next_hyps[n_prefix]
                        next_score2.ns = log_add(next_score2.ns,
                                                 prefix_score.s + prob)
                        # 若当前前缀（含空白符）得分加上当前符号 u 的概率高于 next_score2.v_ns 的已有得分，则将其更新，以保留当前符号路径中最高的得分。
                        if next_score2.v_ns < prefix_score.v_s + prob:
                            next_score2.v_ns = prefix_score.v_s + prob
                            # 将当前符号 u 的概率 prob 存储到 cur_token_prob，更新符号的最大出现概率。
                            next_score2.cur_token_prob = prob
                            # 复制 prefix_score 的时间戳记录 times_s，并将当前时间步 t 添加到 times_ns 中，用于记录符号的具体时间。
                            next_score2.times_ns = prefix_score.times_s.copy()
                            next_score2.times_ns.append(t)
                        # 若 context_graph 存在且 next_score2 尚无上下文状态
                        if context_graph and not next_score2.has_context:
                            # 则调用 update_context，将当前上下文状态更新到 next_score2。
                            next_score2.update_context(context_graph,
                                                       prefix_score, u)
                            # 更新 has_context 标记为 True，表明已完成上下文状态的更新。
                            next_score2.has_context = True
                    else:
                        # 将符号 u 加入当前 prefix，形成新的前缀 n_prefix。
                        n_prefix = prefix + (u, )
                        # 获取或创建该前缀对应的 next_score，用于存储新路径的得分信息。
                        next_score = next_hyps[n_prefix]
                        # 使用 log_add 函数累加 next_score.ns，结合当前前缀得分 prefix_score.score() 与符号 u 的概率 prob，以计算新的非空白符得分。
                        next_score.ns = log_add(next_score.ns,
                                                prefix_score.score() + prob)
                        # 若当前 Viterbi 得分 prefix_score.viterbi_score() + prob 超过已记录的 next_score.v_ns。
                        if next_score.v_ns < prefix_score.viterbi_score(
                        ) + prob:
                            # 则更新 v_ns 为新的最高值。
                            next_score.v_ns = prefix_score.viterbi_score(
                            ) + prob
                            # 将当前符号 u 的概率 prob 存储在 cur_token_prob 中，并记录该符号出现的时间步 t，方便之后生成的解码结果更精确地反映时间信息。
                            next_score.cur_token_prob = prob
                            next_score.times_ns = prefix_score.times().copy()
                            # 记录该符号出现的时间步 t
                            next_score.times_ns.append(t)
                        # 若 context_graph 存在并且 next_score 尚未更新上下文状态，
                        if context_graph and not next_score.has_context:
                            # 则将 prefix_score 的上下文与符号 u 一起更新到 next_score 中。
                            next_score.update_context(context_graph,
                                                      prefix_score, u)
                            # 设置 has_context 为 True，表示该路径已包含上下文信息。
                            next_score.has_context = True

            # 2.2 Second beam prune
            # 按总得分降序排序：sorted 函数将 next_hyps 按照 total_score（总得分）进行排序，total_score() 返回的是前缀的整体得分，用于比较不同前缀的质量。
            # reverse=True 表示按降序排序，即得分最高的前缀将排列在最前面。
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: x[1].total_score(),
                               reverse=True)
            # 保留前 beam_size 个最佳前缀：排序后的 next_hyps 中，得分最高的前 beam_size 个前缀被保留到 cur_hyps 中，用于下一步解码。
            # beam_size 限制了前缀数量，控制解码过程中保留的前缀数量，以在保证解码质量的同时提高效率。
            cur_hyps = next_hyps[:beam_size]

        # We should backoff the context score/state when the context is
        # not fully matched at the last time.
        # 检查 context_graph 是否存在：若 context_graph 不为 None，表示需要对前缀假设的上下文进行处理。
        if context_graph is not None:
            # 遍历当前假设列表 cur_hyps：
            for i, hyp in enumerate(cur_hyps):
                # 调用 context_graph.finalize 进行上下文校正：
                # context_graph.finalize 方法会对每个假设的 context_state（上下文状态）进行校正，
                # 并返回最终的上下文得分 context_score 和更新后的上下文状态 new_context_state。
                context_score, new_context_state = context_graph.finalize(
                    hyp[1].context_state)
                # 更新上下文得分和状态：
                # 将校正后的 context_score 和 context_state 更新到当前假设的 PrefixScore 中，
                # 以保存最终的上下文信息。
                cur_hyps[i][1].context_score = context_score
                cur_hyps[i][1].context_state = new_context_state

        # 提取 nbest 列表：
        # nbest 是一个包含当前假设列表 cur_hyps 中所有前缀的列表。
        nbest = [y[0] for y in cur_hyps]
        # nbest_scores 提取每个假设的总得分。
        nbest_scores = [y[1].total_score() for y in cur_hyps]
        # nbest_times 提取每个假设的时间步信息。
        nbest_times = [y[1].times() for y in cur_hyps]
        # 由于 cur_hyps 已按得分降序排列，nbest[0] 是得分最高的假设。
        # best、best_score 和 best_time 分别保存最佳假设的前缀、得分和时间信息。
        best = nbest[0]
        best_score = nbest_scores[0]
        best_time = nbest_times[0]
        results.append(
            DecodeResult(tokens=best,
                         score=best_score,
                         times=best_time,
                         nbest=nbest,
                         nbest_scores=nbest_scores,
                         nbest_times=nbest_times))
    return results
# 小结：这个是CTC给时间步赋值时间概率然后前缀束搜索高概率的时间步进行解码获得结果


# 实现了基于注意力机制的束搜索。这里用了decoder
def attention_beam_search(
    model,
    encoder_out: torch.Tensor,
    encoder_mask: torch.Tensor,
    beam_size: int = 10,
    length_penalty: float = 0.0,
    infos: Dict[str, List[str]] = None,
) -> List[DecodeResult]:
    print("进入attention_beam_search方法")
    # 获取设备（如 CPU 或 GPU）以及批量大小（batch_size）。
    device = encoder_out.device
    batch_size = encoder_out.shape[0]
    # Let's assume B = batch_size and N = beam_size
    # 1. Encoder
    # 确定编码器输出的最大长度（maxlen）和维度（encoder_dim）。
    maxlen = encoder_out.size(1)
    encoder_dim = encoder_out.size(2)
    # 计算当前的运行大小（running_size），即 batch_size * beam_size。
    running_size = batch_size * beam_size
    # 这部分代码检查 model 对象是否存在 special_tokens 属性（避免属性不存在导致报错），并且检查 special_tokens 中是否包含 "transcribe"。
    # 如果都满足条件，则进行进一步处理。
    if getattr(model, 'special_tokens', None) is not None \
            and "transcribe" in model.special_tokens:
        # 当模型的 special_tokens 属性包含 "transcribe" 时，提取 infos 中的 tasks 和 langs 信息，
        # 并根据 beam_size 对它们进行扩展，以便每个候选路径（由 Beam Search 生成）都可以关联到特定任务和语言。
        tasks, langs = infos["tasks"], infos["langs"]
        # 这里，tasks 和 langs 列表根据 beam_size 进行了扩展，确保每个候选路径都具有一份对应的 task 和 lang 信息。
        tasks = [t for t in tasks for _ in range(beam_size)]
        langs = [l for l in langs for _ in range(beam_size)]
        # hyps 初始化为一个形状为 [running_size, 0] 的张量，表示每个候选路径当前还没有任何生成的 token。
        hyps = torch.ones([running_size, 0], dtype=torch.long,
                          device=device)  # (B*N, 0)
        # 添加特殊 tokens：添加特殊 tokens 的作用是为序列生成任务提供上下文和初始化条件。
        # 这些特殊 tokens 通常包含特定任务（如语音转录、翻译）或语言标识，以确保模型在解码过程中能够按照设定的任务要求生成相应的输出。
        hyps, _ = add_whisper_tokens(model.special_tokens,
                                     hyps,
                                     model.ignore_id,
                                     tasks=tasks,
                                     no_timestamp=True,
                                     langs=langs,
                                     use_prev=False)
    # 否则，初始化 hyps 为全 sos (start of sequence) 的张量，形状为 (B*N, 1)。
    else:
        hyps = torch.ones([running_size, 1], dtype=torch.long,
                          device=device).fill_(model.sos)  # (B*N, 1)
    # prefix_len 代表候选序列 hyps 的当前长度。这里，hyps 刚刚初始化，长度为 1。
    prefix_len = hyps.size(1)
    # 初始化得分张量 scores，其中第一个元素为0，其余元素为负无穷。
    scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                          dtype=torch.float)
    # 将得分张量扩展为 batch_size 大小并调整维度：
    # scores.to(device)：将 scores 张量移动到与模型相同的设备（CPU 或 GPU）
    # repeat([batch_size])：将 scores 按 batch_size 扩展，确保每个 batch 的候选路径初始化得分结构一致。
    # unsqueeze(1)：在第 1 维度上增加一个维度，使 scores 形状变为 (B * N, 1)，其中 B 为 batch_size，N 为 beam_size。
    scores = scores.to(device).repeat([batch_size
                                       ]).unsqueeze(1).to(device)  # (B*N, 1)
    # 初始化结束标志 end_flag，用于指示序列是否已结束。
    # torch.zeros_like(scores, dtype=torch.bool, device=device)：创建一个与 scores 张量形状相同的布尔型张量 end_flag，并设置初始值为 False。
    # 在解码循环中，end_flag 会逐步更新，用于判断哪些候选路径可以停止扩展，以提高解码效率。
    end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
    # 初始化缓存：用于存储解码过程中的注意力缓存的字典
    # cache可以帮助模型高效利用先前计算的注意力权重，从而加速解码过程。
    cache = {
        # 存储自注意力（Self-Attention）层的缓存数据，用于在解码步骤中加速计算，减少重复的注意力计算。
        'self_att_cache': {},
        # 存储交叉注意力（Cross-Attention）层的缓存数据，通常用于解码器关注编码器输出的上下文信息。
        'cross_att_cache': {},
    }
    # 编码器掩码 encoder_mask 转换：该条件语句判断解码器是否启用了 use_sdpa（如使用稀疏或动态注意力机制），
    # 如果启用，则调用 mask_to_bias 函数，将 encoder_mask 转换为适合该注意力机制的形式。
    # mask_to_bias 函数的作用通常是将掩码转换为注意力偏置，以便在特定的注意力实现中（如稀疏注意力）能正确地应用掩码。
    if model.decoder.use_sdpa:
        encoder_mask = mask_to_bias(encoder_mask, encoder_out.dtype)
    # 最大解码长度 maxlen 设置
    # 该条件语句检查模型是否包含 decode_maxlen 属性，如果存在则将 maxlen 设置为 model.decode_maxlen。
    # 目的：在 Beam Search 解码过程中限制生成的最大长度，避免生成过长的序列。
    if hasattr(model, 'decode_maxlen'):
        maxlen = model.decode_maxlen
    # 2. Decoder forward step by step   逐步解码
    # 从已存在的前缀长度 prefix_len 到最大长度 maxlen + 1 进行循环，逐步生成序列的每个时间步。
    for i in range(prefix_len, maxlen + 1):
        # 如果所有候选路径都生成了结束标志 eos，则退出循环。
        # Stop if all batch and all beam produce eos
        if end_flag.sum() == running_size:
            break
        # 2.1 Forward decoder step
        # 创建一个遮掩矩阵，确保当前时间步 i 之后的预测不可见，避免信息泄露。
        hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
            running_size, 1, 1).to(device)  # (B*N, i, i)
        # 如果模型使用 use_sdpa 稀疏注意力机制，则调用 mask_to_bias 进行掩码转换，适配稀疏注意力的实现。
        if model.decoder.use_sdpa:
            hyps_mask = mask_to_bias(hyps_mask, encoder_out.dtype)
        # logp: (B*N, vocab)解码器前向步骤：生成当前时间步的预测概率分布 logp。
        # 在当前时间步生成每个候选路径的预测概率 logp，形状为 (B*N, vocab)。
        logp = model.decoder.forward_one_step(encoder_out, encoder_mask, hyps,
                                              hyps_mask, cache)
        # 2.2 First beam prune: select topk best prob at current time
        # 第一轮束裁剪：选择当前时间步上概率最高的 top_k_logp 和对应的 top_k_index。
        top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
        top_k_logp = mask_finished_scores(top_k_logp, end_flag)
        top_k_index = mask_finished_preds(top_k_index, end_flag, model.eos)
        # 2.3 Second beam prune: select topk score with history第二轮束裁剪：基于历史得分对当前得分进行加权，保留得分最高的 beam_size 个候选项。
        # 更新当前时间步的 scores，选择 beam_size 个最高分的候选路径。
        scores = scores + top_k_logp  # (B*N, N), broadcast add
        scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
        scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
        # Update cache to be consistent with new topk scores / hyps更新缓存：更新自注意力缓存，以保持与新选出的得分和候选项一致。
        cache_index = (offset_k_index // beam_size).view(-1)  # (B*N)
        base_cache_index = (torch.arange(batch_size, device=device).view(
            -1, 1).repeat([1, beam_size]) * beam_size).view(-1)  # (B*N)
        cache_index = base_cache_index + cache_index
        cache['self_att_cache'] = {
            i_layer: (torch.index_select(value[0], dim=0, index=cache_index),
                      torch.index_select(value[1], dim=0, index=cache_index))
            for (i_layer, value) in cache['self_att_cache'].items()
        }
        # NOTE(Mddct): we don't need select cross att here不需要交叉注意力解码
        # 这行代码用于清空 PyTorch 在 GPU 上的缓存。PyTorch 会在使用 GPU 进行计算时分配内存，有时会保留一些未使用的内存以提高后续计算的效率。
        torch.cuda.empty_cache()
        # 这行代码通过 view 方法调整 scores 张量的形状，使其变为 (-1, 1)，即将得分张量转换为二维张量，其中第一维的大小由 PyTorch 自动计算（通常对应于 B*N 的大小），第二维固定为 1。
        # 这种操作通常用于准备数据，使得后续的计算（如广播）能够正确进行。
        scores = scores.view(-1, 1)  # (B*N, 1)
        # 2.4. Compute base index in top_k_index,计算 top_k_index 中的基础索引，
        # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),将 top_k_index 视为 (B*N*N)，将 offset_k_index 视为 (B*N)，
        # then find offset_k_index in top_k_index然后在 top_k_index 中找到 offset_k_index
        # 更新候选序列：根据选出的候选路径 best_k_index 更新 hyps 中的最佳假设。
        # torch.arange(batch_size, device=device) 生成一个包含 batch_size 个元素的张量，表示每个批次的索引。
        base_k_index = torch.arange(batch_size, device=device).view(
            -1, 1).repeat([1, beam_size])  # (B, N)
        # base_k_index 乘以 beam_size * beam_size，用于扩展索引，使其能够正确地映射到束中的每个候选路径,base_k_index 的形状仍为(B, N)
        base_k_index = base_k_index * beam_size * beam_size
        # 将 base_k_index 变形为一维张量，并将其与 offset_k_index（表示在每个束中选择的路径索引）相加，得到 best_k_index。
        best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)  # (B*N)

        # 2.5 Update best hyps更新最佳假设
        # torch.index_select 用于从 top_k_index 中选择最佳的预测词索引。
        # top_k_index 先被展平为一维张量，然后根据 best_k_index 选出最佳的预测结果，形成一个形状为 (B*N) 的张量，表示当前时间步最佳的预测词。
        best_k_pred = torch.index_select(top_k_index.view(-1),
                                         dim=-1,
                                         index=best_k_index)  # (B*N)
        # 获取上一步的最佳假设
        # 计算 best_hyps_index，即每个最佳路径对应的上一步的候选路径索引（通过整除 beam_size 得到）。
        best_hyps_index = best_k_index // beam_size
        # 使用 torch.index_select 从 hyps 中选择对应的上一步最佳假设，形成形状为 (B*N, i) 的张量，表示当前时间步每个束的历史输出。
        last_best_k_hyps = torch.index_select(
            hyps, dim=0, index=best_hyps_index)  # (B*N, i)
        # 合并最佳假设与当前预测：使用 torch.cat 将上一时刻的最佳假设 last_best_k_hyps 与当前时间步的最佳预测 best_k_pred 合并。
        # best_k_pred 被调整为形状 (B*N, 1)，形成一个新的假设序列 hyps，其形状为 (B*N, i+1)，表示在当前时间步每个候选路径的完整输出。
        hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                         dim=1)  # (B*N, i+1)

        # 2.6 Update end flag；上传结束标志
        end_flag = torch.eq(hyps[:, -1], model.eos).view(-1, 1)

    # 3. Select best of best选择最佳中的最佳
    scores = scores.view(batch_size, beam_size)
    # 计算得分时应用长度惩罚，以调整生成序列的长度。
    lengths = hyps.ne(model.eos).sum(dim=1).view(batch_size, beam_size).float()
    scores = scores / lengths.pow(length_penalty)
    # 找到最佳得分和对应的索引，并选择最终的候选序列。
    # 通过 scores.max(dim=-1) 找到每个批次中得分最高的束，并返回最佳得分 best_scores 及其对应的索引 best_index。
    best_scores, best_index = scores.max(dim=-1)
    # 计算 best_hyps_index，它表示在所有束中找到的最佳假设在 hyps 中的索引。
    # 通过 torch.arange(batch_size) 创建一个从 0 到 batch_size-1 的索引，并乘以 beam_size 进行偏移，确保能正确访问每个批次的束。
    best_hyps_index = best_index + torch.arange(
        batch_size, dtype=torch.long, device=device) * beam_size
    # 使用 torch.index_select 根据 best_hyps_index 从 hyps 中选择出最佳的生成序列，形成形状为 (B, i) 的张量。
    best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
    # 移除 prefix_len 之前的部分，保留从 prefix_len 开始的生成结果，这样可以得到最终的生成序列。
    best_hyps = best_hyps[:, prefix_len:]

    # 将最终结果整理为 DecodeResult 对象并返回。
    results = []
    # 遍历每个批次，获取最佳的假设 hyp，并将其中的结束标志（eos）移除。
    for i in range(batch_size):
        hyp = best_hyps[i]
        hyp = hyp[hyp != model.eos]
        # 将处理后的 hyp 转换为列表形式，创建 DecodeResult 对象并添加到 results 中。最后返回 results 列表。
        results.append(DecodeResult(hyp.tolist()))
    return results
# 总结：先注意力给出每个时间步的权重，根据权重进行束搜索优中选优获得结果。


# 这里也用了decoder
# 对 CTC 前缀束搜索的结果进行重评分。通过模型的注意力解码器重新计算每个 n-best 解码结果的得分，并返回最优的解码结果。
# model：模型对象，用于解码的模型。
# ctc_prefix_results：CTC 前缀束搜索的结果，类型为 List[DecodeResult]是个列表，包含多个解码结果。
# encoder_outs：编码器的输出，形状为 (batch_size, max_length, feature_dim)。
# encoder_lens：编码器输出的长度，形状为 (batch_size,)。
# ctc_weight：CTC 分数的权重，控制 CTC 分数在最终评分中的影响程度。
# reverse_weight：反向评分的权重，控制反向解码器分数在最终评分中的影响程度。
# infos：附加信息，包括任务和语言等。
def attention_rescoring(
    model,
    ctc_prefix_results: List[DecodeResult],
    encoder_outs: torch.Tensor,
    encoder_lens: torch.Tensor,
    ctc_weight: float = 0.0,
    reverse_weight: float = 0.0,
    infos: Dict[str, List[str]] = None,
) -> List[DecodeResult]:
    """
        Args:
            ctc_prefix_results(List[DecodeResult]): ctc prefix beam search results
    """
    # sos 和 eos 表示句子起始和结束标记。
    print("进入注意力重打分解码")
    sos, eos = model.sos_symbol(), model.eos_symbol()
    # device 是编码器输出的设备，用于确保张量在同一设备上。
    device = encoder_outs.device
    # 通过断言确保 encoder_outs 和 ctc_prefix_results 的批大小相同。
    assert encoder_outs.shape[0] == len(ctc_prefix_results)
    # 接收批次大小，便于推理加速
    batch_size = encoder_outs.shape[0]
    # 初始化 results 列表，用于保存每个样本的解码结果。
    results = []
    # 处理每个样本
    for b in range(batch_size):
        # encoder_out 选取当前样本的编码器输出。
        encoder_out = encoder_outs[b, :encoder_lens[b], :].unsqueeze(0)
        # 当前样本的假设序列
        hyps = ctc_prefix_results[b].nbest
        # 对应的 CTC 分数。
        ctc_scores = ctc_prefix_results[b].nbest_scores
        # 使用 pad_sequence 将假设序列填充到相同长度。
        hyps_pad = pad_sequence([
            torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps
        ], True, model.ignore_id)  # (beam_size, max_hyps_len)
        # hyps_lens 是每个假设的实际长度。
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        # 先判断模型是否需要特殊标记处理。
        if getattr(model, 'special_tokens', None) is not None \
                and "transcribe" in model.special_tokens:
            # 这里获取当前假设序列（hyps_pad）的长度（即列数），并将其存储在 prev_len 变量中。
            # 这个长度在后续处理中将用于计算添加标记后的变化。
            print("有special_tokens")
            prev_len = hyps_pad.size(1)
            # 为假设序列 hyps_pad 添加特殊标记（例如，转录任务相关的标记）。
            hyps_pad, _ = add_whisper_tokens(
                # 提供要添加的特殊标记。
                model.special_tokens,
                # 假设序列
                hyps_pad,
                # 是忽略非有效标记的位置。
                model.ignore_id,
                # 参数将当前批次的任务信息传递给函数。
                tasks=[infos["tasks"][b]] * len(hyps),
                # 表示不需要时间戳。
                no_timestamp=True,
                # 提供了语言信息。
                langs=[infos["langs"][b]] * len(hyps),
                # 表示不使用之前的标记（即在添加过程中不保留旧的标记）。
                use_prev=False)
            # 计算添加特殊标记后，更新的假设序列长度并存储在 cur_len 中。
            cur_len = hyps_pad.size(1)
            # 更新 hyps_lens，计算添加标记后的新长度。这里将新长度与之前的长度之差相加，以确保 hyps_lens 包含新的序列长度。
            hyps_lens = hyps_lens + cur_len - prev_len
            # 设置 prefix_len 为 4，表示在处理 decoder_out 时，假设序列的前缀长度是 4。这个长度通常用于解码时跳过添加的特殊标记所占用的位置。
            prefix_len = 4
        # 如果不需要，则添加 sos 和 eos。
        else:
            # 将每个假设序列（hyps_pad）前后分别添加 sos（起始符号）和 eos（结束符号）。
            # model.ignore_id 是填充值，用于忽略非有效标记的位置。
            # 返回的 hyps_pad 现在包含了 sos 和 eos。
            print("没有special_tokens")
            hyps_pad, _ = add_sos_eos(hyps_pad, sos, eos, model.ignore_id)
            # 因为在序列开头添加了一个 <sos> 标记，所以所有序列的长度 hyps_lens 都增加 1。
            hyps_lens = hyps_lens + 1  # Add <sos> at begining
            # 设置 prefix_len 为 1，表示在处理 decoder_out 时，解码的第一个标记是 <sos>，所以跳过第一个位置的得分计算。
            prefix_len = 1
        # 调用 model.forward_attention_decoder 进行前向和反向解码，得到解码输出。
        decoder_out, r_decoder_out = model.forward_attention_decoder(
            hyps_pad, hyps_lens, encoder_out, reverse_weight)
        # Only use decoder score for rescoring
        # 初始化 best_score 变量为负无穷大 (-inf)。这个变量用来存储当前找到的最佳得分（即分数最高的假设）。
        # 使用负无穷大是因为在比较时，任何有效的得分都会大于这个初始值，从而确保在第一次计算得分时能够更新 best_score。
        best_score = -float('inf')
        # 记录具有最高得分的假设的索引。这个索引可以在后续的循环中更新，以便找到当前最优的假设。
        best_index = 0
        # 存储每个假设的置信度（confidence）。置信度通常表示模型对每个假设的确定程度，数值越高表示模型越“相信”这个假设是正确的。
        confidences = []
        # 初始化一个空列表 tokens_confidences，用于存储每个假设中每个标记的置信度（token confidence）。
        # 这个列表将为每个假设中的每个标记分别保存对应的置信度值，有助于分析模型在生成过程中对各个标记的信心程度。
        tokens_confidences = []
        # 遍历 hyps 中每个假设，计算总得分 score。
        for i, hyp in enumerate(hyps):
            # 用于累加当前假设的总得分。
            score = 0.0
            # tc 是各个 token 的置信度列表，通过解码概率的指数表示。
            tc = []  # tokens confidences
            # 这个循环遍历当前假设中的每个标记 w，j 是标记的索引。
            for j, w in enumerate(hyp):
                # 从解码器的输出 decoder_out 中提取当前标记的得分 s。
                s = decoder_out[i][j + (prefix_len - 1)][w]
                # 将当前标记的得分 s 累加到总得分 score 中。
                score += s
                # 通过取 s 的指数值，计算出当前标记的置信度并添加到列表 tc 中。
                # 这个指数运算使得得分转化为一种概率形式，表示模型对这个标记的信心程度。
                tc.append(math.exp(s))
            # 将当前假设的结束标记（<eos>）的得分也加入到总得分中。len(hyp) 代表假设中的标记数量，因此这里通过相应的索引提取结束标记的得分。
            score += decoder_out[i][len(hyp) + (prefix_len - 1)][eos]
            # add right to left decoder score
            # 计算反向得分并融合：通过 reverse_weight 控制正反向得分的融合。
            if reverse_weight > 0 and r_decoder_out.dim() > 0:
                # 用于累加反向得分
                r_score = 0.0
                # 这个循环遍历当前假设中的每个标记，以计算反向得分。
                for j, w in enumerate(hyp):
                    # 从反向解码输出 r_decoder_out 中提取当前标记的反向得分 s。
                    s = r_decoder_out[i][len(hyp) - j - 1 +
                                         (prefix_len - 1)][w]
                    # 累加
                    r_score += s
                    # 更新当前标记的置信度 tc[j]，将正向和反向的置信度平均化，以综合考虑两种得分。
                    tc[j] = (tc[j] + math.exp(s)) / 2
                # 将反向解码中结束标记的得分也加入到 r_score 中。
                r_score += r_decoder_out[i][len(hyp) + (prefix_len - 1)][eos]
                # 根据 reverse_weight 加权平均正向得分和反向得分，计算最终得分 score。这使得模型可以根据设置的权重灵活地调节两种得分的影响。
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # 添加 CTC 分数和选择最佳得分：结合 CTC 分数，更新 best_score 和 best_index。
            confidences.append(math.exp(score / (len(hyp) + 1)))
            # add ctc score
            # 将 CTC 分数（ctc_scores[i]）与对应的权重（ctc_weight）相乘，加入到总得分中。
            # CTC（Connectionist Temporal Classification）是用于处理序列数据的一种损失函数，在这里被用来增强假设的评分。
            score += ctc_scores[i] * ctc_weight
            # 如果当前得分 score 超过当前最佳得分 best_score，则更新最佳得分和最佳假设的索引 best_index。
            if score > best_score:
                best_score = score.item()
                best_index = i
            # 将当前假设中各个标记的置信度列表 tc 添加到 tokens_confidences 中，以便后续分析和使用。
            tokens_confidences.append(tc)
        # 构建最终结果并返回
        results.append(
            DecodeResult(hyps[best_index],
                         best_score,
                         confidence=confidences[best_index],
                         times=ctc_prefix_results[b].nbest_times[best_index],
                         tokens_confidence=tokens_confidences[best_index]))
    return results

# 小结：先正向反向计算注意力权重，最后三者一起加权得到最后分数进行解码
# 总结：该代码实现了多种语音识别的解码策略，充分利用了 CTC 和注意力机制的优点，实现了各种搜索。