// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "decoder/ctc_prefix_beam_search.h"

#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "utils/log.h"
#include "utils/utils.h"

namespace wenet {

// 通过维护多个前缀（prefix）来进行CTC解码
CtcPrefixBeamSearch::CtcPrefixBeamSearch(
    const CtcPrefixBeamSearchOptions& opts,
    const std::shared_ptr<ContextGraph>& context_graph)
    : opts_(opts), context_graph_(context_graph) {
  Reset();
}

// 重置解码器
void CtcPrefixBeamSearch::Reset() {
  hypotheses_.clear();
  likelihood_.clear();
  cur_hyps_.clear();
  viterbi_likelihood_.clear();
  times_.clear();
  outputs_.clear();

  abs_time_step_ = 0;
  PrefixScore prefix_score;
  prefix_score.s = 0.0;
  prefix_score.ns = -kFloatMax;
  prefix_score.v_s = 0.0;
  prefix_score.v_ns = 0.0;

  std::vector<int> empty;
  cur_hyps_[empty] = prefix_score;
  outputs_.emplace_back(empty);
  hypotheses_.emplace_back(empty);
  likelihood_.emplace_back(prefix_score.total_score());
  times_.emplace_back(empty);
}

// 前缀分数比较
static bool PrefixScoreCompare(
    const std::pair<std::vector<int>, PrefixScore>& a,
    const std::pair<std::vector<int>, PrefixScore>& b) {
  return a.second.total_score() > b.second.total_score();
}

// 更新当前假设（hypotheses）集合，并基于输入的一组假设（hpys）来更新相关的内部状态
void CtcPrefixBeamSearch::UpdateHypotheses(
    const std::vector<std::pair<std::vector<int>, PrefixScore>>& hpys) {
  cur_hyps_.clear();
  outputs_.clear();
  hypotheses_.clear();
  likelihood_.clear();
  viterbi_likelihood_.clear();
  times_.clear();
  for (auto& item : hpys) {
    cur_hyps_[item.first] = item.second;
    hypotheses_.emplace_back(item.first);
    outputs_.emplace_back(std::move(item.first));
    likelihood_.emplace_back(item.second.total_score());
    viterbi_likelihood_.emplace_back(item.second.viterbi_score());
    times_.emplace_back(item.second.times());
  }
}

// Please refer https://robin1001.github.io/2020/12/11/ctc-search
// for how CTC prefix beam search works, and there is a simple graph demo in
// it.
// 查找。用于处理序列到序列的映射问题，其中输入序列（通常是连续的帧或时间步长的特征向量）和输出序列（通常是离散的标签序列）的长度可能不一致
// 具体来说，Prefix Beam
// Search在每个时间步对K个前缀进行处理，通过TopK选择前k个候选前缀，并根据当前时间步的log概率更新前缀的得分。
// 对于每个前缀，如果当前字符是空白符（blank），则更新空白结束得分；如果当前字符与前缀的最后一个字符相同，则更新非空白结束得分。
void CtcPrefixBeamSearch::Search(const std::vector<std::vector<float>>& logp) {
  if (logp.size() == 0) return;
  int first_beam_size =
      std::min(static_cast<int>(logp[0].size()), opts_.first_beam_size);
  for (int t = 0; t < logp.size(); ++t, ++abs_time_step_) {
    const std::vector<float>& logp_t = logp[t];
    std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> next_hyps;
    // 1. First beam prune, only select topk candidates
    std::vector<float> topk_score;
    std::vector<int32_t> topk_index;
    TopK(logp_t, first_beam_size, &topk_score, &topk_index);

    // 2. Token passing
    for (int i = 0; i < topk_index.size(); ++i) {
      int id = topk_index[i];
      auto prob = topk_score[i];
      for (const auto& it : cur_hyps_) {
        const std::vector<int>& prefix = it.first;
        const PrefixScore& prefix_score = it.second;
        // If prefix doesn't exist in next_hyps, next_hyps[prefix] will insert
        // PrefixScore(-inf, -inf) by default, since the default constructor
        // of PrefixScore will set fields s(blank ending score) and
        // ns(none blank ending score) to -inf, respectively.
        if (id == opts_.blank) {
          // Case 0: *a + ε => *a
          PrefixScore& next_score = next_hyps[prefix];
          next_score.s = LogAdd(next_score.s, prefix_score.score() + prob);
          next_score.v_s = prefix_score.viterbi_score() + prob;
          next_score.times_s = prefix_score.times();
          // Prefix not changed, copy the context from prefix.
          if (context_graph_ && !next_score.has_context) {
            next_score.CopyContext(prefix_score);
            next_score.has_context = true;
          }
        } else if (!prefix.empty() && id == prefix.back()) {
          // Case 1: *a + a => *a
          PrefixScore& next_score1 = next_hyps[prefix];
          next_score1.ns = LogAdd(next_score1.ns, prefix_score.ns + prob);
          if (next_score1.v_ns < prefix_score.v_ns + prob) {
            next_score1.v_ns = prefix_score.v_ns + prob;
            if (next_score1.cur_token_prob < prob) {
              next_score1.cur_token_prob = prob;
              next_score1.times_ns = prefix_score.times_ns;
              CHECK_GT(next_score1.times_ns.size(), 0);
              next_score1.times_ns.back() = abs_time_step_;
            }
          }
          if (context_graph_ && !next_score1.has_context) {
            next_score1.CopyContext(prefix_score);
            next_score1.has_context = true;
          }

          // Case 2: *aε + a => *aa
          std::vector<int> new_prefix(prefix);
          new_prefix.emplace_back(id);
          PrefixScore& next_score2 = next_hyps[new_prefix];
          next_score2.ns = LogAdd(next_score2.ns, prefix_score.s + prob);
          if (next_score2.v_ns < prefix_score.v_s + prob) {
            next_score2.v_ns = prefix_score.v_s + prob;
            next_score2.cur_token_prob = prob;
            next_score2.times_ns = prefix_score.times_s;
            next_score2.times_ns.emplace_back(abs_time_step_);
          }
          if (context_graph_ && !next_score2.has_context) {
            // Prefix changed, calculate the context score.
            next_score2.UpdateContext(context_graph_, prefix_score, id);
            next_score2.has_context = true;
          }
        } else {
          // Case 3: *a + b => *ab, *aε + b => *ab
          std::vector<int> new_prefix(prefix);
          new_prefix.emplace_back(id);
          PrefixScore& next_score = next_hyps[new_prefix];
          next_score.ns = LogAdd(next_score.ns, prefix_score.score() + prob);
          if (next_score.v_ns < prefix_score.viterbi_score() + prob) {
            next_score.v_ns = prefix_score.viterbi_score() + prob;
            next_score.cur_token_prob = prob;
            next_score.times_ns = prefix_score.times();
            next_score.times_ns.emplace_back(abs_time_step_);
          }
          if (context_graph_ && !next_score.has_context) {
            // Calculate the context score.
            next_score.UpdateContext(context_graph_, prefix_score, id);
            next_score.has_context = true;
          }
        }
      }
    }

    // 3. Second beam prune, only keep top n best paths
    std::vector<std::pair<std::vector<int>, PrefixScore>> arr(next_hyps.begin(),
                                                              next_hyps.end());
    int second_beam_size =
        std::min(static_cast<int>(arr.size()), opts_.second_beam_size);
    std::nth_element(arr.begin(), arr.begin() + second_beam_size, arr.end(),
                     PrefixScoreCompare);
    arr.resize(second_beam_size);
    std::sort(arr.begin(), arr.end(), PrefixScoreCompare);

    // 4. Update cur_hyps_ and get new result
    UpdateHypotheses(arr);
  }
}

// 用于完成前缀束搜索（Prefix Beam
// Search）的一个关键步骤，在连接主义时间分类（CTC）解码过程中使用
void CtcPrefixBeamSearch::FinalizeSearch() {
  // 检查上下文图是否为空
  if (context_graph_ == nullptr) return;
  // 一致性检查，确保容器大小相同
  CHECK_EQ(hypotheses_.size(), cur_hyps_.size());
  CHECK_EQ(hypotheses_.size(), likelihood_.size());
  // We should backoff the context score/state when the context is
  // not fully matched at the last time.
  // 回退上下文分数/状态
  for (const auto& prefix : hypotheses_) {
    PrefixScore& prefix_score = cur_hyps_[prefix];
    if (prefix_score.context_state != 0) {
      prefix_score.UpdateContext(context_graph_, prefix_score, -1);
    }
  }
  std::vector<std::pair<std::vector<int>, PrefixScore>> arr(cur_hyps_.begin(),
                                                            cur_hyps_.end());
  // 排序
  std::sort(arr.begin(), arr.end(), PrefixScoreCompare);

  // Update cur_hyps_ and get new result,更新假设获取结果
  UpdateHypotheses(arr);
}

}  // namespace wenet
// 总结：基于CTC和Prefix Beam Search算法的解码器
// Prefix Beam
// Search算法的核心思想是将共有的前缀部分从各个候选序列中分离出来，只保留差异化的部分进行扩展和评分。