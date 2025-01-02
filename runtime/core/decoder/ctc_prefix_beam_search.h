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

#ifndef DECODER_CTC_PREFIX_BEAM_SEARCH_H_
#define DECODER_CTC_PREFIX_BEAM_SEARCH_H_

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "decoder/context_graph.h"
#include "decoder/search_interface.h"
#include "utils/utils.h"

namespace wenet {

struct CtcPrefixBeamSearchOptions {
  int blank = 0;  // blank id
  int first_beam_size = 10;
  int second_beam_size = 10;
};

struct PrefixScore {
  float s = -kFloatMax;               // blank ending score
  float ns = -kFloatMax;              // none blank ending score
  float v_s = -kFloatMax;             // viterbi blank ending score
  float v_ns = -kFloatMax;            // viterbi none blank ending score
  float cur_token_prob = -kFloatMax;  // prob of current token
  std::vector<int> times_s;           // times of viterbi blank path
  std::vector<int> times_ns;          // times of viterbi none blank path

  // 获取得分
  float score() const { return LogAdd(s, ns); }
  // Viterbi算法通过计算每个时间步（或位置）上各个可能状态的得分（通常称为发射得分或一元得分），以及状态之间的转移得分（二元得分），来寻找全局最优的状态序列。
  // 大的就是好的
  float viterbi_score() const { return v_s > v_ns ? v_s : v_ns; }
  // 看谁得分高就用谁的时间
  const std::vector<int>& times() const {
    return v_s > v_ns ? times_s : times_ns;
  }

  bool has_context = false;
  int context_state = 0;
  float context_score = 0;

  // 复制上下文状态和分数
  void CopyContext(const PrefixScore& prefix_score) {
    context_state = prefix_score.context_state;
    context_score = prefix_score.context_score;
  }

  // 更新上下文状态
  void UpdateContext(const std::shared_ptr<ContextGraph>& context_graph,
                     const PrefixScore& prefix_score, int word_id) {
    this->CopyContext(prefix_score);

    float score = 0;
    context_state = context_graph->GetNextState(prefix_score.context_state,
                                                word_id, &score);
    context_score += score;
  }

  float total_score() const { return score() + context_score; }
};

struct PrefixHash {
  size_t operator()(const std::vector<int>& prefix) const {
    size_t hash_code = 0;
    // here we use KB&DR hash code
    for (int id : prefix) {
      hash_code = id + 31 * hash_code;
    }
    return hash_code;
  }
};

class CtcPrefixBeamSearch : public SearchInterface {
 public:
  explicit CtcPrefixBeamSearch(
      const CtcPrefixBeamSearchOptions& opts,
      const std::shared_ptr<ContextGraph>& context_graph = nullptr);

  void Search(const std::vector<std::vector<float>>& logp) override;
  void Reset() override;
  void FinalizeSearch() override;
  SearchType Type() const override { return SearchType::kPrefixBeamSearch; }
  void UpdateHypotheses(
      const std::vector<std::pair<std::vector<int>, PrefixScore>>& hpys);

  // 返回Viterbi似然值
  const std::vector<float>& viterbi_likelihood() const {
    return viterbi_likelihood_;
  }
  // 返回假设输入
  const std::vector<std::vector<int>>& Inputs() const override {
    return hypotheses_;
  }
  // 返回输出
  const std::vector<std::vector<int>>& Outputs() const override {
    return outputs_;
  }
  // 返回预估
  const std::vector<float>& Likelihood() const override { return likelihood_; }
  // 返回时间
  const std::vector<std::vector<int>>& Times() const override { return times_; }

 private:
  int abs_time_step_ = 0;

  // N-best list and corresponding likelihood_, in sorted order
  std::vector<std::vector<int>> hypotheses_;
  std::vector<float> likelihood_;
  std::vector<float> viterbi_likelihood_;
  std::vector<std::vector<int>> times_;

  std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> cur_hyps_;
  std::shared_ptr<ContextGraph> context_graph_ = nullptr;
  // Outputs contain the hypotheses_ and tags like: <context> and </context>
  std::vector<std::vector<int>> outputs_;
  const CtcPrefixBeamSearchOptions& opts_;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(CtcPrefixBeamSearch);
};

}  // namespace wenet

#endif  // DECODER_CTC_PREFIX_BEAM_SEARCH_H_
