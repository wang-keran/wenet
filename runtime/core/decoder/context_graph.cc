// Copyright (c) 2021 Mobvoi Inc (Zhendong Peng)
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

#include "decoder/context_graph.h"

#include <fstream>
#include <queue>
#include <utility>

#include "fst/determinize.h"

#include "utils/string.h"
#include "utils/utils.h"

namespace wenet {

// Split the UTF-8 string into unit ids according to
// unit_table.将输入的字符串根据模块表拆分成块
bool SplitContextToUnits(const std::string& context,
                         const std::shared_ptr<fst::SymbolTable>& unit_table,
                         std::vector<int>* units) {
  std::vector<std::string> chars;
  SplitUTF8StringToChars(context, &chars);

  bool no_oov = true;
  bool beginning = true;
  for (size_t start = 0; start < chars.size();) {
    for (size_t end = chars.size(); end > start; --end) {
      std::string unit;
      for (size_t i = start; i < end; i++) {
        unit += chars[i];
      }
      // Add '▁' at the beginning of English word.
      // TODO(zhendong.peng): Support bpe model
      if (IsAlpha(unit) && beginning) {
        unit = kSpaceSymbol + unit;
      }

      int unit_id = unit_table->Find(unit);
      if (unit_id != -1) {
        units->emplace_back(unit_id);
        start = end;
        beginning = false;
        continue;
      }

      if (end == start + 1) {
        // Matching using '▁' separately for English
        if (unit[0] == kSpaceSymbol[0]) {
          units->emplace_back(unit_table->Find(kSpaceSymbol));
          beginning = false;
          break;
        }
        ++start;
        if (unit == " ") {
          beginning = true;
          continue;
        }
        no_oov = false;
        LOG(WARNING) << unit << " is oov.";
      }
    }
  }
  return no_oov;
}
// 初始化ContextGraph类的对象
ContextGraph::ContextGraph(ContextConfig config) : config_(config) {}

// 用于在给定的当前状态和单元ID下，追踪（或遍历）上下文图（context
// graph），并返回下一个状态。
// 如果达到最终状态，还会通过指针参数返回最终状态。
int ContextGraph::TraceContext(int cur_state, int unit_id, int* final_state) {
  CHECK_GE(cur_state, 0);
  int next_state = 0;
  Matcher matcher(*graph_, fst::MATCH_INPUT);
  matcher.SetState(cur_state);
  if (matcher.Find(unit_id)) {
    next_state = matcher.Value().nextstate;
    if (graph_->Final(next_state) != Weight::Zero()) {
      *final_state = next_state;
    }
    return next_state;
  }
  LOG(FATAL) << "Trace context failed.";
}

// 根据提供的上下文短语列表和单元表，构建一个上下文图并转换为AC（自动机）
// 自动机：AC自动机，全称为Aho-Corasick自动机，是一种用于字符串搜索的算法，由Alfred
// V. Aho和Margaret J. Corasick在1975年提出。
// 该算法主要用于在一个主文本字符串中查找多个模式字符串（或称为“关键词”），并且可以在线性时间内完成搜索，非常高效。
// AC自动机的核心思想是基于Trie树（字典树）构建，并结合了KMP算法的思想。具体来说，它在Trie树的每个节点上添加了一个“失配指针”（fail指针），这些
// 指针允许在查找过程中遇到不匹配时跳转到其他可能匹配的位置，从而避免重复计算。
void ContextGraph::BuildContextGraph(
    const std::vector<std::string>& contexts,
    const std::shared_ptr<fst::SymbolTable>& unit_table) {
  // Split context phrase into unit ids according to the `unit_table`
  std::unordered_map<std::string, std::vector<int>> context_units;
  for (const auto& context : contexts) {
    std::vector<int> units;
    bool no_oov = SplitContextToUnits(context, unit_table, &units);
    if (!no_oov) {
      LOG(WARNING) << "Ignore unknown unit found during compilation.";
      continue;
    }
    context_units[context] = units;
  }

  // Build the context graph,构建上下文图
  std::unique_ptr<fst::StdVectorFst> ofst(new fst::StdVectorFst());
  int start_state = ofst->AddState();
  ofst->SetStart(start_state);
  for (const auto& context : contexts) {
    if (context_units.count(context) == 0) continue;
    std::vector<int> units = context_units[context];
    int state = start_state;
    int next_state = state;
    for (size_t i = 0; i < units.size(); ++i) {
      next_state = ofst->AddState();
      if (i == units.size() - 1) {
        ofst->SetFinal(next_state, Weight::One());
      }
      float score =
          i * config_.incremental_context_score + config_.context_score;
      ofst->AddArc(state, fst::StdArc(units[i], units[i], score, next_state));
      state = next_state;
    }
  }
  graph_ = std::unique_ptr<fst::StdVectorFst>(new fst::StdVectorFst());
  // input/output label are sorted after Determinize
  fst::Determinize(*ofst, graph_.get());

  // Determinize will change the final state id
  for (const auto& context : contexts) {
    if (context_units.count(context) == 0) continue;
    std::vector<int> units = context_units[context];
    int final_state = -1;
    int cur_state = 0;
    for (int unit : units) {
      cur_state = TraceContext(cur_state, unit, &final_state);
    }
    CHECK_GT(final_state, 0);
    context_table_[final_state] = context;
  }

  // Convert context graph to AC automaton
  ConvertToAC();
}

// 将上下文图转换为自动机
void ContextGraph::ConvertToAC() {
  // 初始化
  CHECK(graph_ != nullptr) << "Context graph should not be nullptr!";
  int num_states = graph_->NumStates();
  std::vector<int> fail_states(num_states, 0);
  std::vector<float> total_weights(num_states, 0);
  Matcher matcher(*graph_, fst::MATCH_INPUT);
  // start state
  fail_states[0] = -1;
  total_weights[0] = 0;

  // Please see:
  // https://web.stanford.edu/group/cslipublications/cslipublications/koskenniemi-festschrift/9-mohri.pdf
  std::queue<int> states_queue;
  states_queue.push(0);
  while (!states_queue.empty()) {
    int state = states_queue.front();
    states_queue.pop();

    // 广度优先搜索（BFS）
    for (ArcIterator aiter(*graph_, state); !aiter.Done(); aiter.Next()) {
      const fst::StdArc& arc = aiter.Value();
      int next_state = arc.nextstate;
      total_weights[next_state] = total_weights[state] + arc.weight.Value();
      // Backtracking the failure state for
      // next_state,回溯失败状态以获得下一个状态
      for (int fail_state = fail_states[state]; fail_state != -1;
           fail_state = fail_states[fail_state]) {
        matcher.SetState(fail_state);
        if (matcher.Find(arc.ilabel)) {
          fail_states[next_state] = matcher.Value().nextstate;
          break;
        }
      }
      states_queue.push(next_state);
    }
  }

  // Compute fail weight, add fail arc,计算失败权重并添加失败弧
  for (int state = 0; state < num_states; state++) {
    int fail_state = fail_states[state];
    if (fail_state < 0) continue;
    if (graph_->Final(fail_state) != Weight::Zero()) {
      fallback_finals_[state] = fail_state;
      if (graph_->NumArcs(fail_state) == 0) continue;
    }
    if (graph_->Final(state) != Weight::Zero() && fail_state == 0) continue;

    float fail_weight = total_weights[fail_state] - total_weights[state];
    if (graph_->Final(state) != Weight::Zero()) {
      fail_weight = 0;
    }
    graph_->AddArc(state, fst::StdArc(0, 0, fail_weight, fail_state));
  }
  // Sort arcs by ilabel, means move the fallback arc from last to first for the
  // matcher,排序弧
  fst::ArcSort(graph_.get(), fst::ILabelCompare<fst::StdArc>());
}

// 根据当前状态和单元ID找到下一个状态，并计算得分。
int ContextGraph::GetNextState(int cur_state, int unit_id, float* score,
                               std::unordered_set<std::string>* contexts) {
  // 输入检查
  CHECK_GE(cur_state, 0);
  // Find(0) matches any epsilons on the underlying FST explicitly
  CHECK_NE(unit_id, 0);
  int next_state = 0;

  // 初始化
  Matcher matcher(*graph_, fst::MATCH_INPUT);
  matcher.SetState(cur_state);
  // 匹配弧
  if (matcher.Find(unit_id)) {
    const fst::StdArc& arc = matcher.Value();
    next_state = arc.nextstate;
    *score += arc.weight.Value();
    // Collect all contexts in the decode result，收集上下文
    if (contexts != nullptr) {
      if (graph_->Final(next_state) != Weight::Zero()) {
        contexts->insert(context_table_[next_state]);
      }
      int fallback_final = next_state;
      while (fallback_finals_.count(fallback_final) > 0) {
        fallback_final = fallback_finals_[fallback_final];
        contexts->insert(context_table_[fallback_final]);
      }
    }

    // Leaves go back to the start state，处理无弧状态
    if (graph_->NumArcs(next_state) == 0) {
      return 0;
    }
    // 最终返回 next_state，即下一个状态。
    return next_state;
  }

  // Check whether the first arc is fallback arc，回退弧处理
  ArcIterator aiter(*graph_, cur_state);
  const fst::StdArc& arc = aiter.Value();
  // The start state has no fallback arc
  if (arc.ilabel == 0) {
    next_state = arc.nextstate;
    *score += arc.weight.Value();
    // fallback，回退
    return GetNextState(next_state, unit_id, score);
  }

  return 0;
}

}  // namespace wenet

// 总结：构建图转自动机，获得得分为了接下来的运算，不算decoder的一部分，但是是decoder的基础