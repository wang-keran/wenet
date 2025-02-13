// Copyright (c) 2021 Mobvoi Inc (Binbin Zhang)
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

// 基于CTC的，可以给每个转换分配权重的启发图搜索（比贪心多读进来几个数据比较选最好的）解码器

#include "decoder/ctc_wfst_beam_search.h"

#include <utility>

namespace wenet {

// 重置
void DecodableTensorScaled::Reset() {
  num_frames_ready_ = 0;
  done_ = false;
  // Give an empty initialization, will throw error when
  // AcceptLoglikes is not called
  logp_.clear();
}

// 接收一个对数概率向量logp，并将其存储在成员变量logp_中。同时，增加num_frames_ready_以记录已接收的帧数。
void DecodableTensorScaled::AcceptLoglikes(const std::vector<float>& logp) {
  ++num_frames_ready_;
  // TODO(Binbin Zhang): Avoid copy here
  logp_ = logp;
}

// 根据帧号和索引返回对应的对数似然值，考虑了缩放因子scale_
float DecodableTensorScaled::LogLikelihood(int32 frame, int32 index) {
  CHECK_GT(index, 0);
  CHECK_LT(frame, num_frames_ready_);
  return scale_ * logp_[index - 1];
}

// 判断给定的帧是否是最后一帧
bool DecodableTensorScaled::IsLastFrame(int32 frame) const {
  CHECK_LT(frame, num_frames_ready_);
  return done_ && (frame == num_frames_ready_ - 1);
}

// 当前实现中抛出了一个致命错误，表明这个方法尚未实现。
int32 DecodableTensorScaled::NumIndices() const {
  LOG(FATAL) << "Not implement";
  return 0;
}

// 构造函数
CtcWfstBeamSearch::CtcWfstBeamSearch(
    const fst::Fst<fst::StdArc>& fst, const CtcWfstBeamSearchOptions& opts,
    const std::shared_ptr<ContextGraph>& context_graph)
    : decodable_(opts.acoustic_scale),
      decoder_(fst, opts, context_graph),
      context_graph_(context_graph),
      opts_(opts) {
  Reset();
}

// 重置解码器
void CtcWfstBeamSearch::Reset() {
  num_frames_ = 0;
  decoded_frames_mapping_.clear();
  is_last_frame_blank_ = false;
  last_best_ = 0;
  inputs_.clear();
  outputs_.clear();
  likelihood_.clear();
  times_.clear();
  decodable_.Reset();
  decoder_.InitDecoding();
}

// 接收一系列的对数概率向量（每帧一个），并逐帧进行解码。
// 它根据空白帧的阈值决定是否跳过某些帧，并维护了一个decoded_frames_mapping_来记录哪些帧被实际解码。最后，它获取最佳路径，并准备输出。
void CtcWfstBeamSearch::Search(const std::vector<std::vector<float>>& logp) {
  // 如果对数概率矩阵中没有对数概率，直接返回,检查有效性
  if (0 == logp.size()) {
    return;
  }
  // Every time we get the log posterior, we decode it all before return
  // 遍历每一帧的对数概率
  for (int i = 0; i < logp.size(); i++) {
    // 计算当前帧中空白符号的概率分布.logp[i][opts_.blank]是对数概率，使用std::exp幂指数ex将其转换为线性概率。
    float blank_score = std::exp(logp[i][opts_.blank]);
  // 如果空白符号的分数大于设定的阈值,则跳过当前帧,opts_.blank_skip_thresh * opts_.blank_scale都是设置好的，0.98和1在顶上的构造函数中
    if (blank_score > opts_.blank_skip_thresh * opts_.blank_scale) {
      // 记录日志信息，指示跳过了哪一帧及其分数。
      VLOG(3) << "skipping frame " << num_frames_ << " score " << blank_score;
      // 设置标志is_last_frame_blank_为true，表示上一帧是空白帧。
      is_last_frame_blank_ = true;
      // 将当前帧的对数概率保存到last_frame_prob_，以便后续可能使用。
      last_frame_prob_ = logp[i];
    } else {
      // 如果当前帧不是空白帧，则进行正常的解码操作。不跳过当前帧，则找到当前帧的最佳符号（即具有最高对数概率的符号）。
      // Get the best symbol
      // std::max_element返回指向最大元素的迭代器，减去起始迭代器得到索引。迭代器是什么？减掉是不是就是索引位置，相当于下标数字
      // 是不是相当于从每个帧里挑出数字最大的值的角标？
      int cur_best =
          std::max_element(logp[i].begin(), logp[i].end()) - logp[i].begin();
      // Optional, adding one blank frame if we has skipped it in two same
      // symbols
      // 如果当前最佳符号不是空白符号，并且上一帧是空白帧且与不为空的前一帧的最佳符号相同，则插入一个空白帧：
      if (cur_best != opts_.blank && is_last_frame_blank_ &&
          cur_best == last_best_) {
        // 将上一帧的对数概率传递给解码器。
        decodable_.AcceptLoglikes(last_frame_prob_);
        // 推进解码器状态。
        decoder_.AdvanceDecoding(&decodable_, 1);
        // 更新已解码帧映射，记录插入的空白帧。
        decoded_frames_mapping_.push_back(num_frames_ - 1);
        // 记录日志信息，指示在哪个符号处添加了空白帧。
        VLOG(2) << "Adding blank frame at symbol " << cur_best;
      }
      // 更新last_best_为当前最佳符号。
      last_best_ = cur_best;

      // 将当前帧的对数概率传递给解码器。
      decodable_.AcceptLoglikes(logp[i]);
      // 推进解码器状态。
      decoder_.AdvanceDecoding(&decodable_, 1);
      // 更新已解码帧映射，记录当前帧。
      decoded_frames_mapping_.push_back(num_frames_);
      // 设置标志is_last_frame_blank_为false，表示当前帧不是空白帧。
      is_last_frame_blank_ = false;
    }
    // 每处理完一帧后，增加全局帧计数num_frames_。
    num_frames_++;
  }
  // Get the best path
  // 清空之前存储的输入序列、输出序列和似然值，确保在处理新结果时不会保留旧数据。
  inputs_.clear();
  outputs_.clear();
  likelihood_.clear();
  // 如果已解码帧映射不为空，则说明至少有一帧被解码，可以获取最佳路径。如果没有解码任何帧，则直接跳过后续处理。
  if (decoded_frames_mapping_.size() > 0) {
    // 如果有已解码的帧，则将inputs_、outputs_和likelihood_的大小调整为1，表示只处理一个结果（即最佳路径）。
    inputs_.resize(1);
    outputs_.resize(1);
    likelihood_.resize(1);
    // 创建一个kaldi::Lattice对象lat。
    kaldi::Lattice lat;
    // 调用decoder_.GetBestPath(&lat, true)从解码器中获取最佳路径，并将其存储在lat中。第二个参数true通常表示是否需要详细信息或清理资源。
    decoder_.GetBestPath(&lat, true);
    std::vector<int> alignment;
    // 创建一个kaldi::LatticeWeight类型的weight来存储路径权重。
    kaldi::LatticeWeight weight;
    fst::GetLinearSymbolSequence(lat, &alignment, &outputs_[0], &weight);
    ConvertToInputs(alignment, &inputs_[0]);
    VLOG(3) << weight.Value1() << " " << weight.Value2();
    likelihood_[0] = -(weight.Value1() + weight.Value2());
  }
}

// 在完成所有帧的搜索后，调用此方法以获取最佳路径或N-best路径。
// 它使用解码器的GetBestPath或GetLattice方法，并将结果转换为对齐的输入序列和输出序列，以及对应的似然值。
void CtcWfstBeamSearch::FinalizeSearch() {
  decodable_.SetFinish();
  decoder_.FinalizeDecoding();
  inputs_.clear();
  outputs_.clear();
  likelihood_.clear();
  times_.clear();
  if (decoded_frames_mapping_.size() > 0) {
    std::vector<kaldi::Lattice> nbest_lats;
    if (opts_.nbest == 1) {
      kaldi::Lattice lat;
      decoder_.GetBestPath(&lat, true);
      nbest_lats.push_back(std::move(lat));
    } else {
      // Get N-best path by lattice(CompactLattice)
      kaldi::CompactLattice clat;
      decoder_.GetLattice(&clat, true);
      kaldi::Lattice lat, nbest_lat;
      fst::ConvertLattice(clat, &lat);
      // TODO(Binbin Zhang): it's n-best word lists here, not character n-best
      fst::ShortestPath(lat, &nbest_lat, opts_.nbest);
      fst::ConvertNbestToVector(nbest_lat, &nbest_lats);
    }
    int nbest = nbest_lats.size();
    inputs_.resize(nbest);
    outputs_.resize(nbest);
    likelihood_.resize(nbest);
    times_.resize(nbest);
    for (int i = 0; i < nbest; i++) {
      kaldi::LatticeWeight weight;
      std::vector<int> alignment;
      fst::GetLinearSymbolSequence(nbest_lats[i], &alignment, &outputs_[i],
                                   &weight);
      ConvertToInputs(alignment, &inputs_[i], &times_[i]);
      likelihood_[i] = -(weight.Value1() + weight.Value2());
    }
  }
}

// 将WFST解码器输出的对齐序列转换为输入序列（忽略空白帧和连续相同的标签）。如果提供了time参数，还会记录每个输入对应的时间帧。
void CtcWfstBeamSearch::ConvertToInputs(const std::vector<int>& alignment,
                                        std::vector<int>* input,
                                        std::vector<int>* time) {
  input->clear();
  if (time != nullptr) time->clear();
  for (int cur = 0; cur < alignment.size(); ++cur) {
    // ignore blank
    if (alignment[cur] - 1 == opts_.blank) continue;
    // merge continuous same label
    if (cur > 0 && alignment[cur] == alignment[cur - 1]) continue;

    input->push_back(alignment[cur] - 1);
    if (time != nullptr) {
      time->push_back(decoded_frames_mapping_[cur]);
    }
  }
}

}  // namespace wenet
