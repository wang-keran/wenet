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

#ifndef DECODER_CTC_WFST_BEAM_SEARCH_H_
#define DECODER_CTC_WFST_BEAM_SEARCH_H_

#include <memory>
#include <vector>

#include "decoder/context_graph.h"
#include "decoder/search_interface.h"
#include "kaldi/decoder/lattice-faster-online-decoder.h"
#include "utils/utils.h"

namespace wenet {

// 是一个解码器接口的实现。它允许对解码过程中的对数概率进行缩放，并提供了一系列方法用于解码操作：
class DecodableTensorScaled : public kaldi::DecodableInterface {
 public:
  explicit DecodableTensorScaled(float scale = 1.0) : scale_(scale) { Reset(); }

  void Reset();
  // NumFramesReady(): 返回当前准备好的帧数。
  int32 NumFramesReady() const override { return num_frames_ready_; }
  // IsLastFrame(int32 frame): 判断给定的帧是否是最后一帧。
  bool IsLastFrame(int32 frame) const override;
  // 返回给定帧和索引位置的对数似然值。
  float LogLikelihood(int32 frame, int32 index) override;
  // 返回索引的数量。
  int32 NumIndices() const override;
  // 接受一帧的对数概率。
  void AcceptLoglikes(const std::vector<float>& logp);
  // 标记解码过程完成。
  void SetFinish() { done_ = true; }

 private:
  int num_frames_ready_ = 0;
  float scale_ = 1.0;
  bool done_ = false;
  std::vector<float> logp_;
};

// 用于执行基于WFST的束搜索解码。它接受一个WFST、解码选项和一个上下文图作为输入，并提供了一系列方法用于执行搜索和获取结果：
// LatticeFasterDecoderConfig has the following key members
// beam: decoding beam
// max_active: Decoder max active states
// lattice_beam: Lattice generation beam
struct CtcWfstBeamSearchOptions : public kaldi::LatticeFasterDecoderConfig {
  float acoustic_scale = 1.0;
  float nbest = 10;
  // When blank score is greater than this thresh, skip the frame in viterbi
  // search
  float blank_skip_thresh = 0.98;
  float blank_scale = 1.0;
  int blank = 0;
};

class CtcWfstBeamSearch : public SearchInterface {
 public:
  explicit CtcWfstBeamSearch(
      const fst::Fst<fst::StdArc>& fst, const CtcWfstBeamSearchOptions& opts,
      const std::shared_ptr<ContextGraph>& context_graph);
  // 对给定的对数概率序列执行搜索。
  void Search(const std::vector<std::vector<float>>& logp) override;
  // 重置准备下一轮解码
  void Reset() override;
  // 完成搜索过程，准备输出结果。
  void FinalizeSearch() override;
  // Type(): 返回搜索类型。
  SearchType Type() const override { return SearchType::kWfstBeamSearch; }
  // For CTC prefix beam search, both inputs and outputs are hypotheses_
  // Inputs(), Outputs(), Likelihood(), Times():
  // 分别返回输入序列、输出序列、似然值和时间戳。
  const std::vector<std::vector<int>>& Inputs() const override {
    return inputs_;
  }
  const std::vector<std::vector<int>>& Outputs() const override {
    return outputs_;
  }
  const std::vector<float>& Likelihood() const override { return likelihood_; }
  const std::vector<std::vector<int>>& Times() const override { return times_; }

 private:
  // Sub one and remove <blank>,将对齐转换为输入序列，并可选地返回时间信息。
  void ConvertToInputs(const std::vector<int>& alignment,
                       std::vector<int>* input,
                       std::vector<int>* time = nullptr);

  int num_frames_ = 0;
  std::vector<int> decoded_frames_mapping_;

  int last_best_ = 0;  // last none blank best id
  std::vector<float> last_frame_prob_;
  bool is_last_frame_blank_ = false;
  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<float> likelihood_;
  std::vector<std::vector<int>> times_;
  DecodableTensorScaled decodable_;
  kaldi::LatticeFasterOnlineDecoder decoder_;
  std::shared_ptr<ContextGraph> context_graph_;
  const CtcWfstBeamSearchOptions& opts_;
};

}  // namespace wenet

#endif  // DECODER_CTC_WFST_BEAM_SEARCH_H_
