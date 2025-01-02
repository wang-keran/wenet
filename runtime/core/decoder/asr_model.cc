// Copyright 2022 Binbin Zhang (binbzha@qq.com)

#include "decoder/asr_model.h"

#include <memory>
#include <utility>

namespace wenet {
// 用于计算在处理语音识别（ASR，Automatic Speech Recognition）模型的某个数据块（chunk）时所需的帧数。
// 这个方法考虑了数据块的尺寸（chunk_size_）、子采样率（subsampling_rate_）以及右上下文（right_context_）等因素。

int AsrModel::num_frames_for_chunk(bool start) const {
  int num_required_frames = 0;
  if (chunk_size_ > 0) {
    if (!start) {                        // First batch
      int context = right_context_ + 1;  // Add current frame
      num_required_frames = (chunk_size_ - 1) * subsampling_rate_ + context;
    } else {
      num_required_frames = chunk_size_ * subsampling_rate_;
    }
  } else {
    num_required_frames = std::numeric_limits<int>::max();
  }
  return num_required_frames;
}

// 为了缓存特征，以便为下一个数据块（chunk）准备。这个函数接收一个二维浮点型向量
// chunk_feats，该向量包含了当前数据块的特征。
void AsrModel::CacheFeature(
    const std::vector<std::vector<float>>& chunk_feats) {
  // Cache feature for next chunk
  const int cached_feature_size = 1 + right_context_ - subsampling_rate_;
  if (chunk_feats.size() >= cached_feature_size) {
    // TODO(Binbin Zhang): Only deal the case when
    // chunk_feats.size() > cached_feature_size here, and it's consistent
    // with our current model, refine it later if we have new model or
    // new requirements
    cached_feature_.resize(cached_feature_size);
    for (int i = 0; i < cached_feature_size; ++i) {
      cached_feature_[i] =
          chunk_feats[chunk_feats.size() - cached_feature_size + i];
    }
  }
}

// 它负责将输入的特征（chunk_feats）通过编码器前向传播，并计算连接主义时间分类（CTC）的概率（ctc_prob）。
// 这个函数同时利用了之前缓存的特征（cached_feature_）来增强对当前数据块（chunk_feats）的处理。以下是对该函数的详细解释：
void AsrModel::ForwardEncoder(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>>* ctc_prob) {
  ctc_prob->clear();
  int num_frames = cached_feature_.size() + chunk_feats.size();
  if (num_frames >= right_context_ + 1) {
    this->ForwardEncoderFunc(chunk_feats, ctc_prob);
    this->CacheFeature(chunk_feats);
  }
}

}  // namespace wenet
// 总结：输入模型前的处理，比如切块，缓存特征等