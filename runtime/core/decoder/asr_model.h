// Copyright 2022 Binbin Zhang (binbzha@qq.com)

#ifndef DECODER_ASR_MODEL_H_
#define DECODER_ASR_MODEL_H_

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "utils/timer.h"
#include "utils/utils.h"

namespace wenet {

class AsrModel {
 public:
  // 获取右侧上下文
  virtual int right_context() const { return right_context_; }
  // 获取下采样率
  virtual int subsampling_rate() const { return subsampling_rate_; }
  // 获取开始位置，结束位置
  virtual int sos() const { return sos_; }
  virtual int eos() const { return eos_; }
  // 判断是否为双向上下文解码
  virtual bool is_bidirectional_decoder() const {
    return is_bidirectional_decoder_;
  }
  // 获取offset。一个用于指示数据块或数据片段在完整数据流中位置的偏移量。
  virtual int offset() const { return offset_; }

  // If chunk_size > 0, streaming case. Otherwise, none streaming case
  // 块大小大于0是流式，否则非流式
  virtual void set_chunk_size(int chunk_size) { chunk_size_ = chunk_size; }
  virtual void set_num_left_chunks(int num_left_chunks) {
    num_left_chunks_ = num_left_chunks;
  }
  // start: if it is the start chunk of one sentence
  virtual int num_frames_for_chunk(bool start) const;

  virtual void Reset() = 0;

  virtual void ForwardEncoder(
      const std::vector<std::vector<float>>& chunk_feats,
      std::vector<std::vector<float>>* ctc_prob);

  virtual void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                  float reverse_weight,
                                  std::vector<float>* rescoring_score) = 0;

  virtual std::shared_ptr<AsrModel> Copy() const = 0;

 protected:
  virtual void ForwardEncoderFunc(
      const std::vector<std::vector<float>>& chunk_feats,
      std::vector<std::vector<float>>* ctc_prob) = 0;
  virtual void CacheFeature(const std::vector<std::vector<float>>& chunk_feats);

  int right_context_ = 1;
  int subsampling_rate_ = 1;
  int sos_ = 0;
  int eos_ = 0;
  bool is_bidirectional_decoder_ = false;
  int chunk_size_ = 16;
  int num_left_chunks_ = -1;  // -1 means all left chunks
  int offset_ = 0;

  std::vector<std::vector<float>> cached_feature_;
};

}  // namespace wenet

#endif  // DECODER_ASR_MODEL_H_
