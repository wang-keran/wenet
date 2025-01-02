// Copyright (c) 2017 Personal (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "frontend/feature_pipeline.h"

#include <algorithm>
#include <utility>

namespace wenet {

// 初始化
FeaturePipeline::FeaturePipeline(const FeaturePipelineConfig& config)
    : config_(config),
      feature_dim_(config.num_bins),
      fbank_(config.num_bins, config.sample_rate, config.frame_length,
             config.frame_shift, config.low_freq, config.pre_emphasis,
             config.scale_input_to_unit, config.log_floor, config.log_base,
             config.window_type, config.mel_type, config.norm_type),
      num_frames_(0),
      input_finished_(false) {}

// 接收浮点型音频波形数据，并将其转换为特征向量。
void FeaturePipeline::AcceptWaveform(const float* pcm, const int size) {
  std::vector<std::vector<float>> feats;
  std::vector<float> waves;
  waves.insert(waves.end(), remained_wav_.begin(), remained_wav_.end());
  waves.insert(waves.end(), pcm, pcm + size);
  int num_frames = fbank_.Compute(waves, &feats);
  feature_queue_.Push(std::move(feats));
  num_frames_ += num_frames;

  int left_samples = waves.size() - config_.frame_shift * num_frames;
  remained_wav_.resize(left_samples);
  std::copy(waves.begin() + config_.frame_shift * num_frames, waves.end(),
            remained_wav_.begin());
  // We are still adding wave, notify input is not finished
  finish_condition_.notify_one();
}

// 接收16位整型音频波形数据，并将其转换为浮点型后传递给AcceptWaveform(const
// float* pcm, const int size)方法。
void FeaturePipeline::AcceptWaveform(const int16_t* pcm, const int size) {
  auto* float_pcm = new float[size];
  for (size_t i = 0; i < size; i++) {
    float_pcm[i] = static_cast<float>(pcm[i]);
  }
  this->AcceptWaveform(float_pcm, size);
  delete[] float_pcm;
}

// 该方法用于标记输入数据处理完成，并通知其他线程。
void FeaturePipeline::set_input_finished() {
  CHECK(!input_finished_);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    input_finished_ = true;
  }
  finish_condition_.notify_one();
}

// 从 feature_queue_ 中读取一个特征向量。
// 如果队列不为空，则直接返回队列中的特征向量；如果队列为空，则线程会等待，直到队列中有新的特征向量被添加或输入结束。
bool FeaturePipeline::ReadOne(std::vector<float>* feat) {
  if (!feature_queue_.Empty()) {
    *feat = std::move(feature_queue_.Pop());
    return true;
  } else {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!input_finished_) {
      // This will release the lock and wait for notify_one()
      // from AcceptWaveform() or set_input_finished()
      finish_condition_.wait(lock);
      if (!feature_queue_.Empty()) {
        *feat = std::move(feature_queue_.Pop());
        return true;
      }
    }
    CHECK(input_finished_);
    // Double check queue.empty, see issue#893 for detailed discussions.
    if (!feature_queue_.Empty()) {
      *feat = std::move(feature_queue_.Pop());
      return true;
    } else {
      return false;
    }
  }
}

// 从 feature_queue_
// 中读取指定数量的帧数据。该函数首先检查队列中是否有足够的帧数据，如果没有，则进入等待状态，直到队列中有足够的帧数据或输入结束
bool FeaturePipeline::Read(int num_frames,
                           std::vector<std::vector<float>>* feats) {
  feats->clear();
  if (feature_queue_.Size() >= num_frames) {
    *feats = std::move(feature_queue_.Pop(num_frames));
    return true;
  } else {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!input_finished_) {
      // This will release the lock and wait for notify_one()
      // from AcceptWaveform() or set_input_finished()
      finish_condition_.wait(lock);
      if (feature_queue_.Size() >= num_frames) {
        *feats = std::move(feature_queue_.Pop(num_frames));
        return true;
      }
    }
    CHECK(input_finished_);
    // Double check queue.empty, see issue#893 for detailed discussions.
    if (feature_queue_.Size() >= num_frames) {
      *feats = std::move(feature_queue_.Pop(num_frames));
      return true;
    } else {
      *feats = std::move(feature_queue_.Pop(feature_queue_.Size()));
      return false;
    }
  }
}

// 重置管线避免受到影响
void FeaturePipeline::Reset() {
  input_finished_ = false;
  num_frames_ = 0;
  remained_wav_.clear();
  feature_queue_.Clear();
}

}  // namespace wenet
// 总结：音频预处理，先转特征向量后读取