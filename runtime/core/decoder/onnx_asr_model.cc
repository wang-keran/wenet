// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 ZeXuan Li (lizexuan@huya.com)
//                    Xingchen Song(sxc19@mails.tsinghua.edu.cn)
//                    hamddct@gmail.com (Mddct)
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

#include "decoder/onnx_asr_model.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "utils/string.h"

namespace wenet {

Ort::Env OnnxAsrModel::env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "");
Ort::SessionOptions OnnxAsrModel::session_options_ = Ort::SessionOptions();

// 这个方法设置了会话选项中的
// IntraOpNumThreads，即单个操作内部可以并行执行的线程数。这有助于在具有多核处理器的系统上提高性能。
void OnnxAsrModel::InitEngineThreads(int num_threads) {
  session_options_.SetIntraOpNumThreads(num_threads);
}

// 获取输入输出信息。这个方法接受一个 Ort::Session 的智能指针和两个指向
// std::vector<const char*> 的指针，用于存储输入和输出节点的名称。
void OnnxAsrModel::GetInputOutputInfo(
    const std::shared_ptr<Ort::Session>& session,
    std::vector<std::string>* in_names, std::vector<std::string>* out_names) {
  Ort::AllocatorWithDefaultOptions allocator;
  // Input info
  int num_nodes = session->GetInputCount();
  in_names->resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    Ort::AllocatedStringPtr in_name_ptr =
        session->GetInputNameAllocated(i, allocator);
    Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> node_dims = tensor_info.GetShape();
    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tInput " << i << " : name=" << in_name_ptr.get()
              << " type=" << type << " dims=" << shape.str();
    (*in_names)[i] = std::string(in_name_ptr.get());
  }
  // Output info
  num_nodes = session->GetOutputCount();
  out_names->resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    Ort::AllocatedStringPtr out_name_ptr =
        session->GetOutputNameAllocated(i, allocator);
    Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> node_dims = tensor_info.GetShape();
    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tOutput " << i << " : name=" << out_name_ptr.get()
              << " type=" << type << " dims=" << shape.str();
    (*out_names)[i] = std::string(out_name_ptr.get());
  }
}

// 加载ONNX模型并读取其元数据
void OnnxAsrModel::Read(const std::string& model_dir) {
  std::string encoder_onnx_path = model_dir + "/encoder.onnx";
  std::string rescore_onnx_path = model_dir + "/decoder.onnx";
  std::string ctc_onnx_path = model_dir + "/ctc.onnx";

  // 1. Load sessions
  try {
#ifdef _MSC_VER
    encoder_session_ = std::make_shared<Ort::Session>(
        env_, ToWString(encoder_onnx_path).c_str(), session_options_);
    rescore_session_ = std::make_shared<Ort::Session>(
        env_, ToWString(rescore_onnx_path).c_str(), session_options_);
    ctc_session_ = std::make_shared<Ort::Session>(
        env_, ToWString(ctc_onnx_path).c_str(), session_options_);
#else
    encoder_session_ = std::make_shared<Ort::Session>(
        env_, encoder_onnx_path.c_str(), session_options_);
    rescore_session_ = std::make_shared<Ort::Session>(
        env_, rescore_onnx_path.c_str(), session_options_);
    ctc_session_ = std::make_shared<Ort::Session>(env_, ctc_onnx_path.c_str(),
                                                  session_options_);
#endif
  } catch (std::exception const& e) {
    LOG(ERROR) << "error when load onnx model: " << e.what();
    exit(0);
  }

  // 2. Read metadata
  auto model_metadata = encoder_session_->GetModelMetadata();

  Ort::AllocatorWithDefaultOptions allocator;
  encoder_output_size_ = atoi(
      model_metadata.LookupCustomMetadataMapAllocated("output_size", allocator)
          .get());
  num_blocks_ = atoi(
      model_metadata.LookupCustomMetadataMapAllocated("num_blocks", allocator)
          .get());
  head_ = atoi(
      model_metadata.LookupCustomMetadataMapAllocated("head", allocator).get());
  cnn_module_kernel_ =
      atoi(model_metadata
               .LookupCustomMetadataMapAllocated("cnn_module_kernel", allocator)
               .get());
  subsampling_rate_ =
      atoi(model_metadata
               .LookupCustomMetadataMapAllocated("subsampling_rate", allocator)
               .get());
  right_context_ =
      atoi(model_metadata
               .LookupCustomMetadataMapAllocated("right_context", allocator)
               .get());
  sos_ = atoi(
      model_metadata.LookupCustomMetadataMapAllocated("sos_symbol", allocator)
          .get());
  eos_ = atoi(
      model_metadata.LookupCustomMetadataMapAllocated("eos_symbol", allocator)
          .get());
  is_bidirectional_decoder_ =
      atoi(model_metadata
               .LookupCustomMetadataMapAllocated("is_bidirectional_decoder",
                                                 allocator)
               .get());
  chunk_size_ = atoi(
      model_metadata.LookupCustomMetadataMapAllocated("chunk_size", allocator)
          .get());
  num_left_chunks_ = atoi(
      model_metadata.LookupCustomMetadataMapAllocated("left_chunks", allocator)
          .get());

  LOG(INFO) << "Onnx Model Info:";
  LOG(INFO) << "\tencoder_output_size " << encoder_output_size_;
  LOG(INFO) << "\tnum_blocks " << num_blocks_;
  LOG(INFO) << "\thead " << head_;
  LOG(INFO) << "\tcnn_module_kernel " << cnn_module_kernel_;
  LOG(INFO) << "\tsubsampling_rate " << subsampling_rate_;
  LOG(INFO) << "\tright_context " << right_context_;
  LOG(INFO) << "\tsos " << sos_;
  LOG(INFO) << "\teos " << eos_;
  LOG(INFO) << "\tis bidirectional decoder " << is_bidirectional_decoder_;
  LOG(INFO) << "\tchunk_size " << chunk_size_;
  LOG(INFO) << "\tnum_left_chunks " << num_left_chunks_;

  // 3. Read model nodes
  LOG(INFO) << "Onnx Encoder:";
  GetInputOutputInfo(encoder_session_, &encoder_in_names_, &encoder_out_names_);
  LOG(INFO) << "Onnx CTC:";
  GetInputOutputInfo(ctc_session_, &ctc_in_names_, &ctc_out_names_);
  LOG(INFO) << "Onnx Rescore:";
  GetInputOutputInfo(rescore_session_, &rescore_in_names_, &rescore_out_names_);
}

// 构造函数初始化了一些成员变量，包括 encoder_output_size_、num_blocks_、head_、
OnnxAsrModel::OnnxAsrModel(const OnnxAsrModel& other) {
  // metadatas
  encoder_output_size_ = other.encoder_output_size_;
  num_blocks_ = other.num_blocks_;
  head_ = other.head_;
  cnn_module_kernel_ = other.cnn_module_kernel_;
  right_context_ = other.right_context_;
  subsampling_rate_ = other.subsampling_rate_;
  sos_ = other.sos_;
  eos_ = other.eos_;
  is_bidirectional_decoder_ = other.is_bidirectional_decoder_;
  chunk_size_ = other.chunk_size_;
  num_left_chunks_ = other.num_left_chunks_;
  offset_ = other.offset_;

  // sessions
  encoder_session_ = other.encoder_session_;
  ctc_session_ = other.ctc_session_;
  rescore_session_ = other.rescore_session_;

  // node names
  encoder_in_names_ = other.encoder_in_names_;
  encoder_out_names_ = other.encoder_out_names_;
  ctc_in_names_ = other.ctc_in_names_;
  ctc_out_names_ = other.ctc_out_names_;
  rescore_in_names_ = other.rescore_in_names_;
  rescore_out_names_ = other.rescore_out_names_;
}

// 创建当前 OnnxAsrModel 对象的一个副本，并返回一个指向该副本的
// std::shared_ptr。
std::shared_ptr<AsrModel> OnnxAsrModel::Copy() const {
  auto asr_model = std::make_shared<OnnxAsrModel>(*this);
  // Reset the inner states for new decoding
  asr_model->Reset();
  return asr_model;
}

// 重置状态来准备重新decoder
void OnnxAsrModel::Reset() {
  offset_ = 0;
  encoder_outs_.clear();
  cached_feature_.clear();
  // Reset att_cache
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  if (num_left_chunks_ > 0) {
    int required_cache_size = chunk_size_ * num_left_chunks_;
    offset_ = required_cache_size;
    att_cache_.resize(num_blocks_ * head_ * required_cache_size *
                          encoder_output_size_ / head_ * 2,
                      0.0);
    const int64_t att_cache_shape[] = {num_blocks_, head_, required_cache_size,
                                       encoder_output_size_ / head_ * 2};
    att_cache_ort_ = Ort::Value::CreateTensor<float>(
        memory_info, att_cache_.data(), att_cache_.size(), att_cache_shape, 4);
  } else {
    att_cache_.resize(0, 0.0);
    const int64_t att_cache_shape[] = {num_blocks_, head_, 0,
                                       encoder_output_size_ / head_ * 2};
    att_cache_ort_ = Ort::Value::CreateTensor<float>(
        memory_info, att_cache_.data(), att_cache_.size(), att_cache_shape, 4);
  }

  // Reset cnn_cache
  cnn_cache_.resize(
      num_blocks_ * encoder_output_size_ * (cnn_module_kernel_ - 1), 0.0);
  const int64_t cnn_cache_shape[] = {num_blocks_, 1, encoder_output_size_,
                                     cnn_module_kernel_ - 1};
  cnn_cache_ort_ = Ort::Value::CreateTensor<float>(
      memory_info, cnn_cache_.data(), cnn_cache_.size(), cnn_cache_shape, 4);
}

// 准备输入数据并将其传递给 ONNX 模型进行推理。
void OnnxAsrModel::ForwardEncoderFunc(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>>* out_prob) {
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  // 1. Prepare onnx required data, splice cached_feature_ and chunk_feats
  // chunk
  int num_frames = cached_feature_.size() + chunk_feats.size();
  const int feature_dim = chunk_feats[0].size();
  std::vector<float> feats;
  for (size_t i = 0; i < cached_feature_.size(); ++i) {
    feats.insert(feats.end(), cached_feature_[i].begin(),
                 cached_feature_[i].end());
  }
  for (size_t i = 0; i < chunk_feats.size(); ++i) {
    feats.insert(feats.end(), chunk_feats[i].begin(), chunk_feats[i].end());
  }
  const int64_t feats_shape[3] = {1, num_frames, feature_dim};
  Ort::Value feats_ort = Ort::Value::CreateTensor<float>(
      memory_info, feats.data(), feats.size(), feats_shape, 3);
  // offset
  int64_t offset_int64 = static_cast<int64_t>(offset_);
  Ort::Value offset_ort = Ort::Value::CreateTensor<int64_t>(
      memory_info, &offset_int64, 1, std::vector<int64_t>{}.data(), 0);
  // required_cache_size
  int64_t required_cache_size = chunk_size_ * num_left_chunks_;
  Ort::Value required_cache_size_ort = Ort::Value::CreateTensor<int64_t>(
      memory_info, &required_cache_size, 1, std::vector<int64_t>{}.data(), 0);
  // att_mask
  Ort::Value att_mask_ort{nullptr};
  std::vector<uint8_t> att_mask(required_cache_size + chunk_size_, 1);
  if (num_left_chunks_ > 0) {
    int chunk_idx = offset_ / chunk_size_ - num_left_chunks_;
    if (chunk_idx < num_left_chunks_) {
      for (int i = 0; i < (num_left_chunks_ - chunk_idx) * chunk_size_; ++i) {
        att_mask[i] = 0;
      }
    }
    const int64_t att_mask_shape[] = {1, 1, required_cache_size + chunk_size_};
    att_mask_ort = Ort::Value::CreateTensor<bool>(
        memory_info, reinterpret_cast<bool*>(att_mask.data()), att_mask.size(),
        att_mask_shape, 3);
  }

  // 2. Encoder chunk forward
  std::vector<Ort::Value> inputs;
  for (auto name : encoder_in_names_) {
    if (!strcmp(name.c_str(), "chunk")) {
      inputs.emplace_back(std::move(feats_ort));
    } else if (!strcmp(name.c_str(), "offset")) {
      inputs.emplace_back(std::move(offset_ort));
    } else if (!strcmp(name.c_str(), "required_cache_size")) {
      inputs.emplace_back(std::move(required_cache_size_ort));
    } else if (!strcmp(name.c_str(), "att_cache")) {
      inputs.emplace_back(std::move(att_cache_ort_));
    } else if (!strcmp(name.c_str(), "cnn_cache")) {
      inputs.emplace_back(std::move(cnn_cache_ort_));
    } else if (!strcmp(name.c_str(), "att_mask")) {
      inputs.emplace_back(std::move(att_mask_ort));
    }
  }

  // Convert std::vector<std::string> to std::vector<const char*> for using
  // C-style strings
  std::vector<const char*> encoder_in_names(encoder_in_names_.size());
  std::vector<const char*> encoder_out_names(encoder_out_names_.size());
  std::transform(encoder_in_names_.begin(), encoder_in_names_.end(),
                 encoder_in_names.begin(),
                 [](const std::string& name) { return name.c_str(); });
  std::transform(encoder_out_names_.begin(), encoder_out_names_.end(),
                 encoder_out_names.begin(),
                 [](const std::string& name) { return name.c_str(); });

  std::vector<Ort::Value> ort_outputs = encoder_session_->Run(
      Ort::RunOptions{nullptr}, encoder_in_names.data(), inputs.data(),
      inputs.size(), encoder_out_names.data(), encoder_out_names.size());

  offset_ += static_cast<int>(
      ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1]);
  att_cache_ort_ = std::move(ort_outputs[1]);
  cnn_cache_ort_ = std::move(ort_outputs[2]);

  std::vector<Ort::Value> ctc_inputs;
  ctc_inputs.emplace_back(std::move(ort_outputs[0]));

  // Convert std::vector<std::string> to std::vector<const char*> for using
  // C-style strings
  std::vector<const char*> ctc_in_names(ctc_in_names_.size());
  std::vector<const char*> ctc_out_names(ctc_out_names_.size());
  std::transform(ctc_in_names_.begin(), ctc_in_names_.end(),
                 ctc_in_names.begin(),
                 [](const std::string& name) { return name.c_str(); });
  std::transform(ctc_out_names_.begin(), ctc_out_names_.end(),
                 ctc_out_names.begin(),
                 [](const std::string& name) { return name.c_str(); });

  std::vector<Ort::Value> ctc_ort_outputs = ctc_session_->Run(
      Ort::RunOptions{nullptr}, ctc_in_names.data(), ctc_inputs.data(),
      ctc_inputs.size(), ctc_out_names.data(), ctc_out_names.size());
  encoder_outs_.push_back(std::move(ctc_inputs[0]));

  float* logp_data = ctc_ort_outputs[0].GetTensorMutableData<float>();
  auto type_info = ctc_ort_outputs[0].GetTensorTypeAndShapeInfo();

  int num_outputs = type_info.GetShape()[1];
  int output_dim = type_info.GetShape()[2];
  out_prob->resize(num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    (*out_prob)[i].resize(output_dim);
    memcpy((*out_prob)[i].data(), logp_data + i * output_dim,
           sizeof(float) * output_dim);
  }
}

// 计算注意力分数。该函数通过遍历假设序列（hyp）中的每个元素，并将其对应的概率值累加到
// score 中。最后，它还会加上结束符号（eos）的概率值。
float OnnxAsrModel::ComputeAttentionScore(const float* prob,
                                          const std::vector<int>& hyp, int eos,
                                          int decode_out_len) {
  float score = 0.0f;
  for (size_t j = 0; j < hyp.size(); ++j) {
    score += *(prob + j * decode_out_len + hyp[j]);
  }
  score += *(prob + hyp.size() * decode_out_len + eos);
  return score;
}

// 对输入的假设（hyps）进行重评分。
// 该函数首先检查输入参数的有效性，然后对输入的假设进行预处理，包括计算假设的长度、准备重评分的输入数据，并最终调用
// ONNX Runtime 进行重评分。
void OnnxAsrModel::AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                      float reverse_weight,
                                      std::vector<float>* rescoring_score) {
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  CHECK(rescoring_score != nullptr);
  int num_hyps = hyps.size();
  rescoring_score->resize(num_hyps, 0.0f);

  if (num_hyps == 0) {
    return;
  }
  // No encoder output
  if (encoder_outs_.size() == 0) {
    return;
  }

  std::vector<int64_t> hyps_lens;
  int max_hyps_len = 0;
  for (size_t i = 0; i < num_hyps; ++i) {
    int length = hyps[i].size() + 1;
    max_hyps_len = std::max(length, max_hyps_len);
    hyps_lens.emplace_back(static_cast<int64_t>(length));
  }

  std::vector<float> rescore_input;
  int encoder_len = 0;
  for (int i = 0; i < encoder_outs_.size(); i++) {
    float* encoder_outs_data = encoder_outs_[i].GetTensorMutableData<float>();
    auto type_info = encoder_outs_[i].GetTensorTypeAndShapeInfo();
    for (int j = 0; j < type_info.GetElementCount(); j++) {
      rescore_input.emplace_back(encoder_outs_data[j]);
    }
    encoder_len += type_info.GetShape()[1];
  }

  const int64_t decode_input_shape[] = {1, encoder_len, encoder_output_size_};

  std::vector<int64_t> hyps_pad;

  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hyps[i];
    hyps_pad.emplace_back(sos_);
    size_t j = 0;
    for (; j < hyp.size(); ++j) {
      hyps_pad.emplace_back(hyp[j]);
    }
    if (j == max_hyps_len - 1) {
      continue;
    }
    for (; j < max_hyps_len - 1; ++j) {
      hyps_pad.emplace_back(0);
    }
  }

  const int64_t hyps_pad_shape[] = {num_hyps, max_hyps_len};

  const int64_t hyps_lens_shape[] = {num_hyps};

  Ort::Value decode_input_tensor_ = Ort::Value::CreateTensor<float>(
      memory_info, rescore_input.data(), rescore_input.size(),
      decode_input_shape, 3);
  Ort::Value hyps_pad_tensor_ = Ort::Value::CreateTensor<int64_t>(
      memory_info, hyps_pad.data(), hyps_pad.size(), hyps_pad_shape, 2);
  Ort::Value hyps_lens_tensor_ = Ort::Value::CreateTensor<int64_t>(
      memory_info, hyps_lens.data(), hyps_lens.size(), hyps_lens_shape, 1);

  std::vector<Ort::Value> rescore_inputs;

  rescore_inputs.emplace_back(std::move(hyps_pad_tensor_));
  rescore_inputs.emplace_back(std::move(hyps_lens_tensor_));
  rescore_inputs.emplace_back(std::move(decode_input_tensor_));

  // Convert std::vector<std::string> to std::vector<const char*> for using
  // C-style strings
  std::vector<const char*> rescore_in_names(rescore_in_names_.size());
  std::vector<const char*> rescore_out_names(rescore_out_names_.size());
  std::transform(rescore_in_names_.begin(), rescore_in_names_.end(),
                 rescore_in_names.begin(),
                 [](const std::string& name) { return name.c_str(); });
  std::transform(rescore_out_names_.begin(), rescore_out_names_.end(),
                 rescore_out_names.begin(),
                 [](const std::string& name) { return name.c_str(); });

  std::vector<Ort::Value> rescore_outputs =
      rescore_session_->Run(Ort::RunOptions{nullptr}, rescore_in_names.data(),
                            rescore_inputs.data(), rescore_inputs.size(),
                            rescore_out_names.data(), rescore_out_names.size());

  float* decoder_outs_data = rescore_outputs[0].GetTensorMutableData<float>();
  float* r_decoder_outs_data = rescore_outputs[1].GetTensorMutableData<float>();

  auto type_info = rescore_outputs[0].GetTensorTypeAndShapeInfo();
  int decode_out_len = type_info.GetShape()[2];

  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hyps[i];
    float score = 0.0f;
    // left to right decoder score
    score = ComputeAttentionScore(
        decoder_outs_data + max_hyps_len * decode_out_len * i, hyp, eos_,
        decode_out_len);
    // Optional: Used for right to left score
    float r_score = 0.0f;
    if (is_bidirectional_decoder_ && reverse_weight > 0) {
      std::vector<int> r_hyp(hyp.size());
      std::reverse_copy(hyp.begin(), hyp.end(), r_hyp.begin());
      // right to left decoder score
      r_score = ComputeAttentionScore(
          r_decoder_outs_data + max_hyps_len * decode_out_len * i, r_hyp, eos_,
          decode_out_len);
    }
    // combined left-to-right and right-to-left score
    (*rescoring_score)[i] =
        score * (1 - reverse_weight) + r_score * reverse_weight;
  }
}

}  // namespace wenet

// 总结：处理语音识别的onnx模型类