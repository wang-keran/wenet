// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
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

#include <iomanip>
#include <thread>
#include <utility>

#include "decoder/params.h"
#include "frontend/wav.h"
#include "utils/flags.h"
#include "utils/string.h"
#include "utils/thread_pool.h"
#include "utils/timer.h"
#include "utils/utils.h"

// 初始化各种数据
// 设置是否为流式输入，参数为simulate_streaming，默认false，最后是解释
DEFINE_bool(simulate_streaming, false, "simulate streaming input");
DEFINE_bool(output_nbest, false, "output n-best of decode result");
DEFINE_string(wav_path, "", "single wave path");
DEFINE_string(wav_scp, "", "input wav scp");
DEFINE_string(result, "", "result output file");
// 是否连续解码
DEFINE_bool(continuous_decoding, false, "continuous decoding mode");
DEFINE_int32(thread_num, 1, "num of decode thread");
DEFINE_int32(warmup, 0, "num of warmup decode, 0 means no warmup");

std::shared_ptr<wenet::DecodeOptions> g_decode_config;
std::shared_ptr<wenet::FeaturePipelineConfig> g_feature_config;
std::shared_ptr<wenet::DecodeResource> g_decode_resource;

std::ofstream g_result;
std::mutex g_mutex;
int g_total_waves_dur = 0;
int g_total_decode_time = 0;

// 解码
void Decode(std::pair<std::string, std::string> wav, bool warmup = false) {
  // 读取音频文件和验证音频文件的采样率
  wenet::WavReader wav_reader(wav.second);
  int num_samples = wav_reader.num_samples();
  CHECK_EQ(wav_reader.sample_rate(), FLAGS_sample_rate);

  // 创建音频管道进行特征提取
  auto feature_pipeline =
      std::make_shared<wenet::FeaturePipeline>(*g_feature_config);
  feature_pipeline->AcceptWaveform(wav_reader.data(), num_samples);
  feature_pipeline->set_input_finished();
  LOG(INFO) << "num frames " << feature_pipeline->num_frames();

  // 初始化解码器
  wenet::AsrDecoder decoder(feature_pipeline, g_decode_resource,
                            *g_decode_config);

  // 设置输出结果
  int wave_dur = static_cast<int>(static_cast<float>(num_samples) /
                                  wav_reader.sample_rate() * 1000);
  int decode_time = 0;
  std::string final_result;
  // 循环解码
  while (true) {
    wenet::Timer timer;
    wenet::DecodeState state = decoder.Decode();
    // 表示所有特征已处理完毕，此时进行注意力重评分（Rescoring）。
    if (state == wenet::DecodeState::kEndFeats) {
      decoder.Rescoring();
    }
    // 获取解码时间
    int chunk_decode_time = timer.Elapsed();
    decode_time += chunk_decode_time;
    // 检测出非空句子，将结果存储到日志中
    if (decoder.DecodedSomething()) {
      LOG(INFO) << "Partial result: " << decoder.result()[0].sentence;
    }

    // 检查是否启用了连续解码模式并检测到端点（流式解码）
    if (FLAGS_continuous_decoding && state == wenet::DecodeState::kEndpoint) {
      if (decoder.DecodedSomething()) {
        // 重打分，打印结果，将结果添加到总结果后面
        decoder.Rescoring();
        LOG(INFO) << "Final result (continuous decoding): "
                  << decoder.result()[0].sentence;
        final_result.append(decoder.result()[0].sentence);
      }
      // 重置解码器
      decoder.ResetContinuousDecoding();
    }

    // 如果所有特征已处理完毕（即解码器已经处理完所有音频特征），则退出解码循环。
    if (state == wenet::DecodeState::kEndFeats) {
      break;
    }
    // 检查是否启用模拟流式输入 
    else if (FLAGS_chunk_size > 0 && FLAGS_simulate_streaming) {
      // 计算帧移的时间长度
      float frame_shift_in_ms =
          static_cast<float>(g_feature_config->frame_shift) /
          wav_reader.sample_rate() * 1000;
      // 计算等待时间
      auto wait_time =
          decoder.num_frames_in_current_chunk() * frame_shift_in_ms -
          chunk_decode_time;
        // 如果计算出的等待时间大于 0，则记录日志并让当前线程休眠指定的时间。通过休眠模拟真实的流式输入延迟，使解码过程更接近实际的流式处理场景。
      if (wait_time > 0) {
        LOG(INFO) << "Simulate streaming, waiting for " << wait_time << "ms";
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>(wait_time)));
      }
    }
  }
  // 在解码完成后，检查是否有最终解码结果，并将这些结果追加到 final_result 字符串中。
  if (decoder.DecodedSomething()) {
    final_result.append(decoder.result()[0].sentence);
  }
  LOG(INFO) << wav.first << " Final result: " << final_result << std::endl;
  LOG(INFO) << "Decoded " << wave_dur << "ms audio taken " << decode_time
            << "ms.";

  // 在非预热解码完成后，将解码结果输出到指定文件或控制台，并更新全局统计信息。
  if (!warmup) {
    g_mutex.lock();
    std::ostream& buffer = FLAGS_result.empty() ? std::cout : g_result;
    if (!FLAGS_output_nbest) {
      buffer << wav.first << " " << final_result << std::endl;
    } else {
      buffer << "wav " << wav.first << std::endl;
      auto& results = decoder.result();
      for (auto& r : results) {
        if (r.sentence.empty()) continue;
        buffer << "candidate " << r.score << " " << r.sentence << std::endl;
      }
    }
    g_total_waves_dur += wave_dur;
    g_total_decode_time += decode_time;
    g_mutex.unlock();
  }
}

int main(int argc, char* argv[]) {
  // 解析输入参数，初始化日志
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  // 初始化解码选项配置，初始化特征提取配置，初始化解码资源
  g_decode_config = wenet::InitDecodeOptionsFromFlags();
  g_feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  g_decode_resource = wenet::InitDecodeResourceFromFlags();

  // 判断是否有音频文件
  if (FLAGS_wav_path.empty() && FLAGS_wav_scp.empty()) {
    LOG(FATAL) << "Please provide the wave path or the wav scp.";
  }
  // 初始化音频文件列表
  std::vector<std::pair<std::string, std::string>> waves;
  // 处理单个音频文件路径
  if (!FLAGS_wav_path.empty()) {
    waves.emplace_back(make_pair("test", FLAGS_wav_path));
  }
  // 处理 wav_scp 音频列表文件 
  else {
    std::ifstream wav_scp(FLAGS_wav_scp);
    std::string line;
    while (getline(wav_scp, line)) {
      std::vector<std::string> strs;
      wenet::SplitString(line, &strs);
      CHECK_GE(strs.size(), 2);
      waves.emplace_back(make_pair(strs[0], strs[1]));
    }

    // 如果音频列表为空，报错
    if (waves.empty()) {
      LOG(FATAL) << "Please provide non-empty wav scp.";
    }
  }

  // 打开结果文件
  if (!FLAGS_result.empty()) {
    g_result.open(FLAGS_result, std::ios::out);
  }

  // Warmup预热代码，提前加载模型等资源，避免冷启动带来的性能波动
  if (FLAGS_warmup > 0) {
    LOG(INFO) << "Warming up...";
    {
      ThreadPool pool(FLAGS_thread_num);
      auto wav = waves[0];
      for (int i = 0; i < FLAGS_warmup; i++) {
        pool.enqueue(Decode, wav, true);
      }
    }
    LOG(INFO) << "Warmup done.";
  }

  // 创建线程池进行多线程解码
  {
    ThreadPool pool(FLAGS_thread_num);
    // 提交解码任务到线程池
    for (auto& wav : waves) {
      pool.enqueue(Decode, wav, false);
    }
  }

  // 打印解码统计信息
  LOG(INFO) << "Total: decoded " << g_total_waves_dur << "ms audio taken "
            << g_total_decode_time << "ms.";
  LOG(INFO) << "RTF: " << std::setprecision(4)
            << static_cast<float>(g_total_decode_time) / g_total_waves_dur;
  return 0;
}
