// Copyright (c) 2021 Ximalaya Speech Team (Xiang Lyu)
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

#include "frontend/wav.h"
#include "grpc/grpc_client.h"
#include "utils/flags.h"
#include "utils/timer.h"

// 设置服务端和客户端的信息
DEFINE_string(hostname, "127.0.0.1", "hostname of websocket server");
DEFINE_int32(port, 10086, "port of websocket server");
DEFINE_int32(nbest, 1, "n-best of decode result");
DEFINE_string(wav_path, "", "test wav file path");
DEFINE_bool(continuous_decoding, false, "continuous decoding mode");

int main(int argc, char* argv[]) {
  // 初始化数据
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  // 初始化日志系统
  google::InitGoogleLogging(argv[0]);
  // 初始化客户端
  wenet::GrpcClient client(FLAGS_hostname, FLAGS_port, FLAGS_nbest,
                           FLAGS_continuous_decoding);

  // 读取音频文件
  wenet::WavReader wav_reader(FLAGS_wav_path);
  const int sample_rate = 16000;
  // Only support 16K检查采样率
  CHECK_EQ(wav_reader.sample_rate(), sample_rate);
  const int num_samples = wav_reader.num_samples();
  // 指向要复制的元素范围的起始和结束位置
  std::vector<float> pcm_data(wav_reader.data(),
                              wav_reader.data() + num_samples);
  // Send data every 0.5 second，0.5秒发一次信息
  const float interval = 0.5;
  const int sample_interval = interval * sample_rate;
  for (int start = 0; start < num_samples; start += sample_interval) {
    if (client.done()) {
      break;
    }
    int end = std::min(start + sample_interval, num_samples);
    // Convert to short，转为短数据
    std::vector<int16_t> data;
    data.reserve(end - start);
    for (int j = start; j < end; j++) {
      data.push_back(static_cast<int16_t>(pcm_data[j]));
    }
    // Send PCM data，发送字节流
    client.SendBinaryData(data.data(), data.size() * sizeof(int16_t));
    VLOG(2) << "Send " << data.size() << " samples";
    std::this_thread::sleep_for(
        std::chrono::milliseconds(static_cast<int>(interval * 1000)));
  }
  wenet::Timer timer;

  client.Join();
  VLOG(2) << "Total latency: " << timer.Elapsed() << "ms.";
  return 0;
}

// 总结：使用wenet框架的grpc通信的GRPC客户端的工作流程