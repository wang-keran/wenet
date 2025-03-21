// Copyright (c) 2023 Ximalaya Speech Team (Xiang Lyu)
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

#include "decoder/params.h"
#include "http/http_server.h"
#include "utils/log.h"

DEFINE_int32(port, 10086, "http listening port");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  // 初始化解码选项
  auto decode_config = wenet::InitDecodeOptionsFromFlags();
  // 初始化特征处理管道
  auto feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  // 初始化解码资源（模型）
  auto decode_resource = wenet::InitDecodeResourceFromFlags();

  wenet::HttpServer server(FLAGS_port, feature_config, decode_config,
                           decode_resource);
  LOG(INFO) << "Listening at port " << FLAGS_port;
  server.Start();
  return 0;
}
